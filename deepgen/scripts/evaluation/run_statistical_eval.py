import os
import sys
import json
import time
import argparse
import math
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from mmengine.config import Config
from xtuner.registry import BUILDER
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel
from scipy import stats

def parse_args():
    parser = argparse.ArgumentParser(description="Multi-Seed Statistical Benchmark Suite & Component Ablation")
    parser.add_argument("--config", default="configs/models/deepgen_sft_llm_adapter.py", type=str)
    parser.add_argument("--checkpoint", default="work_dirs/sft_llm_ablation/iter_50000.pth", type=str)
    parser.add_argument("--output_dir", default="outputs/eval_results/statistical_run", type=str)
    parser.add_argument("--num_samples", default=30, type=int)
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--cfg_scale", default=4.0, type=float)
    parser.add_argument("--num_steps", default=50, type=int)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 999])
    return parser.parse_args()

def load_clean_model(config_path, checkpoint_path, device, dtype, ablation_mode="full"):
    print(f"\n========================================================")
    print(f"Loading Model: Config={config_path}, Ckpt={checkpoint_path}, Mode={ablation_mode}")
    print(f"========================================================")
    cfg = Config.fromfile(config_path)
    model = BUILDER.build(cfg.model)

    resolved_ckpt_path = None
    if checkpoint_path:
        if os.path.exists(checkpoint_path):
            resolved_ckpt_path = checkpoint_path
        else:
            try:
                from huggingface_hub import hf_hub_download
                print(f"Attempting to download adapter checkpoint from Hugging Face Hub: {checkpoint_path}...")
                resolved_ckpt_path = hf_hub_download(
                    repo_id=checkpoint_path,
                    filename="deepgen_sft_llm_adapter_50k.pt"
                )
                print(f"Downloaded Hugging Face Hub weight to: {resolved_ckpt_path}")
            except Exception as e:
                print(f"[WARN] Hugging Face Hub download failed or path not found: {e}")
                resolved_ckpt_path = None

    is_trained_adapter = resolved_ckpt_path and os.path.exists(resolved_ckpt_path)
    
    if is_trained_adapter:
        print(f"Loading state dict from: {resolved_ckpt_path}")
        raw_ckpt = torch.load(resolved_ckpt_path, map_location="cpu", weights_only=False)
        raw_sd = raw_ckpt.get("state_dict", raw_ckpt)
        
        # Cleanly strip 'module.' or 'model.' prefixes from DDP/MMEngine
        clean_sd = {}
        for k, v in raw_sd.items():
            clean_k = k
            if clean_k.startswith("module."):
                clean_k = clean_k[7:]
            if clean_k.startswith("model."):
                clean_k = clean_k[6:]
            if "adapter" in clean_k:
                clean_sd[clean_k] = v
            
        load_res = model.load_state_dict(clean_sd, strict=False)
        print(f"Matched loaded adapter keys count: {len(clean_sd)}, Missing keys in model: {len(load_res.missing_keys)}")
    else:
        print("Initializing Step 0 / Base model (No adapter checkpoint loaded)")

    # Runtime verification of adapter weights
    adapter_norm = sum(p.abs().sum().item() for name, p in model.named_parameters() if "adapter" in name)
    print(f"-> Verified Loaded Adapter Weights L1 Norm: {adapter_norm:.4f}")
    
    if is_trained_adapter:
        assert adapter_norm > 100.0, f"FATAL: Adapter weights L1 norm is {adapter_norm:.4f} <= 100.0. Checkpoint loading failed!"
        print("[SUCCESS] Trained non-zero adapter weights verified successfully!")
    else:
        print("[INFO] Step 0 condition active (Adapter output is zero-initialized identity).")

    # Apply Component Ablation Hooks if requested
    if ablation_mode == "seq_only":
        print("[ABLATION MODE] Sequence Cross-Attention ONLY (adapter_pool output forced to 0)", flush=True)
        def zero_pool_hook(module, args, output):
            return torch.zeros_like(output)
        model.adapter_pool[-1].register_forward_hook(zero_pool_hook)
    elif ablation_mode == "pool_only":
        print("[ABLATION MODE] Pooled MLP ONLY (adapter_seq output forced to 0)", flush=True)
        def zero_seq_hook(module, args, output):
            return torch.zeros_like(output)
        model.adapter_seq.to_out.register_forward_hook(zero_seq_hook)
    elif ablation_mode == "random_noise_control":
        print("[CONTROL ABLATION] Replacing LLM hidden states with Gaussian Noise N(0, 1)", flush=True)
        def noisy_inject(pooled_base, seq_base, prompts):
            text_inputs_llm = model.tokenizer_llm(
                prompts,
                padding=True,
                truncation=True,
                max_length=model.max_length,
                return_tensors="pt"
            ).to(model.device)
            b, l = text_inputs_llm['input_ids'].shape
            h_llm = torch.randn(b, l, 2048, device=model.device, dtype=model.dtype)
            delta_pooled = model.adapter_pool(h_llm.mean(dim=1))
            delta_seq = model.adapter_seq(seq_base, h_llm, context_mask=text_inputs_llm['attention_mask'])
            return pooled_base + delta_pooled, seq_base + delta_seq
        model.inject_llm_reasoning = noisy_inject

    model = model.to(device=device, dtype=dtype).eval()
    return model

def save_tensor_image(tensor_img, save_path):
    clamped = torch.clamp(127.5 * tensor_img + 128.0, 0, 255).to("cpu", dtype=torch.uint8)
    pil_img = Image.fromarray(clamped.permute(1, 2, 0).numpy())
    pil_img.save(save_path)
    return pil_img

def run_benchmark_sweep(model, dataset_type, prompts_list, out_dir, seed, batch_size, cfg_scale, num_steps, device):
    os.makedirs(out_dir, exist_ok=True)
    generator = torch.Generator(device=device).manual_seed(seed)
    images_saved = []

    # Check if all images already exist (resume capability)
    all_exist = True
    for idx, item in enumerate(prompts_list):
        p_id = item.get("id", f"{idx:04d}")
        save_path = os.path.join(out_dir, f"sample_{p_id}_seed{seed}.png")
        if not os.path.exists(save_path):
            all_exist = False
            break

    if all_exist:
        print(f"  [CACHE HIT] All {len(prompts_list)} samples exist in {out_dir}, loading from disk...", flush=True)
        for idx, item in enumerate(prompts_list):
            p_id = item.get("id", f"{idx:04d}")
            save_path = os.path.join(out_dir, f"sample_{p_id}_seed{seed}.png")
            pil_img = Image.open(save_path).convert("RGB")
            images_saved.append({"id": p_id, "text": item["text"], "path": save_path, "img": pil_img})
        return images_saved

    for i in range(0, len(prompts_list), batch_size):
        batch = prompts_list[i:i+batch_size]
        prompts = [item["text"] for item in batch]
        cfg_prompts = [""] * len(prompts)

        with torch.no_grad():
            images = model.generate(
                prompt=prompts,
                cfg_prompt=cfg_prompts,
                pixel_values_src=None,
                cfg_scale=cfg_scale,
                num_steps=num_steps,
                generator=generator,
                height=512,
                width=512
            )

        for j, item in enumerate(batch):
            idx = i + j
            p_id = item.get("id", f"{idx:04d}")
            save_path = os.path.join(out_dir, f"sample_{p_id}_seed{seed}.png")
            pil_img = save_tensor_image(images[j], save_path)
            images_saved.append({"id": p_id, "text": item["text"], "path": save_path, "img": pil_img})

    return images_saved

def compute_clip_scores(clip, proc, items, device):
    scores = []
    for item in items:
        img = item["img"]
        inputs = proc(text=[item["text"][:200]], images=[img], return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            outputs = clip(**inputs)
            sim = (outputs.logits_per_image / 100.0).item()
            scores.append(sim)
    return np.array(scores)

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    print("="*80)
    print("STATISTICALLY RIGOROUS MULTI-SEED BENCHMARK EVALUATION & ABLATION")
    print(f"Trained Checkpoint: {args.checkpoint}")
    print(f"Seeds:              {args.seeds}")
    print(f"Samples per bench:  {args.num_samples}")
    print(f"Device:             {device} ({dtype})")
    print("="*80)

    # 1. Prepare Datasets
    # GenEval
    with open("evaluation/geneval/geneval_prompt.jsonl", "r") as f:
        geneval_raw = [json.loads(l) for l in f if l.strip()][:args.num_samples]
        geneval_data = [{"id": f"geneval_{i:03d}", "text": item["prompt"]} for i, item in enumerate(geneval_raw)]

    # DPGBench
    dpg_dir = "evaluation/DPG-Bench/prompts"
    dpg_files = sorted([f for f in os.listdir(dpg_dir) if f.endswith(".txt")])[:args.num_samples]
    dpg_data = []
    for f in dpg_files:
        with open(os.path.join(dpg_dir, f), "r") as fp:
            dpg_data.append({"id": f"dpg_{f.replace('.txt','')}", "text": fp.read().strip()})

    # WISE
    with open("evaluation/wise/data/spatio-temporal_reasoning.json", "r") as f:
        wise_raw = json.load(f)[:args.num_samples]
        wise_data = [{"id": f"wise_{item.get('prompt_id', i)}", "text": item["Prompt"]} for i, item in enumerate(wise_raw)]

    benchmarks = {
        "GenEval": geneval_data,
        "DPGBench": dpg_data,
        "WISE": wise_data
    }

    # Load CLIP for evaluation scoring
    print("Loading CLIP evaluator (openai/clip-vit-base-patch32)...")
    clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # -------------------------------------------------------------------------
    # PART 1: Multi-Seed Benchmark Evaluation (Condition 1 vs. Condition 2)
    # -------------------------------------------------------------------------
    results_matrix = {}

    conditions = [
        ("Step0_Baseline", "checkpoints/model.pt", "full"),
        ("Trained_Adapter_50k", args.checkpoint, "full"),
        ("Ablation_Seq_Only", args.checkpoint, "seq_only"),
        ("Ablation_Pool_Only", args.checkpoint, "pool_only"),
        ("Control_Random_Noise", args.checkpoint, "random_noise_control")
    ]

    for cond_name, ckpt_path, mode in conditions:
        print(f"\n========================================================", flush=True)
        print(f"EXECUTING EXPERIMENTAL CONDITION: {cond_name}", flush=True)
        print(f"========================================================", flush=True)
        model = load_clean_model(args.config, ckpt_path, device, dtype, ablation_mode=mode)
        
        results_matrix[cond_name] = {}

        for bench_name, dataset in benchmarks.items():
            results_matrix[cond_name][bench_name] = {"seed_scores": [], "per_prompt_scores_by_seed": []}

            for seed in args.seeds:
                out_dir = os.path.join(args.output_dir, cond_name, bench_name, f"seed_{seed}")
                t0 = time.time()
                saved_items = run_benchmark_sweep(model, bench_name, dataset, out_dir, seed, args.batch_size, args.cfg_scale, args.num_steps, device)
                elapsed = time.time() - t0
                
                scores = compute_clip_scores(clip, proc, saved_items, device)
                mean_score = float(np.mean(scores))
                results_matrix[cond_name][bench_name]["seed_scores"].append(mean_score)
                results_matrix[cond_name][bench_name]["per_prompt_scores_by_seed"].append(scores.tolist())
                print(f"  [{cond_name}] {bench_name} (Seed {seed}): Mean Score = {mean_score:.4f} (Time: {elapsed:.2f}s)", flush=True)

        del model
        torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # PART 2: Statistical Significance Analysis (Paired t-tests & Effect Sizes)
    # -------------------------------------------------------------------------
    stats_summary = {}

    for bench_name in benchmarks.keys():
        s0_scores = np.array(results_matrix["Step0_Baseline"][bench_name]["seed_scores"])
        adap_scores = np.array(results_matrix["Trained_Adapter_50k"][bench_name]["seed_scores"])
        seq_scores = np.array(results_matrix["Ablation_Seq_Only"][bench_name]["seed_scores"])
        pool_scores = np.array(results_matrix["Ablation_Pool_Only"][bench_name]["seed_scores"])
        noise_scores = np.array(results_matrix["Control_Random_Noise"][bench_name]["seed_scores"])

        # Flatten per-prompt across all seeds for high-power paired t-test
        s0_all = np.array(results_matrix["Step0_Baseline"][bench_name]["per_prompt_scores_by_seed"]).flatten()
        adap_all = np.array(results_matrix["Trained_Adapter_50k"][bench_name]["per_prompt_scores_by_seed"]).flatten()
        
        t_stat, p_val = stats.ttest_rel(adap_all, s0_all)
        cohen_d = (np.mean(adap_all) - np.mean(s0_all)) / np.std(adap_all - s0_all)

        stats_summary[bench_name] = {
            "Step0_Baseline": {"mean": float(np.mean(s0_scores)), "std": float(np.std(s0_scores))},
            "Trained_Adapter_50k": {"mean": float(np.mean(adap_scores)), "std": float(np.std(adap_scores))},
            "Ablation_Seq_Only": {"mean": float(np.mean(seq_scores)), "std": float(np.std(seq_scores))},
            "Ablation_Pool_Only": {"mean": float(np.mean(pool_scores)), "std": float(np.std(pool_scores))},
            "Control_Random_Noise": {"mean": float(np.mean(noise_scores)), "std": float(np.std(noise_scores))},
            "delta_vs_step0": float(np.mean(adap_scores) - np.mean(s0_scores)),
            "t_statistic": float(t_stat),
            "p_value": float(p_val),
            "cohens_d": float(cohen_d),
            "statistically_significant": bool(p_val < 0.05)
        }

    # Save detailed JSON artifacts
    with open(os.path.join(args.output_dir, "statistical_results.json"), "w") as f:
        json.dump({"raw_runs": results_matrix, "summary_stats": stats_summary}, f, indent=2)

    # -------------------------------------------------------------------------
    # PART 3: Print Publication-Grade Statistical Table
    # -------------------------------------------------------------------------
    print("\n" + "="*100, flush=True)
    print("STATISTICAL BENCHMARK COMPARISON TABLE (Mean ± Std over Seeds [42, 123, 999])", flush=True)
    print("="*100, flush=True)
    print(f"{'Benchmark':<12} | {'Step 0 Baseline':<16} | {'Trained Adapter':<16} | {'Delta':<8} | {'p-value':<12} | {'Cohen d':<8} | {'Sig (p<0.05)?'}", flush=True)
    print("-" * 100, flush=True)
    for bench_name, st in stats_summary.items():
        s0_str = f"{st['Step0_Baseline']['mean']:.4f} ± {st['Step0_Baseline']['std']:.4f}"
        ad_str = f"{st['Trained_Adapter_50k']['mean']:.4f} ± {st['Trained_Adapter_50k']['std']:.4f}"
        delta_str = f"+{st['delta_vs_step0']:.4f}" if st['delta_vs_step0'] >= 0 else f"{st['delta_vs_step0']:.4f}"
        sig_str = "YES (p < 0.05)" if st['statistically_significant'] else "NO"
        print(f"{bench_name:<12} | {s0_str:<16} | {ad_str:<16} | {delta_str:<8} | {st['p_value']:<12.4e} | {st['cohens_d']:<8.3f} | {sig_str}", flush=True)
    
    print("\n" + "="*100, flush=True)
    print("COMPONENT & CONTROL ABLATION TABLE", flush=True)
    print("="*100, flush=True)
    print(f"{'Benchmark':<12} | {'Full Adapter':<16} | {'Seq Only (Cross-Attn)':<22} | {'Pool Only (MLP)':<18} | {'Noise Control':<16}", flush=True)
    print("-" * 100, flush=True)
    for bench_name, st in stats_summary.items():
        ad_str = f"{st['Trained_Adapter_50k']['mean']:.4f}"
        seq_str = f"{st['Ablation_Seq_Only']['mean']:.4f}"
        pool_str = f"{st['Ablation_Pool_Only']['mean']:.4f}"
        noise_str = f"{st['Control_Random_Noise']['mean']:.4f}"
        print(f"{bench_name:<12} | {ad_str:<16} | {seq_str:<22} | {pool_str:<18} | {noise_str:<16}", flush=True)
    print("="*100, flush=True)

if __name__ == "__main__":
    main()
