import os
import sys
import json
import time
import argparse
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from mmengine.config import Config
from xtuner.registry import BUILDER
from xtuner.model.utils import guess_load_checkpoint
from torchvision import transforms
from einops import rearrange

def parse_args():
    parser = argparse.ArgumentParser(description="Comprehensive Benchmark Evaluation Suite for DeepGen SFT + LLM Adapter")
    parser.add_argument("--config", default="configs/models/deepgen_sft_llm_adapter.py", type=str)
    parser.add_argument("--checkpoint", default="work_dirs/sft_llm_ablation/iter_50000.pth", type=str)
    parser.add_argument("--output_dir", default="outputs/eval_results/real_run", type=str)
    parser.add_argument("--num_samples_per_bench", default=50, type=int, help="Number of benchmark samples to evaluate per domain")
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--cfg_scale", default=4.0, type=float)
    parser.add_argument("--num_steps", default=50, type=int)
    parser.add_argument("--seed", default=42, type=int)
    return parser.parse_args()

def load_model(config_path, checkpoint_path, device, dtype):
    print(f"[1/4] Loading model configuration from: {config_path}")
    cfg = Config.fromfile(config_path)
    model = BUILDER.build(cfg.model)

    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"[2/4] Loading trained checkpoint weights: {checkpoint_path}")
        state_dict = guess_load_checkpoint(checkpoint_path)
        load_res = model.load_state_dict(state_dict, strict=False)
        print(f"      Missing keys: {len(load_res.missing_keys)}, Unexpected keys: {len(load_res.unexpected_keys)}")
    else:
        print(f"[WARNING] Checkpoint not found at {checkpoint_path}, running baseline initialization!")

    model = model.to(device=device, dtype=dtype).eval()
    print(f"[3/4] Model successfully loaded on {device} ({dtype})")
    return model

def save_tensor_image(tensor_img, save_path):
    # tensor_img: (C, H, W) in [-1, 1]
    clamped = torch.clamp(127.5 * tensor_img + 128.0, 0, 255).to("cpu", dtype=torch.uint8)
    pil_img = Image.fromarray(clamped.permute(1, 2, 0).numpy())
    pil_img.save(save_path)
    return pil_img

def eval_geneval(model, output_dir, num_samples, batch_size, cfg_scale, num_steps, seed, device, dtype):
    print("\n" + "="*70)
    print(f"--> [EVAL 1/4] Executing GenEval Benchmark ({num_samples} prompts)")
    print("="*70)
    geneval_path = "evaluation/geneval/geneval_prompt.jsonl"
    out_dir = os.path.join(output_dir, "geneval")
    os.makedirs(out_dir, exist_ok=True)

    with open(geneval_path, "r") as f:
        prompts_data = [json.loads(line) for line in f if line.strip()]

    selected = prompts_data[:num_samples]
    generator = torch.Generator(device=device).manual_seed(seed)
    
    categories = {}
    total_generated = 0
    t0 = time.time()

    for i in range(0, len(selected), batch_size):
        batch = selected[i:i+batch_size]
        prompts = [item["prompt"] for item in batch]
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
            tag = item.get("tag", "general")
            save_path = os.path.join(out_dir, f"geneval_{idx:04d}_{tag}.png")
            save_tensor_image(images[j], save_path)

            if tag not in categories:
                categories[tag] = 0
            categories[tag] += 1
            total_generated += 1

    elapsed = time.time() - t0
    print(f"GenEval Complete! Generated {total_generated} samples in {elapsed:.2f}s ({elapsed/max(1,total_generated):.2f}s/sample)")
    print(f"Evaluated Categories: {categories}")
    return {"total_samples": total_generated, "categories": categories, "latency_sec": elapsed}

def eval_dpgbench(model, output_dir, num_samples, batch_size, cfg_scale, num_steps, seed, device, dtype):
    print("\n" + "="*70)
    print(f"--> [EVAL 2/4] Executing DPGBench Benchmark ({num_samples} dense prompts)")
    print("="*70)
    dpg_dir = "evaluation/DPG-Bench/prompts"
    out_dir = os.path.join(output_dir, "dpgbench")
    os.makedirs(out_dir, exist_ok=True)

    files = sorted([f for f in os.listdir(dpg_dir) if f.endswith(".txt")])[:num_samples]
    prompts_data = []
    for f in files:
        with open(os.path.join(dpg_dir, f), "r") as fp:
            prompts_data.append({"id": f.replace(".txt", ""), "prompt": fp.read().strip()})

    generator = torch.Generator(device=device).manual_seed(seed + 1)
    total_generated = 0
    t0 = time.time()

    for i in range(0, len(prompts_data), batch_size):
        batch = prompts_data[i:i+batch_size]
        prompts = [item["prompt"] for item in batch]
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
            idx = item["id"]
            save_path = os.path.join(out_dir, f"dpg_{idx}.png")
            save_tensor_image(images[j], save_path)
            total_generated += 1

    elapsed = time.time() - t0
    print(f"DPGBench Complete! Generated {total_generated} dense prompt images in {elapsed:.2f}s")
    return {"total_samples": total_generated, "latency_sec": elapsed}

def eval_wise(model, output_dir, num_samples, batch_size, cfg_scale, num_steps, seed, device, dtype):
    print("\n" + "="*70)
    print(f"--> [EVAL 3/4] Executing WISE Commonsense Benchmark ({num_samples} prompts)")
    print("="*70)
    wise_path = "evaluation/wise/data/spatio-temporal_reasoning.json"
    out_dir = os.path.join(output_dir, "wise")
    os.makedirs(out_dir, exist_ok=True)

    with open(wise_path, "r") as f:
        wise_data = json.load(f)[:num_samples]

    generator = torch.Generator(device=device).manual_seed(seed + 2)
    total_generated = 0
    t0 = time.time()

    for i in range(0, len(wise_data), batch_size):
        batch = wise_data[i:i+batch_size]
        prompts = [item["Prompt"] for item in batch]
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
            p_id = item.get("prompt_id", i + j)
            save_path = os.path.join(out_dir, f"wise_{p_id}.png")
            save_tensor_image(images[j], save_path)
            total_generated += 1

    elapsed = time.time() - t0
    print(f"WISE Complete! Generated {total_generated} commonsense reasoning images in {elapsed:.2f}s")
    return {"total_samples": total_generated, "latency_sec": elapsed}

def eval_editing(model, output_dir, num_samples, batch_size, cfg_scale, num_steps, seed, device, dtype):
    print("\n" + "="*70)
    print(f"--> [EVAL 4/4] Executing Instruction-Based Image Editing Suite ({num_samples} tasks)")
    print("="*70)
    out_dir = os.path.join(output_dir, "editing")
    os.makedirs(out_dir, exist_ok=True)

    # Benchmark instruction editing cases (source generation -> instruction edit)
    test_cases = [
        {"id": "edit_01", "src_prompt": "A modern sports car parked in a showroom.", "instruction": "Change the sports car into a classic red vintage roadster."},
        {"id": "edit_02", "src_prompt": "A peaceful green forest under bright daylight.", "instruction": "Make it a snowy winter night with a starry sky."},
        {"id": "edit_03", "src_prompt": "A ceramic coffee mug on a wooden desk.", "instruction": "Replace the ceramic coffee mug with an antique brass hourglass."},
        {"id": "edit_04", "src_prompt": "A golden retriever sitting on a green lawn.", "instruction": "Add a pair of sunglasses and a party hat on the dog."},
        {"id": "edit_05", "src_prompt": "A medieval castle on top of a rocky cliff.", "instruction": "Add fire breathing dragons flying around the castle towers."}
    ][:num_samples]

    generator = torch.Generator(device=device).manual_seed(seed + 3)
    src_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    total_generated = 0
    t0 = time.time()

    for item in test_cases:
        # Step A: generate source
        with torch.no_grad():
            src_img = model.generate(
                prompt=[item["src_prompt"]],
                cfg_prompt=[""],
                pixel_values_src=None,
                cfg_scale=cfg_scale,
                num_steps=num_steps,
                generator=generator,
                height=512,
                width=512
            )
        src_save_path = os.path.join(out_dir, f"{item['id']}_src.png")
        pil_src = save_tensor_image(src_img[0], src_save_path)

        # Step B: execute instruction edit conditioned on source
        src_tensor = src_transform(pil_src).unsqueeze(0).to(device=device, dtype=dtype)
        with torch.no_grad():
            edited_img = model.generate(
                prompt=[item["instruction"]],
                cfg_prompt=["blurry, low quality, artifact"],
                pixel_values_src=[src_tensor],
                cfg_scale=cfg_scale,
                num_steps=num_steps,
                generator=generator,
                height=512,
                width=512
            )
        edit_save_path = os.path.join(out_dir, f"{item['id']}_edited.png")
        save_tensor_image(edited_img[0], edit_save_path)
        total_generated += 1

    elapsed = time.time() - t0
    print(f"Editing Complete! Executed {total_generated} paired edit benchmarks in {elapsed:.2f}s")
    return {"total_samples": total_generated, "latency_sec": elapsed}

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    print("="*70)
    print("DEEPGEN SFT + LLM ADAPTER: REAL EMPIRICAL EVALUATION SUITE")
    print(f"Checkpoint:  {args.checkpoint}")
    print(f"Config:      {args.config}")
    print(f"Output Dir:  {args.output_dir}")
    print(f"Device:      {device} ({dtype})")
    print(f"Per-Bench N: {args.num_samples_per_bench}")
    print("="*70)

    model = load_model(args.config, args.checkpoint, device, dtype)

    results = {}
    results["geneval"] = eval_geneval(model, args.output_dir, args.num_samples_per_bench, args.batch_size, args.cfg_scale, args.num_steps, args.seed, device, dtype)
    results["dpgbench"] = eval_dpgbench(model, args.output_dir, args.num_samples_per_bench, args.batch_size, args.cfg_scale, args.num_steps, args.seed, device, dtype)
    results["wise"] = eval_wise(model, args.output_dir, args.num_samples_per_bench, args.batch_size, args.cfg_scale, args.num_steps, args.seed, device, dtype)
    results["editing"] = eval_editing(model, args.output_dir, min(10, args.num_samples_per_bench), 1, args.cfg_scale, args.num_steps, args.seed, device, dtype)

    summary_file = os.path.join(args.output_dir, "eval_summary.json")
    with open(summary_file, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*70)
    print(f"ALL EVALUATIONS SUCCESSFULLY COMPLETED!")
    print(f"Summary metrics saved to: {summary_file}")
    print("="*70)

if __name__ == "__main__":
    main()
