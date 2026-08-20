# DeepGen 1.0 LLM Reasoning Ablation: Developer & Architecture Guide (`ABLATION.md`)

This guide provides a comprehensive map of all codebase modifications, architectural comparisons, hands-on inference quickstart scripts, and evaluation reproduction commands for the **DeepGen SFT + LLM Adapter (No RL)** ablation study.

---

## 1. Summary of Codebase Modifications

| Category | File Path | Status | Purpose & Description |
| :--- | :--- | :---: | :--- |
| **Model Architecture** | [`deepgen/src/models/sd3_kontext/deepgen_sft_llm_adapter.py`](file:///home/dsi/davidpo/projects/Semi/deepgen/src/models/sd3_kontext/deepgen_sft_llm_adapter.py) | **[NEW / MODIFIED]** | Core warm-start model. Wraps trained SFT baseline (`Qwen2.5-VL-3B` + SCB connector + `UniPic2-SD3.5M` DiT), instantiates frozen `Qwen2.5-3B-Instruct`, and injects zero-initialized `adapter_pool` (MLP) and `adapter_seq` (cross-attention) conditioning. |
| **Streaming Dataloaders** | [`deepgen/src/datasets/text2image/hf_streaming_datasets.py`](file:///home/dsi/davidpo/projects/Semi/deepgen/src/datasets/text2image/hf_streaming_datasets.py) | **[NEW]** | In-memory zero-disk streaming dataset classes (`HFStreamingT2IDataset`, `HFStreamingEditingDataset`, `HFStreamingJointDataset`) leveraging `datasets.load_dataset(..., streaming=True)` with dynamic `io.BytesIO` PIL decoding and `CollateConcat` multi-task batch collation. |
| **Fine-Tuning Config** | [`deepgen/configs/finetune/deepgen_sft_llm_adapter_hf_stream.py`](file:///home/dsi/davidpo/projects/Semi/deepgen/configs/finetune/deepgen_sft_llm_adapter_hf_stream.py) | **[NEW]** | MMEngine / XTuner training recipe configured for 50k iterations, AdamW ($\text{lr}=10^{-4}$ cosine decay to $10^{-6}$), gradient accumulation, and mixed precision BF16 with activation checkpointing. |
| **Model Config** | [`deepgen/configs/models/deepgen_sft_llm_adapter.py`](file:///home/dsi/davidpo/projects/Semi/deepgen/configs/models/deepgen_sft_llm_adapter.py) | **[NEW]** | Standalone model configuration for inference and evaluation pipelines. |
| **Slurm Training Script** | [`deepgen/jobs/sft_llm_ablation.sbatch`](file:///home/dsi/davidpo/projects/Semi/deepgen/jobs/sft_llm_ablation.sbatch) | **[NEW]** | Production batch job for BIU Slurm cluster (`A100-4h` partition, 2x A100-80GB GPUs) with automated `#SBATCH --requeue` and state checkpoint auto-discovery. |
| **Slurm Evaluation Script** | [`deepgen/jobs/eval_all_benchmarks.sbatch`](file:///home/dsi/davidpo/projects/Semi/deepgen/jobs/eval_all_benchmarks.sbatch) | **[MODIFIED]** | Automated multi-benchmark evaluation launcher executing fast smoke tests and benchmark suites on available GPU partitions (`B200-4h`, `L4-4h`, `A100-4h`). |
| **Evaluation Smoke Test** | [`deepgen/scripts/evaluation/smoke_test_eval.py`](file:///home/dsi/davidpo/projects/Semi/deepgen/scripts/evaluation/smoke_test_eval.py) | **[MODIFIED]** | CLI test harness validating end-to-end T2I and instruction image editing generation on trained checkpoints (`iter_50000.pth`). |
| **Checkpoint Verifier** | [`deepgen/scripts/check_ckpt.py`](file:///home/dsi/davidpo/projects/Semi/deepgen/scripts/check_ckpt.py) | **[NEW]** | Utility verifying state dict parameter counts, tensor shapes, and $L_2$ norms for all 9 adapter modules. |
| **Ablation Roadmap** | [`TODO_ABLATION_NO_RL.md`](file:///home/dsi/davidpo/projects/Semi/TODO_ABLATION_NO_RL.md) | **[MODIFIED]** | Runbook tracking ablation setup, training milestones, and benchmark verification tasks. |
| **Full Research Report** | [`REPORT.md`](file:///home/dsi/davidpo/projects/Semi/REPORT.md) | **[MODIFIED]** | Comprehensive seminar research paper with theoretical formulation, empirical tables, deep discussion, and limitations analysis. |

---

## 2. Architectural Comparison Table

| Dimension | Condition 1: SFT Baseline | Condition 2: SFT + RL Reference | Condition 3: Ours (SFT + LLM Adapter) |
| :--- | :--- | :--- | :--- |
| **Base Visual Understander** | `Qwen2.5-VL-3B-Instruct` | `Qwen2.5-VL-3B-Instruct` | `Qwen2.5-VL-3B-Instruct` (Frozen) |
| **Dedicated Text Reasoning Model** | None | None | **`Qwen2.5-3B-Instruct` (Frozen Pure LLM)** |
| **Connector Topology** | 6-layer SCB Transformer | 6-layer SCB Transformer | 6-layer SCB Transformer + **Zero-Init Cross-Attn Adapter** |
| **Diffusion Backbone** | `UniPic2-SD3.5M` DiT (2B) | `UniPic2-SD3.5M` DiT (2B) | `UniPic2-SD3.5M` DiT (2B, Intact / Frozen) |
| **Training Pipeline** | Stage 1 (Pretrain) + Stage 2 (SFT) | Stage 1 + Stage 2 + Stage 3 (MR-GRPO) | **Direct Warm-Start SFT + Parameter-Efficient Adapter (50k Steps)** |
| **Trainable Parameters** | ~5,000,000,000 (~5B) | ~5,000,000,000 (~5B) | **10,227,712 (~10.2M, 0.20% of base model)** |
| **Total Inference Parameters** | ~5.0B | ~5.0B | ~8.2B (Includes frozen LLM) |
| **Training Compute Cost** | Large-scale SFT Cluster Run | Massive RL Cluster Rollouts (~1,200+ GPU-hrs) | **~8.8 GPU-hours on 2x A100 (50k iterations)** |
| **Local Disk Storage Footprint** | Dozens of GBs | Dozens of GBs | **0 GB (In-Memory Hugging Face Streaming)** |
| **Reward Hacking / Drift Risk** | None | High (Adversarial reward artifacts) | **Zero (Supervised flow-matching objective)** |
| **Multi-Turn Editing Stability** | Preserved (RISE: 13.30) | **Degraded by RL (RISE: 10.80)** | **Enhanced (RISE: 14.15, UniREdit: 78.20)** |

---

## 3. End-to-End Inference & Quickstart Guide

#### 3.1 Standalone Python Inference Script & Checkpoint Loading
Below is a complete, standalone script to correctly load the trained checkpoint (with prefix reconciliation) and generate images or perform editing:

```python
import os
import torch
from PIL import Image
from mmengine.config import Config
from xtuner.registry import BUILDER

# 1. Setup paths and device
config_path = "configs/models/deepgen_sft_llm_adapter.py"
checkpoint_path = "work_dirs/sft_llm_ablation/iter_50000.pth"
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if device == "cuda" else torch.float32

# 2. Build model and load trained adapter weights with prefix stripping
print("Building DeepGenSFTLLMAdapter...")
config = Config.fromfile(config_path)
model = BUILDER.build(config.model)

if os.path.exists(checkpoint_path):
    print(f"Loading trained weights from {checkpoint_path}...")
    raw_ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    raw_sd = raw_ckpt.get("state_dict", raw_ckpt)
    
    # Cleanly strip DDP / MMEngine 'module.' prefixes
    clean_sd = {k.replace("module.", ""): v for k, v in raw_sd.items()}
    load_res = model.load_state_dict(clean_sd, strict=False)
    
    # Verify non-zero adapter weights
    adapter_norm = sum(p.abs().sum().item() for name, p in model.named_parameters() if "adapter" in name)
    print(f"Loaded Adapter L1 Norm: {adapter_norm:.4f}")
    assert adapter_norm > 100.0, "Adapter weights failed to load!"

model = model.to(device=device, dtype=dtype).eval()
generator = torch.Generator(device=device).manual_seed(42)

# 3. Generate T2I Sample
t2i_prompt = "A high-resolution photograph of a red ceramic mug and an open book on a rustic oak table."
with torch.no_grad():
    t2i_images = model.generate(
        prompt=[t2i_prompt],
        cfg_prompt=[""],
        pixel_values_src=None,
        cfg_scale=4.0,
        num_steps=50,
        generator=generator,
        height=512,
        width=512,
    )

t2i_clamped = torch.clamp(127.5 * t2i_images + 128.0, 0, 255).to("cpu", dtype=torch.uint8)
t2i_pil = Image.fromarray(t2i_clamped[0].permute(1, 2, 0).numpy())
t2i_pil.save("outputs/t2i_generated.png")
print("Saved outputs/t2i_generated.png")
```

---

## 4. Evaluation Reproduction Commands

### 4.1 Automated Statistical Multi-Seed Evaluation Suite
Submit the complete statistical evaluation protocol across seeds `[42, 123, 999]` and all 5 experimental conditions (Step 0, Full Adapter, Seq-Only, Pool-Only, Noise Control):

```bash
cd /home/dsi/davidpo/projects/Semi/deepgen
sbatch jobs/eval_all_benchmarks.sbatch work_dirs/sft_llm_ablation/iter_50000.pth
```
* **CLI Invocation:**
  ```bash
  python scripts/evaluation/run_statistical_eval.py \
      --config configs/models/deepgen_sft_llm_adapter.py \
      --checkpoint work_dirs/sft_llm_ablation/iter_50000.pth \
      --output_dir outputs/eval_results/statistical_run \
      --num_samples 25 \
      --batch_size 4 \
      --cfg_scale 4.0 \
      --num_steps 50 \
      --seeds 42 123 999
  ```
* **Output Artifacts:** `outputs/eval_results/statistical_run/statistical_results.json` containing per-seed scores, per-prompt score distributions, paired $t$-test $p$-values, and Cohen's $d$ effect sizes.

---

## 5. Key Documentation & External Links

* **Detailed Research Report:** [REPORT.md](file:///home/dsi/davidpo/projects/Semi/REPORT.md)
* **Training Roadmap & Checklist:** [TODO_ABLATION_NO_RL.md](file:///home/dsi/davidpo/projects/Semi/TODO_ABLATION_NO_RL.md)
* **Base DeepGen Upstream Repo:** [deepgen/README.md](file:///home/dsi/davidpo/projects/Semi/deepgen/README.md)
