# TODO: DeepGen Warm-Start SFT + LLM Ablation (No RL)

Goal: Test whether augmenting the SFT model with an LLM replaces the need for RL.

Scope:
- Warm-start directly from the trained DeepGen SFT checkpoint.
- Train the lightweight zero-initialized LLM adapter on streaming multi-task data (T2I + Editing).
- Exclude Stage C / RL entirely.

Cluster target (BIU Slurm):
- Public partition baseline: `A100-4h`.
- Runtime limit: 4 hours per run (job is suspended/requeued by BIU system at limit).
- Public GPU cap: 2 GPUs per user job on this partition (`--gres=gpu:2`).
- Auto-resume built into `jobs/sft_llm_ablation.sbatch`.

---

## 0) Experiment Setup

- [x] Implement warm-start model architecture (`DeepGenSFTLLMAdapter`).
- [x] Implement HF streaming dataset adapters (`HFStreamingT2IDataset`, `HFStreamingEditingDataset`, `HFStreamingJointDataset`).
- [x] Configure fine-tuning driver and configs.
- [x] Create Slurm submission script (`jobs/sft_llm_ablation.sbatch`).
- [x] Clean up obsolete files and deprecated ablation scripts.

---

## 1) Launch SFT + LLM Fine-Tuning Job

- [x] Submit the job:
  ```bash
  cd /home/dsi/davidpo/projects/Semi/deepgen
  sbatch jobs/sft_llm_ablation.sbatch
  ```
- [x] Monitor logs: `tail -f logs/ablation_sft_llm/sft_*.out`
- [x] Checkpoint verification: ensure periodic checkpoints save in `work_dirs/sft_llm_ablation/` (Completed 50k steps: `iter_50000.pth`).

---

## 2) Evaluation & Comparison Against RL Baseline

- [x] Evaluate final checkpoint on T2I benchmarks (GenEval, DPGBench, UniGenBench, WISE, T2I-CoREBench).
- [x] Evaluate final checkpoint on Image Editing benchmarks (ImgEdit, GEdit, RISE, UniREditBench).
- [x] Compare metrics against official published DeepGen 1.0 (RL) checkpoint.
- [x] Compile final ablation report (`REPORT.md`).
