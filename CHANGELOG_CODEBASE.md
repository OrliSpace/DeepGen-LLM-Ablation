# DeepGen Ablation Study: Codebase Architecture & Modifications Map

This document provides a comprehensive audit of all codebase additions, modifications to original DeepGen framework files, and critical engineering resolutions implemented during this research.

---

## 1. Original DeepGen Repository Files Modified

| File Path | Original Framework Role | Modifications Introduced & Purpose |
| :--- | :--- | :--- |
| [`deepgen/src/models/sd3_kontext/`](file:///home/dsi/davidpo/projects/Semi/deepgen/src/models/sd3_kontext/) (Core DiT / Connector Modules) | Base DeepGen SCB conditioning & DiT forward pipeline | Modified the conditioning forward pass to accept and apply residual adapter deltas ($\Delta y_{\text{pool}}$ into AdaLN blocks and $\Delta c_{\text{seq}}$ into sequence cross-attention) alongside the original base SCB outputs. |
| [`deepgen/src/models/`](file:///home/dsi/davidpo/projects/Semi/deepgen/src/models/) (Model Builder / Registry Interfaces) | Model instantiation & weight loading registries | Registered the new `DeepGenSFTLLMAdapter` class into the MMEngine / XTuner model registry to allow configuration-driven loading without breaking base model pipelines. |
| [`deepgen/evaluation/geneval/`](file:///home/dsi/davidpo/projects/Semi/deepgen/evaluation/geneval/) & [`deepgen/evaluation/wise/`](file:///home/dsi/davidpo/projects/Semi/deepgen/evaluation/wise/) (Evaluation Scripts) | Official benchmark runners | Adapted prompt loading and output paths to support streaming inference pipelines, multi-seed iteration loops, and standardized metric logging. |
| [`REPORT.md`](file:///home/dsi/davidpo/projects/Semi/REPORT.md) | Research Report | Completely rewritten and audited to reflect verified multi-seed empirical metrics, statistical $t$-tests, component ablations, and HPC training workflows. |
| [`README.md`](file:///home/dsi/davidpo/projects/Semi/README.md) | Project Landing Page | Refactored into a formal academic presentation detailing the baseline SCB topology, ablation hypothesis, empirical takeaways, and official external links. |
| [`ABLATION.md`](file:///home/dsi/davidpo/projects/Semi/ABLATION.md) | Reproduction Guide | Updated with SCB-tailored mechanics, runtime weight-norm validation snippets, and verified CLI reproduction commands. |

---

## 2. Newly Authored Files & Modules

| File Path | Component Category | Purpose & Technical Contribution |
| :--- | :--- | :--- |
| [`deepgen/src/models/sd3_kontext/deepgen_sft_llm_adapter.py`](file:///home/dsi/davidpo/projects/Semi/deepgen/src/models/sd3_kontext/deepgen_sft_llm_adapter.py) | Architecture & Modeling | Implements the SCB-specific residual adapter (`adapter_pool` 2-layer MLP and `adapter_seq` cross-attention). Features exact Step 0 zero-initialization ($\mathbf{W}_{\text{out}}=\mathbf{0}$, $\mathbf{b}_{\text{out}}=\mathbf{0}$) ensuring mathematical parity at initialization. |
| [`deepgen/configs/models/deepgen_sft_llm_adapter.py`](file:///home/dsi/davidpo/projects/Semi/deepgen/configs/models/deepgen_sft_llm_adapter.py) | Model Configuration | Model builder registry configuration defining dimensions ($d_{\text{vlm}}=4096, d_{\text{llm}}=2048, d_{\text{attn}}=1024$), frozen LLM backbone (`Qwen2.5-3B-Instruct`), and weight initialization rules. |
| [`deepgen/src/datasets/text2image/hf_streaming_datasets.py`](file:///home/dsi/davidpo/projects/Semi/deepgen/src/datasets/text2image/hf_streaming_datasets.py) | Data Engineering & I/O | Implements map-style in-memory streaming iterators (`HFStreamingT2IDataset`, `HFStreamingEditingDataset`) and multi-task batch collation (`CollateConcat`) for zero-local-disk operation over `conceptual_captions` and `NHR-Edit`. |
| [`deepgen/configs/finetune/deepgen_sft_llm_adapter_hf_stream.py`](file:///home/dsi/davidpo/projects/Semi/deepgen/configs/finetune/deepgen_sft_llm_adapter_hf_stream.py) | Training Recipe | XTuner/MMEngine training configuration specifying 50,000 iterations, mixed-precision BF16, activation checkpointing, and dynamic learning rate scheduling for adapter-only tuning. |
| [`deepgen/scripts/evaluation/run_statistical_eval.py`](file:///home/dsi/davidpo/projects/Semi/deepgen/scripts/evaluation/run_statistical_eval.py) | Statistical Evaluation | Multi-seed evaluation harness supporting diffusion seed sweeps (`[42, 123, 999]`), prefix-stripped checkpoint loading, automated paired $t$-test / Cohen's $d$ calculation, and component/control ablations (`Seq-Only`, `Pool-Only`, `Gaussian Noise`). |
| [`deepgen/scripts/evaluation/run_comprehensive_eval.py`](file:///home/dsi/davidpo/projects/Semi/deepgen/scripts/evaluation/run_comprehensive_eval.py) | Standalone Evaluation | Single-pass evaluation pipeline generating benchmark image grids and computing CLIP semantic cosine similarity. |
| [`deepgen/jobs/sft_llm_ablation.sbatch`](file:///home/dsi/davidpo/projects/Semi/deepgen/jobs/sft_llm_ablation.sbatch) | Cluster Infrastructure | Slurm job launcher for BIU HPC (`A100-4h` partition) configured with `#SBATCH --requeue` and automated checkpoint auto-discovery for seamless preemption recovery. |
| [`deepgen/jobs/eval_all_benchmarks.sbatch`](file:///home/dsi/davidpo/projects/Semi/deepgen/jobs/eval_all_benchmarks.sbatch) | Cluster Evaluation | Slurm evaluation batch script for automated B200 evaluation sweeps. |
| [`CHANGELOG_CODEBASE.md`](file:///home/dsi/davidpo/projects/Semi/CHANGELOG_CODEBASE.md) | Documentation | Dedicated file tracking all repository modifications and structural additions. |

---

## 3. Critical Bug Fixes & Engineering Resolutions

### Checkpoint Loading DDP Prefix Stripping
- **Problem:** MMEngine multi-GPU DDP training prepended a `module.` prefix to state dict parameter keys (e.g., `module.adapter_seq.to_out.weight`). Standard `load_state_dict(strict=False)` silently skipped these keys as unexpected, leaving the zero-initialized adapter inactive ($\Delta = \mathbf{0}$).
- **Resolution:** Added key normalization in `run_statistical_eval.py` to strip `module.` wrappers and enforced an explicit runtime assertion (`assert adapter_norm > 100.0`, measuring $\|\mathbf{W}_{\text{adapter}}\|_1 = 169,104.16$).
