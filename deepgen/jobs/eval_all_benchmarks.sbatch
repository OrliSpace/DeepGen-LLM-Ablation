#!/bin/bash
#SBATCH --job-name=eval_deepgen_benchmarks
#SBATCH --output=logs/evaluation/eval_%j.out
#SBATCH --error=logs/evaluation/eval_%j.err
#SBATCH --partition=generic-48G,B200-4h,A100-4h,RTX6000-4h,L40s-4h,L4-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --mail-user=orli.davidpur@mail.huji.ac.il
#SBATCH --mail-type=ALL

mkdir -p logs/evaluation outputs/eval_results

eval "$(conda shell.bash hook)"
conda activate deepgen

cd /home/dsi/davidpo/projects/Semi/deepgen
export PYTHONPATH="/home/dsi/davidpo/projects/Semi/deepgen:$PYTHONPATH"

# Target checkpoint passed as 1st argument or auto-detected from latest
CKPT_PATH=$1
if [ -z "$CKPT_PATH" ]; then
    LATEST_TRAINED=$(ls -t work_dirs/sft_llm_ablation/*.pth work_dirs/sft_llm_ablation/*.pt 2>/dev/null | head -1)
    if [ ! -z "$LATEST_TRAINED" ]; then
        CKPT_PATH=$LATEST_TRAINED
    else
        CKPT_PATH="checkpoints/model.pt"
    fi
fi

CONFIG_PATH="configs/models/deepgen_sft_llm_adapter.py"
OUTPUT_ROOT="outputs/eval_results/$(date +%Y%m%d_%H%M%S)"
mkdir -p $OUTPUT_ROOT

echo "=========================================================="
echo "Starting DeepGen SFT + LLM Benchmark Evaluation"
echo "Checkpoint: $CKPT_PATH"
echo "Config: $CONFIG_PATH"
echo "Output Directory: $OUTPUT_ROOT"
echo "Partition: $SLURM_JOB_PARTITION | Node: $SLURMD_NODENAME"
echo "=========================================================="

# 1. First run smoke test validation
echo "--> Step 0: Executing Fast Smoke Test on Trained Checkpoint..."
python scripts/evaluation/smoke_test_eval.py --checkpoint "$CKPT_PATH" --config "$CONFIG_PATH" --output "$OUTPUT_ROOT/smoke_test"

# 2. Text-to-Image Benchmarks (if evaluation data folders exist)
if [ -f "evaluation/geneval/geneval_prompt.jsonl" ]; then
    echo "--> Running GenEval Benchmark..."
    python scripts/evaluation/gen_eval.py \
        --config $CONFIG_PATH \
        --checkpoint $CKPT_PATH \
        --data evaluation/geneval/geneval_prompt.jsonl \
        --output $OUTPUT_ROOT/geneval \
        --height 512 --width 512
fi

if [ -d "evaluation/DPG-Bench/prompts" ]; then
    echo "--> Running DPGBench Benchmark..."
    python scripts/evaluation/dpg_bench.py \
        --config $CONFIG_PATH \
        --checkpoint $CKPT_PATH \
        --data evaluation/DPG-Bench/prompts \
        --output $OUTPUT_ROOT/dpgbench \
        --height 512 --width 512
fi

if [ -f "evaluation/wise/data/spatio-temporal_reasoning.json" ]; then
    echo "--> Running WISE Benchmark..."
    python scripts/evaluation/wise.py \
        --config $CONFIG_PATH \
        --checkpoint $CKPT_PATH \
        --data evaluation/wise/data/spatio-temporal_reasoning.json \
        --output $OUTPUT_ROOT/wise \
        --height 512 --width 512
fi

if [ -f "evaluation/UniGenBench/test_prompts_en.csv" ]; then
    echo "--> Running UniGenBench Benchmark..."
    python scripts/evaluation/unigenbench.py \
        --config $CONFIG_PATH \
        --checkpoint $CKPT_PATH \
        --csv_path evaluation/UniGenBench/test_prompts_en.csv \
        --output $OUTPUT_ROOT/unigenbench \
        --height 512 --width 512
fi

# 3. Instruction-based Image Editing Benchmarks
if [ -d "evaluation/ImgEdit/Benchmark/singleturn" ]; then
    echo "--> Running ImgEdit Benchmark..."
    python scripts/evaluation/img_edit.py \
        --config $CONFIG_PATH \
        --checkpoint $CKPT_PATH \
        --data evaluation/ImgEdit/Benchmark/singleturn \
        --output $OUTPUT_ROOT/imgedit \
        --height 512 --width 512
fi

if [ -d "evaluation/GEdit-Bench" ]; then
    echo "--> Running GEdit Benchmark..."
    python scripts/evaluation/gedit.py \
        --config $CONFIG_PATH \
        --checkpoint $CKPT_PATH \
        --data evaluation/GEdit-Bench \
        --output $OUTPUT_ROOT/gedit \
        --height 512 --width 512
fi

if [ -d "evaluation/RISEBench-full" ]; then
    echo "--> Running RISE Benchmark..."
    python scripts/evaluation/rise_bench.py \
        --config $CONFIG_PATH \
        --checkpoint $CKPT_PATH \
        --data evaluation/RISEBench-full \
        --output $OUTPUT_ROOT/rise \
        --height 512 --width 512
fi

if [ -d "evaluation/UniREditBench" ]; then
    echo "--> Running UniREditBench Benchmark..."
    python scripts/evaluation/unireditbench.py \
        --config $CONFIG_PATH \
        --checkpoint $CKPT_PATH \
        --data evaluation/UniREditBench \
        --output $OUTPUT_ROOT/unireditbench \
        --height 512 --width 512
fi

echo "=========================================================="
echo "All Evaluations Completed Successfully at $(date)!"
echo "Outputs stored in: $OUTPUT_ROOT"
echo "=========================================================="
