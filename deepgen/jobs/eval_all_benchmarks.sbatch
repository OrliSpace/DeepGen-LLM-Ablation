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
if [ ! -z "$2" ]; then
    OUTPUT_ROOT="$2"
else
    OUTPUT_ROOT="outputs/eval_results/$(date +%Y%m%d_%H%M%S)"
fi
mkdir -p $OUTPUT_ROOT

echo "=========================================================="
echo "Starting DeepGen SFT + LLM Benchmark Evaluation"
echo "Checkpoint: $CKPT_PATH"
echo "Config: $CONFIG_PATH"
echo "Output Directory: $OUTPUT_ROOT"
echo "Partition: $SLURM_JOB_PARTITION | Node: $SLURMD_NODENAME"
echo "=========================================================="

# 1. Run Statistical Multi-Seed Benchmark Suite & Component Ablation
echo "--> Executing Statistical Multi-Seed Benchmark Suite & Component Ablation..."
python scripts/evaluation/run_statistical_eval.py \
    --config "$CONFIG_PATH" \
    --checkpoint "$CKPT_PATH" \
    --output_dir "$OUTPUT_ROOT" \
    --num_samples 25 \
    --batch_size 4 \
    --cfg_scale 4.0 \
    --num_steps 50 \
    --seeds 42 123 999

echo "=========================================================="
echo "All Statistical Evaluations Completed Successfully at $(date)!"
echo "Outputs stored in: $OUTPUT_ROOT"
echo "=========================================================="
