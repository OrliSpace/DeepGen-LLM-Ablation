#!/bin/bash
#SBATCH --job-name=deepgen_sft_llm
#SBATCH --output=logs/ablation_sft_llm/sft_%j.out
#SBATCH --error=logs/ablation_sft_llm/sft_%j.err
#SBATCH --partition=B200-4h,generic-48G,H200-4h,L40s-4h,A100-4h,RTX6000-4h
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --mail-user=orli.davidpur@mail.huji.ac.il
#SBATCH --mail-type=ALL

# Create output and logging directories
mkdir -p logs/ablation_sft_llm work_dirs/sft_llm_ablation

# Load BIU Slurm conda environment
eval "$(conda shell.bash hook)"
conda activate deepgen

cd /home/dsi/davidpo/projects/Semi/deepgen
export PYTHONPATH="/home/dsi/davidpo/projects/Semi/deepgen:$PYTHONPATH"
export HF_HUB_ENABLE_HF_TRANSFER=0

# Detect number of allocated GPUs
NUM_GPUS=$SLURM_GPUS_ON_NODE
if [ -z "$NUM_GPUS" ]; then
    NUM_GPUS=2
fi

MASTER_PORT=$((RANDOM % 1000 + 27000))

# Automatic checkpoint discovery for seamless resumption upon Slurm job requeue / preemption
LATEST_CKPT=$(ls -t ./work_dirs/sft_llm_ablation/*.pth ./work_dirs/sft_llm_ablation/*.pt 2>/dev/null | head -1)
RESUME_FLAG=""
if [ ! -z "$LATEST_CKPT" ]; then
    echo "Auto-resuming from latest checkpoint: $LATEST_CKPT"
    RESUME_FLAG="--resume $LATEST_CKPT"
else
    echo "No prior checkpoint found. Warm-starting from official DeepGen SFT baseline with Zero-Init Adapter."
fi

echo "=========================================================="
echo "Starting DeepGen SFT + LLM Adapter Training (No RL)"
echo "Partition: $SLURM_JOB_PARTITION | Node: $SLURMD_NODENAME | GPUs: $NUM_GPUS"
echo "Job ID: $SLURM_JOB_ID | Port: $MASTER_PORT"
echo "=========================================================="

torchrun --nproc_per_node=$NUM_GPUS \
         --nnodes=1 \
         --master_port=$MASTER_PORT \
         scripts/train.py \
         configs/finetune/deepgen_sft_llm_adapter_hf_stream.py \
         --work-dir ./work_dirs/sft_llm_ablation \
         --launcher pytorch \
         $RESUME_FLAG

echo "Job finished / suspended at $(date)"
