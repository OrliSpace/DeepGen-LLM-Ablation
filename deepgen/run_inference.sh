#!/bin/bash
#SBATCH --job-name=deepgen_inference
#SBATCH --output=logs/out_%j.out
#SBATCH --error=logs/err_%j.err
#SBATCH --partition=A100-4h          
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1        # Request 1 GPU
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00

# 1. Load any necessary modules (uncomment and adjust if required by your cluster)
# module load anaconda3
# module load cuda/12.1

# 2. Activate your virtual environment
eval "$(conda shell.bash hook)"
conda activate deepgen

# FIX: Force bitsandbytes to use a compatible CUDA binary
export BNB_CUDA_VERSION=122

# 3. Navigate to the official repository
export PYTHONPATH=.

# 4. Generate base images (Text-to-Image)
echo "Generating base images..."
python scripts/text2image.py \
    --checkpoint checkpoints/model.pt \
    --prompt "A person with a blue shirt" \
    --output "generated_blue_shirt" \
    --height 512 --width 512 \
    --seed 42

python scripts/text2image.py \
    --checkpoint checkpoints/model.pt \
    --prompt "A messy room" \
    --output "generated_messy_room" \
    --height 512 --width 512 \
    --seed 42

# 5. Run Image-to-Image edits using the generated base images
# (The text2image script saves 4 examples as case_0.png to case_3.png, so we use case_0)
echo "Running Image-to-Image edits..."
python scripts/image2image.py \
    --checkpoint checkpoints/model.pt \
    --prompt "Make the person’s shirt red instead of blue." \
    --src_img "generated_blue_shirt/case_0.png" \
    --output "edited_red_shirt" \
    --height 512 --width 512 \
    --seed 42

python scripts/image2image.py \
    --checkpoint checkpoints/model.pt \
    --prompt "Edit the image to show what the room looked like before the mess happened." \
    --src_img "generated_messy_room/case_0.png" \
    --output "edited_clean_room" \
    --height 512 --width 512 \
    --seed 42

# 6. Generate meme (Text-to-Image)
echo "Generating meme..."
python scripts/text2image.py \
    --checkpoint checkpoints/model.pt \
    --prompt "Make a meme with the caption: When the code finally works." \
    --output "generated_meme" \
    --height 512 --width 512 \
    --seed 42
