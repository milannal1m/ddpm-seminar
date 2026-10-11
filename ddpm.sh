#!/bin/bash
#SBATCH --job-name=logo_ddpm
#SBATCH --output=logs/logo_ddpm_%j.out
#SBATCH --error=logs/logo_ddpm_%j.err
#SBATCH --partition=gpu_a100_il           # <-- change to your cluster's GPU partition name
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64000
#SBATCH --time=03:10:00            # HH:MM:SS, adjust based on --steps and resolution

# --- environment setup ---
# Uncomment/adjust whichever your cluster uses:
# module load anaconda
# module load cuda/12.1

module load devel/miniforge/25.3.1-python-3.12
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ddpm

# --- run ---
# Paths are resolved by the script itself: --image/--out relative to images/,
# --save_weights relative to the repo root. Submit from the repo root.
python -u src/denoising.py \
    --image unnoised.jpg \
    --out figure_reverse_process.png \
    --save_weights ddpm_weights.pt \
    --steps 18000 \
    --max_side 512
