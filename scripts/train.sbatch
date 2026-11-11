#!/bin/bash
# Submit:   sbatch scripts/train.sbatch configs/prefix_frozen.yaml
# Watch:    tail -f logs/train_<jobid>.out
# Partition / module names follow the Northeastern Explorer docs; adjust if yours differ.
#SBATCH --job-name=caption-train
#SBATCH --partition=gpu,courses-gpu
# GPU: V100 only (courses-gpu also has older P100s). Several partitions = whichever has a V100 free first.
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/train_%j.out

set -euo pipefail
CONFIG=${1:?usage: sbatch scripts/train.sbatch configs/<experiment>.yaml}
cd "$SLURM_SUBMIT_DIR"

module load anaconda3 2>/dev/null || true
source activate captioning 2>/dev/null || conda activate captioning
export HF_HOME=${HF_HOME:-$PWD/.hf_cache}

nvidia-smi
python -m src.train --config "$CONFIG" --resume
