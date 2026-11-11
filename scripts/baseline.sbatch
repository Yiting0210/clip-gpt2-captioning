#!/bin/bash
# Submit:   sbatch scripts/baseline.sbatch retrieval     |   sbatch scripts/baseline.sbatch blip2
# BLIP-2 in 8-bit needs: pip install bitsandbytes accelerate  (falls back to fp16, ~8 GB GPU memory)
#SBATCH --job-name=caption-baseline
#SBATCH --partition=gpu,courses-gpu
# GPU: V100 only (courses-gpu also has older P100s). Several partitions = whichever has a V100 free first.
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/baseline_%j.out

set -euo pipefail
WHICH=${1:?usage: sbatch scripts/baseline.sbatch retrieval|blip2}
cd "$SLURM_SUBMIT_DIR"

module load anaconda3 2>/dev/null || true
module load java 2>/dev/null || true
source activate captioning 2>/dev/null || conda activate captioning
export HF_HOME=${HF_HOME:-$PWD/.hf_cache}

python -m src.evaluate --baseline "$WHICH" --split test
