#!/bin/bash
# Submit:   sbatch scripts/eval.sbatch runs/prefix_frozen "greedy beam_w3 nucleus_p0.9_t0.7"
#           sbatch scripts/eval.sbatch runs/prefix_frozen sweep
#SBATCH --job-name=caption-eval
#SBATCH --partition=gpu,courses-gpu
# GPU: V100 only (courses-gpu also has older P100s). Several partitions = whichever has a V100 free first.
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/eval_%j.out

set -euo pipefail
RUN=${1:?usage: sbatch scripts/eval.sbatch runs/<experiment> "<decode specs>"}
DECODE=${2:-"greedy beam_w3 nucleus_p0.9_t0.7"}
cd "$SLURM_SUBMIT_DIR"

module load anaconda3 2>/dev/null || true
module load java 2>/dev/null || true   # METEOR + PTB tokenizer
source activate captioning 2>/dev/null || conda activate captioning
export HF_HOME=${HF_HOME:-$PWD/.hf_cache}

python -m src.evaluate --runs "$RUN" --decode $DECODE --split test
