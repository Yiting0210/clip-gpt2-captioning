#!/bin/bash
# Round 2 analyses that need finished runs:
#   attention  cross-attention heatmaps + grounding score for the xattn models
#   cycle      rescore the baselines, then image -> caption -> image with Stable Diffusion v1.5
# Submit:   sbatch scripts/analysis.sbatch attention     |   sbatch scripts/analysis.sbatch cycle
#SBATCH --job-name=caption-analysis
#SBATCH --partition=gpu,courses-gpu
# GPU: V100 only (courses-gpu also has older P100s). Several partitions = whichever has a V100 free first.
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=logs/analysis_%j.out

set -euo pipefail
WHAT=${1:?usage: sbatch scripts/analysis.sbatch attention|cycle}
cd "$SLURM_SUBMIT_DIR"

module load anaconda3 2>/dev/null || true
module load java 2>/dev/null || true
source activate captioning 2>/dev/null || conda activate captioning
export HF_HOME=${HF_HOME:-$PWD/.hf_cache}
nvidia-smi --query-gpu=name,memory.total --format=csv

case "$WHAT" in
  attention)
    python -m src.attention --runs runs/xattn_frozen_s0 runs/xattn_lora_s0 runs/xattn_full_s0 --figure 6
    ;;
  cycle)
    # baselines get the new columns (bootstrap intervals, repetition); BLIP-2 captions are reused, not regenerated
    python -m src.evaluate --baseline retrieval --split test
    python -m src.evaluate --baseline blip2 --split test
    # Stable Diffusion weights (~2.5 GB) go to scratch, not the home quota
    HF_HOME=/scratch/$USER/hf_cache python -m src.cycle --n 100 --sources reference \
        results/prefix_full_s0/beam_w3_test.json results/prefix_lora_s0/beam_w3_test.json \
        results/baseline_blip2/blip2_test.json results/baseline_retrieval/retrieval_test.json
    ;;
  *) echo "unknown analysis $WHAT" >&2; exit 1 ;;
esac
echo "ANALYSIS_DONE $WHAT"
