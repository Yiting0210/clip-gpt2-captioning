#!/bin/bash
# Stage 1 as one batch job: unit tests, data prep, CLIP embeddings, sanity checks.
# Expects data/raw/ to hold COCO val2017 (python -m src.data download, or wget + unzip).
# Submit:   sbatch scripts/stage1.sbatch        Watch:  tail -f logs/stage1_<jobid>.out
#SBATCH --job-name=caption-stage1
#SBATCH --partition=gpu-short,gpu,courses-gpu
# GPU: V100 only (courses-gpu also has older P100s). Several partitions = whichever has a V100 free first.
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/stage1_%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module load anaconda3 2>/dev/null || true
source activate captioning 2>/dev/null || conda activate captioning
export HF_HOME=${HF_HOME:-/scratch/$USER/hf_cache}

nvidia-smi --query-gpu=name,memory.total --format=csv
echo "== unit tests";        python -m pytest -q -p no:cacheprovider
echo "== prepare";           python -m src.data download && python -m src.data prepare
echo "== CLIP embeddings";   python -m src.data embed
echo "== caption lengths";   python -m src.data stats
echo "== overfit one batch"; python -m src.train --config configs/prefix_frozen.yaml --overfit-batch
echo "== debug run";         python -m src.train --config configs/prefix_frozen.yaml --debug
echo "STAGE1_DONE"
