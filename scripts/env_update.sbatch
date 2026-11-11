#!/bin/bash
# Installs the packages round 2 added to requirements.txt (diffusers for the cycle experiment,
# nbconvert + ipykernel for the report job) into the existing conda env. CPU only.
# pip on the NFS home directory is slow (> 20 min), hence a batch job with a long time limit.
# Submit:   sbatch --partition=courses scripts/env_update.sbatch
#SBATCH --job-name=caption-env
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=logs/env_%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module load anaconda3 2>/dev/null || true
source activate captioning 2>/dev/null || conda activate captioning

export PIP_CACHE_DIR=/scratch/$USER/pip_cache
# transformers < 5 needs huggingface-hub < 1.0; without the pin, a new diffusers pulls hub 2.x and breaks transformers.
python -m pip install --disable-pip-version-check "huggingface-hub>=0.34,<1.0" "diffusers>=0.30,<1" \
    "accelerate>=0.30" "nbconvert>=7.0" "ipykernel>=6.0"
python -m ipykernel install --sys-prefix --name captioning --display-name "captioning"
# the whole stack must import together before any GPU job starts
python -c "import transformers, peft, diffusers, nbconvert, accelerate, huggingface_hub as h; \
from transformers import GPT2LMHeadModel, CLIPModel; from diffusers import StableDiffusionPipeline; \
print('transformers', transformers.__version__, '| hub', h.__version__, '| diffusers', diffusers.__version__)"
echo "ENV_DONE"
