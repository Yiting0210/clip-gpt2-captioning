#!/bin/bash
# Re-executes notebooks/03_evaluation.ipynb in place so its tables and plots show the latest results.
# Submitted by scripts/pipeline.sh after every training / evaluation job has ended.
#SBATCH --job-name=caption-report
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/report_%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

module load anaconda3 2>/dev/null || true
source activate captioning 2>/dev/null || conda activate captioning

echo "== results/metrics.csv"
cat results/metrics.csv || echo "(no metrics yet)"
echo "== failed or cancelled jobs in this pipeline, if any, are listed in logs/*.out"
python -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1200 \
    --ExecutePreprocessor.kernel_name=captioning notebooks/03_evaluation.ipynb
echo "notebooks/03_evaluation.ipynb updated"
