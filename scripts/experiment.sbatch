#!/bin/bash
# Round 2, one job per config: train 3 seeds (up to 40 epochs, early stopping), evaluate each seed,
# optionally the decoding sweep on seed 0 and the data-size runs (1k / 2k training images x 3 seeds).
# Submit:   sbatch scripts/experiment.sbatch configs/prefix_lora.yaml [--sweep] [--scale]
# Watch:    tail -f logs/exp_<jobid>.out
#SBATCH --job-name=caption-exp
#SBATCH --partition=gpu,courses-gpu
# GPU: V100 only (courses-gpu also has older P100s). Several partitions = whichever has a V100 free first.
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/exp_%j.out

set -euo pipefail
CONFIG=${1:?usage: sbatch scripts/experiment.sbatch configs/<experiment>.yaml [--sweep] [--scale]}
shift
SWEEP=0 SCALE=0
for a in "$@"; do
  case "$a" in --sweep) SWEEP=1 ;; --scale) SCALE=1 ;; *) echo "unknown option $a" >&2; exit 1 ;; esac
done
EXP=$(basename "$CONFIG" .yaml)
SEEDS=${SEEDS:-"0 1 2"}
DECODE=${DECODE:-"greedy beam_w3 greedy_nr3 beam_w3_nr3 nucleus_p0.9_t0.7"}
cd "$SLURM_SUBMIT_DIR"

module load anaconda3 2>/dev/null || true
module load java 2>/dev/null || true   # METEOR + PTB tokenizer
source activate captioning 2>/dev/null || conda activate captioning
export HF_HOME=${HF_HOME:-$PWD/.hf_cache}
nvidia-smi --query-gpu=name,memory.total --format=csv

for s in $SEEDS; do
  name="${EXP}_s${s}"
  echo "== train $name"
  python -m src.train --config "$CONFIG" --resume --set experiment="$name" train.seed="$s"
  echo "== evaluate $name"
  python -m src.evaluate --runs "runs/$name" --decode $DECODE --split test
done

if (( SWEEP )); then
  echo "== decoding sweep on ${EXP}_s0"
  python -m src.evaluate --runs "runs/${EXP}_s0" --decode sweep --split test --gallery 0
fi

if (( SCALE )); then
  for n in 1000 2000; do
    for s in $SEEDS; do
      name="${EXP}_n${n}_s${s}"
      echo "== train $name ($n training images)"
      python -m src.train --config "$CONFIG" --resume --set experiment="$name" train.seed="$s" train_images="$n"
      python -m src.evaluate --runs "runs/$name" --decode beam_w3 --split test --gallery 0
    done
  done
fi
echo "EXPERIMENT_DONE $EXP"
