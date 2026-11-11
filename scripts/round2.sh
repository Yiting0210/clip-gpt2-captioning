#!/bin/bash
# Round 2 of the experiments, chained with Slurm dependencies:
#   env     (CPU)  install the new packages (diffusers, nbconvert)          -> every other job waits (afterok)
#   6 x exp (GPU)  each config: 3 seeds + eval; prefix_lora also gets the decoding sweep;
#                  prefix_lora and xattn_lora also get the data-size runs
#   attention (GPU) after the three xattn jobs        cycle (GPU) after prefix_full + prefix_lora
#   report  (CPU)  after everything: re-executes notebooks/03_evaluation.ipynb
# 8 GPU jobs in total, within the gpu partition's limit of 8 submitted jobs per user.
#
#   bash scripts/round2.sh            DRY_RUN=1 bash scripts/round2.sh   (print, submit nothing)
# Env: CPU_PARTITION (default "courses").
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
CPU=${CPU_PARTITION:-courses}

FAKE=2000 JOB=""
submit() {  # submit [sbatch options...] script [args...]  -> sets $JOB
  if [[ -n "${DRY_RUN:-}" ]]; then
    FAKE=$((FAKE + 1)); JOB=$FAKE
    printf '  [%s] sbatch' "$JOB"; printf ' %q' "$@"; echo
  else
    JOB=$(sbatch --parsable "$@" | cut -d';' -f1)
  fi
}
dep() { local IFS=:; echo "$*"; }

submit --job-name=env --partition="$CPU" scripts/env_update.sbatch; ENV=$JOB
echo "env update: $ENV"

declare -A EXP
for spec in "prefix_frozen" "prefix_lora --sweep --scale" "prefix_full" \
            "xattn_frozen" "xattn_lora --scale" "xattn_full"; do
  read -r name opts <<<"$spec"
  # shellcheck disable=SC2086
  # wait for the env job: a pip install while jobs run swaps libraries under them
  submit --job-name="r2-$name" --dependency=afterok:$ENV --kill-on-invalid-dep=yes \
         scripts/experiment.sbatch "configs/$name.yaml" $opts
  EXP[$name]=$JOB
  echo "$name: $JOB"
done

submit --job-name=r2-attention --dependency=afterok:$(dep "${EXP[xattn_frozen]}" "${EXP[xattn_lora]}" "${EXP[xattn_full]}") \
       --kill-on-invalid-dep=yes scripts/analysis.sbatch attention; ATT=$JOB
submit --job-name=r2-cycle --dependency=afterok:$(dep "$ENV" "${EXP[prefix_full]}" "${EXP[prefix_lora]}") \
       --kill-on-invalid-dep=yes scripts/analysis.sbatch cycle; CYC=$JOB
echo "attention: $ATT   cycle: $CYC"

submit --job-name=r2-report --partition="$CPU" \
       --dependency=afterany:$(dep "$ENV" "${EXP[@]}" "$ATT" "$CYC") scripts/report.sbatch
echo "report: $JOB"
echo "watch with: squeue -u \$USER   |   logs in logs/"
