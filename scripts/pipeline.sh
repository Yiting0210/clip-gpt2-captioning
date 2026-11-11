#!/bin/bash
# Chain Slurm jobs: each config trains, then its evaluation starts automatically (afterok);
# when every job has ended, a report job re-runs notebooks/03_evaluation.ipynb (afterany).
#
#   bash scripts/pipeline.sh configs/prefix_frozen.yaml --sweep prefix_frozen        # stage 2
#   bash scripts/pipeline.sh configs/prefix_lora.yaml configs/prefix_full.yaml \
#        configs/xattn_frozen.yaml configs/xattn_lora.yaml configs/xattn_full.yaml \
#        --retrieval --blip2                                                          # stage 3
#   DRY_RUN=1 bash scripts/pipeline.sh ...      # print the sbatch commands, submit nothing
#
# Options
#   --decode "<specs>"   decoding specs for every eval job (default: greedy beam_w3 nucleus_p0.9_t0.7)
#   --sweep <experiment> also run the full decoding sweep for that experiment
#   --retrieval          CLIP retrieval baseline (starts immediately)
#   --blip2              zero-shot BLIP-2 baseline (starts immediately; may fail without bitsandbytes)
#   --no-report          skip the final notebook job
# Env: CPU_PARTITION (default "courses") for the report job.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

DECODE="greedy beam_w3 nucleus_p0.9_t0.7"
SWEEP="" RETRIEVAL=0 BLIP2=0 REPORT=1 CONFIGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --decode)    DECODE="$2"; shift 2 ;;
    --sweep)     SWEEP="$2"; shift 2 ;;
    --retrieval) RETRIEVAL=1; shift ;;
    --blip2)     BLIP2=1; shift ;;
    --no-report) REPORT=0; shift ;;
    -*)          echo "unknown option $1" >&2; exit 1 ;;
    *)           [[ -f "$1" ]] || { echo "no such config: $1" >&2; exit 1; }; CONFIGS+=("$1"); shift ;;
  esac
done

FAKE=1000 JOB=""
submit() {  # submit [sbatch options...] script [args...]  -> sets $JOB to the new job id
  if [[ -n "${DRY_RUN:-}" ]]; then
    FAKE=$((FAKE + 1)); JOB=$FAKE
    printf '  [%s] sbatch' "$JOB"; printf ' %q' "$@"; echo
  else
    JOB=$(sbatch --parsable "$@" | cut -d';' -f1)
  fi
}

ALL=()
for cfg in "${CONFIGS[@]}"; do
  exp=$(basename "$cfg" .yaml)
  submit --job-name="tr-$exp" scripts/train.sbatch "$cfg"; t=$JOB
  submit --job-name="ev-$exp" --dependency=afterok:$t --kill-on-invalid-dep=yes \
         scripts/eval.sbatch "runs/$exp" "$DECODE"; e=$JOB
  ALL+=("$t" "$e")
  echo "$exp: train $t -> eval $e"
  if [[ "$exp" == "$SWEEP" ]]; then
    submit --job-name="sw-$exp" --dependency=afterok:$t --kill-on-invalid-dep=yes \
           scripts/eval.sbatch "runs/$exp" sweep
    ALL+=("$JOB"); echo "$exp: sweep $JOB (after $t)"
  fi
done

if (( RETRIEVAL )); then submit --job-name=bl-retrieval scripts/baseline.sbatch retrieval; ALL+=("$JOB"); echo "retrieval baseline: $JOB"; fi
if (( BLIP2 ));     then submit --job-name=bl-blip2 scripts/baseline.sbatch blip2;         ALL+=("$JOB"); echo "BLIP-2 baseline: $JOB"; fi

if (( REPORT )) && (( ${#ALL[@]} )); then
  deps=$(IFS=:; echo "${ALL[*]}")
  submit --job-name=report --partition="${CPU_PARTITION:-courses}" --dependency=afterany:$deps scripts/report.sbatch
  echo "report (after all of the above): $JOB"
fi
echo "watch with: squeue -u \$USER   |   logs in logs/"
