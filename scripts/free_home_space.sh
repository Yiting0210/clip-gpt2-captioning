#!/bin/bash
# Frees home-directory quota without deleting anything (run when no training job is running):
#   .hf_cache  -> /scratch/$USER/caption/hf_cache, symlinked back (model downloads; re-downloadable)
#   last.pt of finished runs -> /scratch/$USER/caption/last_pt_backup/<run>/ (only needed to resume)
# A run counts as finished when it has summary.json and best.pt.
#   srun -p courses --time=01:00:00 bash scripts/free_home_space.sh
set -euo pipefail
cd "$(dirname "$0")/.."
DEST=/scratch/$USER/caption
mkdir -p "$DEST/last_pt_backup"

if [[ -d .hf_cache && ! -L .hf_cache ]]; then
  echo "moving .hf_cache ($(du -sh .hf_cache | cut -f1)) to $DEST/hf_cache"
  mkdir -p "$DEST/hf_cache"
  cp -a .hf_cache/. "$DEST/hf_cache/"
  rm -rf .hf_cache
  ln -s "$DEST/hf_cache" .hf_cache
fi

for run in runs/*/; do
  run=${run%/}
  if [[ -f $run/last.pt && -f $run/summary.json && -f $run/best.pt ]]; then
    mkdir -p "$DEST/last_pt_backup/$(basename "$run")"
    mv "$run/last.pt" "$DEST/last_pt_backup/$(basename "$run")/"
    echo "backed up $run/last.pt"
  fi
done
rm -f runs/*/*.pt.tmp
du -sh runs .hf_cache/ 2>/dev/null
echo FREE_DONE
