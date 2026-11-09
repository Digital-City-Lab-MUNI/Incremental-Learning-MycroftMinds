#!/bin/bash
# WES pre-script: runs on the HPC login node inside the job directory, before sbatch.
# wes has already cloned THIS repo there as ./Incremental-Learning-MycroftMinds.
set -euo pipefail

# Datasets are gitignored in this repo. Point FASTODT_DATA_SHARED at a directory
# whose children are named first_batch/, other_data/, second_batch/ (e.g. a
# staging dir with symlinks into ../datasets/<name>(current)/<name>/) to link
# them into the cloned repo's data/ dir. job.sh then uses <repo>/data as root.
if [ -n "${FASTODT_DATA_SHARED:-}" ]; then
  echo "[pre] linking datasets from $FASTODT_DATA_SHARED"
  mkdir -p Incremental-Learning-MycroftMinds/data
  for d in "$FASTODT_DATA_SHARED"/*/; do
    ln -sfn "$d" "Incremental-Learning-MycroftMinds/data/$(basename "$d")"
  done
fi

echo "[pre] done"
