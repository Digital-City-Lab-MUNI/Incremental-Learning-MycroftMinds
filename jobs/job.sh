#!/bin/bash
#SBATCH --output=job_output.txt
#SBATCH --error=job_error.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --array=0-8%9

# WES job script: runs on a compute node with CWD = the wes job directory.
# wes cloned this repo here as ./Incremental-Learning-MycroftMinds (see jobs/fastodt_rerun.wes).
#
# One SLURM array task per dataset. Override via environment before launching:
#   MODELS="FastODT xFastODT IncubationBoost"  (default: all models of the dataset)
#   BENCH_MODELS="..."                          (default: all models)
#   MAX_LENGTH=20000                            (cap samples per split, -1 = all)
#   JOBS=8                                      (parallel model workers per stage, default 1)
#   FASTODT_DATA_ROOT=/path/to/data_root        (default: <repo>/data)
set -uo pipefail

REPO_DIR="Incremental-Learning-MycroftMinds"
DATASETS=(fb_1h fb_15m sb_com_1h sb_com_15m sb_ind_1h sb_ind_15m od_aq_1h od_electricity_1h od_friedman_1h)
DATASET="${DATASETS[$SLURM_ARRAY_TASK_ID]}"

echo "=== job $SLURM_JOB_ID array $SLURM_ARRAY_TASK_ID dataset=$DATASET ==="
echo "host: $(hostname)"

cd "$REPO_DIR"
DATA_ROOT="${FASTODT_DATA_ROOT:-$PWD/data}"
RESULTS="results/rerun"

# --- Python environment -----------------------------------------------------
# The pipeline needs Python >= 3.10 (the pinned ipython==9.2.0 in
# requirements.txt needs >= 3.11). The cluster may provide uv via modules (as in
# the ../../wes examples), so load it first. Prefer a modern interpreter for the
# venv; fall back to the system python3 with system site-packages; install pinned
# requirements, then unpinned core deps; verify imports and fail fast otherwise.
if command -v module >/dev/null 2>&1; then
  module load uv 2>/dev/null || true
fi
CORE_DEPS="numpy pandas scipy scikit-learn river tqdm Pympler pyyaml"
if command -v uv >/dev/null 2>&1; then
  VENV_DIR="${SLURM_SUBMIT_DIR}/venv-$SLURM_JOB_ID"
  rm -rf "$VENV_DIR"
  uv venv --python 3.12 "$VENV_DIR" 2>/dev/null \
    || uv venv --python "$(command -v python3)" --system-site-packages "$VENV_DIR" \
    || uv venv "$VENV_DIR"
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
  uv pip install -r requirements.txt \
    || uv pip install $CORE_DEPS \
    || echo "[warn] uv install failed; relying on preinstalled packages"
else
  python3 -m venv --system-site-packages .venv 2>/dev/null || python3 -m venv .venv
  # shellcheck disable=SC1091
  source .venv/bin/activate
  pip install -r requirements.txt \
    || pip install $CORE_DEPS \
    || pip install --user $CORE_DEPS \
    || echo "[warn] pip install failed; relying on system packages"
fi
echo "python: $(python -c 'import sys; print(sys.version)') at $(command -v python)"
python -c "import numpy, pandas, scipy, sklearn, river, tqdm, pympler, yaml; print('deps ok')" \
  || { echo "[ERROR] missing required packages (numpy pandas scipy sklearn river tqdm pympler yaml); job will fail"; exit 1; }

# --- evaluation (valid splits + test set) -----------------------------------
MODELS_ARG=()
if [ -n "${MODELS:-}" ]; then
  # shellcheck disable=SC2206
  MODELS_ARG=(--models $MODELS)
fi
python -m experiments.run_evaluation \
  --dataset "$DATASET" "${MODELS_ARG[@]}" \
  --data-root "$DATA_ROOT" \
  --results-dir "$RESULTS/raw" --max-length "${MAX_LENGTH:--1}" \
  --jobs "${JOBS:-1}"

# --- computational benchmark -------------------------------------------------
BENCH_ARG=()
if [ -n "${BENCH_MODELS:-}" ]; then
  # shellcheck disable=SC2206
  BENCH_ARG=(--models $BENCH_MODELS)
fi
python -m experiments.run_benchmark \
  --dataset "$DATASET" "${BENCH_ARG[@]}" \
  --data-root "$DATA_ROOT" \
  --results-dir "$RESULTS/raw" --max-length "${MAX_LENGTH:--1}" \
  --jobs "${JOBS:-1}"

# --- aggregate whatever raw results exist so far (idempotent) ----------------
python -m experiments.prepare_results --results-dir "$RESULTS/raw" --out "$RESULTS/tables"

echo "=== job $SLURM_JOB_ID dataset=$DATASET completed ==="
