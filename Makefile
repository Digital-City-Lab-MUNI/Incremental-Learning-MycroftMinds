.PHONY: install format lint all clean test exp-run exp-benchmark exp-prepare exp-all exp-matrix wes-launch wes-sync

# Python command: prefer the uv project env (`uv run python` uses .venv created
# by `uv sync` / `make install`); fall back to system python3 (run `make install`
# first in that case). Override with e.g. `make exp-all PYTHON=python3`.
UV := $(shell command -v uv 2>/dev/null)
ifeq ($(UV),)
    PYTHON ?= python3
else
    PYTHON ?= uv run python
endif

# Experiment rerun (see experiments/README.md); override with e.g.
#   make exp-run DATASET=od_friedman_1h DATA_ROOT=data_staging
DATASET ?= od_friedman_1h
DATA_ROOT ?= data
RESULTS ?= results/rerun
MAX_LENGTH ?= -1

# Paper rerun: six hourly datasets, eight table models plus matched xFastODT.
PAPER_DATASETS ?= od_aq_1h od_friedman_1h od_electricity_1h sb_com_1h fb_1h sb_ind_1h
PAPER_MODELS ?= BL_Mean GBDT RF HoeffdingTree ARF_Hoeffding FastODT ARF_FastODT IncubationBoost xFastODT
DATASET_JOBS ?= 1

# Install Python dependencies: uv project env when uv is available, else pip.
install:
ifeq ($(UV),)
	pip install -r requirements.txt
else
	uv sync --extra dev
endif

# Format code with Ruff (uses parameters from pyproject.toml)
format:
	ruff format src tests

# Lint code with Ruff (uses parameters from pyproject.toml)
# Docstring style configured in pyproject.toml (NumPy)
lint:
	ruff check src

# Run format then lint
all: format lint

# Remove Ruff cache if needed
clean:
	rm -rf .ruff_cache

test:
	${PYTHON} -m unittest tests/test_*

# --- experiment reruns (notebook-equivalent, see experiments/README.md) ---
# MODELS="FastODT IncubationBoost" to subset models, MAX_LENGTH=N to cap rows/split,
# SKIP_EXISTING=1 to skip models whose results already exist (resume), JOBS=N for parallel model workers
exp-run:
	${PYTHON} -m experiments.run_evaluation --dataset $(DATASET) --data-root $(DATA_ROOT) --results-dir $(RESULTS)/raw $(if $(MODELS),--models $(MODELS),) --max-length $(MAX_LENGTH) $(if $(JOBS),--jobs $(JOBS),) $(if $(SKIP_EXISTING),--skip-existing,)

exp-benchmark:
	${PYTHON} -m experiments.run_benchmark --dataset $(DATASET) --data-root $(DATA_ROOT) --results-dir $(RESULTS)/raw $(if $(MODELS),--models $(MODELS),) --max-length $(MAX_LENGTH) $(if $(JOBS),--jobs $(JOBS),) $(if $(SKIP_EXISTING),--skip-existing,)

exp-prepare:
	${PYTHON} -m experiments.prepare_results --results-dir $(RESULTS)/raw --out $(RESULTS)/tables

# Paper rerun: evaluation followed by fresh-process, serial resource benchmarks.
# JOBS=N runs N models in parallel within each dataset (datasets stay sequential;
# set DATASET_JOBS>1 to run datasets in parallel as well).
exp-all:
	${PYTHON} -m experiments.run_matrix \
	  --datasets $(PAPER_DATASETS) \
	  --models $(PAPER_MODELS) \
	  --data-root $(DATA_ROOT) \
	  --results-dir $(RESULTS) \
	  --max-length $(MAX_LENGTH) \
	  --jobs $(DATASET_JOBS) \
	  --model-jobs $(if $(JOBS),$(JOBS),1) \
	  $(if $(SKIP_EXISTING),--skip-existing,)

# full 9-dataset x all-model matrix (evaluation + benchmark), parallel across datasets
# without SLURM (plain server): JOBS=N runs N dataset workers concurrently,
# MODEL_JOBS=M parallelizes models within each dataset
# (e.g. make exp-matrix JOBS=4 MODEL_JOBS=2 SKIP_EXISTING=1)
exp-matrix:
	${PYTHON} -m experiments.run_matrix --data-root $(DATA_ROOT) --results-dir $(RESULTS) --max-length $(MAX_LENGTH) $(if $(JOBS),--jobs $(JOBS),) $(if $(MODEL_JOBS),--model-jobs $(MODEL_JOBS),) $(if $(MODELS),--models $(MODELS),) $(if $(SKIP_EXISTING),--skip-existing,)

# --- HPC rerun via WES (see jobs/fastodt_rerun.wes) ---
WES ?= ../wes/.venv/bin/wes
wes-launch:
	$(WES) dispatch jobs/fastodt_rerun.wes

wes-sync:
	$(WES) sync --ssh hpc
