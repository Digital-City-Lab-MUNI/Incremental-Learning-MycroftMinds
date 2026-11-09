.PHONY: install format lint all clean test exp-run exp-benchmark exp-prepare exp-all wes-launch wes-sync

# Python command: prefer uv (ephemeral pinned env on python 3.12, like the wes
# examples); fall back to system python3 (run `make install` first in that case).
# Override with e.g. `make exp-all PYTHON=python3`.
UV := $(shell command -v uv 2>/dev/null)
ifeq ($(UV),)
    PYTHON ?= python3
else
    PYTHON ?= uv run --with-requirements requirements.txt --python 3.12 python
endif

# Experiment rerun (see experiments/README.md); override with e.g.
#   make exp-run DATASET=od_friedman_1h DATA_ROOT=data_staging
DATASET ?= od_friedman_1h
DATA_ROOT ?= data
RESULTS ?= results/rerun
MAX_LENGTH ?= -1

# Install Python dependencies (including dev dependencies for now)
install:
ifeq ($(UV),)
	pip install -r requirements.txt
else
	uv pip install --system -r requirements.txt
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
# MODELS="FastODT IncubationBoost" to subset models, MAX_LENGTH=N to cap rows/split
exp-run:
	${PYTHON} -m experiments.run_evaluation --dataset $(DATASET) --data-root $(DATA_ROOT) --results-dir $(RESULTS)/raw $(if $(MODELS),--models $(MODELS),) --max-length $(MAX_LENGTH)

exp-benchmark:
	${PYTHON} -m experiments.run_benchmark --dataset $(DATASET) --data-root $(DATA_ROOT) --results-dir $(RESULTS)/raw $(if $(MODELS),--models $(MODELS),) --max-length $(MAX_LENGTH)

exp-prepare:
	${PYTHON} -m experiments.prepare_results --results-dir $(RESULTS)/raw --out $(RESULTS)/tables

# full experiment matrix: every dataset in configs/experiments.yaml, all models;
# a failing dataset is reported but does not abort the rest
exp-all:
	@fail=0; for ds in $$(${PYTHON} -c "import yaml; print(' '.join(yaml.safe_load(open('configs/experiments.yaml'))['datasets']))"); do \
	  echo "=== dataset $$ds ==="; \
	  ${PYTHON} -m experiments.run_evaluation --dataset $$ds --data-root $(DATA_ROOT) --results-dir $(RESULTS)/raw --max-length $(MAX_LENGTH) || { echo "[FAIL] eval $$ds"; fail=1; }; \
	  ${PYTHON} -m experiments.run_benchmark --dataset $$ds --data-root $(DATA_ROOT) --results-dir $(RESULTS)/raw --max-length $(MAX_LENGTH) || { echo "[FAIL] bench $$ds"; fail=1; }; \
	done; \
	${PYTHON} -m experiments.prepare_results --results-dir $(RESULTS)/raw --out $(RESULTS)/tables; \
	exit $$fail

# --- HPC rerun via WES (see jobs/fastodt_rerun.wes) ---
WES ?= ../wes/.venv/bin/wes
wes-launch:
	$(WES) dispatch jobs/fastodt_rerun.wes

wes-sync:
	$(WES) sync --ssh hpc
