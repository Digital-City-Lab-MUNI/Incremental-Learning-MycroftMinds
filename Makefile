.PHONY: install format lint all clean test exp-run exp-benchmark exp-prepare wes-launch wes-sync

# Detect OS and set PYTHON command accordingly
ifeq ($(OS),Windows_NT)
    PYTHON := $(shell where python3)
else
    PYTHON := $(shell which python3)
endif

# Experiment rerun (see experiments/README.md); override with e.g.
#   make exp-run DATASET=od_friedman_1h DATA_ROOT=data_staging
DATASET ?= od_friedman_1h
DATA_ROOT ?= data
RESULTS ?= results/rerun

# Install Python dependencies (including dev dependencies for now)
install:
	pip install -r requirements.txt

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
exp-run:
	${PYTHON} -m experiments.run_evaluation --dataset $(DATASET) --data-root $(DATA_ROOT) --results-dir $(RESULTS)/raw

exp-benchmark:
	${PYTHON} -m experiments.run_benchmark --dataset $(DATASET) --data-root $(DATA_ROOT) --results-dir $(RESULTS)/raw

exp-prepare:
	${PYTHON} -m experiments.prepare_results --results-dir $(RESULTS)/raw --out $(RESULTS)/tables

# --- HPC rerun via WES (see jobs/fastodt_rerun.wes) ---
WES ?= ../wes/.venv/bin/wes
wes-launch:
	$(WES) dispatch jobs/fastodt_rerun.wes

wes-sync:
	$(WES) sync --ssh hpc
