.PHONY: install format lint all clean

# Detect OS and set PYTHON command accordingly
ifeq ($(OS),Windows_NT)
    PYTHON := $(shell where python3)
else
    PYTHON := $(shell which python3)
endif

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
