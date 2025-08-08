# Makefile for code formatting and linting

.PHONY: install format lint all

# Install all required dependencies
install:
	pip install black flake8 darglint

# Format code with black
# Parameters:
# 1. line-length = 88, 
# 2. excluding .venv and __*.py files
format:
	black --line-length 88 --exclude '/(\.venv|__.*\.py)/' src

# Lint code with flake8 and darglint (
# Parameters:
# 1. line-length = 88, 
# 2. excluding .venv and __*.py files 
# 3. strictness set to short
# 4. docstring style set to sphinx)
lint:
	flake8 --max-line-length 88 --exclude .venv,__*.py --strictness=short --docstring-style=sphinx src
	darglint -v 2 src

# Run both format and lint
all: format lint