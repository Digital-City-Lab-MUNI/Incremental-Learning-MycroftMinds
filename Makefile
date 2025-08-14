.PHONY: install format lint all clean

# Install ruff (formatter + linter)
install:
	pip install ruff

# Format code with Ruff (uses parameters from pyproject.toml)
format:
	ruff format src

# Lint code with Ruff (uses parameters from pyproject.toml)
# Docstring style configured in pyproject.toml (NumPy)
lint:
	ruff check src

# Run format then lint
all: format lint

# Remove Ruff cache if needed
clean:
	rm -rf .ruff_cache
