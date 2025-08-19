.PHONY: install format lint all clean

# Install Python dependencies (including dev dependencies for now)
install:
	pip install -r requirements.txt

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
