.PHONY: init lint format test check clean help

# Default target
.DEFAULT_GOAL := help

# Variables
PYTHON := python
PIP := pip
VENV := .venv

## help: Show this help message
help:
	@echo "OpenCR - Available targets:"
	@echo ""
	@echo "  init     Create venv and install dependencies"
	@echo "  lint     Run ruff linter"
	@echo "  format   Format code with black and ruff"
	@echo "  test     Run pytest"
	@echo "  check    Run all checks (lint + test)"
	@echo "  clean    Remove build artifacts and caches"
	@echo ""

## init: Create virtual environment and install package
init:
	$(PYTHON) -m venv $(VENV)
	$(VENV)/Scripts/pip install --upgrade pip
	$(VENV)/Scripts/pip install -e ".[dev]"
	$(VENV)/Scripts/pre-commit install
	@echo ""
	@echo "✅ Setup complete! Activate with: .\.venv\Scripts\Activate.ps1"

## lint: Run ruff linter
lint:
	ruff check src/ tests/
	black --check src/ tests/

## format: Format code with black and ruff
format:
	ruff check --fix src/ tests/
	black src/ tests/

## test: Run pytest
test:
	pytest tests/ -v

## check: Run all checks
check: lint test
	@echo "✅ All checks passed!"

## clean: Remove build artifacts and caches
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf src/*.egg-info/
	rm -rf .pytest_cache/
	rm -rf .ruff_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "✅ Cleaned!"
