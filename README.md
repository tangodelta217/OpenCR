# OpenCR

Open Compensatory Reserve (OpenCR) is a reproducible ML/signal processing pipeline to
estimate the Compensatory Reserve Index (0-100) from PPG and bioimpedance signals.

[![CI](https://github.com/your-org/opencr/actions/workflows/ci.yml/badge.svg)](https://github.com/your-org/opencr/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

## Features

- Reproducible: one command generates results and figures
- Auditable: subject-level splits with automated anti-leakage
- Transferable: dataset adapters for lab-to-lab portability
- Edge-ready: ONNX export or pickle stub, budgets, host benchmarks (no int8 for baseline)
- Demo-ready: fuel-gauge visualization anyone understands in 20 seconds

## Quickstart

```bash
# 1. Clone
git clone https://github.com/your-org/opencr.git && cd opencr

# 2. Create venv
python -m venv .venv

# 3. Activate (Windows PowerShell)
.\.venv\Scripts\Activate.ps1
#    Activate (Linux/macOS)
#    source .venv/bin/activate

# 4. Install (dev tools included)
pip install -e ".[dev]"

# 5. Verify
python -m opencr --help
```

## Manual Installation

```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# Activate (Linux/macOS)
source .venv/bin/activate

# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Verify
opencr --help
```

## Development

```bash
# Format code
make format

# Run linters
make lint

# Run tests
make test

# Run all checks
make check

# If make is not available (e.g., Windows)
ruff check --fix src tests
black src tests
ruff check src tests
black --check src tests
pytest
```

## Project Structure

```
OpenCR/
  src/opencr/          # Main package
    cli.py             # Typer CLI application
    logging.py         # Structured logging
    data/              # Dataset adapters + anti-leakage
    preprocess/        # Filtering, SQI, windowing, targets
    features/          # Feature extraction
    evaluation/        # LOSO splits + metrics
    models/            # Model training
    report/            # Annex A and metrics reports
    edge/              # Export + benchmark artifacts
    demo/              # Fuel-gauge demos
    targets/           # OpenCR target mapping
    repro/             # Manifests and hashes
  tests/               # pytest test suite
  docs/                # Documentation
  data/                # Dataset references (gitignored)
  runs/                # Experiment outputs (gitignored)
```

## Anti-Leakage Policy

**Non-negotiable**: All train/validation/test splits are performed at the subject
level, never at the sample level. This prevents data contamination where samples
from the same subject appear in both training and test sets.

## License

MIT License - see [LICENSE](LICENSE) for details.
