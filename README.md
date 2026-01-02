# OpenCR 🎯

[![CI](https://github.com/your-org/opencr/actions/workflows/ci.yml/badge.svg)](https://github.com/your-org/opencr/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

**Open Cognitive Radio** — A professional, reproducible ML/Signal Processing pipeline for TFM research.

## ✨ Features

- 🔬 **Reproducible**: One command generates all results and figures
- 🔒 **Auditable**: Subject-level splits with automated anti-leakage
- 🔄 **Transferable**: Dataset adapters for easy lab-to-lab portability
- ⚡ **Edge-ready**: ONNX export + INT8 quantization + benchmarks
- 📊 **Demo-ready**: Fuel-gauge visualization anyone understands in 20 seconds

## 🚀 Quickstart (3 commands)

```bash
# 1. Clone and setup
git clone https://github.com/your-org/opencr.git && cd opencr

# 2. Create environment and install
make init

# 3. Verify installation
opencr --help
```

## 📦 Manual Installation

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

## 🛠️ Development

```bash
# Format code
make format

# Run linters
make lint

# Run tests
make test

# Run all checks
make check
```

## 📁 Project Structure

```
OpenCR/
├── src/opencr/          # Main package
│   ├── cli.py           # Typer CLI application
│   ├── logging.py       # Structured logging
│   ├── data/            # Dataset adapters + anti-leakage
│   ├── features/        # Feature extraction
│   ├── models/          # Model training
│   └── export/          # ONNX + quantization
├── tests/               # pytest test suite
├── docs/                # Documentation
├── data/                # Dataset references (gitignored)
└── runs/                # Experiment outputs (gitignored)
```

## 🔐 Anti-Leakage Policy

> ⚠️ **Non-negotiable**: All train/validation/test splits are performed at the **subject level**, never at the sample level. This prevents data contamination where samples from the same subject appear in both training and test sets.

## 📜 License

MIT License - see [LICENSE](LICENSE) for details.
