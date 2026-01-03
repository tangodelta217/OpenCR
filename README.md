# OpenCR

Open Compensatory Reserve (OpenCR) is a reproducible ML/signal processing pipeline to
estimate the Compensatory Reserve Index (0-100) from PPG and bioimpedance signals.

[![CI](https://github.com/tangodelta217/OpenCR/actions/workflows/ci.yml/badge.svg)](https://github.com/tangodelta217/OpenCR/actions/workflows/ci.yml)
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

<p align="center">
  <strong>Demo Output (ASCII mode):</strong>
</p>

```
+----------------------- OpenCR -----------------------+

+--------------------- Fuel Gauge ---------------------+
| Sample: 4/4                                          |
|                                                      |
| CRI: 72%                                             |
| ######################--------                       |
|                                                      |
| Status: STABLE                                       |
+------------------------------------------------------+
```

## Quickstart

```bash
# 1. Clone
git clone https://github.com/tangodelta217/OpenCR.git && cd OpenCR

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

## Quickstart Windows (PowerShell)

```powershell
git clone https://github.com/tangodelta217/OpenCR.git
cd OpenCR
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
python -m opencr --help
```

## Quick demo (synthetic dataset)

```powershell
python gen_data.py --output data/demo --n-subjects 4 --duration-sec 600 --fs 100
python run_all.py data/demo --run-dir runs/demo --report-dir docs
# Add --with-edge to also run edge export + benchmark.
```

Outputs:
- `runs/demo/` (fetch, preprocess, baseline artifacts + manifests)
- `docs/figures/annexA` and `docs/tables/annexA`
- With `--with-edge`: `runs/demo/edge` (export manifest, budget, benchmark)

## ASCII-safe demo (Windows-friendly)

```powershell
python -m opencr demo fuel-gauge --duration 10 --ascii
```

## Data policy

- Do not commit datasets or generated artifacts. Keep inputs in `data/` and outputs in `runs/`.
- Generated reports in `docs/figures/` and `docs/tables/` are local outputs. Only place static assets in
  `docs/assets/`.

## Limitations

- Edge export uses a pickle stub if ONNX tooling is not installed. Install `skl2onnx` to enable ONNX export.
- Baseline sklearn edge exports do not report RAM usage, so RAM budgets may be shown as unknown.
- Small-sample folds can emit `UndefinedMetricWarning` for R^2 in synthetic/demo runs; tests still pass.

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

## Dev setup (Windows-friendly)

Dev extras include `pytest`, `ruff`, `black`, and `pre-commit`.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"

python -m pytest
python -m ruff check src tests
python -m black --check src tests
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
python -m ruff check --fix src tests
python -m black src tests
python -m ruff check src tests
python -m black --check src tests
python -m pytest
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

## Signal Format

PPG and BioZ inputs are single-channel per subject. Supported shapes are
`(T,)` or `(1, T)`. Multi-channel inputs are rejected in v0.x; select a
channel or average before loading.

## Targets

Preprocessing builds a dataset-level target map (`target_map.json`) from all
available protocol steps. This keeps `y_opencr` and `y_ord` on a consistent
global scale across subjects instead of per-subject rescaling. Subjects missing
steps are recorded in the preprocess manifest and training fails early if a
step-based target is requested.

## Anti-Leakage Policy

**Non-negotiable**: All train/validation/test splits are performed at the subject
level, never at the sample level. This prevents data contamination where samples
from the same subject appear in both training and test sets.

## License

MIT License - see [LICENSE](LICENSE) for details.
