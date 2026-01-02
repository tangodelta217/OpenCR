# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0-proposal] - 2026-01-02

### Added
- **Core**: Repository structure, `pyproject.toml` configuration, `Makefile`.
- **CLI**: `opencr` command-line interface with `data`, `baseline`, `report`, `demo`, `edge` subcommands.
- **Data**: `LocalNpzAdapter` for dataset loading, data validation, and leakage checks.
- **Preprocessing**: Signal processing pipeline (filtering, resampling, windowing) and SQI (Signal Quality Index).
- **Modeling**: 
  - Feature extraction (Time/Frequency domain) for PPG and BioZ.
  - Baseline models (RandomForest, GradientBoosting, XGBoost).
  - LOSO (Leave-One-Subject-Out) cross-validation evaluation.
- **Reporting**: Annex A report generator (CSV tables, ROC curves, Fuel Gauge figures).
- **Demo**: Live terminal visualization with Rich (`opencr demo`).
- **Edge**: Model export infrastructure (ONNX, Pickle Stub) and resource budget estimation.
- **CI**: GitHub Actions workflow for linting (Ruff), formatting (Black), and testing (Pytest).
- **Docs**: Comprehensive README, Edge deployment strategy, Demo guide.

### Fixed
- Matplotlib backend configuration for headless CI environments.
- Windows encoding issues in CLI output (removed problematic emojis).
