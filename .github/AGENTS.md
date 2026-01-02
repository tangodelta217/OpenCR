# CI/CD rules (OpenCR)

## Goal
CI must match the local gates from root AGENTS.md.

## Requirements
- Use Python 3.11 (and optionally 3.12).
- Install with: `python -m pip install -e ".[dev]"`.
- Run:
  - `python -m ruff check src tests`
  - `python -m black --check src tests`
  - `python -m pytest`

## Notes
- Do not rely on `make` (Windows users).
- Keep workflow steps minimal and deterministic.
