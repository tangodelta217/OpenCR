# Contributing to OpenCR

Thank you for your interest in contributing to OpenCR!

## Development Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-org/opencr.git
   cd opencr
   ```

2. **Initialize environment**
   ```bash
   make init
   # Or manually:
   # python -m venv .venv
   # pip install -e ".[dev]"
   # pre-commit install
   ```

3. **Activate environment**
   - Windows: `.\.venv\Scripts\Activate.ps1`
   - Linux/Mac: `source .venv/bin/activate`

## Workflow

1. **Create a branch** for your feature (`feat/amazing-feature`) or fix (`fix/annoying-bug`).
2. **Write code** following the style guide (Black/Ruff).
3. **Add tests** in `tests/` directory.
4. **Run checks**:
   ```bash
   make check
   ```
5. **Commit** using conventional commit messages (e.g., `feat: add new filter`, `fix: resolve numpy error`).

## Code Style

- **Formatter**: [Black](https://github.com/psf/black)
- **Linter**: [Ruff](https://github.com/astral-sh/ruff)
- **Type Checking**: Mypy (optional but recommended)

Run `make format` to automatically format your code.

## Pull Request Process

1. Ensure all tests pass locally (`make test`).
2. Update documentation if necessary.
3. Open a PR against `main`.
4. The CI pipeline will run automatically to verify your changes.
