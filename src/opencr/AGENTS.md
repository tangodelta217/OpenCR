# Core package rules (src/opencr)

## Constraints
- Keep APIs stable; do not break CLI without updating docs/tests.
- Prefer small, testable functions.
- Avoid over-engineering; introduce new abstractions only with clear benefits.

## Style
- Type hints required for public functions.
- Prefer built-in generics (PEP 585) under Python 3.11+ (e.g., list[int], dict[str, float]).
- Import order: standard library, third-party, then local (opencr), separated by blank lines.
- Comments: explain WHY (decisions/assumptions), not WHAT.

## Scientific integrity
- Never silently change targets, gating, or metrics. If you change behavior:
  - update tests
  - update manifests/schema
  - update docs (limits and assumptions)
