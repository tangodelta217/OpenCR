# OpenCR - Agent Operating Manual (Repository)

## North Star (WOW_RELEASE)
Ship a "paper-grade" repository: CI green, reproducible end-to-end demo, honest reporting, clean Git state, and professional GitHub readiness.

## Non-negotiables
1) Never commit real datasets or generated artifacts:
   - Do NOT commit: *.npz (except tiny protocol/example JSON), data/**, runs/**, docs/tables/**, docs/figures/**.
   - Only commit: source code, tests, docs, small static assets under docs/assets/** (if explicitly intended).
2) Always keep Git clean:
   - `git status -sb` must be clean before declaring DONE.
   - No untracked files except those ignored by .gitignore.
3) Changes must be safe and minimal:
   - Patch plan first (files + rationale + risk + verification).
   - Then implement.
   - Then verify.

## Mandatory verification gates (must pass)
Run and fix until all pass:
- Install: `python -m pip install -e ".[dev]"`
- Tests: `python -m pytest`
- Lint: `python -m ruff check src tests`
- Format check: `python -m black --check src tests`
- Repo audit: `python scripts/repo_audit.py --strict`

## Troubleshooting (when a gate fails)
- Re-run only the failing command to get a clean error signal, then fix and re-run the full gate set.
- Keep generated outputs in ignored paths (data/**, runs/**, docs/figures/**, docs/tables/**) to avoid repo_audit failures.
- If repo_audit fails for a dirty tree, clean outputs or update .gitignore before proceeding.

## End-to-End demo gate (must pass for WOW_RELEASE)
Run the demo without external data:
1) Generate demo dataset (ignored by git):
   `python gen_data.py --output data/_demo --n-subjects 4 --duration-sec 120 --fs 100 --seed 123`
2) Run full pipeline:
   `python run_all.py data/_demo --run-dir runs/_demo_run --report-dir runs/_demo_run/report`
3) Edge P0:
   `python -m opencr edge export --run runs/_demo_run/baseline --out runs/_demo_run/edge --format auto`
   `python -m opencr edge benchmark --run runs/_demo_run/baseline --out runs/_demo_run/edge`

If flags/commands are missing, implement them as part of the sprint before claiming "one-command".

## Language and style
- Keep CLI/log messages in ONE language (prefer English).
- Remove "what comments"; keep "why comments".
- Docstrings for public APIs should be concise and actionable.

## Directory-specific instructions
Before editing files in an area, open and follow the nearest AGENTS.md:
- CI/workflows: `.github/AGENTS.md`
- Core package: `src/opencr/AGENTS.md`
- Reports: `src/opencr/report/AGENTS.md`
- Edge: `src/opencr/edge/AGENTS.md`
- Repro/manifest: `src/opencr/repro/AGENTS.md`
- Repo audit & tooling: `scripts/AGENTS.md`
- Documentation: `docs/AGENTS.md`
- Tests: `tests/AGENTS.md`

## Definition of Done (WOW_RELEASE)
DONE means ALL:
- CI green (ruff + black --check + pytest).
- Annex A produces ROC when applicable and never silently skips due to a bug.
- run_all covers fetch/preprocess/train/evaluate/report AND (optionally) edge export/benchmark, or clearly documents how to add edge.
- repo_audit PASS and no placeholders in README.
- No data / runs outputs committed; `.gitignore` prevents dirty tree after running demos.
- GitHub README links and badges reflect the real repo (no "your-org" placeholders).
