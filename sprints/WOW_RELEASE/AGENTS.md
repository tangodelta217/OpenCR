# WOW_RELEASE Sprint - Close the remaining release gates

## Goal
Make the repo publicly presentable with CI green and one-command demo, and ensure GitHub professionalism.

## Work order (do in this order)
1) GREEN CI (H1)
   - Run black formatting (apply) and fix ruff errors.
   - Ensure CI matches local gates.
   Acceptance:
   - `python -m ruff check src tests` PASS
   - `python -m black --check src tests` PASS
   - `python -m pytest` PASS

2) FIX Annex A ROC (H2)
   - Remove silent skipping; fix undefined variables (class_order).
   - Ensure ROC appears for classification (binary/multiclass with proba), and is explicitly "not applicable" for regression.
   Acceptance:
   - annexA generates expected figures and table for demo run
   - no NameError/undefined references
   - tests updated (if any)

3) TRUE one-command (H3)
   - Update run_all.py to optionally execute:
     edge export + benchmark
   - Prefer flags: --with-edge / --skip-edge
   Acceptance:
   - running demo pipeline produces edge_budget.json + benchmark.json under run_dir

4) GitHub professionalism (H4 + M2 + L1 + L2)
   - repo_audit must PASS (and enforce clean git + no artifacts tracked).
   - Fix README placeholders ("your-org") and broken badges.
   - Remove invalid dependency extras (typer[all]) or correct them.
   - Ensure docs show ASCII demo usage.
   Acceptance:
   - `python scripts/repo_audit.py --strict` PASS
   - `git status -sb` clean
   - README has no placeholders and links are correct
   - data/**, runs/**, docs/figures/**, docs/tables/** are not tracked

5) Optional WOW upgrades (M1 + M3)
   - Add optional extra for ONNX (e.g., opencr[onnx]) and document in EDGE.md.
   - Propagate target_map_source into baseline manifest/results for traceability.

## Final verification gate
Run:
- python -m pytest
- python -m ruff check src tests
- python -m black --check src tests
- python scripts/repo_audit.py --strict
- python gen_data.py ...; python run_all.py ...; python -m opencr edge export ...; python -m opencr edge benchmark ...
