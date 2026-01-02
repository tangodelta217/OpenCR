# Tooling and GitHub readiness rules

## repo_audit.py is a release gate
- repo_audit must fail when:
  - git status is dirty
  - untracked files exist (unless ignored)
  - data artifacts are tracked
  - docs outputs directories are tracked (runs/, data/, docs/figures, docs/tables)
  - README contains placeholders (e.g., "your-org")
  - CI workflow is missing key checks

## GitHub professionalism checks
- README:
  - correct repo URLs (or neutral if remote not set)
  - working Quickstart
  - no broken links to missing files
- Ensure .gitignore prevents dirty tree after running demo pipeline.
