# GitHub Release Checklist

Use this checklist before publishing to GitHub.

## Pre-release

- [ ] Local repo clean: `python scripts/repo_audit.py` passes
- [ ] Branch is up to date with `main`
- [ ] CI is green on GitHub Actions
- [ ] Version tag prepared: `v0.1-proposal`

## Release assets (if applicable)

- [ ] One-pager PDF attached
- [ ] Proposal PDF attached
- [ ] Annex/Appendix PDFs attached

## Compliance and safety

- [ ] No datasets committed (only `data/README.md` tracked)
- [ ] No credentials, tokens, or secrets committed
- [ ] `runs/` contains only `.gitkeep`

## Post-release

- [ ] GitHub Release notes published
- [ ] Verify badges render correctly on README
