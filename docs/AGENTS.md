# Documentation rules

## Tone
- Professional, concise, consistent language (prefer English across docs).

## Required sections in README
- Quickstart (demo: gen_data + run_all)
- Dev setup (pip install -e ".[dev]", pytest, ruff, black)
- Data policy (no datasets committed)
- Limitations: if LBNP context applies, include "LBNP != trauma real" disclaimer.

## Demo guide
- Must include an ASCII-safe demo example (flag --ascii).
- Avoid Unicode-only characters that break Windows consoles.

## Assets
- Store static images in `docs/assets/` (PNG or SVG preferred).
- Do not commit generated plots from runs unless intentionally copied into `docs/assets/`.
