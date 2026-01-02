#!/usr/bin/env python
"""
Local repository audit for GitHub-ready checks.

Run: python scripts/repo_audit.py
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run_git(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )


def _print_result(ok: bool, title: str, details: str | None = None) -> None:
    status = "PASS" if ok else "FAIL"
    line = f"[{status}] {title}"
    print(line)
    if details:
        print(f"  - {details}")


def _check_git_clean() -> tuple[bool, str | None]:
    result = _run_git(["status", "--porcelain"])
    if result.returncode != 0:
        return False, f"git status failed: {result.stderr.strip() or result.stdout.strip()}"
    if result.stdout.strip():
        summary = result.stdout.strip().splitlines()[0]
        return False, f"working tree not clean (example: {summary})"
    return True, None


def _check_required_files() -> tuple[bool, str | None]:
    required = [
        "README.md",
        "LICENSE",
        "pyproject.toml",
        ".gitignore",
        ".github/workflows/ci.yml",
    ]
    missing = [path for path in required if not (ROOT / path).exists()]
    if missing:
        return False, f"missing: {', '.join(missing)}"
    return True, None


def _check_tracked_data() -> tuple[bool, str | None]:
    data_files = _run_git(["ls-files", "data"]).stdout.strip().splitlines()
    runs_files = _run_git(["ls-files", "runs"]).stdout.strip().splitlines()

    allowed_data = {"data/README.md"}
    allowed_runs = {"runs/.gitkeep"}

    unexpected_data = sorted(set(data_files) - allowed_data)
    unexpected_runs = sorted(set(runs_files) - allowed_runs)

    messages = []
    if unexpected_data:
        messages.append(f"unexpected tracked data files: {', '.join(unexpected_data)}")
    if unexpected_runs:
        messages.append(f"unexpected tracked runs files: {', '.join(unexpected_runs)}")

    if messages:
        return False, " | ".join(messages)
    return True, None


def _check_large_files(max_mb: int = 10) -> tuple[bool, str | None]:
    files = _run_git(["ls-files"]).stdout.strip().splitlines()
    too_large: list[str] = []
    limit_bytes = max_mb * 1024 * 1024
    for rel in files:
        path = ROOT / rel
        if not path.exists():
            continue
        size = path.stat().st_size
        if size > limit_bytes:
            too_large.append(f"{rel} ({size / (1024 * 1024):.1f} MB)")
    if too_large:
        return False, "tracked files > 10MB: " + ", ".join(too_large)
    return True, None


def _check_readme() -> tuple[bool, str | None]:
    readme = ROOT / "README.md"
    if not readme.exists():
        return False, "README.md missing"
    text = readme.read_text(encoding="utf-8", errors="replace")
    missing = []
    if not re.search(r"quickstart", text, flags=re.I):
        missing.append("Quickstart section")
    if 'pip install -e ".[dev]"' not in text:
        missing.append("dev install command")
    if "run_all.py" not in text:
        missing.append("run_all demo command")
    if missing:
        return False, "README missing: " + ", ".join(missing)
    return True, None


def _check_badges() -> tuple[bool, str | None]:
    text = (ROOT / "README.md").read_text(encoding="utf-8", errors="replace")
    badge_ok = (
        "actions/workflows/ci.yml/badge.svg" in text
        or re.search(r"\[!\[CI\]\(", text) is not None
    )
    if not badge_ok:
        return False, "CI badge not found in README"
    return True, None


def _check_doc_links() -> tuple[bool, str | None]:
    md_files = [ROOT / "README.md"]
    md_files.extend(sorted((ROOT / "docs").rglob("*.md")))

    link_re = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
    missing_links: list[str] = []

    for md_file in md_files:
        text = md_file.read_text(encoding="utf-8", errors="replace")
        for match in link_re.finditer(text):
            raw = match.group(1).strip()
            if raw.startswith("<") and raw.endswith(">"):
                raw = raw[1:-1].strip()
            if not raw or raw.startswith("#"):
                continue
            if "://" in raw or raw.startswith("mailto:"):
                continue
            path_part = raw.split("#", 1)[0].split("?", 1)[0].strip()
            if not path_part:
                continue
            if path_part.startswith("/"):
                target = ROOT / path_part.lstrip("/")
            else:
                target = md_file.parent / path_part
            if not target.exists():
                missing_links.append(f"{md_file.relative_to(ROOT)} -> {raw}")

    if missing_links:
        preview = ", ".join(missing_links[:5])
        suffix = " ..." if len(missing_links) > 5 else ""
        return False, f"broken doc links: {preview}{suffix}"
    return True, None


def main() -> int:
    checks = [
        ("git status clean", _check_git_clean),
        ("required files present", _check_required_files),
        ("no tracked data/runs files", _check_tracked_data),
        ("no tracked files > 10MB", _check_large_files),
        ("README content", _check_readme),
        ("CI badge present", _check_badges),
        ("doc links resolve", _check_doc_links),
    ]

    failures = 0
    for title, fn in checks:
        ok, details = fn()
        _print_result(ok, title, details)
        if not ok:
            failures += 1

    print()
    if failures:
        print(f"Repo audit FAILED ({failures} issue(s)).")
        print("Fix the issues above and re-run the audit before pushing.")
        return 1

    print("Repo audit PASSED.")
    print("Safe to push and open a release PR.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
