from __future__ import annotations

from pathlib import Path

from setuptools import find_packages, setup

try:
    import tomllib
except ModuleNotFoundError as exc:  # pragma: no cover - requires Python >=3.11
    raise SystemExit("Python 3.11+ is required to install OpenCR.") from exc

ROOT = Path(__file__).resolve().parent


def _load_pyproject() -> dict:
    with (ROOT / "pyproject.toml").open("rb") as handle:
        return tomllib.load(handle)


def _read_readme(readme: str) -> str:
    readme_path = ROOT / readme
    if not readme_path.exists():
        return ""
    return readme_path.read_text(encoding="utf-8")


def _parse_authors(authors: list[dict]) -> tuple[str, str]:
    names = [author.get("name", "") for author in authors if author.get("name")]
    emails = [author.get("email", "") for author in authors if author.get("email")]
    return ", ".join(names), ", ".join(emails)


def _parse_entry_points(scripts: dict[str, str]) -> dict[str, list[str]]:
    if not scripts:
        return {}
    return {"console_scripts": [f"{name}={target}" for name, target in scripts.items()]}


config = _load_pyproject()
project = config.get("project", {})
urls = project.get("urls", {})
author, author_email = _parse_authors(project.get("authors", []))

setup(
    name=project.get("name", "opencr"),
    version=project.get("version", "0.0.0"),
    description=project.get("description", ""),
    long_description=_read_readme(project.get("readme", "README.md")),
    long_description_content_type="text/markdown",
    license=project.get("license", {}).get("text", ""),
    python_requires=project.get("requires-python", ""),
    author=author,
    author_email=author_email,
    keywords=" ".join(project.get("keywords", [])) or None,
    classifiers=project.get("classifiers", []),
    install_requires=project.get("dependencies", []),
    extras_require=project.get("optional-dependencies", {}),
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    entry_points=_parse_entry_points(project.get("scripts", {})),
    url=urls.get("Repository") or urls.get("Homepage") or None,
    project_urls=urls or None,
)
