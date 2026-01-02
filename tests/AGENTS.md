# Tests rules

- Tests must be deterministic (seeded).
- No network access.
- Keep synthetic datasets small and fast.
- Name tests `test_<feature>_<scenario>`; use classes only when needed for shared fixtures.
- Use `pytest.mark.slow` for long-running tests; keep default suite fast.
- Prefer tmp_path fixtures for generated files; avoid writing outside temp dirs.
- Use conftest.py for shared fixtures; keep fixtures small and single-purpose.
- If you change schema/manifest, update tests accordingly.
