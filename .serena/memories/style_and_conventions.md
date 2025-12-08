# SciPy style and conventions
- Python style follows PEP8 with 88-char lines; linted via `ruff`/`pycodestyle` (`spin lint` or `python tools/lint.py --files ...`). Prefer touching style only in modified lines.
- Docstrings follow numpydoc; public functions need parameter/returns sections, examples, and `.. versionadded::` for new APIs. Ensure docs render correctly.
- Tests should follow NumPy/SciPy testing guidelines using pytest/hypothesis; enable deterministic hypothesis profile by default. Add focused unit tests for new/changed code.
- Pre-commit hook available (`cp tools/pre-commit-hook.py .git/hooks/pre-commit`) to run lint before commits.
- PR checklist emphasizes: license compatibility, tests passing, docstrings + docs updates, style compliance, benchmarks if relevant, meson build integration for new files.