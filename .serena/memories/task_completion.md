# Task completion checklist
- Build/test env: `source .venv/bin/activate` then use pixi tasks (`~/.pixi/bin/pixi run build` followed by `~/.pixi/bin/pixi run test [./scipy/integrate/]`) to ensure meson/ninja build is current; delete `scipy/build` if tests misbehave after changes.
- Run relevant tests: prefer scoped runs (`spin test -s <submodule>` or `spin test -t <path>::<case>`); use `spin test -m full` when slow tests matter.
- Lint/type-check touched files: `spin lint --files <paths>` (ruff/pycodestyle) and `spin mypy` when typing is affected.
- Docs: update numpydoc docstrings/examples; ensure new/changed APIs include `.. versionadded::` and docs render (spot-check with `spin docs` or `spin smoke_docs`).
- Build integration: add new sources to `meson.build` where required; keep line length ≤88 and follow SciPy PEP8 guidance.
- Before PR: verify license compatibility, add/adjust tests, consider benchmarks for performance changes, craft clear commit/PR messages (see contributor workflow).