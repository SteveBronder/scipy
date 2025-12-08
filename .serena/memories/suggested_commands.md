# Common commands
- Activate env: `source .venv/bin/activate` before running tooling (pixi/spin installed via pixi envs).
- Build/install: `source .venv/bin/activate && ~/.pixi/bin/pixi run build` (runs `spin build --setup-args ...`, meson/ninja); `spin shell` for a dev shell.
- Testing (pixi): `source .venv/bin/activate && ~/.pixi/bin/pixi run test` for default tests; scope to a module with `~/.pixi/bin/pixi run test ./scipy/integrate/` (uses `spin test --no-build`); targeted: `spin test -s <submodule>` or `spin test -t scipy.<module>.tests.<file>::<TestClass>::<test>`; add `-- -n 4` for xdist, `-m full` for slow tests.
- Lint/type: `spin lint --files <paths>` (ruff/pycodestyle) and `spin mypy`.
- Docs: `spin docs`, `spin smoke_docs`, `spin refguide_check`; tutorials via `spin smoke_tutorials`.
- Benchmarks: `spin bench` (ASV). Quick search/tools: `rg '<pattern>'`, `git status`, `git diff`.