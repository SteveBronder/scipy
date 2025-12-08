# Design Doc: Adding `solve_ivp_osc` (riccaticpp) to `scipy.integrate`

## Goal
Expose an oscillatory ODE solver `solve_ivp_osc` in `scipy.integrate`, powered by the riccaticpp algorithm. The API should align with SciPy expectations (rtol/atol, `OdeResult`/`DenseOutput`, float64/complex128 only) while keeping riccati-specific controls (`omega_fun`, `gamma_fun`, step-size tuning).

## Non-goals (initially)
- LowLevelCallable/native callbacks (Python omega/gamma only in v1).
- Non-float64/complex128 dtypes.
- Benchmarks in-tree (focus on correctness tests first).
- Adding a new method to `solve_ivp`; this stays a standalone function.

## Target Python API
- `solve_ivp_osc(t_span, y0, *, omega_fun, gamma_fun, rtol=1e-3, atol=1e-6, t_eval=None, dense_output=False, max_step=None, first_step=None, epsilon_h=None, ...)`.
- `omega_fun`, `gamma_fun`: required callables (scalar or vector inputs; return real or complex arrays).
- Map SciPy `rtol`/`atol` to the riccati `eps`; choose `epsilon_h` default from `rtol` (e.g., `epsilon_h = max(1e-6, rtol)`) unless provided.
- Return an `OdeResult`; riccati extras (step types, phases, success flags) stored under `result.extra`.
- If `dense_output=True`, provide an `OdeSolution`-like object with `sol(t)`; otherwise respect `t_eval` if given.

## Current riccaticpp snapshot (inside `scipy/integrate/riccaticpp`)
- Header-only C++ (Eigen) implementing oscillatory/non-osc steps, adaptive step sizes (`choose_osc_stepsize`, `choose_nonosc_stepsize`), optional dense evaluation when `x_eval` is supplied.
- Pybind11 module `pyriccaticpp` exposes `evolve`, `osc_evolve`, `nonosc_evolve`, step-size helpers, and four `Init_*` solver types (omega/gamma real/complex combinations). Returns large tuples, not `OdeResult`.
- Builds via CMake/scikit-build; fetches Eigen. Not meson/Cython-integrated.

## Design choices
- **Wrapper tech:** use Cython (SciPy standard), not pybind11. Compile C++17 with Eigen headers.
- **Module placement/name:** add a Cython extension `_riccati` under `scipy/integrate/_ivp`; public API `scipy.integrate.solve_ivp_osc`.
- **Dense output:** needs a C++ evaluator to compute `y(t)` per step. Current code only returns dense arrays when `x_eval` is provided. We should add a small C++ entry point to evaluate a saved step (or return step records) so Python can implement a `DenseOutput` subclass that delegates to C++.
- **Callback scope:** Python omega/gamma only in v1; vectorized NumPy inputs allowed. Defer LowLevelCallable.
- **rtol/atol mapping:** document mapping to `eps`/`epsilon_h`; enforce float64/complex128 only.
- **Eigen:** reuse vendored Eigen (already in SciPy) via meson include paths; no new third-party deps.

## Phased migration
### Phase 0 — Stub & plumbing
- Add meson target for a new Cython C++ extension `_riccati` under `_ivp`.
- Add `solve_ivp_osc` stub (NotImplemented or trivial) and docs skeleton; export from `scipy.integrate`.
- Smoke tests for import/argument validation.

### Phase 1 — Minimal working solver (no true dense_output)
- Wrap C++ `evolve`/`osc_evolve` via a thin C++ facade + Cython, mapping `rtol`/`atol`→`eps`; derive `epsilon_h` from `rtol` if not passed.
- Support `t_eval`; if provided, pass as `x_eval` to C++ to fill `y` along the trajectory. If `dense_output=False`, return without interpolation.
- Return `OdeResult` with extras in `result.extra`.
- Port riccaticpp Python tests (Schrodinger, Bremer) to call `solve_ivp_osc`.

### Phase 2 — DenseOutput / OdeSolution
- Extend C++ to expose per-step interpolation data or an evaluator `eval_step(record, t)`.
- Store step records in Python; implement a `DenseOutput` subclass that calls the C++ evaluator.
- Tests: verify `sol(t_query)` matches C++ dense outputs within tolerance.

### Phase 3 — Polish
- Documentation and examples in `scipy.integrate`.
- Expose riccati diagnostics in `extra`.
- Optional: add benchmark harness; consider LowLevelCallable in a follow-up.

## Build/meson sketch
- Sources: header-only `riccati` code + a small C++ wrapper TU for Cython.
- Cython module `_riccati.pyx` under `_ivp`, compiled with `cpp_std=c++17`, `dependencies: [np_dep]` (no ccallback needed initially), Eigen include dirs (reuse SciPy’s xsf/Eigen).
- Update `scipy/integrate/meson.build` to build/install the extension; install `solve_ivp_osc` Python wrapper and export via `scipy/integrate/__init__.py`.

## Risks / caveats
- Dense output needs C++ support for on-demand evaluation; current code only returns arrays for explicit `x_eval`.
- Template-heavy C++ may need a thin facade to simplify the Cython interface.
- Mapping `rtol/atol` to `eps`/`epsilon_h` must be explained clearly in docs to avoid user confusion.

## Open minor choice
- Extension name/location: recommendation is `_ivp/_riccati` (keeps IVP logic together) with public `solve_ivp_osc` in `scipy.integrate`. If we hit layout constraints, a sibling `_riccati` under `scipy/integrate` is an alternate.
