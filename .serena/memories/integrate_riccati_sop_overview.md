# integrate/riccaticpp → solve_ivp_osc migration overview

This project is migrating the riccaticpp oscillatory ODE solver into SciPy as `scipy.integrate.solve_ivp_osc`.
Key expectations:

- API shape:
  - Public function: `scipy.integrate.solve_ivp_osc(omega_fun, gamma_fun, t_span, y0, *, t_eval=None, dense_output=False, rtol=1e-3, atol=1e-6, events=None, vectorized=False, args=None, **options)`.
  - `omega_fun`, `gamma_fun` are required Python callables describing the frequency and friction; they may accept scalar or vector inputs.
  - `y0` must eventually encode both `y` and `y'` stacked; the exact convention should be documented clearly.
  - Only `float64` and `complex128` are supported.
  - Return value is an `OdeResult`-like object; riccati-specific details (step types, phases, statuses, possibly derivatives) go into `result.extra`.
  - `events` support is explicitly out-of-scope for the first implementation, but the parameter should be accepted for forward compatibility.

- Integration strategy:
  - Use Cython + C++17 + Eigen, following existing SciPy patterns. pybind11/CMake build machinery from the standalone riccaticpp project will not be used directly.
  - The compiled core lives in `_ivp._riccati` (Cython module + C++ facade). The Python-facing function and dense-output logic live in `_ivp/_osc.py` (and possibly a small helper module).
  - riccati headers are to be placed under `scipy/integrate/include/riccati`, and Eigen is reused from SciPy’s existing meson configuration.

- Phased SOPs:
  - Phase 1: stub `solve_ivp_osc` (argument validation only, raising `NotImplementedError`), wired into `scipy.integrate` with tests.
  - Phase 2: dummy `_ivp._riccati` Cython extension with a trivial function, fully meson-integrated and tested.
  - Phase 3: wire riccati C++ core and implement a working `solve_ivp_osc` without `dense_output`, returning `OdeResult` and passing ported riccaticpp tests.
  - Phase 4: add dense-output support using per-step interpolation data and a `DenseOutput` subclass; `sol(t)` should be accurate according to the tests.
  - Phase 5: documentation, exports, and release notes polish; treat `solve_ivp_osc` as a first-class solver, clearly described as specialized for oscillatory problems.

- Testing and parity:
  - All existing riccaticpp tests (Schrodinger, Bremer, etc.) should be ported and made to call `solve_ivp_osc`.
  - Tolerances in SciPy tests should match the original riccaticpp expectations as strictly as feasible; any deviation should be justified.
  - Dense-output accuracy should be governed by tests derived from the original project.

- Error handling & UX:
  - Where possible, match `solve_ivp`’s error types and message wording for invalid inputs.
  - `events` and other advanced features may be accepted as arguments but are not implemented in the first version; they should either be ignored with clear documentation or raise explicit errors once the core is stable.

There are per-phase SOPs in `design-docs/riccati_solve_ivp_osc_phase*.md` that subagents should follow; they give concrete file paths, build commands, and per-phase exit criteria. Future agents should lean on those SOPs instead of improvising the process, to keep the migration predictable and reviewable.