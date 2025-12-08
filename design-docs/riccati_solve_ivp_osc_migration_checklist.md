# Checklist: Migrating `riccaticpp` into `scipy.integrate` as `solve_ivp_osc`

This is a task-oriented checklist for subagents to implement `solve_ivp_osc` in stages.
You are a senior developer familiar with SciPy's codebase and development practices. Your goal is to execute the implementation of `solve_ivp_osc`, a new ODE solver specialized for oscillatory problems, by migrating and adapting the existing `riccaticpp` codebase into SciPy.

---

## 0. Context & constraints

- [ ] Public API must be `scipy.integrate.solve_ivp_osc` (standalone, not a `method=` for `solve_ivp`).
- [ ] Arguments must include specialized pieces: `omega_fun`, `gamma_fun`.
- [ ] Only support `float64` / `complex128` for now.
- [ ] Use SciPy-style `rtol`/`atol`; map to riccati’s tolerances (`eps`, `epsilon_h`).
- [ ] Return an `OdeResult`-like object; place riccati-specific diagnostics in `result.extra`.
- [ ] Implementation should use Cython + C++ (C++17) with Eigen; no pybind11.
- [ ] Start simple and iterate: minimal stub → buildable extension → working solver → dense output → docs/polish.

This section is informational; implementation work happens in the phases below.

### Questions / clarifications for context

- [ ] Events are out-of-scope for the initial implementation; plan event support only after the core solver is stable.
- [ ] In docs, give `solve_ivp_osc` the same prominence as `solve_ivp`, but explicitly note that it is specialized for oscillatory problems.
---

## Phase 1 – Public Python stub (no compiled code)

**Goal:** Make `solve_ivp_osc` importable from `scipy.integrate` with a reasonable signature and argument checks, but no solver logic yet.

### 1.1 Add the stub function

- [ ] Create a new module file, e.g. `scipy/integrate/_ivp/_osc.py`.
- [ ] In that file, define:
  - [ ] `def solve_ivp_osc(omega_fun, gamma_fun, t_span, y0, *, t_eval=None, dense_output=False, rtol=1e-3, atol=1e-6, **options):`
  - [ ] Add a docstring describing the function, its parameters, return value, and how it differs from `solve_ivp` (use of `omega_fun`/`gamma_fun`, restricted dtypes, etc.).
  - [ ] Basic checks:
    - [ ] `t_span` is a 2-element sequence of floats.
    - [ ] `y0` is array-like and convertible to a 1-D NumPy array.
    - [ ] `omega_fun` and `gamma_fun` are callables.
    - [ ] `rtol` and `atol` are positive scalars.
  - [ ] For now, return None.

### 1.2 Wire into the package namespace

- [ ] In `scipy/integrate/_ivp/__init__.py`, import and re-export:
  - [ ] `from ._osc import solve_ivp_osc`
- [ ] In `scipy/integrate/__init__.py`:
  - [ ] Add `solve_ivp_osc` to the imports from `._ivp`.
  - [ ] Ensure `solve_ivp_osc` appears in `__all__`.

### 1.3 Add stub tests

- [ ] Create a test file, e.g. `scipy/integrate/_ivp/tests/test_solve_ivp_osc_stub.py`.
- [ ] Tests:
  - [ ] `from scipy.integrate import solve_ivp_osc` does not raise.
  - [ ] Calling with obviously wrong args (e.g. `omega_fun=None`) raises `TypeError` / `ValueError` as implemented.
  - [ ] Calling with valid-looking args raises `NotImplementedError`.

### 1.4 Verify build & tests
- [ ] Ensure `.venv` is active.
- [ ] Run `pixi run build`.
- [ ] Run `pixi run test ./scipy/integrate/`.

**Exit criteria Phase 1:** `solve_ivp_osc` is importable, argument-checked, and gated behind `NotImplementedError`. All integrate tests still pass.

### Questions / clarifications for Phase 1

- [ ] The stub (and final implementation) should accept additional keywords for forward compatibility: `events=None`, `vectorized=False`, `args=None`, plus `**options`, even if they are not yet used internally.
- [ ] Argument validation and error messages should match `solve_ivp`’s behavior and wording as closely as practical.
---

## Phase 2 – Cython extension skeleton `_riccati` (dummy implementation)

**Goal:** Add a Cython C++ extension to the build so that tooling and meson wiring are in place before introducing the actual riccati algorithm.

Note: `_riccati.pyx` is a Cython module **within** `_ivp` (no extra package level); it exists solely to host the compiled core while keeping the Python-facing API logic in `_osc.py`.

### 2.1 Create Cython module

- [ ] Add `scipy/integrate/_ivp/_riccati.pyx` with a minimal content:
  - [ ] `cpdef int _dummy_riccati(int x):` returning `x`.
  - [ ] No riccati headers or Eigen yet.

### 2.2 Meson build integration

- [ ] In `scipy/integrate/meson.build`:
  - [ ] Add a new extension, e.g.:
    - [ ] `py3.extension_module('_ivp._riccati', [ lib_cython_gen.process('_ivp/_riccati.pyx') ], dependencies: [np_dep], subdir: 'scipy/integrate')`
    - [ ] Use `link_args: version_link_args` as for other extensions.
  - [ ] Ensure install path matches `scipy/integrate/_ivp/_riccati.*.so`.

### 2.3 Exercise the extension from Python

- [ ] In `scipy/integrate/_ivp/_osc.py`:
  - [ ] Add `from . import _riccati as _ric` near the top.
  - [ ] In `solve_ivp_osc` stub, call `_ric._dummy_riccati(1)` (ignore result) to ensure the extension imports and runs.

### 2.4 Tests

- [ ] Add a small test in, e.g. `scipy/integrate/_ivp/tests/test_riccati_dummy.py`:
  - [ ] Import `_ivp._riccati` and assert `_dummy_riccati(5) == 5`.
- [ ] Confirm existing stub tests still pass.

### 2.5 Verify build & tests

- [ ] Run `pixi run build`.
- [ ] Run `pixi run test ./scipy/integrate/`.

**Exit criteria Phase 2:** `_ivp._riccati` builds and is callable from Python; `solve_ivp_osc` still raises `NotImplementedError` but now exercises the extension lightly.

### Questions / clarifications for Phase 2

- [ ] Follow existing SciPy conventions: keep Python API modules (`_ivp/ivp.py`, `_ivp/base.py`, etc.) separate from Cython/compiled modules, and use a simple name like `_riccati` under `_ivp` for this solver’s core.
- [ ] No special platform constraints beyond normal SciPy CI; keep the dummy extension minimal and portable C++17.
---

## Phase 3 – Wire riccati C++ core (minimal solver, no dense_output)

**Goal:** Replace the dummy extension with a minimal working wrapper to the riccaticpp core and implement a basic `solve_ivp_osc` that returns an `OdeResult` without dense output.

### 3.1 Prepare C++ code inside SciPy

- [ ] Decide where to place the riccati headers (or reference them):
  - [ ] Place the riccati headers under `scipy/integrate/include/riccati` and adjust include paths accordingly.
- [ ] Ensure Eigen is available:
  - [ ] Add necessary meson dependency for Eigen.
  - [ ] Confirm existing SciPy Eigen/xsf subproject and include paths (do not re-fetch with CMake).
  - [ ] Add the include directory for Eigen and the riccati headers to the `_ivp._riccati` meson target.
- [ ] Write a C++ facade (e.g. `scipy/integrate/src/riccati_wrapper.cpp`) that:
  - [ ] Includes the necessary riccati headers (`solver.hpp`, `evolve.hpp`, `stepsize.hpp`, etc.).
  - [ ] Provides C-friendly functions wrapping the main solve routine:
    - [ ] Example: `int riccati_solve(..., /* arrays for t, y, etc. */)`.
  - [ ] Restricts template exposure at the Cython boundary (convert to concrete types like `double` / `std::complex<double>`).

### 3.2 Extend Cython `_riccati.pyx` to call the facade

- [ ] In `_riccati.pyx`:
  - [ ] Add `cdef extern from "riccati_wrapper.hpp"` (or similar) declarations for the C++ facade.
  - [ ] Implement a helper:
    - [ ] `cpdef _riccati_solve(double xi, double xf, np.ndarray y0, object omega_fun, object gamma_fun, double eps, double epsilon_h, np.ndarray t_eval_or_none):`
    - [ ] Responsibilities:
      - [ ] Validate and convert `y0` to `np.ndarray` of `complex128` (if needed).
      - [ ] Manage calls to `omega_fun`, `gamma_fun` as required by the facade.
      - [ ] Allocate output arrays for `t`, `y`, (optionally `y'`), status/step-type info.
      - [ ] Call the `riccati_solve` facade and fill NumPy arrays.
      - [ ] Return `(t, y, ydot, successes, phases, steptypes)` (or subset as agreed).

### 3.3 Implement `solve_ivp_osc` using `_riccati_solve`

- [ ] In `_osc.solve_ivp_osc`:
  - [ ] Replace `NotImplementedError` with real logic:
    - [ ] Convert `t_span` to `xi`, `xf` (float64).
    - [ ] Convert `y0` to `np.ndarray` of `complex128` or `float64` depending on input.
    - [ ] Map `rtol`/`atol` to riccati’s `eps`:
      - [ ] Choose an initial mapping rule, e.g. `eps = max(rtol, 1e-15)`; document it in code comments.
    - [ ] Determine `epsilon_h`:
      - [ ] If user passes `epsilon_h` in `**options`, use it.
      - [ ] Else set `epsilon_h = max(1e-6, rtol)` or similar.
    - [ ] If `t_eval` is not `None`, convert it to a 1-D `np.ndarray` and pass to `_riccati_solve`:
      - [ ] For this phase, treat `t_eval` as the only requested output points.
    - [ ] Otherwise, allow the C++ core to choose its own internal mesh and return those times.
  - [ ] Construct an `OdeResult`:
    - [ ] Use `scipy.integrate._ivp.ivp.OdeResult` (same as `solve_ivp`).
    - [ ] Fill `t`, `y`, and simple `status`/`message`.
    - [ ] Leave `t_events` and `y_events` empty.
    - [ ] Add `extra` dict: `{"successes": successes, "phases": phases, "steptypes": steptypes}`.
  - [ ] For now, if `dense_output=True` is requested:
    - [ ] Raise `NotImplementedError("dense_output=True is not yet supported for solve_ivp_osc")`.

### 3.4 Port and adapt riccaticpp tests (no dense_output assumptions)

- [ ] Take existing tests from `scipy/integrate/riccaticpp/tests/python/test.py`:
  - [ ] Schrodinger equations tests.
  - [ ] Bremer equation tests.
- [ ] Rewrite them to call `solve_ivp_osc`:
  - [ ] Define `omega_fun` and `gamma_fun` as before.
  - [ ] Choose appropriate `rtol`, `atol`, and any explicit `epsilon_h` so numerical behavior is consistent with the original tests.
  - [ ] Compute derived quantities (energy residuals, relative errors) using the `OdeResult` `t`/`y`.
  - [ ] Keep tolerance checks similar to the original tests, but adjust as needed for any small differences introduced by the mapping to `rtol`/`atol`.
- [ ] Place these tests under `scipy/integrate/_ivp/tests/` or `scipy/integrate/tests/` per SciPy conventions.

### 3.5 Verify build & tests

- [ ] Run `pixi run build`.
- [ ] Run `pixi run test ./scipy/integrate/`.

**Exit criteria Phase 3:** `solve_ivp_osc` runs, returns an `OdeResult`, and passes ported riccaticpp tests (without `dense_output` support). Extension builds across supported configurations.

### Questions / clarifications for Phase 3

- [ ] The riccati core currently evolves both `y` and `y'`. Should `solve_ivp_osc`:
  - [ ] Require `y0` to encode both `y` and `y'` stacked together (document the expected layout clearly in the docstring and tests).
- [ ] Ensure parity with all existing riccaticpp tests (Schrodinger, Bremer, and any others) when ported to `solve_ivp_osc`.
- [ ] Match the original riccaticpp test tolerances as strictly as possible; differences should only come from unavoidable numerical or API changes.
---

## Phase 4 – Dense output (`dense_output=True` / `sol(t)`)

**Goal:** Support `dense_output=True` and an `OdeSolution`-like `sol(t)` similar to `solve_ivp`.

### 4.1 Extend C++ core to expose interpolation data

- [ ] Identify the minimal per-step data needed to evaluate `y(t)` inside a step (Chebyshev coefficients, Riccati step state).
- [ ] Define a C++ struct, e.g. `RiccatiStepRecord`, containing:
  - [ ] `x_start`, `x_end`.
  - [ ] Coefficients needed for interpolation.
  - [ ] Flags for oscillatory vs non-oscillatory step, etc.
- [ ] Modify the C++ facade so that:
  - [ ] When requested (e.g. a flag), it records and returns a vector of `RiccatiStepRecord`s alongside the discrete solution.
  - [ ] Provide a function `eval_step(const RiccatiStepRecord&, double t, complex<double>* y_out)` (and optionally derivative output).

### 4.2 Cython bindings for step records and evaluation

- [ ] In `_riccati.pyx`:
  - [ ] Add `cdef` declarations for `RiccatiStepRecord` and `eval_step`.
  - [ ] Wrap step records as opaque Python objects (e.g. using `cdef class _RiccatiStepRecord` that holds a pointer or index).
  - [ ] Extend `_riccati_solve` to optionally return a list of step records when dense output is requested.
- [ ] Add a small helper:
  - [ ] `_riccati_eval(record, double t)` returning `np.ndarray` for `y(t)` at a single `t` (and maybe `dy(t)` if needed).

### 4.3 Implement a `DenseOutput` subclass

- [ ] In Python (probably under `_ivp`), define `RiccatiDenseOutput(DenseOutput)`:
  - [ ] Store `t_min`, `t_max`, and the list of step records (and any global info needed).
  - [ ] Implement `_call_impl(self, t)`:
    - [ ] Convert `t` to a NumPy array.
    - [ ] For each `t`, find the step record where `x_start <= t <= x_end` (or the integration direction equivalent).
    - [ ] Call `_riccati_eval(record, t)` and stack results into an array shaped like `y`.
  - [ ] Support vectorized `t` inputs.

### 4.4 Wire `dense_output=True` into `solve_ivp_osc`

- [ ] In `solve_ivp_osc`:
  - [ ] If `dense_output=True`:
    - [ ] Call `_riccati_solve` in a mode that collects step records.
    - [ ] Build a `RiccatiDenseOutput` instance.
    - [ ] Attach it to the `OdeResult` `sol` attribute.
  - [ ] Optionally, if `t_eval` is given:
    - [ ] Either pass `t_eval` directly to the C++ core as `x_eval`, or
    - [ ] Use `result.sol(t_eval)` to compute `y(t_eval)` post-hoc (for consistency with `solve_ivp`).

### 4.5 Tests for dense output

- [ ] Add tests that:
  - [ ] Call `solve_ivp_osc(..., dense_output=True)` on one or two representative problems.
  - [ ] Sample `sol(t_eval2)` at multiple points and compare to:
    - [ ] High-resolution reference solutions, or
    - [ ] Direct C++ dense outputs where available.
  - [ ] Verify shapes and dtypes of outputs.

### 4.6 Verify build & tests

- [ ] Run `pixi run build`.
- [ ] Run `pixi run test ./scipy/integrate/`.

**Exit criteria Phase 4:** `solve_ivp_osc` supports `dense_output=True`, returning a functioning `sol(t)` consistent with the underlying riccati solver.

### Questions / clarifications for Phase 4

- [ ] Treat the tests as the source of truth for required dense-output accuracy; dense evaluations should satisfy the same expectations encoded in the riccaticpp-derived tests.
- [ ] It is acceptable if dense output is somewhat more expensive per point than `solve_ivp`’s built-in dense solvers, given the specialized algorithm.
- [ ] Expose derivative information `y'(t)` via the result’s `extra` dictionary (e.g. storing per-step derivative data or allowing evaluation of `y'` at dense points).
---

## Phase 5 – Documentation, polish, and optional follow-ups

### 5.1 Documentation

- [ ] Add `solve_ivp_osc` to the integrate reference docs (e.g. `doc/source/reference/integrate.rst`).
- [ ] Document:
  - [ ] Full signature, including `omega_fun`, `gamma_fun`, `epsilon_h`.
  - [ ] Mapping from `rtol`/`atol` to internal error controls.
  - [ ] Supported dtypes and limitations.
  - [ ] Example(s) for oscillatory problems (e.g. brief Schrodinger example).
- [ ] Ensure docstrings follow numpydoc style and integrate into API docs.

### 5.2 Final polish

- [ ] Make sure `__all__` in `scipy.integrate` includes `solve_ivp_osc`.
- [ ] Confirm no leftover references to standalone `pyriccaticpp` packaging; core is fully integrated via meson/Cython.
- [ ] Check error messages for clarity (argument validation, unsupported features).

### 5.3 Optional future work

- [ ] Investigate LowLevelCallable-style support if performance of Python callbacks is a bottleneck.
- [ ] Add more benchmarks and possibly integrate them in `benchmarks/`.

**Exit criteria Phase 5:** `solve_ivp_osc` is a documented, tested, first-class solver in `scipy.integrate`, with a clear roadmap for future enhancements.

### Questions / clarifications for Phase 5

- [ ] Should `solve_ivp_osc` be highlighted in release notes as a new feature once merged, and if so, under which section (integrate/ODE solvers)?
- [ ] Highlight `solve_ivp_osc` in the release notes under the `scipy.integrate` / ODE solvers section.
- [ ] No specific figures are required initially; examples can remain minimal for the first version.
- [ ] Do not label the API as “experimental”, but clearly note in docs that `solve_ivp_osc` is specialized for oscillatory problems.
