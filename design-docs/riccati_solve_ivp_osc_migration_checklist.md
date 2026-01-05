# Checklist: Migrating `riccaticpp` into `scipy.integrate` as `solve_ivp_osc`

Status: Phase 1-4 complete. Phase 5 in progress.

This is a task-oriented checklist for subagents to implement `solve_ivp_osc` in stages.
You are a senior developer familiar with SciPy's codebase and development practices. Your goal is to execute the implementation of `solve_ivp_osc`, a new ODE solver specialized for oscillatory problems, by migrating and adapting the existing `riccaticpp` codebase into SciPy.

---

## 0. Context & constraints

- [x] Public API must be `scipy.integrate.solve_ivp_osc` (standalone, not a `method=` for `solve_ivp`).
- [x] Arguments must include specialized pieces: `omega_fun`, `gamma_fun`.
- [x] Only support `float64` / `complex128` for now.
- [x] Use SciPy-style `rtol`/`atol`; map to riccati’s tolerances (`eps`, `epsilon_h`).
- [x] Return an `OdeResult`-like object; attach riccati diagnostics as attributes.
- [x] Implementation uses pybind11 + C++17 with Eigen.
- [x] Start simple and iterate: minimal stub → buildable extension → working solver → dense output → docs/polish.

This section is informational; implementation work happens in the phases below.

### Questions / clarifications for context

- [x] Events are out-of-scope for the initial implementation; plan event support only after the core solver is stable.
- [x] In docs, give `solve_ivp_osc` the same prominence as `solve_ivp`, but explicitly note that it is specialized for oscillatory problems.
---

## Phase 1 – Public Python stub (no compiled code)

**Goal:** Make `solve_ivp_osc` importable from `scipy.integrate` with a reasonable signature and argument checks, but no solver logic yet.

### 1.1 Add the stub function

- [x] Create a new module file, e.g. `scipy/integrate/_ivp/_osc.py`.
- [x] In that file, define:
  - [x] `def solve_ivp_osc(omega_fun, gamma_fun, t_span, y0, *, t_eval=None, dense_output=False, rtol=1e-3, atol=1e-6, **options):`
  - [x] Add a docstring describing the function, its parameters, return value, and how it differs from `solve_ivp` (use of `omega_fun`/`gamma_fun`, restricted dtypes, etc.).
  - [x] Basic checks:
    - [x] `t_span` is a 2-element sequence of floats.
    - [x] `y0` is array-like and convertible to a 1-D NumPy array.
    - [x] `omega_fun` and `gamma_fun` are callables.
    - [x] `rtol` and `atol` are positive scalars.
  - [x] For now, return None.

### 1.2 Wire into the package namespace

- [x] In `scipy/integrate/_ivp/__init__.py`, import and re-export:
  - [x] `from ._osc import solve_ivp_osc`
- [x] In `scipy/integrate/__init__.py`:
  - [x] Add `solve_ivp_osc` to the imports from `._ivp`.
  - [x] Ensure `solve_ivp_osc` appears in `__all__`.

### 1.3 Add stub tests

- [x] Create a test file, e.g. `scipy/integrate/_ivp/tests/test_solve_ivp_osc_stub.py`.
- [x] Tests:
  - [x] `from scipy.integrate import solve_ivp_osc` does not raise.
  - [x] Calling with obviously wrong args (e.g. `omega_fun=None`) raises `TypeError` / `ValueError` as implemented.
  - [x] Calling with valid-looking args raises `NotImplementedError`.

### 1.4 Verify build & tests
- [x] Ensure `.venv` is active.
- [x] Run `pixi run build`.
- [x] Run `pixi run test ./scipy/integrate/`.

**Exit criteria Phase 1:** `solve_ivp_osc` is importable, argument-checked, and gated behind `NotImplementedError`. All integrate tests still pass.

### Questions / clarifications for Phase 1

- [x] The stub (and final implementation) should accept additional keywords for forward compatibility: `events=None`, `vectorized=False`, `args=None`, plus `**options`, even if they are not yet used internally.
- [x] Argument validation and error messages should match `solve_ivp`’s behavior and wording as closely as practical.
---

## Phase 2 – Cython extension skeleton `_riccati` (dummy implementation)

**Goal:** Add a Cython C++ extension to the build so that tooling and meson wiring are in place before introducing the actual riccati algorithm.

Note: `_riccati.pyx` is a Cython module **within** `_ivp` (no extra package level); it exists solely to host the compiled core while keeping the Python-facing API logic in `_osc.py`.

### 2.1 Create Cython module

- [x] Add `scipy/integrate/_ivp/_riccati.pyx` with a minimal content:
  - [x] `cpdef int _dummy_riccati(int x):` returning `x`.
  - [x] No riccati headers or Eigen yet.

### 2.2 Meson build integration

- [x] In `scipy/integrate/meson.build`:
  - [x] Add a new extension, e.g.:
    - [x] `py3.extension_module('_ivp._riccati', [ lib_cython_gen.process('_ivp/_riccati.pyx') ], dependencies: [np_dep], subdir: 'scipy/integrate')`
    - [x] Use `link_args: version_link_args` as for other extensions.
  - [x] Ensure install path matches `scipy/integrate/_ivp/_riccati.*.so`.

### 2.3 Exercise the extension from Python

- [x] In `scipy/integrate/_ivp/_osc.py`:
  - [x] Add `from . import _riccati as _ric` near the top.
  - [x] In `solve_ivp_osc` stub, call `_ric._dummy_riccati(1)` (ignore result) to ensure the extension imports and runs.

### 2.4 Tests

- [x] Add a small test in, e.g. `scipy/integrate/_ivp/tests/test_riccati_dummy.py`:
  - [x] Import `_ivp._riccati` and assert `_dummy_riccati(5) == 5`.
- [x] Confirm existing stub tests still pass.

### 2.5 Verify build & tests

- [x] Run `pixi run build`.
- [x] Run `pixi run test ./scipy/integrate/`.

**Exit criteria Phase 2:** `_ivp._riccati` builds and is callable from Python; `solve_ivp_osc` still raises `NotImplementedError` but now exercises the extension lightly.

### Questions / clarifications for Phase 2

- [x] Follow existing SciPy conventions: keep Python API modules (`_ivp/ivp.py`, `_ivp/base.py`, etc.) separate from Cython/compiled modules, and use a simple name like `_riccati` under `_ivp` for this solver’s core.
- [x] No special platform constraints beyond normal SciPy CI; keep the dummy extension minimal and portable C++17.
---

## Phase 3 – Wire riccati C++ core (minimal solver, no dense_output)

**Goal:** Replace the dummy extension with a minimal working wrapper to the riccaticpp core and implement a basic `solve_ivp_osc` that returns an `OdeResult` without dense output.

### 3.1 Prepare C++ code inside SciPy

- [x] Decide where to place the riccati headers (or reference them):
  - [x] Place the riccati headers under `scipy/integrate/include/riccati` and adjust include paths accordingly.
- [x] Ensure Eigen is available:
  - [x] Add necessary meson dependency for Eigen.
  - [x] Confirm existing SciPy Eigen/xsf subproject and include paths (do not re-fetch with CMake).
  - [x] Add the include directory for Eigen and the riccati headers to the `_ivp._riccati` meson target.
- [x] Write a C++ facade (e.g. `scipy/integrate/src/riccati_wrapper.cpp`) that:
  - [x] Includes the necessary riccati headers (`solver.hpp`, `evolve.hpp`, `stepsize.hpp`, etc.).
  - [x] Provides C-friendly functions wrapping the main solve routine:
    - [x] Example: `int riccati_solve(..., /* arrays for t, y, etc. */)`.
  - [x] Restricts template exposure at the Cython boundary (convert to concrete types like `double` / `std::complex<double>`).

### 3.2 Extend Cython `_riccati.pyx` to call the facade

- [x] In `_riccati.pyx`:
  - [x] Add `cdef extern from "riccati_wrapper.hpp"` (or similar) declarations for the C++ facade.
  - [x] Implement a helper:
    - [x] `cpdef _riccati_solve(double xi, double xf, np.ndarray y0, object omega_fun, object gamma_fun, double eps, double epsilon_h, np.ndarray t_eval_or_none):`
    - [x] Responsibilities:
      - [x] Validate and convert `y0` to `np.ndarray` of `complex128` (if needed).
      - [x] Manage calls to `omega_fun`, `gamma_fun` as required by the facade.
      - [x] Allocate output arrays for `t`, `y`, (optionally `y'`), status/step-type info.
      - [x] Call the `riccati_solve` facade and fill NumPy arrays.
      - [x] Return `(t, y, ydot, successes, phases, steptypes)` (or subset as agreed).

### 3.3 Implement `solve_ivp_osc` using `_riccati_solve`

- [x] In `_osc.solve_ivp_osc`:
  - [x] Replace `NotImplementedError` with real logic:
    - [x] Convert `t_span` to `xi`, `xf` (float64).
    - [x] Convert `y0` to `np.ndarray` of `complex128` or `float64` depending on input.
    - [x] Map `rtol`/`atol` to riccati’s `eps`:
      - [x] Choose an initial mapping rule, e.g. `eps = max(rtol, 1e-15)`; document it in code comments.
    - [x] Determine `epsilon_h`:
      - [x] If user passes `epsilon_h` in `**options`, use it.
      - [x] Else set `epsilon_h = max(1e-6, rtol)` or similar.
    - [x] If `t_eval` is not `None`, convert it to a 1-D `np.ndarray` and pass to `_riccati_solve`:
      - [x] For this phase, treat `t_eval` as the only requested output points.
    - [x] Otherwise, allow the C++ core to choose its own internal mesh and return those times.
  - [x] Construct an `OdeResult`:
    - [x] Use `scipy.integrate._ivp.ivp.OdeResult` (same as `solve_ivp`).
    - [x] Fill `t`, `y`, and simple `status`/`message`.
    - [x] Leave `t_events` and `y_events` empty.
    - [x] Add `extra` dict: `{"successes": successes, "phases": phases, "steptypes": steptypes}`.
  - [x] For now, if `dense_output=True` is requested:
    - [x] Raise `NotImplementedError("dense_output=True is not yet supported for solve_ivp_osc")`.

### 3.4 Port and adapt riccaticpp tests (no dense_output assumptions)

- [x] Take existing tests from `scipy/integrate/riccaticpp/tests/python/test.py`:
  - [x] Schrodinger equations tests.
  - [x] Bremer equation tests.
- [x] Rewrite them to call `solve_ivp_osc`:
  - [x] Define `omega_fun` and `gamma_fun` as before.
  - [x] Choose appropriate `rtol`, `atol`, and any explicit `epsilon_h` so numerical behavior is consistent with the original tests.
  - [x] Compute derived quantities (energy residuals, relative errors) using the `OdeResult` `t`/`y`.
  - [x] Keep tolerance checks similar to the original tests, but adjust as needed for any small differences introduced by the mapping to `rtol`/`atol`.
- [x] Place these tests under `scipy/integrate/_ivp/tests/` or `scipy/integrate/tests/` per SciPy conventions.

### 3.5 Verify build & tests

- [x] Run `pixi run build`.
- [x] Run `pixi run test ./scipy/integrate/`.

**Exit criteria Phase 3:** `solve_ivp_osc` runs, returns an `OdeResult`, and passes ported riccaticpp tests (without `dense_output` support). Extension builds across supported configurations.

### Questions / clarifications for Phase 3

- [x] The riccati core currently evolves both `y` and `y'`. Should `solve_ivp_osc`:
  - [x] Require `y0` to encode both `y` and `y'` stacked together (document the expected layout clearly in the docstring and tests).
- [x] Ensure parity with all existing riccaticpp tests (Schrodinger, Bremer, and any others) when ported to `solve_ivp_osc`.
- [x] Match the original riccaticpp test tolerances as strictly as possible; differences should only come from unavoidable numerical or API changes.
---

## Phase 4 – Dense output (`dense_output=True` / `sol(t)`)

**Goal:** Support `dense_output=True` and an `OdeSolution`-like `sol(t)` similar to `solve_ivp`.

Note: Dense output is implemented by re-evaluating the solver at requested points,
not by per-step record interpolation.

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

- [x] In Python (under `_ivp`), define `RiccatiDenseOutput(DenseOutput)`:
  - [x] Store `t_min`, `t_max`, and solver parameters needed for re-evaluation.
  - [x] Implement `_call_impl(self, t)`:
    - [x] Convert `t` to a NumPy array.
    - [x] Re-evaluate the solver using `t_eval=t` and return the dense values.
  - [x] Support vectorized `t` inputs.

### 4.4 Wire `dense_output=True` into `solve_ivp_osc`

- [x] In `solve_ivp_osc`:
  - [x] If `dense_output=True`:
    - [x] Build a `RiccatiDenseOutput` instance.
    - [x] Attach it to the `OdeResult` `sol` attribute.
  - [x] If `t_eval` is given:
    - [x] Pass `t_eval` directly to the C++ core as `x_eval`.

### 4.5 Tests for dense output

- [x] Add tests that:
  - [x] Call `solve_ivp_osc(..., dense_output=True)` on one or two representative problems.
  - [x] Sample `sol(t_eval2)` at multiple points and compare to:
    - [x] High-resolution reference solutions, or
    - [x] Direct C++ dense outputs where available.
  - [x] Verify shapes and dtypes of outputs.

### 4.6 Verify build & tests

- [x] Run `pixi run build`.
- [x] Run `pixi run test ./scipy/integrate/`.

**Exit criteria Phase 4:** `solve_ivp_osc` supports `dense_output=True`, returning a functioning `sol(t)` consistent with the underlying riccati solver.

### Questions / clarifications for Phase 4

- [x] Treat the tests as the source of truth for required dense-output accuracy; dense evaluations should satisfy the same expectations encoded in the riccaticpp-derived tests.
- [x] It is acceptable if dense output is somewhat more expensive per point than `solve_ivp`’s built-in dense solvers, given the specialized algorithm.
- [x] Expose derivative information `y'(t)` via the result’s `extra` dictionary (e.g. storing per-step derivative data or allowing evaluation of `y'` at dense points).
---

## Phase 5 – Documentation, polish, and optional follow-ups

### 5.1 Documentation

- [x] Add `solve_ivp_osc` to the integrate reference docs (e.g. `doc/source/reference/integrate.rst`).
- [x] Document:
  - [x] Full signature, including `omega_fun`, `gamma_fun`, `epsilon_h`.
  - [x] Mapping from `rtol`/`atol` to internal error controls.
  - [x] Supported dtypes and limitations.
  - [x] Example(s) for oscillatory problems (e.g. brief Schrodinger example).
- [x] Ensure docstrings follow numpydoc style and integrate into API docs.

### 5.2 Final polish

- [x] Make sure `__all__` in `scipy.integrate` includes `solve_ivp_osc`.
- [ ] Confirm no leftover references to standalone `pyriccaticpp` packaging; core is fully integrated via meson/Cython.
- [x] Check error messages for clarity (argument validation, unsupported features).

### 5.3 Optional future work

- [ ] Investigate LowLevelCallable-style support if performance of Python callbacks is a bottleneck.
- [ ] Add more benchmarks and possibly integrate them in `benchmarks/`.

**Exit criteria Phase 5:** `solve_ivp_osc` is a documented, tested, first-class solver in `scipy.integrate`, with a clear roadmap for future enhancements.

### Questions / clarifications for Phase 5

- [x] Should `solve_ivp_osc` be highlighted in release notes as a new feature once merged, and if so, under which section (integrate/ODE solvers)?
- [x] Highlight `solve_ivp_osc` in the release notes under the `scipy.integrate` / ODE solvers section.
- [x] No specific figures are required initially; examples can remain minimal for the first version.
- [x] Do not label the API as “experimental”, but clearly note in docs that `solve_ivp_osc` is specialized for oscillatory problems.
