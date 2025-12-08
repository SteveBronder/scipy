# SOP: Phase 3 – Wire riccati C++ core (minimal solver, no dense_output)

This is a task-oriented checklist for subagents to implement `solve_ivp_osc` in stages.
You are a senior developer familiar with SciPy's codebase and development practices. Your goal is to execute the implementation of `solve_ivp_osc`, a new ODE solver specialized for oscillatory problems, by migrating and adapting the existing `riccaticpp` codebase into SciPy.

**Goal:** Replace the dummy `_riccati` extension with a minimal working wrapper around the riccaticpp core and implement `solve_ivp_osc` to return an `OdeResult` without dense output support.

---

## Preconditions

- [ ] Phase 1 and Phase 2 SOPs completed and passing tests.
- [ ] riccaticpp headers are available in the repo (under `scipy/integrate/riccaticpp/include/riccati` or ready to move).
- [ ] You can run `pixi run build` and `pixi run test ./scipy/integrate/`.

---

## Step 1 – Prepare C++ headers and includes

- [ ] Move or symlink riccati headers into `scipy/integrate/include/riccati` (per project decision).
- [ ] Ensure Eigen is available via SciPy’s existing setup:
  - [ ] Identify the correct include path(s) for Eigen (from SciPy’s subprojects).
  - [ ] Do **not** introduce new FetchContent/CMake logic; reuse the existing meson-based configuration.
- [ ] Update `scipy/integrate/meson.build` for `_ivp._riccati`:
  - [ ] Add the riccati include directory and Eigen include directories to the extension’s `include_directories`.

---

## Step 2 – Implement a C++ facade (`riccati_wrapper`)

- [ ] Create a header `scipy/integrate/include/riccati/riccati_wrapper.hpp` and implementation `scipy/integrate/src/riccati_wrapper.cpp`.
- [ ] In the wrapper:
  - [ ] Include `riccati/solver.hpp`, `riccati/evolve.hpp`, `riccati/stepsize.hpp`, and any other required headers.
  - [ ] Define concrete, non-templated entry points restricted to `double` / `std::complex<double>`, for example:
    ```cpp
    int riccati_solve(
        /* scalar xi, xf */,
        /* stacked y0 array (y, y') */,
        /* omega_fun / gamma_fun callback hooks */,
        /* eps, epsilon_h */,
        /* optional t_eval */,
        /* output arrays for t, y, ydot, statuses, phases, steptypes */,
        /* error/status reporting */
    );
    ```
  - [ ] Ensure this interface is Cython-friendly (no templates or complex types in the signature).

---

## Step 3 – Extend `_riccati.pyx` to call the facade

- [ ] In `scipy/integrate/_ivp/_riccati.pyx`:
  - [ ] Add `cdef extern from "riccati/riccati_wrapper.hpp":` declarations for `riccati_solve` (and any helpers).
  - [ ] Implement:
    ```cython
    cpdef _riccati_solve(
        double xi,
        double xf,
        np.ndarray y0,
        object omega_fun,
        object gamma_fun,
        double eps,
        double epsilon_h,
        np.ndarray t_eval_or_none
    ):
    ```
  - [ ] Inside `_riccati_solve`:
    - [ ] Validate `y0` is 1-D complex128 (or convertible), representing `[y, y']` stacked as specified.
    - [ ] Convert `t_eval_or_none` to either `NULL`/empty or a 1-D double array.
    - [ ] Provide a mechanism to call `omega_fun` and `gamma_fun` from C++ (e.g. via function pointer shims or Python callback wrappers).
    - [ ] Allocate NumPy arrays for outputs: `t`, `y`, `ydot`, `successes`, `phases`, `steptypes`.
    - [ ] Call `riccati_solve` and fill these arrays.
    - [ ] Return `(t, y, ydot, successes, phases, steptypes)`.

---

## Step 4 – Implement real `solve_ivp_osc` (no dense_output)

- [ ] In `_osc.solve_ivp_osc`:
  - [ ] Keep the signature from Phase 1 (including `events`, `vectorized`, `args`, `**options`).
  - [ ] Replace `NotImplementedError` with:
    - [ ] Conversion of `t_span` to `xi` and `xf` (float64).
    - [ ] Conversion of `y0` to a 1-D `np.ndarray` of complex128 or float64:
      - [ ] Enforce that it contains both `y` and `y'` stacked (document the exact expected shape and ordering).
    - [ ] Mapping `rtol`/`atol` to riccati’s `eps`:
      - [ ] Choose a rule, e.g. `eps = max(rtol, 1e-15)`, and document it.
    - [ ] Determining `epsilon_h`:
      - [ ] If `epsilon_h` is present in `options`, use it.
      - [ ] Otherwise, set `epsilon_h = max(1e-6, rtol)` (or similar).
    - [ ] Handling `t_eval`:
      - [ ] If `t_eval` is provided, convert to a 1-D double array and pass to `_riccati_solve`.
      - [ ] Otherwise, allow the C++ code to choose its internal time grid.
    - [ ] Call `_riccati_solve` and get `(t, y, ydot, successes, phases, steptypes)`.
    - [ ] Build an `OdeResult`:
      - [ ] Use `scipy.integrate._ivp.ivp.OdeResult`.
      - [ ] Fill `t`, `y`, `status`, `message` similarly to `solve_ivp`.
      - [ ] Set `t_events`, `y_events` empty.
      - [ ] Attach an `extra` dict with at least `successes`, `phases`, `steptypes`, and potentially `ydot`.
    - [ ] If `dense_output=True`:
      - [ ] Raise `NotImplementedError("dense_output=True is not yet supported for solve_ivp_osc")` in this phase.

---

## Step 5 – Port and adapt riccaticpp tests (no dense_output)

- [ ] From `scipy/integrate/riccaticpp/tests/python/test.py`, identify:
  - [ ] Schrodinger-related tests.
  - [ ] Bremer equation tests.
  - [ ] Any other riccaticpp-specific tests.
- [ ] Create equivalent tests under `scipy/integrate/_ivp/tests/` (or `scipy/integrate/tests/`) that:
  - [ ] Use the same `omega_fun` and `gamma_fun` definitions.
  - [ ] Use `solve_ivp_osc` instead of the old `pyriccaticpp` interface.
  - [ ] Configure `rtol`, `atol`, and any explicit `epsilon_h` so numerical behavior remains consistent.
  - [ ] Compute the same derived quantities (e.g. energy differences, relative errors).
  - [ ] Match the original tolerances as closely as possible.

---

## Step 6 – Build and test

- [ ] Run:
  - [ ] `pixi run build`
  - [ ] `pixi run test ./scipy/integrate/`
- [ ] Ensure:
  - [ ] New tests pass.
  - [ ] No regressions in existing `scipy.integrate` tests.

---

## Exit criteria for Phase 3

- [ ] `_ivp._riccati` wraps the real riccati C++ core.
- [ ] `solve_ivp_osc` executes without `NotImplementedError` and returns an `OdeResult`.
- [ ] `y0` is documented and enforced as `[y, y']` stacked.
- [ ] Ported riccaticpp tests pass with comparable tolerances.

---

## Additional questions for Phase 3

- [ ] For `y0` stacking, do we want a strict convention (e.g. first half `y`, second half `y'`) with explicit shape checks, or a more flexible scheme validated at runtime?
- [ ] Should failures in the C++ core (e.g. step-size underflow) map to specific Python exceptions or to a nonzero `status` with a descriptive `message` in `OdeResult`?

