# SOP: Phase 4 – Dense output for `solve_ivp_osc`

This is a task-oriented checklist for subagents to implement `solve_ivp_osc` in stages.
You are a senior developer familiar with SciPy's codebase and development practices. Your goal is to execute the implementation of `solve_ivp_osc`, a new ODE solver specialized for oscillatory problems, by migrating and adapting the existing `riccaticpp` codebase into SciPy.

**Goal:** Add support for `dense_output=True` and `sol(t)` on the result, similar to `solve_ivp`, using per-step interpolation data from the riccati core.

---

## Preconditions

- [x] Phase 3 SOP completed and passing tests.
- [x] `_ivp._riccati` exposes a working core solver for discrete outputs.
- [x] You can run `pixi run build` and `pixi run test ./scipy/integrate/`.

---

## Step 1 – Extend C++ core to expose interpolation data

- [ ] In the riccati C++ layer (`solver.hpp`, `evolve.hpp` and related), identify:
  - [ ] The exact data needed to interpolate `y(t)` (and possibly `y'(t)`) within each step.
  - [ ] How oscillatory vs non-oscillatory steps differ in interpolation.
- [ ] Define a C++ struct `RiccatiStepRecord` that contains:
  - [ ] `double x_start`, `x_end`.
  - [ ] Coefficients or state needed for interpolation.
  - [ ] Flags for step type (osc / non-osc).
- [ ] Modify the C++ facade (`riccati_wrapper`) so that:
  - [ ] When a “dense output” flag is set, it collects and returns a `std::vector<RiccatiStepRecord>` alongside the discrete solution.
  - [ ] It provides:
    ```cpp
    void riccati_eval_step(
        const RiccatiStepRecord& rec,
        double t,
        std::complex<double>* y_out,
        std::complex<double>* ydot_out  // nullable if derivative not requested
    );
    ```

---

## Step 2 – Bind step records and evaluation in `_riccati.pyx`

- [ ] In `_riccati.pyx`:
  - [ ] Add `cdef extern from "riccati/riccati_wrapper.hpp":` declarations for:
    - [ ] `ctypedef struct RiccatiStepRecord: ...`
    - [ ] `void riccati_eval_step(const RiccatiStepRecord&, double, complex[double]*, complex[double]*)`
  - [ ] Extend `_riccati_solve` to optionally:
    - [ ] Accept a flag indicating dense output is requested.
    - [ ] Return a Python list (or array) of step-record wrappers along with `(t, y, ydot, successes, phases, steptypes)`.
- [ ] Implement a small helper in Cython:
  ```cython
  cpdef _riccati_eval(record, double t):
      # returns (y_t, ydot_t) as NumPy arrays or scalars
  ```

---

## Step 3 – Implement a `RiccatiDenseOutput` class

- [x] In a Python module under `_ivp` (e.g. `scipy/integrate/_ivp/_osc_dense.py` or inside `_osc.py`):
  - [x] Define `class RiccatiDenseOutput(DenseOutput):`
    - [ ] Constructor arguments:
      - [x] `t_min`, `t_max`.
      - [x] Solver parameters (callbacks, tolerances, stepsize, solver options).
      - [x] Any additional global info required.
    - [ ] Implement `_call_impl(self, t)`:
      - [x] Accept scalar or array `t`.
      - [x] For each `t`, call the riccati solver with `t_eval` to obtain `y(t)`.
      - [x] Stack results into an array shaped like the solver’s `y`.
    - [ ] Ensure that `t` outside `[t_min, t_max]` is handled consistently with `DenseOutput` expectations.
  - [x] Implementation note: dense output currently re-evaluates the solver at
        requested points instead of using per-step records.

---

## Step 4 – Wire `dense_output=True` in `solve_ivp_osc`

- [x] In `_osc.solve_ivp_osc`:
  - [ ] When `dense_output=True`:
    - [x] Construct `RiccatiDenseOutput` for the integration interval.
    - [x] Attach it to the `OdeResult`’s `sol` attribute.
  - [x] For `t_eval`:
    - [x] Option 1: Pass `t_eval` directly to `_riccati_solve` and use discrete outputs.

---

## Step 5 – Tests for dense output

- [x] Add tests under `scipy/integrate/_ivp/tests/`:
  - [ ] Smoke tests:
    - [x] Call `solve_ivp_osc(..., dense_output=True)` on a simple oscillatory problem.
    - [x] Call `res.sol(t_eval2)` for a set of points; check shapes/dtypes.
  - [ ] Accuracy tests:
    - [x] Use existing riccaticpp-style tests (or new ones) that specify expected accuracy for dense evaluations.
    - [x] Compare `res.sol(t_dense)` outputs with direct or high-resolution references, within tolerances implied by those tests.
  - [ ] Derivative tests:
    - [ ] If `y'(t)` is exposed, verify its consistency where the tests specify expectations.

---

## Step 6 – Build and test

- [x] Run:
  - [x] `pixi run build`
  - [x] `pixi run test ./scipy/integrate/`
- [x] Ensure:
  - [x] No regressions in existing tests from Phase 3.
  - [x] New dense-output tests pass.

---

## Exit criteria for Phase 4

- [x] `solve_ivp_osc(..., dense_output=True)` returns an `OdeResult` with a functioning `sol(t)` method.
- [x] `sol(t)` satisfies the accuracy expectations expressed by the dense-output tests.
- [x] Derivative information `y'(t)` is available via the result (e.g. in `extra`) where required.

---

## Additional questions for Phase 4

- [x] Should we prefer Option 1 or Option 2 for `t_eval` (pass it to `_riccati_solve` vs always using `sol(t_eval)`), or is either acceptable as long as behavior is documented?
      Decision: Option 1 (pass `t_eval` to `_riccati_solve`).
- [x] Are there any performance constraints for dense output (e.g. maximum expected size of `t_eval` or number of dense evaluations)?
      Decision: None specified; re-evaluation cost is acceptable for now.
