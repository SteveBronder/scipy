# SOP: Phase 4 – Dense output for `solve_ivp_osc`

This is a task-oriented checklist for subagents to implement `solve_ivp_osc` in stages.
You are a senior developer familiar with SciPy's codebase and development practices. Your goal is to execute the implementation of `solve_ivp_osc`, a new ODE solver specialized for oscillatory problems, by migrating and adapting the existing `riccaticpp` codebase into SciPy.

**Goal:** Add support for `dense_output=True` and `sol(t)` on the result, similar to `solve_ivp`, using per-step interpolation data from the riccati core.

---

## Preconditions

- [ ] Phase 3 SOP completed and passing tests.
- [ ] `_ivp._riccati` exposes a working core solver for discrete outputs.
- [ ] You can run `pixi run build` and `pixi run test ./scipy/integrate/`.

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

- [ ] In a Python module under `_ivp` (e.g. `scipy/integrate/_ivp/_osc_dense.py` or inside `_osc.py`):
  - [ ] Define `class RiccatiDenseOutput(DenseOutput):`
    - [ ] Constructor arguments:
      - [ ] `t_min`, `t_max`.
      - [ ] List/sequence of step records.
      - [ ] Any additional global info required.
    - [ ] Implement `_call_impl(self, t)`:
      - [ ] Accept scalar or array `t`.
      - [ ] For each `t`, locate the step record with the appropriate interval.
      - [ ] Call `_riccati_eval(record, t)` to obtain `y(t)` (and optionally `y'(t)`).
      - [ ] Stack results into an array shaped like the solver’s `y`.
    - [ ] Ensure that `t` outside `[t_min, t_max]` is handled consistently with `DenseOutput` expectations.

---

## Step 4 – Wire `dense_output=True` in `solve_ivp_osc`

- [ ] In `_osc.solve_ivp_osc`:
  - [ ] When `dense_output=True`:
    - [ ] Call `_riccati_solve` in a mode that returns step records.
    - [ ] Construct `RiccatiDenseOutput` with the step records and integration interval.
    - [ ] Attach it to the `OdeResult`’s `sol` attribute.
  - [ ] For `t_eval`:
    - [ ] Option 1: Pass `t_eval` directly to `_riccati_solve` and use discrete outputs.
    - [ ] Option 2: Always derive `y(t_eval)` as `result.sol(t_eval)`; choose one consistent strategy and document it.

---

## Step 5 – Tests for dense output

- [ ] Add tests under `scipy/integrate/_ivp/tests/`:
  - [ ] Smoke tests:
    - [ ] Call `solve_ivp_osc(..., dense_output=True)` on a simple oscillatory problem.
    - [ ] Call `res.sol(t_eval2)` for a set of points; check shapes/dtypes.
  - [ ] Accuracy tests:
    - [ ] Use existing riccaticpp-style tests (or new ones) that specify expected accuracy for dense evaluations.
    - [ ] Compare `res.sol(t_dense)` outputs with direct or high-resolution references, within tolerances implied by those tests.
  - [ ] Derivative tests:
    - [ ] If `y'(t)` is exposed, verify its consistency where the tests specify expectations.

---

## Step 6 – Build and test

- [ ] Run:
  - [ ] `pixi run build`
  - [ ] `pixi run test ./scipy/integrate/`
- [ ] Ensure:
  - [ ] No regressions in existing tests from Phase 3.
  - [ ] New dense-output tests pass.

---

## Exit criteria for Phase 4

- [ ] `solve_ivp_osc(..., dense_output=True)` returns an `OdeResult` with a functioning `sol(t)` method.
- [ ] `sol(t)` satisfies the accuracy expectations expressed by the dense-output tests.
- [ ] Derivative information `y'(t)` is available via the result (e.g. in `extra`) where required.

---

## Additional questions for Phase 4

- [ ] Should we prefer Option 1 or Option 2 for `t_eval` (pass it to `_riccati_solve` vs always using `sol(t_eval)`), or is either acceptable as long as behavior is documented?
- [ ] Are there any performance constraints for dense output (e.g. maximum expected size of `t_eval` or number of dense evaluations)?

