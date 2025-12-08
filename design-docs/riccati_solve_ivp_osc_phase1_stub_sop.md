# SOP: Phase 1 – Public Python stub for `solve_ivp_osc`

This is a task-oriented checklist for subagents to implement `solve_ivp_osc` in stages.
You are a senior developer familiar with SciPy's codebase and development practices. Your goal is to execute the implementation of `solve_ivp_osc`, a new ODE solver specialized for oscillatory problems, by migrating and adapting the existing `riccaticpp` codebase into SciPy.

**Goal:** Make `solve_ivp_osc` importable from `scipy.integrate` with a reasonable signature and argument checks, but no solver logic yet.

---

## Preconditions

- [x] Worktree is clean or on a feature branch.
- [x] `.venv` is available and can be activated.
- [x] You can run `pixi run build` and `pixi run test ./scipy/integrate/`.

---

## Step 1 – Add the stub function

- [x] Create a new module file: `scipy/integrate/_ivp/_osc.py` (if it does not exist).
- [x] In that file, define:
  - [x] A function with signature:
    ```python
    def solve_ivp_osc(
        omega_fun,
        gamma_fun,
        t_span,
        y0,
        *,
        t_eval=None,
        dense_output=False,
        rtol=1e-3,
        atol=1e-6,
        events=None,
        vectorized=False,
        args=None,
        **options,
    ):
    ```
  - [x] Add a docstring describing:
    - [x] Purpose of the solver (oscillatory problems).
    - [x] How the signature relates to `solve_ivp` (extra `omega_fun`, `gamma_fun`; same-style `rtol`/`atol`).
    - [x] Expected types/shapes of `t_span`, `y0`.
    - [x] That it currently supports only `float64` / `complex128`.
  - [x] Implement basic validation:
    - [x] `t_span` is a length-2 sequence convertible to floats.
    - [x] `y0` is array-like and convertible to a 1-D NumPy array.
    - [x] `omega_fun` and `gamma_fun` are callables.
    - [x] `rtol` and `atol` are positive scalars.
    - [x] Accept `events`, `vectorized`, `args`, `**options` but do not use them yet.
  - [x] At the end of the function, raise:
    ```python
    raise NotImplementedError("solve_ivp_osc is not implemented yet")
    ```
  - [x] Where practical, mimic `solve_ivp`’s error types and messages for invalid inputs.

---

## Step 2 – Wire into the package namespace

- [x] In `scipy/integrate/_ivp/__init__.py`:
  - [x] Add `from ._osc import solve_ivp_osc` if not already present.
- [x] In `scipy/integrate/__init__.py`:
  - [x] Ensure `solve_ivp_osc` is imported from `._ivp`.
  - [x] Ensure `solve_ivp_osc` is added to `__all__`.

---

## Step 3 – Add stub tests

- [x] Create `scipy/integrate/_ivp/tests/test_solve_ivp_osc_stub.py`.
- [x] Add tests:
  - [x] Import:
    - [x] `from scipy.integrate import solve_ivp_osc` does not raise.
  - [x] Invalid arguments:
    - [x] Calling with `omega_fun=None` or `gamma_fun=None` raises `TypeError` or `ValueError` matching `solve_ivp`-style messages.
    - [x] Invalid `t_span` or `y0` shapes/types raise appropriate errors.
  - [x] Valid-looking arguments:
    - [x] A simple call with minimal valid arguments raises `NotImplementedError`.

---

## Step 4 – Build and test

- [x] Activate `.venv` if required.
- [x] Run:
  - [x] `pixi run build`
  - [x] `pixi run test ./scipy/integrate/`
- [x] Confirm no regressions in existing integrate tests.

---

## Exit criteria for Phase 1

- [x] `solve_ivp_osc` is importable from `scipy.integrate`.
- [x] It performs basic argument validation (mirroring `solve_ivp` semantics/messages where practical).
- [x] For all valid-looking inputs it raises `NotImplementedError`.
- [x] All `scipy.integrate` tests pass.

---

## Additional questions for Phase 1

- [x] Should the stub already validate that `y0` contains stacked `[y, y']` (shape-wise), or should that be deferred until Phase 3 when the core solver is wired in?
      Decision: defer strict `[y, y']` shape validation to Phase 3; Phase 1 only requires that `y0` is 1-D with dtype `float64` or `complex128`.
- [x] Do we want any initial warning (e.g. `UserWarning`) when calling the stub, or is `NotImplementedError` sufficient for early stages?
      Decision: `NotImplementedError` is sufficient for the Phase 1 stub; no additional warning is emitted.
