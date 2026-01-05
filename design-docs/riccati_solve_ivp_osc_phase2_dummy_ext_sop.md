# SOP: Phase 2 – Cython `_riccati` dummy extension

This is a task-oriented checklist for subagents to implement `solve_ivp_osc` in stages.
You are a senior developer familiar with SciPy's codebase and development practices. Your goal is to execute the implementation of `solve_ivp_osc`, a new ODE solver specialized for oscillatory problems, by migrating and adapting the existing `riccaticpp` codebase into SciPy.

**Goal:** Add a minimal Cython C++ extension to the SciPy build so the meson and Cython plumbing are proven before integrating the riccati algorithm.

**Status (post-Phase 3):** The dummy extension is now provided by the pybind11 module
(`scipy/integrate/_ivp/_riccati_pybind.cpp`), and `solve_ivp_osc` no longer raises
`NotImplementedError`.

---

## Preconditions

- [x] Phase 1 SOP completed and tests passing.
- [x] You can successfully run `pixi run build` and `pixi run test ./scipy/integrate/`.

---

## Step 1 – Create the Cython module

- [x] Create `scipy/integrate/_ivp/_riccati.pyx` with content similar to:
  ```cython
  cpdef int _dummy_riccati(int x):
      return x
  ```
- [x] No Eigen or riccati headers should be used yet—keep it pure Cython for now.

---

## Step 2 – Integrate `_riccati` into meson

- [x] Open `scipy/integrate/meson.build`.
- [x] Add a new extension module definition for `_ivp._riccati`, for example:
  ```meson
  py3.extension_module(
    '_ivp._riccati',
    [lib_cython_gen.process('_ivp/_riccati.pyx')],
    dependencies: [np_dep],
    link_args: version_link_args,
    install: true,
    subdir: 'scipy/integrate',
  )
  ```
- [x] Ensure the resulting shared library installs to `scipy/integrate/_ivp/_riccati.*.so`.

---

## Step 3 – Call the dummy extension from Python

- [x] In `scipy/integrate/_ivp/_osc.py`:
  - [x] Add `from . import _riccati as _ric` near the top.
  - [x] Inside `solve_ivp_osc`, before raising `NotImplementedError`, add a call like:
    ```python
    _ = _ric._dummy_riccati(1)
    ```
    (ignore the result). This ensures the extension is imported and callable.

---

## Step 4 – Add tests for the dummy extension

- [x] Create `scipy/integrate/_ivp/tests/test_riccati_dummy.py`.
- [x] Add tests:
  - [x] Import test:
    ```python
    from scipy.integrate._ivp import _riccati
    ```
    should not raise.
  - [x] Behavior test:
    - [x] Assert `_riccati._dummy_riccati(5) == 5`.

---

## Step 5 – Build and test

- [x] Run:
  - [x] `pixi run build`
  - [x] `pixi run test ./scipy/integrate/`
- [x] Ensure that:
  - [x] `_ivp._riccati` builds on all CI platforms supported by SciPy.
  - [x] `solve_ivp_osc` still raises `NotImplementedError` but now successfully imports and calls `_dummy_riccati`.

---

## Exit criteria for Phase 2

- [x] `_ivp._riccati` is built and installed by meson.
- [x] The dummy function `_dummy_riccati` can be imported and called from Python.
- [x] Existing `solve_ivp_osc` stub test suite still passes.

---

## Additional questions for Phase 2

- [x] Are there any SciPy-internal naming/preferences for compiled modules beyond the `_ivp._name` pattern that we should conform to?
      Decision: stick with `_ivp._riccati` to match existing SciPy compiled module naming.
- [x] Should this phase also add minimal type annotations in `_riccati.pyx` (e.g. `int` vs `Py_ssize_t`) for consistency with other Cython modules, or is that premature?
      Decision: N/A; the module migrated to pybind11 in Phase 3, so no Cython annotations are needed.
