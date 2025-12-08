# SOP: Phase 5 – Documentation, polish, and follow-ups for `solve_ivp_osc`

This is a task-oriented checklist for subagents to implement `solve_ivp_osc` in stages.
You are a senior developer familiar with SciPy's codebase and development practices. Your goal is to execute the implementation of `solve_ivp_osc`, a new ODE solver specialized for oscillatory problems, by migrating and adapting the existing `riccaticpp` codebase into SciPy.

**Goal:** Document `solve_ivp_osc` as a first-class solver in `scipy.integrate`, ensure exports and messaging are correct, and prepare it for release.

---

## Preconditions

- [ ] Phases 1–4 SOPs completed and all tests passing.
- [ ] Public API and behavior of `solve_ivp_osc` are stable.

---

## Step 1 – Update reference documentation

- [ ] Add `solve_ivp_osc` to the integrate reference docs (e.g. `doc/source/reference/integrate.rst`):
  - [ ] Include it in the appropriate autosummary or function listing.
  - [ ] Provide a short section describing:
    - [ ] That it is specialized for oscillatory ODE problems.
    - [ ] Its relationship to `solve_ivp`.
    - [ ] The role of `omega_fun`, `gamma_fun`, and `epsilon_h`.
- [ ] Ensure the function’s docstring is numpydoc-compliant:
  - [ ] Parameters section for all arguments.
  - [ ] Returns section describing `OdeResult` and its `extra` fields.
  - [ ] Notes section explaining specialization and limitations (float64/complex128, no events in v1).
  - [ ] Examples section with at least one minimal example.

---

## Step 2 – Public API and exports

- [ ] Verify `scipy.integrate.__all__` includes `solve_ivp_osc`.
- [ ] Verify that `solve_ivp_osc` appears in generated API docs where expected.
- [ ] Check that imports like:
  ```python
  from scipy.integrate import solve_ivp_osc
  ```
  work without warnings.

---

## Step 3 – Release notes

- [ ] Add an entry in the release notes (e.g. `doc/source/release/1.xx.0-notes.rst`) under the `scipy.integrate` / ODE solvers section:
  - [ ] Briefly describe `solve_ivp_osc` and its purpose.
  - [ ] Mention that it is specialized for oscillatory problems.
  - [ ] Optionally reference the underlying methodology or prior art (e.g. riccati-based oscillatory integration).

---

## Step 4 – Final polish

- [ ] Review error messages raised by `solve_ivp_osc` for:
  - [ ] Consistency with `solve_ivp`.
  - [ ] Clear guidance when arguments are invalid or unsupported (e.g. `events` currently ignored).
- [ ] Ensure that the `extra` dictionary in the result is documented:
  - [ ] Explain keys like `successes`, `phases`, `steptypes`, and any derivative-related entries.
- [ ] Confirm there are no remaining references to the old standalone `pyriccaticpp` packaging:
  - [ ] All relevant code paths should now go through SciPy’s meson/Cython integration.

---

## Step 5 – Build and test

- [ ] Run:
  - [ ] `pixi run build`
  - [ ] `pixi run test ./scipy/integrate/`
- [ ] Optionally run a subset of full SciPy test suite to ensure no cross-module regressions.

---

## Exit criteria for Phase 5

- [ ] `solve_ivp_osc` is fully documented and appears in the integrate reference.
- [ ] Release notes mention `solve_ivp_osc` under ODE solvers.
- [ ] Public exports and error messages are polished and consistent with the rest of SciPy.

---

## Additional questions for Phase 5

- [ ] Should we add an explicit cross-reference from the `solve_ivp` docs to `solve_ivp_osc` (e.g. “for oscillatory problems, see `solve_ivp_osc`”) to guide users?
- [ ] Are there any internal maintainers or domain experts who should be listed as code owners/reviewers for `solve_ivp_osc` in documentation or contribution guides?

