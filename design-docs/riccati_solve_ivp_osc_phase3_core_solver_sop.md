# SOP: Phase 3 – Wire riccati C++ core (minimal solver, no dense_output)

This is a task-oriented checklist for subagents to implement `solve_ivp_osc` in stages.
You are a senior developer familiar with SciPy's codebase and development practices. Your goal is to execute the implementation of `solve_ivp_osc`, a new ODE solver specialized for oscillatory problems, by migrating and adapting the existing `riccaticpp` codebase into SciPy.

**Goal:** Replace the dummy `_riccati` extension with a minimal working wrapper around the riccaticpp core (via pybind11) and implement `solve_ivp_osc` to return an `OdeResult` without dense output support.

---

## Preconditions

 - [x] Phase 1 and Phase 2 SOPs completed and passing tests.
 - [x] riccaticpp headers are available in the repo (under `scipy/integrate/riccaticpp/include/riccati` or ready to move).
 - [x] You can run `pixi run build` and `pixi run test ./scipy/integrate/`.

---

## Step 1 – Prepare C++ headers and includes

- [x] **Copy** riccati headers from `scipy/integrate/riccaticpp/include/riccati/` into `scipy/integrate/include/riccati/`.
  - [x] Do NOT use symlinks (cross-platform issues) or hard move (need to preserve for revert).
  - [x] Copy all header files: `solver.hpp`, `evolve.hpp`, `stepsize.hpp`, `step.hpp`, `chebyshev.hpp`, `utils.hpp`, `macros.hpp`, `logger.hpp`, `memory.hpp`, `arena_matrix.hpp`, `vectorizer.hpp`, `riccati.hpp`.
- [x] Add Eigen3 as a meson subproject:
  - [x] Do **not** vendor Eigen (to keep diffs clean).
  - [x] Create `subprojects/eigen.wrap` with appropriate wrap file configuration.
  - [x] Use `dependency('eigen3')` in meson to get Eigen include paths.
- [x] Update `scipy/integrate/_ivp/meson.build` for `_riccati`:
  - [x] Add dependency on `eigen3_dep = dependency('eigen3')`.
  - [x] Add `include_directories('../../include')` to access `riccati/` headers.
  - [x] Update the extension module definition to include both dependencies and include paths.

---

## Step 2 – Implement a C++ facade (`riccati_wrapper`) with pybind11 entry points

**Pybind11 usage guidance (for this project)**
- Build system: add a pybind11 extension in `scipy/integrate/_ivp/meson.build` using `pybind11_dep` (already defined in top-level `meson.build`), e.g.:
  ```meson
  py3.extension_module(
    '_riccati',
    ['_riccati_bindings.cpp', riccati_core_sources...],
    dependencies: [np_dep, pybind11_dep, eigen_dep],
    include_directories: [riccati_inc],
    link_args: version_link_args,
    install: true,
    subdir: 'scipy/integrate/_ivp',
    cpp_args: ['-std=c++17'],
  )
  ```
  Keep sources in `scipy/integrate/_ivp/` (for bindings) and `scipy/integrate/src/` (core). Avoid Cython for this module.
- Module name/path: the Python import should remain `scipy.integrate._ivp._riccati`.
- Binding style: use pybind11 to wrap a lightweight C++ shim (not the entire riccaticpp library) that calls `riccati::evolve` and friends; expose only the Python surface needed by `_osc.solve_ivp_osc`.
- NumPy interop: use `pybind11::array`/`pybind11::buffer_info` for passing `t_eval`; return `pybind11::array` for `t`, `y`, `ydot`, `successes`, `phases`, `steptypes` without extra copies (wrap existing buffers or move `std::vector` via `pybind11::array` constructors with `pybind11::capsule` deleters).
- GIL: release GIL around the numeric solve (`py::gil_scoped_release`) and re-acquire for callback invocations; manage callbacks with `py::function` and explicit type detection.
- Error handling: translate C++ exceptions to Python exceptions via pybind11 (or return status codes and raise in bindings).

### Step 2.0 – Research existing SciPy patterns

- [x] Study existing SciPy code that calls Python from C++:
  - [x] Look at `scipy/_lib/src/_test_ccallback.c` for Python C API patterns
  - [x] Examine how other extensions handle GIL (Global Interpreter Lock)
  - [x] Review NumPy array creation and data copying patterns
  - [x] Identify reference counting best practices in SciPy codebase

### Step 2.1 – Setup and structure

- [x] Create a header `scipy/integrate/include/riccati/riccati_wrapper.hpp` and implementation `scipy/integrate/src/riccati_wrapper.cpp`.
- [x] In the wrapper:
  - [x] Include `riccati/solver.hpp`, `riccati/evolve.hpp`, `riccati/stepsize.hpp`, and any other required headers.
  - [x] Include `<Python.h>` and `<numpy/arrayobject.h>` for Python C API.
  - [x] Add `import_array()` call in module initialization (or use `NO_IMPORT_ARRAY` if called from Cython).

### Step 2.2 – Python C API callback utilities

- [x] Implement helper functions for calling Python callbacks:
  - [x] `call_python_scalar_callback`:
    ```cpp
    template<typename Scalar>
    Scalar call_python_scalar_callback(PyObject* func, Scalar x) {
        // 1. Acquire GIL (PyGILState_Ensure)
        // 2. Create Python float/complex from x
        // 3. Call PyObject_CallFunctionObjArgs(func, py_x, NULL)
        // 4. Extract result and convert to Scalar
        // 5. Release GIL (PyGILState_Release)
        // 6. Return result
    }
    ```
  - [x] `call_python_vectorized_callback`:
    ```cpp
    template<typename Scalar, typename EigenVec>
    Eigen::Matrix<Scalar, -1, 1>
    call_python_vectorized_callback(PyObject* func, const EigenVec& x_vec) {
        // 1. Acquire GIL
        // 2. Convert Eigen vector to NumPy array (PyArray_SimpleNewFromData)
        // 3. Call func(numpy_array)
        // 4. Convert result NumPy array back to Eigen vector
        // 5. Release GIL
        // 6. Return Eigen result
    }
    ```
  - [x] `detect_callback_return_type`:
    ```cpp
    enum CallbackReturnType { RETURN_DOUBLE = 0, RETURN_COMPLEX = 1 };

    CallbackReturnType detect_callback_return_type(PyObject* func) {
        // 1. Call func with test value (e.g., 0.5)
        // 2. Check result type, handling both Python and NumPy scalars:
        //    - PyFloat_Check(result) → RETURN_DOUBLE
        //    - PyComplex_Check(result) → RETURN_COMPLEX
        //    - PyArray_IsScalar(result, Float) → RETURN_DOUBLE
        //    - PyArray_IsScalar(result, ComplexFloating) → RETURN_COMPLEX
        //    - PyArray_CheckScalar(result) and dtype is floating → RETURN_DOUBLE
        //    - PyArray_CheckScalar(result) and dtype is complex → RETURN_COMPLEX
        // 3. Use PyNumber_Float/PyNumber_Complex as fallback for type coercion
        // 4. Raise clear error if type cannot be determined
        // 5. Return appropriate enum value
    }
    ```

### Step 2.3 – SolverInfoWrapper class (pybind11-facing)

 - [x] Define a `SolverInfoWrapper` class that:
  - [x] Stores PyObject* pointers for omega_fun and gamma_fun (with INCREF in constructor)
  - [x] Stores detected return types (omega_type, gamma_type)
  - [x] Uses `std::variant` or union to hold one of 4 SolverInfo template variants:
    - [x] `SolverInfo<OmegaCallbackWrapper, GammaCallbackWrapper, double, int64_t>` for each combination
  - [x] Wraps callback functions to bridge Python C API calls:
    ```cpp
    struct OmegaCallbackWrapper {
        PyObject* py_func;
        CallbackReturnType return_type;

        // Operator() for scalar calls
        auto operator()(double x) const {
            return call_python_scalar_callback(py_func, x);
        }

        // Operator() for vectorized calls
        template<typename EigenVec>
        auto operator()(const EigenVec& x) const {
            return call_python_vectorized_callback(py_func, x);
        }
    };
    ```
  - [x] Manages arena allocator lifetime in constructor/destructor
  - [x] Stores solver configuration (nini, nmax, n, p parameters)
  - [x] Has destructor that DECREF's Python objects

### Step 2.4 – Type dispatch implementation

 - [x] Implement `riccati_solver_init`:
  - [x] Detect omega_fun and gamma_fun return types using `detect_callback_return_type`
  - [x] Based on detected types, create the appropriate SolverInfo variant:
    - [x] omega=double, gamma=double → variant index 0
    - [x] omega=complex, gamma=double → variant index 1
    - [x] omega=double, gamma=complex → variant index 2
    - [x] omega=complex, gamma=complex → variant index 3
  - [x] Wrap callbacks in OmegaCallbackWrapper/GammaCallbackWrapper
  - [x] Construct SolverInfo with wrapped callbacks and solver parameters (nini, nmax, n, p)
  - [x] Store in std::variant and return as void* handle
  - [x] Set output type parameters

### Step 2.5 – Main solve function (C++ core, exposed via pybind11)

 - [x] Implement `riccati_solve`:
  - [x] Cast void* handle back to SolverInfoWrapper*
  - [x] Use std::visit or switch on variant index to dispatch to correct template instance
  - [x] Call riccati::evolve with appropriate parameters:
    - [x] Convert t_eval array (or pass empty Eigen matrix if NULL)
    - [x] Call with hard_stop flag
  - [x] Collect results into std::vectors (letting them grow dynamically)
  - [x] After solve completes, allocate NumPy arrays for outputs:
    - [x] Use `PyArray_SimpleNew` to create arrays
    - [x] Copy from std::vector to NumPy array data
    - [x] Set output pointers to NumPy array data
  - [x] Return status code (0=success, negative=error)

### Step 2.6 – C API function signatures

 - [x] Define concrete, non-templated entry points restricted to `double` / `std::complex<double>`:
  ```cpp
  // Initialize solver and return handle
  void* riccati_solver_init(
      PyObject* omega_fun,
      PyObject* gamma_fun,
      int nini, int nmax, int n, int p,
      int* omega_return_type,  // OUT: detected type (0=double, 1=complex)
      int* gamma_return_type   // OUT: detected type (0=double, 1=complex)
  );

  // Free solver
  void riccati_solver_free(void* solver_handle);

  // Main solve function
  int riccati_solve(
      void* solver_handle,
      double xi, double xf,
      double complex yi, double complex dyi,  // y0 = [yi, dyi]
      double eps,
      double epsilon_h,
      double init_stepsize,
      int n_eval, double* t_eval,  // optional dense output points
      bool hard_stop,
      // OUTPUT arrays (allocated by this function):
      int* n_out,  // number of output points
      double** t_out,
      double complex** y_out,
      double complex** ydot_out,
      int** success_out,
      double** phase_out,
      int** steptype_out
  );
  ```

### Step 2.7 – Error handling and GIL management

- [x] Implement comprehensive error handling for Python callbacks:
  - [x] Check for Python exceptions after **every** PyObject_Call using `PyErr_Occurred()`
  - [x] If exception occurs:
    - [x] Clear Python error state with `PyErr_Clear()` (after logging)
    - [x] Clean up all allocated resources (DECREF Python objects, free C++ memory)
    - [x] Reset arena allocator to recover memory
    - [x] Set an error flag/status code to propagate to caller
    - [x] Return early with error status
  - [x] Use RAII guard objects for automatic cleanup:
    - [x] Create `PythonObjectGuard` class that DECREFs in destructor
    - [x] Create `ArenaAllocatorGuard` class that calls `recover_memory()` in destructor
    - [x] Create `GILGuard` class for GIL acquire/release

- [x] Implement RAII wrappers for exception safety:
  ```cpp
  class GILGuard {
      PyGILState_STATE gstate;
  public:
      GILGuard() : gstate(PyGILState_Ensure()) {}
      ~GILGuard() { PyGILState_Release(gstate); }
      GILGuard(const GILGuard&) = delete;
      GILGuard& operator=(const GILGuard&) = delete;
  };

  class PythonObjectGuard {
      PyObject* obj;
  public:
      explicit PythonObjectGuard(PyObject* o) : obj(o) {}
      ~PythonObjectGuard() { Py_XDECREF(obj); }
      PyObject* get() { return obj; }
      PyObject* release() { auto tmp = obj; obj = nullptr; return tmp; }
  };
  ```

- [x] Ensure GIL is properly managed:
  - [x] Acquire GIL before any Python C API call using `GILGuard`
  - [x] Release after completing Python operations (automatic with RAII)
  - [x] Never hold GIL during C++ numerical computations (only during callbacks)
  - [x] Document GIL invariants in code comments

- [x] Handle memory cleanup on error:
  - [x] In `riccati_solve`, wrap entire body in try-catch
  - [x] On exception, ensure:
    - [x] All allocated NumPy arrays are DECREFed
    - [x] std::vectors are freed (automatic with RAII)
    - [x] Arena allocator is reset
    - [x] Python error state is checked and cleared if needed
  - [x] Return error status code (e.g., -1 for general error, -2 for callback error)

---

## Step 3 – Expose `_riccati` via pybind11

 - [x] Add a pybind11 module `scipy.integrate._ivp._riccati`:
  - [x] Bind a `RiccatiSolver` class that wraps the C++ `SolverInfo` variant handle.
  - [x] Constructor takes `(omega_fun, gamma_fun, nini, nmax, n, p)` and performs type detection.
  - [x] `solve` method mirrors the C++ `riccati_solve` entry point and returns `(t, y, ydot, successes, phases, steptypes)` as NumPy arrays.
  - [x] Expose a convenience function `_riccati_solve_default(...)` that constructs a temporary solver and calls `solve`.
  - [x] Ensure NumPy arrays are returned without extra copies and with correct ownership semantics (capsules or pybind11 buffer management).

---

## Step 4 – Implement real `solve_ivp_osc` (no dense_output)

- [x] In `_osc.solve_ivp_osc`:
  - [x] Update signature to expose new parameters:
    ```python
    def solve_ivp_osc(
        omega_fun, gamma_fun, t_span, y0, *,
        t_eval=None, dense_output=False,
        rtol=1e-3, atol=1e-6,
        epsilon_h=None,      # NEW: stepsize tolerance parameter
        init_stepsize=0.01,  # NEW: initial stepsize
        events=None, vectorized=False, args=None,
        **options
    ):
    ```
  - [x] Extract solver configuration from `**options`:
    - [x] `nini = options.get('nini', 16)` - minimum Chebyshev nodes
    - [x] `nmax = options.get('nmax', 32)` - maximum Chebyshev nodes
    - [x] `n = options.get('n', 32)` - Chebyshev nodes for collocation steps
    - [x] `p = options.get('p', 32)` - Chebyshev nodes for Riccati steps
    - [x] Document these in docstring as advanced options
  - [x] Replace `NotImplementedError` with:
    - [x] Conversion of `t_span` to `xi` and `xf` (float64).
    - [x] Conversion of `y0` to a length-2 `np.ndarray` of complex128:
      - [x] Validate shape is exactly `(2,)` with elements `[y, y']`.
      - [x] Document: "y0 must be a 2-element array containing [y(t0), y'(t0)]" with examples in docstring.
    - [x] Mapping `rtol`/`atol` to riccati's `eps`:
      - [x] Use rule: `eps = max(rtol, 1e-15)` (document in docstring).
      - [x] Note: `atol` is validated but not directly used (riccati is relative-tolerance based).
    - [x] Determining `epsilon_h`:
      - [x] If `epsilon_h` is explicitly provided by user, use it.
      - [x] Otherwise, set `epsilon_h = max(1e-6, rtol)` as default.
      - [x] Document this heuristic in docstring.
    - [x] Handling `t_eval`:
      - [x] If `t_eval` is provided, convert to a 1-D double array and pass to `_riccati_solve_default`.
      - [x] Otherwise, pass `None` to allow C++ code to choose its internal time grid.
    - [x] Call `_riccati_solve_default` with all parameters and get `(t, y, ydot, successes, phases, steptypes)`.
    - [x] Build an `OdeResult`:
      - [x] Use `scipy.integrate._ivp.ivp.OdeResult`.
      - [x] Fill `t`, `y`, `status`, `message` following `solve_ivp` conventions.
      - [x] Use **status codes** (not exceptions): 0=success, -1=failure, etc.
      - [x] Set `t_events`, `y_events` to empty lists (events not supported yet).
      - [x] Set `sol` to `None` (dense output not supported yet).
      - [x] Attach `successes`, `phases`, `steptypes`, `ydot` as additional attributes or in an `extra` dict.
    - [x] If `dense_output=True`:
      - [x] Raise `NotImplementedError("dense_output=True is not yet supported for solve_ivp_osc")` in this phase.

---

## Step 5 – Port and adapt riccaticpp tests (no dense_output)

- [x] From `scipy/integrate/riccaticpp/tests/python/test.py`, identify:
  - [x] Schrodinger-related tests (`test_schrodinger_nondense_fwd_path_optimize`, `test_schrodinger_nondense_fwd_full_optimize`).
  - [x] Bremer equation tests (`test_bremer_nondense`).
  - [x] Airy function tests (`test_solve_airy`, `test_solve_airy_backwards`).
  - [x] Other non-dense tests (`test_solve_burst`, `test_osc_evolve`, `test_nonosc_evolve`).
  - [x] **Skip** all dense_output tests for Phase 3.
- [x] Create equivalent tests under `scipy/integrate/_ivp/tests/test_osc.py`:
  - [x] Use the same `omega_fun` and `gamma_fun` definitions.
  - [x] Adapt from `ric.Init()` + `ric.evolve()` to `solve_ivp_osc()` with appropriate `y0=[yi, dyi]` stacking.
  - [x] Map solver parameters: `eps` → `rtol`, `epsilon_h` → `epsilon_h`, etc.
  - [x] Configure `rtol`, `atol`, and explicit `epsilon_h` to match original numerical behavior.
  - [x] Compute the same derived quantities (e.g. energy differences, relative errors).
  - [x] Match the original tolerances as closely as possible.
- [x] Copy Bremer reference data:
  - [x] Copy `scipy/integrate/riccaticpp/tests/python/data/eq237.txt` to `scipy/integrate/_ivp/tests/data/eq237.csv` (or embed directly in test code).
  - [x] Document the source of reference values in test docstrings.

---

## Step 6 – Build and test

- [x] Run:
  - [x] `pixi run build`
  - [x] `pixi run test ./scipy/integrate/`
- [x] Ensure:
  - [x] New tests pass.
  - [x] No regressions in existing `scipy.integrate` tests.

---

## Exit criteria for Phase 3

- [x] `_ivp._riccati` wraps the real riccati C++ core.
- [x] `solve_ivp_osc` executes without `NotImplementedError` and returns an `OdeResult`.
- [x] `y0` is documented and enforced as `[y, y']` stacked.
- [x] Ported riccaticpp tests pass with comparable tolerances.

---

## Design Decisions (Resolved)

The following design questions have been resolved:

1. **Eigen Integration:** Add Eigen3 as a meson subproject (not vendored). Use wrap file pointing to `https://github.com/meson-library/eigen.git` with `dependency('eigen3')` in meson build.

2. **pybind11 vs Cython:** Use pybind11 to expose the C++ core (consistent with other SciPy bindings like pocketfft/fast_matrix_market).

3. **Header organization:** Copy (not symlink, not move) riccati headers from `riccaticpp/include/riccati/` to `scipy/integrate/include/riccati/`.

4. **y0 stacking:** Use `[y, y']` as 2-element array. Complex values handled as complex types directly (not split into real/imag).

5. **epsilon_h parameter:** Expose as user parameter in `solve_ivp_osc(epsilon_h=None)` with default `max(1e-6, rtol)`.

6. **init_stepsize parameter:** Expose as user parameter in `solve_ivp_osc(init_stepsize=0.01)`.

7. **Error handling:** Use status codes in `OdeResult` (consistent with `solve_ivp`), not exceptions.

8. **Callback mechanism:** Use Python C API to call omega_fun/gamma_fun directly (like riccaticpp's pybind11 approach). **DO NOT use ccallback.h** - it's designed for scalar callbacks but riccati needs vectorized callbacks (Eigen vectors).

9. **Memory management:** Expose `RiccatiSolver` class that manages arena allocator lifetime. For `solve_ivp_osc`, create a temporary solver internally with default parameters.

10. **Type dispatch:** Detect omega/gamma return types at runtime in `riccati_solver_init` by calling with test values, then create appropriate SolverInfo variant.

11. **Solver configuration:** Expose `nini`, `nmax`, `n`, `p` as advanced options via `**options` in `solve_ivp_osc`, with sensible defaults (16, 32, 32, 32).

12. **Test data:** Copy `eq237.txt` to SciPy test directory as `eq237.csv` (or embed in code).

13. **Error handling strategy:** Use RAII guards for all resource management. Check for Python exceptions after every callback. On error: clean up resources, reset arena allocator, return error status code.

14. **NumPy scalar handling:** Type detection must handle both Python scalars (`float`, `complex`) and NumPy scalars (`numpy.float64`, `numpy.complex128`) using `PyArray_IsScalar()` and `PyArray_CheckScalar()`.
