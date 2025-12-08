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

- [ ] **Copy** riccati headers from `scipy/integrate/riccaticpp/include/riccati/` into `scipy/integrate/include/riccati/`.
  - [ ] Do NOT use symlinks (cross-platform issues) or hard move (need to preserve for revert).
  - [ ] Copy all header files: `solver.hpp`, `evolve.hpp`, `stepsize.hpp`, `step.hpp`, `chebyshev.hpp`, `utils.hpp`, `macros.hpp`, `logger.hpp`, `memory.hpp`, `arena_matrix.hpp`, `vectorizer.hpp`, `riccati.hpp`.
- [ ] Add Eigen3 as a meson subproject:
  - [ ] Do **not** vendor Eigen (to keep diffs clean).
  - [ ] Create `subprojects/eigen.wrap` with appropriate wrap file configuration.
  - [ ] Use `dependency('eigen3')` in meson to get Eigen include paths.
- [ ] Update `scipy/integrate/_ivp/meson.build` for `_riccati`:
  - [ ] Add dependency on `eigen3_dep = dependency('eigen3')`.
  - [ ] Add `include_directories('../../include')` to access `riccati/` headers.
  - [ ] Update the extension module definition to include both dependencies and include paths.

---

## Step 2 – Implement a C++ facade (`riccati_wrapper`)

### Step 2.0 – Research existing SciPy patterns

- [ ] Study existing SciPy code that calls Python from C++:
  - [ ] Look at `scipy/_lib/src/_test_ccallback.c` for Python C API patterns
  - [ ] Examine how other extensions handle GIL (Global Interpreter Lock)
  - [ ] Review NumPy array creation and data copying patterns
  - [ ] Identify reference counting best practices in SciPy codebase

### Step 2.1 – Setup and structure

- [ ] Create a header `scipy/integrate/include/riccati/riccati_wrapper.hpp` and implementation `scipy/integrate/src/riccati_wrapper.cpp`.
- [ ] In the wrapper:
  - [ ] Include `riccati/solver.hpp`, `riccati/evolve.hpp`, `riccati/stepsize.hpp`, and any other required headers.
  - [ ] Include `<Python.h>` and `<numpy/arrayobject.h>` for Python C API.
  - [ ] Add `import_array()` call in module initialization (or use `NO_IMPORT_ARRAY` if called from Cython).

### Step 2.2 – Python C API callback utilities

- [ ] Implement helper functions for calling Python callbacks:
  - [ ] `call_python_scalar_callback`:
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
  - [ ] `call_python_vectorized_callback`:
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
  - [ ] `detect_callback_return_type`:
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

### Step 2.3 – SolverInfoWrapper class

- [ ] Define a `SolverInfoWrapper` class that:
  - [ ] Stores PyObject* pointers for omega_fun and gamma_fun (with INCREF in constructor)
  - [ ] Stores detected return types (omega_type, gamma_type)
  - [ ] Uses `std::variant` or union to hold one of 4 SolverInfo template variants:
    - [ ] `SolverInfo<OmegaCallbackWrapper, GammaCallbackWrapper, double, int64_t>` for each combination
  - [ ] Wraps callback functions to bridge Python C API calls:
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
  - [ ] Manages arena allocator lifetime in constructor/destructor
  - [ ] Stores solver configuration (nini, nmax, n, p parameters)
  - [ ] Has destructor that DECREF's Python objects

### Step 2.4 – Type dispatch implementation

- [ ] Implement `riccati_solver_init`:
  - [ ] Detect omega_fun and gamma_fun return types using `detect_callback_return_type`
  - [ ] Based on detected types, create the appropriate SolverInfo variant:
    - [ ] omega=double, gamma=double → variant index 0
    - [ ] omega=complex, gamma=double → variant index 1
    - [ ] omega=double, gamma=complex → variant index 2
    - [ ] omega=complex, gamma=complex → variant index 3
  - [ ] Wrap callbacks in OmegaCallbackWrapper/GammaCallbackWrapper
  - [ ] Construct SolverInfo with wrapped callbacks and solver parameters (nini, nmax, n, p)
  - [ ] Store in std::variant and return as void* handle
  - [ ] Set output type parameters

### Step 2.5 – Main solve function

- [ ] Implement `riccati_solve`:
  - [ ] Cast void* handle back to SolverInfoWrapper*
  - [ ] Use std::visit or switch on variant index to dispatch to correct template instance
  - [ ] Call riccati::evolve with appropriate parameters:
    - [ ] Convert t_eval array (or pass empty Eigen matrix if NULL)
    - [ ] Call with hard_stop flag
  - [ ] Collect results into std::vectors (letting them grow dynamically)
  - [ ] After solve completes, allocate NumPy arrays for outputs:
    - [ ] Use `PyArray_SimpleNew` to create arrays
    - [ ] Copy from std::vector to NumPy array data
    - [ ] Set output pointers to NumPy array data
  - [ ] Return status code (0=success, negative=error)

### Step 2.6 – C API function signatures

- [ ] Define concrete, non-templated entry points restricted to `double` / `std::complex<double>`:
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

- [ ] Implement comprehensive error handling for Python callbacks:
  - [ ] Check for Python exceptions after **every** PyObject_Call using `PyErr_Occurred()`
  - [ ] If exception occurs:
    - [ ] Clear Python error state with `PyErr_Clear()` (after logging)
    - [ ] Clean up all allocated resources (DECREF Python objects, free C++ memory)
    - [ ] Reset arena allocator to recover memory
    - [ ] Set an error flag/status code to propagate to caller
    - [ ] Return early with error status
  - [ ] Use RAII guard objects for automatic cleanup:
    - [ ] Create `PythonObjectGuard` class that DECREFs in destructor
    - [ ] Create `ArenaAllocatorGuard` class that calls `recover_memory()` in destructor
    - [ ] Create `GILGuard` class for GIL acquire/release

- [ ] Implement RAII wrappers for exception safety:
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

- [ ] Ensure GIL is properly managed:
  - [ ] Acquire GIL before any Python C API call using `GILGuard`
  - [ ] Release after completing Python operations (automatic with RAII)
  - [ ] Never hold GIL during C++ numerical computations (only during callbacks)
  - [ ] Document GIL invariants in code comments

- [ ] Handle memory cleanup on error:
  - [ ] In `riccati_solve`, wrap entire body in try-catch
  - [ ] On exception, ensure:
    - [ ] All allocated NumPy arrays are DECREFed
    - [ ] std::vectors are freed (automatic with RAII)
    - [ ] Arena allocator is reset
    - [ ] Python error state is checked and cleared if needed
  - [ ] Return error status code (e.g., -1 for general error, -2 for callback error)

---

## Step 3 – Extend `_riccati.pyx` to call the facade

- [ ] In `scipy/integrate/_ivp/_riccati.pyx`:
  - [ ] Add `cdef extern from "riccati/riccati_wrapper.hpp":` declarations for wrapper functions.
  - [ ] Define a Cython class `RiccatiSolver` that:
    - [ ] Wraps the opaque `void* solver_handle`
    - [ ] Stores omega_fun, gamma_fun as Python object references
    - [ ] Stores solver parameters (nini, nmax, n, p)
    - [ ] Has `__init__` method to call `riccati_solver_init`
    - [ ] Has `__dealloc__` method to call `riccati_solver_free`
    - [ ] Has a `solve` method that calls `riccati_solve`
  - [ ] Implement the `RiccatiSolver.__init__` method:
    ```cython
    def __init__(self, omega_fun, gamma_fun, int nini, int nmax, int n, int p):
        # Store Python references
        self.omega_fun = omega_fun
        self.gamma_fun = gamma_fun
        # Call C++ init
        cdef int omega_type, gamma_type
        self.handle = riccati_solver_init(
            <PyObject*>omega_fun, <PyObject*>gamma_fun,
            nini, nmax, n, p,
            &omega_type, &gamma_type
        )
        # Store detected types for later reference if needed
    ```
  - [ ] Implement the `RiccatiSolver.solve` method:
    ```cython
    cpdef solve(self, double xi, double xf, np.ndarray y0,
                double eps, double epsilon_h, double init_stepsize,
                np.ndarray t_eval_or_none, bool hard_stop):
    ```
    - [ ] Validate `y0` is exactly length 2, containing `[y, y']` where each element is complex128 (or convertible).
    - [ ] Convert `t_eval_or_none` to either `NULL`/0-length or a 1-D double array.
    - [ ] Call `riccati_solve` with the solver handle and parameters.
    - [ ] Allocate/receive NumPy arrays for outputs: `t`, `y`, `ydot`, `successes`, `phases`, `steptypes`.
    - [ ] Return `(t, y, ydot, successes, phases, steptypes)`.
  - [ ] Implement a convenience function for `solve_ivp_osc`:
    ```cython
    cpdef _riccati_solve_default(
        double xi, double xf, np.ndarray y0,
        object omega_fun, object gamma_fun,
        double eps, double epsilon_h, double init_stepsize,
        np.ndarray t_eval_or_none, bool hard_stop,
        int nini, int nmax, int n, int p
    ):
        # Create temporary solver, call solve, return results
    ```

---

## Step 4 – Implement real `solve_ivp_osc` (no dense_output)

- [ ] In `_osc.solve_ivp_osc`:
  - [ ] Update signature to expose new parameters:
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
  - [ ] Extract solver configuration from `**options`:
    - [ ] `nini = options.get('nini', 16)` - minimum Chebyshev nodes
    - [ ] `nmax = options.get('nmax', 32)` - maximum Chebyshev nodes
    - [ ] `n = options.get('n', 32)` - Chebyshev nodes for collocation steps
    - [ ] `p = options.get('p', 32)` - Chebyshev nodes for Riccati steps
    - [ ] Document these in docstring as advanced options
  - [ ] Replace `NotImplementedError` with:
    - [ ] Conversion of `t_span` to `xi` and `xf` (float64).
    - [ ] Conversion of `y0` to a length-2 `np.ndarray` of complex128:
      - [ ] Validate shape is exactly `(2,)` with elements `[y, y']`.
      - [ ] Document: "y0 must be a 2-element array containing [y(t0), y'(t0)]" with examples in docstring.
    - [ ] Mapping `rtol`/`atol` to riccati's `eps`:
      - [ ] Use rule: `eps = max(rtol, 1e-15)` (document in docstring).
      - [ ] Note: `atol` is validated but not directly used (riccati is relative-tolerance based).
    - [ ] Determining `epsilon_h`:
      - [ ] If `epsilon_h` is explicitly provided by user, use it.
      - [ ] Otherwise, set `epsilon_h = max(1e-6, rtol)` as default.
      - [ ] Document this heuristic in docstring.
    - [ ] Handling `t_eval`:
      - [ ] If `t_eval` is provided, convert to a 1-D double array and pass to `_riccati_solve_default`.
      - [ ] Otherwise, pass `None` to allow C++ code to choose its internal time grid.
    - [ ] Call `_riccati_solve_default` with all parameters and get `(t, y, ydot, successes, phases, steptypes)`.
    - [ ] Build an `OdeResult`:
      - [ ] Use `scipy.integrate._ivp.ivp.OdeResult`.
      - [ ] Fill `t`, `y`, `status`, `message` following `solve_ivp` conventions.
      - [ ] Use **status codes** (not exceptions): 0=success, -1=failure, etc.
      - [ ] Set `t_events`, `y_events` to empty lists (events not supported yet).
      - [ ] Set `sol` to `None` (dense output not supported yet).
      - [ ] Attach `successes`, `phases`, `steptypes`, `ydot` as additional attributes or in an `extra` dict.
    - [ ] If `dense_output=True`:
      - [ ] Raise `NotImplementedError("dense_output=True is not yet supported for solve_ivp_osc")` in this phase.

---

## Step 5 – Port and adapt riccaticpp tests (no dense_output)

- [ ] From `scipy/integrate/riccaticpp/tests/python/test.py`, identify:
  - [ ] Schrodinger-related tests (`test_schrodinger_nondense_fwd_path_optimize`, `test_schrodinger_nondense_fwd_full_optimize`).
  - [ ] Bremer equation tests (`test_bremer_nondense`).
  - [ ] Airy function tests (`test_solve_airy`, `test_solve_airy_backwards`).
  - [ ] Other non-dense tests (`test_solve_burst`, `test_osc_evolve`, `test_nonosc_evolve`).
  - [ ] **Skip** all dense_output tests for Phase 3.
- [ ] Create equivalent tests under `scipy/integrate/_ivp/tests/test_osc.py`:
  - [ ] Use the same `omega_fun` and `gamma_fun` definitions.
  - [ ] Adapt from `ric.Init()` + `ric.evolve()` to `solve_ivp_osc()` with appropriate `y0=[yi, dyi]` stacking.
  - [ ] Map solver parameters: `eps` → `rtol`, `epsilon_h` → `epsilon_h`, etc.
  - [ ] Configure `rtol`, `atol`, and explicit `epsilon_h` to match original numerical behavior.
  - [ ] Compute the same derived quantities (e.g. energy differences, relative errors).
  - [ ] Match the original tolerances as closely as possible.
- [ ] Copy Bremer reference data:
  - [ ] Copy `scipy/integrate/riccaticpp/tests/python/data/eq237.txt` to `scipy/integrate/_ivp/tests/data/eq237.csv` (or embed directly in test code).
  - [ ] Document the source of reference values in test docstrings.

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

## Design Decisions (Resolved)

The following design questions have been resolved:

1. **Eigen Integration:** Add Eigen3 as a meson subproject (not vendored). Use wrap file pointing to `https://github.com/meson-library/eigen.git` with `dependency('eigen3')` in meson build.

2. **pybind11 vs Cython:** Migrate from pybind11 to Cython for consistency with SciPy's existing extensions and build system.

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

