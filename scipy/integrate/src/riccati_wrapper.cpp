/*
 * Riccati C++ Wrapper Implementation
 *
 * This file implements the Python/Cython-friendly interface to the riccati solver.
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include <complex>
#include <vector>
#include <variant>
#include <memory>
#include <stdexcept>

#include <Eigen/Dense>
#include <riccati/solver.hpp>
#include <riccati/evolve.hpp>
#include <riccati/stepsize.hpp>
#include <riccati/riccati_wrapper.hpp>

// Helper for consistent complex type
using complex_t = std::complex<double>;

/*
 * RAII Guard for GIL Management
 */
class GILGuard {
    PyGILState_STATE gstate;
public:
    GILGuard() : gstate(PyGILState_Ensure()) {}
    ~GILGuard() { PyGILState_Release(gstate); }
    GILGuard(const GILGuard&) = delete;
    GILGuard& operator=(const GILGuard&) = delete;
};

/*
 * RAII Guard for Python Object References
 */
class PythonObjectGuard {
    PyObject* obj;
public:
    explicit PythonObjectGuard(PyObject* o) : obj(o) {}
    ~PythonObjectGuard() { Py_XDECREF(obj); }
    PyObject* get() const { return obj; }
    PyObject* release() {
        auto tmp = obj;
        obj = nullptr;
        return tmp;
    }
    PythonObjectGuard(const PythonObjectGuard&) = delete;
    PythonObjectGuard& operator=(const PythonObjectGuard&) = delete;
};

/*
 * Callback return type enumeration
 */
enum CallbackReturnType {
    RETURN_DOUBLE = 0,
    RETURN_COMPLEX = 1
};

/*
 * Helper function to call Python callback with scalar argument
 */
template<typename Scalar>
inline Scalar call_python_scalar_callback(PyObject* func, Scalar x, bool& error_flag) {
    GILGuard gil;
    error_flag = false;

    // Create Python argument
    PyObject* py_x;
    if constexpr (std::is_same_v<Scalar, double>) {
        py_x = PyFloat_FromDouble(x);
    } else {
        py_x = PyComplex_FromDoubles(x.real(), x.imag());
    }

    if (!py_x) {
        error_flag = true;
        return Scalar(0);
    }

    PythonObjectGuard x_guard(py_x);

    // Call function
    PyObject* result = PyObject_CallFunctionObjArgs(func, py_x, NULL);
    if (!result) {
        error_flag = true;
        return Scalar(0);
    }

    PythonObjectGuard result_guard(result);

    // Extract result
    if constexpr (std::is_same_v<Scalar, double>) {
        // Try to convert to double
        PyObject* float_result = PyNumber_Float(result);
        if (!float_result) {
            error_flag = true;
            return 0.0;
        }
        PythonObjectGuard float_guard(float_result);
        return PyFloat_AsDouble(float_result);
    } else {
        // Try to convert to complex
        if (PyComplex_Check(result)) {
            return complex_t(PyComplex_RealAsDouble(result),
                           PyComplex_ImagAsDouble(result));
        } else if (PyFloat_Check(result)) {
            return complex_t(PyFloat_AsDouble(result), 0.0);
        } else {
            // Try number conversion
            PyObject* complex_result = PyNumber_Complex(result);
            if (!complex_result) {
                error_flag = true;
                return complex_t(0, 0);
            }
            PythonObjectGuard complex_guard(complex_result);
            return complex_t(PyComplex_RealAsDouble(complex_result),
                           PyComplex_ImagAsDouble(complex_result));
        }
    }
}

/*
 * Helper function to call Python callback with vector argument
 */
template<typename Scalar, typename EigenVec>
inline Eigen::Matrix<Scalar, -1, 1>
call_python_vectorized_callback(PyObject* func, const EigenVec& x_vec, bool& error_flag) {
    GILGuard gil;
    error_flag = false;

    // Create NumPy array from Eigen vector
    npy_intp dims[1] = {x_vec.size()};
    PyObject* x_array = PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    if (!x_array) {
        error_flag = true;
        return Eigen::Matrix<Scalar, -1, 1>();
    }

    PythonObjectGuard array_guard(x_array);

    // Copy data
    double* data = (double*)PyArray_DATA((PyArrayObject*)x_array);
    for (int i = 0; i < x_vec.size(); ++i) {
        data[i] = x_vec(i);
    }

    // Call function
    PyObject* result = PyObject_CallFunctionObjArgs(func, x_array, NULL);
    if (!result) {
        error_flag = true;
        return Eigen::Matrix<Scalar, -1, 1>();
    }

    PythonObjectGuard result_guard(result);

    // Convert result back to Eigen
    if (!PyArray_Check(result)) {
        error_flag = true;
        return Eigen::Matrix<Scalar, -1, 1>();
    }

    PyArrayObject* result_array = (PyArrayObject*)result;
    int n = PyArray_SIZE(result_array);

    Eigen::Matrix<Scalar, -1, 1> output(n);

    if constexpr (std::is_same_v<Scalar, double>) {
        // Extract double values
        for (int i = 0; i < n; ++i) {
            PyObject* item = PyArray_GETITEM(result_array, PyArray_GETPTR1(result_array, i));
            if (!item) {
                error_flag = true;
                return Eigen::Matrix<Scalar, -1, 1>();
            }
            PythonObjectGuard item_guard(item);

            PyObject* float_item = PyNumber_Float(item);
            if (!float_item) {
                error_flag = true;
                return Eigen::Matrix<Scalar, -1, 1>();
            }
            PythonObjectGuard float_guard(float_item);

            output(i) = PyFloat_AsDouble(float_item);
        }
    } else {
        // Extract complex values
        for (int i = 0; i < n; ++i) {
            PyObject* item = PyArray_GETITEM(result_array, PyArray_GETPTR1(result_array, i));
            if (!item) {
                error_flag = true;
                return Eigen::Matrix<Scalar, -1, 1>();
            }
            PythonObjectGuard item_guard(item);

            if (PyComplex_Check(item)) {
                output(i) = complex_t(PyComplex_RealAsDouble(item),
                                    PyComplex_ImagAsDouble(item));
            } else if (PyFloat_Check(item)) {
                output(i) = complex_t(PyFloat_AsDouble(item), 0.0);
            } else {
                PyObject* complex_item = PyNumber_Complex(item);
                if (!complex_item) {
                    error_flag = true;
                    return Eigen::Matrix<Scalar, -1, 1>();
                }
                PythonObjectGuard complex_guard(complex_item);
                output(i) = complex_t(PyComplex_RealAsDouble(complex_item),
                                    PyComplex_ImagAsDouble(complex_item));
            }
        }
    }

    return output;
}

/*
 * Detect callback return type
 */
CallbackReturnType detect_callback_return_type(PyObject* func) {
    GILGuard gil;

    // Test with value 0.5
    PyObject* test_val = PyFloat_FromDouble(0.5);
    if (!test_val) {
        throw std::runtime_error("Failed to create test value");
    }
    PythonObjectGuard test_guard(test_val);

    PyObject* result = PyObject_CallFunctionObjArgs(func, test_val, NULL);
    if (!result) {
        PyErr_Print();
        throw std::runtime_error("Callback function failed during type detection");
    }
    PythonObjectGuard result_guard(result);

    // Check result type
    if (PyComplex_Check(result)) {
        return RETURN_COMPLEX;
    } else if (PyFloat_Check(result)) {
        return RETURN_DOUBLE;
    } else if (PyArray_IsScalar(result, ComplexFloating)) {
        return RETURN_COMPLEX;
    } else if (PyArray_IsScalar(result, Floating)) {
        return RETURN_DOUBLE;
    } else {
        // Try to convert
        PyObject* complex_result = PyNumber_Complex(result);
        if (complex_result) {
            Py_DECREF(complex_result);
            return RETURN_COMPLEX;
        }
        PyErr_Clear();

        PyObject* float_result = PyNumber_Float(result);
        if (float_result) {
            Py_DECREF(float_result);
            return RETURN_DOUBLE;
        }
        PyErr_Clear();

        throw std::runtime_error("Cannot determine callback return type");
    }
}

/*
 * Callback wrapper class
 */
template<typename ReturnType>
struct CallbackWrapper {
    PyObject* py_func;

    CallbackWrapper(PyObject* func) : py_func(func) {
        Py_INCREF(py_func);
    }

    ~CallbackWrapper() {
        GILGuard gil;
        Py_DECREF(py_func);
    }

    // Scalar call
    auto operator()(double x) const {
        bool error = false;
        auto result = call_python_scalar_callback<ReturnType>(py_func, x, error);
        if (error) {
            throw std::runtime_error("Callback error");
        }
        return result;
    }

    // Vector call
    template<typename EigenVec>
    auto operator()(const EigenVec& x) const {
        bool error = false;
        auto result = call_python_vectorized_callback<ReturnType, EigenVec>(py_func, x, error);
        if (error) {
            throw std::runtime_error("Callback error");
        }
        return result;
    }
};

/*
 * SolverInfo type aliases for each omega/gamma combination
 */
using SolverInfo_DD = riccati::SolverInfo<
    CallbackWrapper<double>, CallbackWrapper<double>,
    double, int64_t,
    riccati::arena_allocator<double, riccati::arena_alloc>,
    riccati::EmptyLogger, double, double>;

using SolverInfo_CD = riccati::SolverInfo<
    CallbackWrapper<complex_t>, CallbackWrapper<double>,
    double, int64_t,
    riccati::arena_allocator<double, riccati::arena_alloc>,
    riccati::EmptyLogger, complex_t, double>;

using SolverInfo_DC = riccati::SolverInfo<
    CallbackWrapper<double>, CallbackWrapper<complex_t>,
    double, int64_t,
    riccati::arena_allocator<double, riccati::arena_alloc>,
    riccati::EmptyLogger, double, complex_t>;

using SolverInfo_CC = riccati::SolverInfo<
    CallbackWrapper<complex_t>, CallbackWrapper<complex_t>,
    double, int64_t,
    riccati::arena_allocator<double, riccati::arena_alloc>,
    riccati::EmptyLogger, complex_t, complex_t>;

/*
 * Solver wrapper using std::variant for type dispatch
 */
struct RiccatiSolverHandle {
    std::variant<SolverInfo_DD, SolverInfo_CD, SolverInfo_DC, SolverInfo_CC> solver_info;
    int omega_type;
    int gamma_type;

    template<typename OmegaType, typename GammaType>
    RiccatiSolverHandle(
        PyObject* omega_fun, PyObject* gamma_fun,
        int nini, int nmax, int n, int p,
        CallbackReturnType omega_ret, CallbackReturnType gamma_ret)
        : omega_type(omega_ret), gamma_type(gamma_ret)
    {
        CallbackWrapper<OmegaType> omega_wrapper(omega_fun);
        CallbackWrapper<GammaType> gamma_wrapper(gamma_fun);

        using SolverType = riccati::SolverInfo<
            CallbackWrapper<OmegaType>, CallbackWrapper<GammaType>,
            double, int64_t,
            riccati::arena_allocator<double, riccati::arena_alloc>,
            riccati::EmptyLogger, OmegaType, GammaType>;

        solver_info = SolverType(std::move(omega_wrapper), std::move(gamma_wrapper),
                                nini, nmax, n, p);
    }
};

/*
 * C API Implementation
 */

extern "C" {

void* riccati_solver_init(
    PyObject* omega_fun,
    PyObject* gamma_fun,
    int nini, int nmax, int n, int p,
    int* omega_return_type,
    int* gamma_return_type)
{
    try {
        // Detect callback return types
        CallbackReturnType omega_type = detect_callback_return_type(omega_fun);
        CallbackReturnType gamma_type = detect_callback_return_type(gamma_fun);

        *omega_return_type = omega_type;
        *gamma_return_type = gamma_type;

        // Create appropriate solver based on types
        RiccatiSolverHandle* handle = nullptr;

        if (omega_type == RETURN_DOUBLE && gamma_type == RETURN_DOUBLE) {
            handle = new RiccatiSolverHandle(omega_fun, gamma_fun, nini, nmax, n, p,
                                            omega_type, gamma_type);
        } else if (omega_type == RETURN_COMPLEX && gamma_type == RETURN_DOUBLE) {
            handle = new RiccatiSolverHandle(omega_fun, gamma_fun, nini, nmax, n, p,
                                            omega_type, gamma_type);
        } else if (omega_type == RETURN_DOUBLE && gamma_type == RETURN_COMPLEX) {
            handle = new RiccatiSolverHandle(omega_fun, gamma_fun, nini, nmax, n, p,
                                            omega_type, gamma_type);
        } else { // both complex
            handle = new RiccatiSolverHandle(omega_fun, gamma_fun, nini, nmax, n, p,
                                            omega_type, gamma_type);
        }

        return handle;
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    }
}

void riccati_solver_free(void* solver_handle) {
    if (solver_handle) {
        delete static_cast<RiccatiSolverHandle*>(solver_handle);
    }
}

int riccati_solve(
    void* solver_handle,
    double xi, double xf,
    double complex yi, double complex dyi,
    double eps,
    double epsilon_h,
    double init_stepsize,
    int n_eval, double* t_eval,
    int hard_stop,
    int* n_out,
    double** t_out,
    double complex** y_out,
    double complex** ydot_out,
    int** success_out,
    double** phase_out,
    int** steptype_out)
{
    if (!solver_handle) {
        return -1;
    }

    try {
        RiccatiSolverHandle* handle = static_cast<RiccatiSolverHandle*>(solver_handle);

        // Convert t_eval to Eigen vector if provided
        Eigen::VectorXd t_eval_vec;
        if (n_eval > 0 && t_eval) {
            t_eval_vec = Eigen::Map<Eigen::VectorXd>(t_eval, n_eval);
        }

        // Storage for results
        std::vector<double> t_result;
        std::vector<complex_t> y_result;
        std::vector<complex_t> ydot_result;
        std::vector<int> success_result;
        std::vector<double> phase_result;
        std::vector<int> steptype_result;

        // Dispatch based on variant type
        std::visit([&](auto& info) {
            using SolverInfo = std::decay_t<decltype(info)>;

            // Call evolve
            auto result = (n_eval > 0)
                ? riccati::evolve(info, xi, xf,
                                complex_t(__real__(yi), __imag__(yi)),
                                complex_t(__real__(dyi), __imag__(dyi)),
                                eps, epsilon_h, init_stepsize, t_eval_vec, hard_stop != 0)
                : riccati::evolve(info, xi, xf,
                                complex_t(__real__(yi), __imag__(yi)),
                                complex_t(__real__(dyi), __imag__(dyi)),
                                eps, epsilon_h, init_stepsize,
                                Eigen::Matrix<double, 0, 0>{}, hard_stop != 0);

            // Extract results
            auto [xs, ys, dys, successes, phases, steptypes, yeval, dyeval, dense_start, dense_size] = result;

            // Copy to std::vectors
            t_result.assign(xs.begin(), xs.end());
            y_result.assign(ys.begin(), ys.end());
            ydot_result.assign(dys.begin(), dys.end());
            success_result.assign(successes.begin(), successes.end());
            phase_result.assign(phases.begin(), phases.end());
            steptype_result.assign(steptypes.begin(), steptypes.end());

            // Recover arena memory
            info.alloc_.recover_memory();
        }, handle->solver_info);

        // Allocate NumPy arrays and copy data
        int n = t_result.size();
        *n_out = n;

        // Create arrays using Python C API
        GILGuard gil;

        npy_intp dims[1] = {n};

        // Time array
        PyObject* t_array = PyArray_SimpleNew(1, dims, NPY_DOUBLE);
        if (!t_array) return -1;
        double* t_data = (double*)PyArray_DATA((PyArrayObject*)t_array);
        std::copy(t_result.begin(), t_result.end(), t_data);
        *t_out = t_data;

        // Y array (complex)
        PyObject* y_array = PyArray_SimpleNew(1, dims, NPY_COMPLEX128);
        if (!y_array) { Py_DECREF(t_array); return -1; }
        complex_t* y_data = (complex_t*)PyArray_DATA((PyArrayObject*)y_array);
        std::copy(y_result.begin(), y_result.end(), y_data);
        *y_out = (double complex*)y_data;

        // Ydot array (complex)
        PyObject* ydot_array = PyArray_SimpleNew(1, dims, NPY_COMPLEX128);
        if (!ydot_array) { Py_DECREF(t_array); Py_DECREF(y_array); return -1; }
        complex_t* ydot_data = (complex_t*)PyArray_DATA((PyArrayObject*)ydot_array);
        std::copy(ydot_result.begin(), ydot_result.end(), ydot_data);
        *ydot_out = (double complex*)ydot_data;

        // Success array
        PyObject* success_array = PyArray_SimpleNew(1, dims, NPY_INT);
        if (!success_array) {
            Py_DECREF(t_array); Py_DECREF(y_array); Py_DECREF(ydot_array);
            return -1;
        }
        int* success_data = (int*)PyArray_DATA((PyArrayObject*)success_array);
        std::copy(success_result.begin(), success_result.end(), success_data);
        *success_out = success_data;

        // Phase array
        PyObject* phase_array = PyArray_SimpleNew(1, dims, NPY_DOUBLE);
        if (!phase_array) {
            Py_DECREF(t_array); Py_DECREF(y_array); Py_DECREF(ydot_array);
            Py_DECREF(success_array);
            return -1;
        }
        double* phase_data = (double*)PyArray_DATA((PyArrayObject*)phase_array);
        std::copy(phase_result.begin(), phase_result.end(), phase_data);
        *phase_out = phase_data;

        // Steptype array
        PyObject* steptype_array = PyArray_SimpleNew(1, dims, NPY_INT);
        if (!steptype_array) {
            Py_DECREF(t_array); Py_DECREF(y_array); Py_DECREF(ydot_array);
            Py_DECREF(success_array); Py_DECREF(phase_array);
            return -1;
        }
        int* steptype_data = (int*)PyArray_DATA((PyArrayObject*)steptype_array);
        std::copy(steptype_result.begin(), steptype_result.end(), steptype_data);
        *steptype_out = steptype_data;

        return 0;  // Success

    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return -2;  // Callback/runtime error
    }
}

} // extern "C"

