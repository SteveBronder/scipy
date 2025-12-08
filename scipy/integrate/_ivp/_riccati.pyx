# cython: language_level=3
"""
Riccati ODE solver extension module.

This module provides the compiled Riccati solver backend for solve_ivp_osc.
"""

import numpy as np
cimport numpy as np
from cpython.object cimport PyObject

np.import_array()

# External declarations from C++ wrapper
cdef extern from "riccati/riccati_wrapper.hpp":
    void* riccati_solver_init(
        PyObject* omega_fun,
        PyObject* gamma_fun,
        int nini, int nmax, int n, int p,
        int* omega_return_type,
        int* gamma_return_type
    ) noexcept nogil

    void riccati_solver_free(void* solver_handle) noexcept nogil

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
        int** steptype_out
    ) noexcept nogil


cdef class RiccatiSolver:
    """
    Riccati ODE solver object.

    This class wraps the C++ riccati solver and manages its lifetime.

    Parameters
    ----------
    omega_fun : callable
        Frequency function omega(x)
    gamma_fun : callable
        Friction function gamma(x)
    nini : int
        Minimum number of Chebyshev nodes
    nmax : int
        Maximum number of Chebyshev nodes
    n : int
        Number of Chebyshev nodes for collocation steps
    p : int
        Number of Chebyshev nodes for Riccati steps
    """
    cdef void* _handle
    cdef object _omega_fun
    cdef object _gamma_fun
    cdef int _omega_type
    cdef int _gamma_type

    def __cinit__(self, omega_fun, gamma_fun, int nini, int nmax, int n, int p):
        self._omega_fun = omega_fun
        self._gamma_fun = gamma_fun

        cdef int omega_type, gamma_type
        self._handle = riccati_solver_init(
            <PyObject*>omega_fun,
            <PyObject*>gamma_fun,
            nini, nmax, n, p,
            &omega_type,
            &gamma_type
        )

        if self._handle == NULL:
            raise RuntimeError("Failed to initialize riccati solver")

        self._omega_type = omega_type
        self._gamma_type = gamma_type

    def __dealloc__(self):
        if self._handle != NULL:
            riccati_solver_free(self._handle)
            self._handle = NULL

    cpdef solve(
        self,
        double xi, double xf,
        np.ndarray[np.complex128_t, ndim=1] y0,
        double eps,
        double epsilon_h,
        double init_stepsize,
        np.ndarray[np.float64_t, ndim=1] t_eval=None,
        bint hard_stop=False
    ):
        """
        Solve the Riccati ODE.

        Parameters
        ----------
        xi, xf : float
            Integration interval [xi, xf]
        y0 : ndarray of complex128, shape (2,)
            Initial conditions [y(xi), y'(xi)]
        eps : float
            Relative tolerance for steps
        epsilon_h : float
            Relative tolerance for stepsize selection
        init_stepsize : float
            Initial stepsize
        t_eval : ndarray of float64, optional
            Evaluation points
        hard_stop : bool
            Whether to force stop exactly at xf

        Returns
        -------
        t : ndarray of float64
            Time points
        y : ndarray of complex128
            Solution values
        ydot : ndarray of complex128
            Derivative values
        success : ndarray of int
            Success flags
        phase : ndarray of float64
            Phase values
        steptype : ndarray of int
            Step type indicators
        """
        # Validate y0
        if y0.shape[0] != 2:
            raise ValueError("y0 must have shape (2,) containing [y, y']")

        cdef double complex yi = y0[0]
        cdef double complex dyi = y0[1]

        cdef int n_eval = 0
        cdef double* t_eval_ptr = NULL
        if t_eval is not None:
            n_eval = t_eval.shape[0]
            t_eval_ptr = &t_eval[0]

        cdef int n_out
        cdef double* t_out
        cdef double complex* y_out
        cdef double complex* ydot_out
        cdef int* success_out
        cdef double* phase_out
        cdef int* steptype_out

        cdef int status = riccati_solve(
            self._handle,
            xi, xf, yi, dyi,
            eps, epsilon_h, init_stepsize,
            n_eval, t_eval_ptr,
            1 if hard_stop else 0,
            &n_out,
            &t_out,
            &y_out,
            &ydot_out,
            &success_out,
            &phase_out,
            &steptype_out
        )

        if status != 0:
            raise RuntimeError(f"Riccati solve failed with status {status}")

        # Wrap output pointers in NumPy arrays
        # Note: these arrays reference data owned by NumPy, created in the C++ wrapper
        cdef np.npy_intp dims[1]
        dims[0] = n_out

        t_array = np.PyArray_SimpleNewFromData(1, dims, np.NPY_DOUBLE, t_out)
        y_array = np.PyArray_SimpleNewFromData(1, dims, np.NPY_COMPLEX128, y_out)
        ydot_array = np.PyArray_SimpleNewFromData(1, dims, np.NPY_COMPLEX128, ydot_out)
        success_array = np.PyArray_SimpleNewFromData(1, dims, np.NPY_INT, success_out)
        phase_array = np.PyArray_SimpleNewFromData(1, dims, np.NPY_DOUBLE, phase_out)
        steptype_array = np.PyArray_SimpleNewFromData(1, dims, np.NPY_INT, steptype_out)

        return (t_array, y_array, ydot_array, success_array, phase_array, steptype_array)


cpdef _riccati_solve_default(
    double xi, double xf,
    np.ndarray[np.complex128_t, ndim=1] y0,
    object omega_fun,
    object gamma_fun,
    double eps,
    double epsilon_h,
    double init_stepsize,
    np.ndarray[np.float64_t, ndim=1] t_eval=None,
    bint hard_stop=False,
    int nini=16,
    int nmax=32,
    int n=32,
    int p=32
):
    """
    Convenience function to solve with default solver configuration.

    This creates a temporary solver, solves, and returns the results.
    For repeated solves with the same omega/gamma functions, create a
    RiccatiSolver object directly for better performance.

    Parameters are the same as RiccatiSolver.solve(), plus solver configuration.
    """
    solver = RiccatiSolver(omega_fun, gamma_fun, nini, nmax, n, p)
    return solver.solve(xi, xf, y0, eps, epsilon_h, init_stepsize, t_eval, hard_stop)
