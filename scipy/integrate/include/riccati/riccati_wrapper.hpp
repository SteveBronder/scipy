/*
 * Riccati C++ Wrapper for Python/Cython Integration
 *
 * This file provides a C-compatible interface to the templated riccati solver,
 * handling Python callbacks and type dispatch.
 */

#ifndef RICCATI_WRAPPER_HPP
#define RICCATI_WRAPPER_HPP

#include <Python.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Opaque handle to the solver state
 */
typedef struct RiccatiSolverHandle RiccatiSolverHandle;

/*
 * Initialize a riccati solver with Python callback functions.
 *
 * Parameters:
 * -----------
 * omega_fun : PyObject*
 *     Python callable for omega(x) frequency function
 * gamma_fun : PyObject*
 *     Python callable for gamma(x) friction function
 * nini : int
 *     Minimum number of Chebyshev nodes
 * nmax : int
 *     Maximum number of Chebyshev nodes
 * n : int
 *     Number of Chebyshev nodes for collocation steps
 * p : int
 *     Number of Chebyshev nodes for Riccati steps
 * omega_return_type : int*
 *     OUTPUT: detected return type (0=double, 1=complex)
 * gamma_return_type : int*
 *     OUTPUT: detected return type (0=double, 1=complex)
 *
 * Returns:
 * --------
 * void* : Opaque handle to solver, or NULL on error
 */
void* riccati_solver_init(
    PyObject* omega_fun,
    PyObject* gamma_fun,
    int nini, int nmax, int n, int p,
    int* omega_return_type,
    int* gamma_return_type
);

/*
 * Free a riccati solver handle
 *
 * Parameters:
 * -----------
 * solver_handle : void*
 *     Opaque solver handle from riccati_solver_init
 */
void riccati_solver_free(void* solver_handle);

/*
 * Solve the riccati ODE
 *
 * Parameters:
 * -----------
 * solver_handle : void*
 *     Opaque solver handle
 * xi, xf : double
 *     Integration interval [xi, xf]
 * yi, dyi : double complex
 *     Initial conditions y(xi) and y'(xi)
 * eps : double
 *     Relative tolerance for steps
 * epsilon_h : double
 *     Relative tolerance for stepsize selection
 * init_stepsize : double
 *     Initial stepsize
 * n_eval : int
 *     Number of evaluation points in t_eval (0 if NULL)
 * t_eval : double*
 *     Optional evaluation points (can be NULL)
 * hard_stop : int
 *     Whether to force stop exactly at xf (boolean)
 * n_out : int*
 *     OUTPUT: number of output points
 * t_out : double**
 *     OUTPUT: time points (allocated by this function)
 * y_out : double complex**
 *     OUTPUT: solution values (allocated by this function)
 * ydot_out : double complex**
 *     OUTPUT: derivative values (allocated by this function)
 * success_out : int**
 *     OUTPUT: success flags (allocated by this function)
 * phase_out : double**
 *     OUTPUT: phase values (allocated by this function)
 * steptype_out : int**
 *     OUTPUT: step type indicators (allocated by this function)
 *
 * Returns:
 * --------
 * int : Status code (0=success, negative=error)
 */
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
);

#ifdef __cplusplus
}
#endif

#endif // RICCATI_WRAPPER_HPP
