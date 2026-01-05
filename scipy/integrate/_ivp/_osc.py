import inspect

import numpy as np

from .base import DenseOutput
from .ivp import OdeResult
from . import _riccati as _ric


def _validate_t_span(t_span):
    try:
        t0, tf = t_span
    except (TypeError, ValueError) as exc:
        raise ValueError("`t_span` must be a 2-element sequence.") from exc

    try:
        t0 = float(t0)
        tf = float(tf)
    except (TypeError, ValueError) as exc:
        raise ValueError("Values in `t_span` must be real numbers.") from exc

    return t0, tf


def _validate_y0(y0):
    try:
        y0 = np.asarray(y0, dtype=np.complex128)
    except (TypeError, ValueError) as exc:
        raise TypeError("`y0` must be array_like.") from exc

    if y0.ndim != 1:
        raise ValueError("`y0` must be 1-dimensional.")

    if len(y0) != 2:
        raise ValueError("`y0` must have exactly 2 elements: [y, y'].")

    return y0


def _validate_tolerances(rtol, atol):
    try:
        rtol = float(rtol)
    except (TypeError, ValueError) as exc:
        raise TypeError("`rtol` must be a positive float.") from exc

    try:
        atol = float(atol)
    except (TypeError, ValueError) as exc:
        raise TypeError("`atol` must be a positive float.") from exc

    if not (rtol > 0):
        raise ValueError("`rtol` must be positive.")
    if not (atol > 0):
        raise ValueError("`atol` must be positive.")

    return rtol, atol


def _validate_callable(name, fun):
    if not callable(fun):
        raise TypeError(f"`{name}` must be callable.")


class RiccatiDenseOutput(DenseOutput):
    """Dense output using riccati solver evaluation at requested points."""

    def __init__(
        self,
        t_old,
        t,
        y0,
        omega_fun,
        gamma_fun,
        eps,
        epsilon_h,
        init_stepsize,
        hard_stop,
        nini,
        nmax,
        n,
        p,
    ):
        super().__init__(t_old, t)
        self._y0 = y0
        self._omega_fun = omega_fun
        self._gamma_fun = gamma_fun
        self._eps = eps
        self._epsilon_h = epsilon_h
        self._init_stepsize = init_stepsize
        self._hard_stop = hard_stop
        self._nini = nini
        self._nmax = nmax
        self._n = n
        self._p = p
        self._direction = -1.0 if t < t_old else 1.0 if t > t_old else 0.0

    def _evaluate(self, t, return_derivative=False):
        t_eval = np.asarray(t, dtype=np.float64)
        t_eval_flat = t_eval.ravel()
        if t_eval_flat.size == 0:
            return np.empty((1, 0), dtype=np.complex128)

        if self._direction >= 0:
            order = np.argsort(t_eval_flat)
        else:
            order = np.argsort(-t_eval_flat)
        reverse = np.empty_like(order)
        reverse[order] = np.arange(order.size)
        t_sorted = t_eval_flat[order]

        _, _, _, _, _, _, y_eval, ydot_eval = _ric._riccati_solve_default(
            xi=self.t_old,
            xf=self.t,
            y0_0=self._y0[0],
            y0_1=self._y0[1],
            omega_fun=self._omega_fun,
            gamma_fun=self._gamma_fun,
            eps=self._eps,
            epsilon_h=self._epsilon_h,
            init_stepsize=self._init_stepsize,
            t_eval=t_sorted,
            hard_stop=self._hard_stop,
            nini=self._nini,
            nmax=self._nmax,
            n=self._n,
            p=self._p,
        )

        eval_array = ydot_eval if return_derivative else y_eval
        y_unsorted = np.asarray(eval_array)[reverse]
        if t.ndim == 0:
            return np.array([y_unsorted[0]])
        return y_unsorted[np.newaxis, :]

    def _call_impl(self, t):
        return self._evaluate(t, return_derivative=False)

    def eval_derivative(self, t):
        """Evaluate y'(t) at points t."""
        return self._evaluate(t, return_derivative=True)


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
    epsilon_h=None,
    init_stepsize=0.01,
    events=None,
    vectorized=False,
    args=None,
    **options,
):
    """Solve an initial value problem for oscillatory systems.

    This function is a specialized variant of :func:`solve_ivp` for
    oscillatory problems. It integrates second-order oscillatory systems
    using a Riccati-based formulation.

    Parameters
    ----------
    omega_fun, gamma_fun : callable
        Callables describing the oscillation frequency omega(x) and
        friction gamma(x). Both must accept a scalar or array argument
        and return values of the same shape. They can return either
        real (float) or complex values.
    t_span : 2-member sequence of float
        Interval of integration ``(t0, tf)``. Both entries must be
        finite, real numbers.
    y0 : array_like, shape (2,)
        Initial conditions as ``[y(t0), y'(t0)]``. Must be a 2-element
        array containing the initial value and its derivative. Can be
        complex-valued.
    t_eval : array_like or None, optional
        Times at which to store the computed solution. If None, the
        solver chooses time points automatically. Must be strictly monotonic
        and lie within `t_span`.
    dense_output : bool, optional
        Whether to construct a continuous solution. The returned `sol(t)`
        re-evaluates the solver at requested points.
    rtol, atol : float, optional
        Relative and absolute tolerances. The solver maps these to its
        internal tolerance `eps = max(rtol, 1e-15)`. Note that `atol`
        is validated but not directly used (riccati is relative-tolerance
        based).
    epsilon_h : float, optional
        Tolerance for stepsize selection. Controls how accurately the
        frequency and friction functions are interpolated. If not provided,
        defaults to `max(1e-6, rtol)`.
    init_stepsize : float, optional
        Initial stepsize for integration. Default is 0.01.
    events : callable or list of callables, optional
        Event functions. Not yet implemented.
    vectorized : bool, optional
        Whether omega_fun and gamma_fun support vectorized evaluation.
        The riccati solver calls these functions with both scalars and
        arrays, so they should handle both cases.
    args : tuple, optional
        Additional arguments passed to omega_fun and gamma_fun.
        Not yet implemented.
    **options
        Advanced solver options:

        - nini : int, default 16
            Minimum number of Chebyshev nodes
        - nmax : int, default 32
            Maximum number of Chebyshev nodes
        - n : int, default 32
            Number of Chebyshev nodes for collocation steps
        - p : int, default same as n (32)
            Number of Chebyshev nodes for Riccati steps

    Returns
    -------
    OdeResult
        Object with fields:

        - t : ndarray, shape (n_points,)
            Time points
        - y : ndarray, shape (n_points,)
            Solution values at t
        - status : int
            Reason for algorithm termination (0 = success)
        - message : str
            Verbal description of status
        - success : bool
            True if solver reached end of interval
        - t_events : list
            Empty (events not yet supported)
        - y_events : list
            Empty (events not yet supported)
        - nfev : int
            Number of omega_fun/gamma_fun evaluations
        - njev : int
            Always 0 (not applicable)
        - nlu : int
            Always 0 (not applicable)
        - sol : callable or None
            Continuous solution if dense_output=True, otherwise None
        - extra : dict
            Riccati diagnostics and helpers. Includes `successes`, `phases`,
            `steptypes`, `ydot`, and `sol_ydot` (callable) when dense_output
            is requested.

    .. versionadded:: 1.18.0

    Notes
    -----
    The riccati solver is specialized for second-order ODEs of the form
    ``y'' + 2*gamma(x)*y' + omega(x)**2 * y = 0``. It uses adaptive
    Chebyshev spectral collocation and can efficiently handle both
    oscillatory and non-oscillatory regions.
    When `dense_output=True`, evaluating `sol(t)` re-runs the solver at the
    requested points.

    Examples
    --------
    Solve the Airy equation y'' - x*y = 0:

    >>> from scipy.integrate import solve_ivp_osc
    >>> import numpy as np
    >>> omega_fun = lambda x: np.sqrt(np.abs(x))
    >>> gamma_fun = lambda x: np.zeros_like(x) if hasattr(x, '__len__') else 0.0
    >>> y0 = [1.0, 0.0]  # y(0) = 1, y'(0) = 0
    >>> result = solve_ivp_osc(omega_fun, gamma_fun, (0, 10), y0)
    """
    # Validate callables
    _validate_callable("omega_fun", omega_fun)
    _validate_callable("gamma_fun", gamma_fun)
    
    # Validate and extract t_span
    t0, tf = _validate_t_span(t_span)
    direction = -1.0 if tf < t0 else 1.0 if tf > t0 else 0.0
    
    # Validate and convert y0 to complex128 array
    y0 = _validate_y0(y0)
    if y0.shape[0] != 2:
        raise ValueError("`y0` must have exactly 2 elements [y(t0), y'(t0)].")
    
    # Convert to complex128 if needed
    if y0.dtype == np.float64:
        y0 = y0.astype(np.complex128)
    
    # Validate tolerances
    rtol, atol = _validate_tolerances(rtol, atol)
    
    # Convert rtol/atol to eps (riccati uses relative tolerance)
    eps = max(rtol, 1e-15)
    
    # Determine epsilon_h
    if epsilon_h is None:
        epsilon_h = max(1e-6, rtol)
    else:
        try:
            epsilon_h = float(epsilon_h)
        except (TypeError, ValueError) as exc:
            raise TypeError("`epsilon_h` must be a positive float.") from exc
        if not (epsilon_h > 0):
            raise ValueError("`epsilon_h` must be positive.")
    
    # Validate init_stepsize
    try:
        init_stepsize = float(init_stepsize)
    except (TypeError, ValueError) as exc:
        raise TypeError("`init_stepsize` must be a float.") from exc
    if init_stepsize == 0:
        raise ValueError("`init_stepsize` must be non-zero.")

    if direction != 0.0:
        init_stepsize = np.copysign(abs(init_stepsize), direction)
    
    # Validate t_eval
    t_eval_array = None
    if t_eval is not None:
        t_eval_array = np.asarray(t_eval, dtype=np.float64)
        if t_eval_array.ndim != 1:
            raise ValueError("`t_eval` must be 1-dimensional.")
        if direction != 0.0:
            t_eval_min = min(t0, tf)
            t_eval_max = max(t0, tf)
            if np.any(t_eval_array < t_eval_min) or np.any(t_eval_array > t_eval_max):
                raise ValueError("`t_eval` must be within `t_span`.")
            diffs = np.diff(t_eval_array)
            if direction > 0 and np.any(diffs <= 0):
                raise ValueError("`t_eval` must be strictly increasing.")
            if direction < 0 and np.any(diffs >= 0):
                raise ValueError("`t_eval` must be strictly decreasing.")
    
    if events is not None:
        if callable(events):
            events = (events,)
        try:
            for event in events:
                if not callable(event):
                    raise TypeError("Each event must be callable.")
                if hasattr(event, "direction"):
                    float(event.direction)
        except TypeError as exc:
            raise TypeError("Each event must be callable.") from exc
        raise NotImplementedError(
            "events are not yet supported for solve_ivp_osc"
        )
    
    if args is not None:
        try:
            _ = [*(args)]
        except TypeError as exc:
            suggestion_tuple = (
                "Supplied 'args' cannot be unpacked. Please supply `args`"
                f" as a tuple (e.g. `args=({args},)`)"
            )
            raise TypeError(suggestion_tuple) from exc
        raise NotImplementedError(
            "args are not yet supported for solve_ivp_osc"
        )
    
    if not isinstance(dense_output, (bool, np.bool_)):
        raise TypeError("`dense_output` must be bool.")
    
    if not isinstance(vectorized, (bool, np.bool_)):
        raise TypeError("`vectorized` must be bool.")

    hard_stop = options.get('hard_stop', False)
    if not isinstance(hard_stop, (bool, np.bool_)):
        raise TypeError("`hard_stop` must be bool.")

    # Extract solver options
    nini = options.get('nini', 16)
    nmax = options.get('nmax', 32)
    user_set_n = 'n' in options
    user_set_p = 'p' in options
    n = options.get('n', 32)
    p = options.get('p', 32)
    if user_set_n and not user_set_p:
        p = n
    elif user_set_p and not user_set_n:
        n = p
    
    # Validate solver options
    for name, value in [('nini', nini), ('nmax', nmax), ('n', n), ('p', p)]:
        try:
            value = int(value)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"`{name}` must be an integer.") from exc
        if value <= 0:
            raise ValueError(f"`{name}` must be positive.")
    
    # Call the riccati solver
    try:
        (t_steps, y_steps, ydot_steps, success_out, phase_out, steptype_out,
         y_eval, ydot_eval) = (
            _ric._riccati_solve_default(
                xi=t0,
                xf=tf,
                y0_0=y0[0],
                y0_1=y0[1],
                omega_fun=omega_fun,
                gamma_fun=gamma_fun,
                eps=eps,
                epsilon_h=epsilon_h,
                init_stepsize=init_stepsize,
                t_eval=t_eval_array if t_eval_array is not None else None,
                hard_stop=hard_stop,
                nini=nini,
                nmax=nmax,
                n=n,
                p=p,
            )
        )
    except Exception as e:
        # Build failure result
        return OdeResult(
            t=np.array([t0]),
            y=y0[0:1],
            sol=None,
            t_events=[],
            y_events=[],
            nfev=0,
            njev=0,
            nlu=0,
            status=-1,
            message=f"Riccati solver failed: {str(e)}",
            success=False,
        )
    
    # Check if solver reached the end successfully
    # Status code 0 indicates success
    if len(success_out) > 0 and success_out[-1] == 1:
        status = 0
        message = "The solver successfully reached the end of the integration interval."
        success = True
    else:
        status = -1
        message = "The solver did not reach the end of the integration interval."
        success = False
    
    if t_eval_array is not None and y_eval.size:
        t_out = t_eval_array
        y_out = y_eval
        ydot_out = ydot_eval
    else:
        t_out = t_steps
        y_out = y_steps
        ydot_out = ydot_steps

    sol = None
    if dense_output:
        sol = RiccatiDenseOutput(
            t0,
            tf,
            y0,
            omega_fun,
            gamma_fun,
            eps,
            epsilon_h,
            init_stepsize,
            hard_stop,
            nini,
            nmax,
            n,
            p,
        )

    # Build OdeResult
    # Note: y_out contains the solution values (not [y, y'])
    result = OdeResult(
        t=t_out,
        y=y_out,
        sol=sol,
        t_events=[],
        y_events=[],
        nfev=len(t_out),  # Approximate: each step evaluates omega/gamma
        njev=0,
        nlu=0,
        status=status,
        message=message,
        success=success,
    )
    # Attach diagnostics from the Riccati core
    result.successes = success_out
    result.phases = phase_out
    result.steptypes = steptype_out
    result.ydot = ydot_out
    result.extra = {
        "successes": success_out,
        "phases": phase_out,
        "steptypes": steptype_out,
        "ydot": ydot_out,
    }
    if sol is not None:
        result.extra["sol_ydot"] = sol.eval_derivative

    return result
