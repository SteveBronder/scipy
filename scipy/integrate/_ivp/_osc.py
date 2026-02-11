"""Oscillatory initial value problem solver based on a Riccati formulation."""

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .base import DenseOutput
from .ivp import OdeResult
from . import _riccati as _ric


def _validate_t_span(t_span: Sequence[float]) -> Tuple[float, float]:
    """Validate and coerce `t_span` to a pair of floats.

    Parameters
    ----------
    t_span : 2-member sequence of float
        Integration limits ``(t0, tf)``.

    Returns
    -------
    t0, tf : float
        Start and end times.

    Raises
    ------
    ValueError
        If `t_span` is not a 2-element sequence or contains non-real values.
    """
    try:
        t_span_values: Tuple[float, float] = tuple(t_span)
    except (TypeError, ValueError) as exc:
        raise ValueError("`t_span` must be a 2-element sequence.") from exc

    if len(t_span_values) != 2:
        raise ValueError("`t_span` must be a 2-element sequence.")

    try:
        t0: float = float(t_span_values[0])
        tf: float = float(t_span_values[1])
    except (TypeError, ValueError) as exc:
        raise ValueError("Values in `t_span` must be real numbers.") from exc

    return t0, tf


def _validate_time_span(t_span: Sequence[float]) -> Tuple[float, float, float]:
    """Validate time span and compute integration direction.

    Parameters
    ----------
    t_span : 2-member sequence of float
        Integration limits ``(t0, tf)``.

    Returns
    -------
    t0, tf : float
        Start and end times.
    direction : float
        Integration direction (-1.0, 0.0, or 1.0).

    Raises
    ------
    ValueError
        If `t_span` is not a 2-element sequence or contains non-real values.
    """
    t_span_values: Tuple[float, float] = _validate_t_span(t_span)
    t0: float = t_span_values[0]
    tf: float = t_span_values[1]
    direction: float = -1.0 if tf < t0 else 1.0 if tf > t0 else 0.0
    return t0, tf, direction


def _validate_y0(y0: ArrayLike) -> NDArray[np.complex128]:
    """Validate and coerce initial conditions to a complex 1-D array.

    Parameters
    ----------
    y0 : array_like, shape (2,)
        Initial conditions ``[y(t0), y'(t0)]``.

    Returns
    -------
    y0 : ndarray, shape (2,), dtype complex128
        Validated initial conditions.

    Raises
    ------
    TypeError
        If `y0` cannot be converted to an array.
    ValueError
        If `y0` is not 1-D or does not have exactly two elements.
    """
    try:
        y0: NDArray[np.complex128] = np.asarray(y0, dtype=np.complex128)
    except (TypeError, ValueError) as exc:
        raise TypeError("`y0` must be array_like.") from exc

    if y0.ndim != 1:
        raise ValueError("`y0` must be 1-dimensional.")

    if len(y0) != 2:
        raise ValueError("`y0` must have exactly 2 elements: [y, y'].")

    return y0


def _validate_tolerances_and_eps(
    rtol: float,
    atol: float,
) -> Tuple[float, float, float]:
    """Validate tolerances and compute solver-relative epsilon.

    Parameters
    ----------
    rtol, atol : float
        Relative and absolute tolerances.

    Returns
    -------
    rtol, atol : float
        Validated positive tolerances.
    eps : float
        Relative tolerance used by the Riccati solver.

    Raises
    ------
    TypeError
        If either tolerance cannot be converted to float.
    ValueError
        If either tolerance is non-positive.
    """
    tolerances: Tuple[float, float] = _validate_tolerances(rtol, atol)
    rtol_value: float = tolerances[0]
    atol_value: float = tolerances[1]
    eps: float = max(rtol_value, 1e-15)
    return rtol_value, atol_value, eps


def _validate_epsilon_h(epsilon_h: Optional[float], rtol: float) -> float:
    """Validate epsilon_h or derive a default.

    Parameters
    ----------
    epsilon_h : float or None
        Stepsize selection tolerance.
    rtol : float
        Relative tolerance for defaulting when `epsilon_h` is None.

    Returns
    -------
    epsilon_h_value : float
        Validated stepsize selection tolerance.

    Raises
    ------
    TypeError
        If `epsilon_h` cannot be converted to float.
    ValueError
        If `epsilon_h` is non-positive.
    """
    if epsilon_h is None:
        epsilon_h_value: float = max(1e-6, rtol)
    else:
        try:
            epsilon_h_value: float = float(epsilon_h)
        except (TypeError, ValueError) as exc:
            raise TypeError("`epsilon_h` must be a positive float.") from exc
        if not (epsilon_h_value > 0):
            raise ValueError("`epsilon_h` must be positive.")
    return epsilon_h_value


def _validate_init_stepsize(init_stepsize: float, direction: float) -> float:
    """Validate and normalize the initial stepsize.

    Parameters
    ----------
    init_stepsize : float
        Initial stepsize.
    direction : float
        Integration direction (-1.0, 0.0, or 1.0).

    Returns
    -------
    init_stepsize_value : float
        Validated stepsize with sign matching the integration direction.

    Raises
    ------
    TypeError
        If `init_stepsize` cannot be converted to float.
    ValueError
        If `init_stepsize` is zero.
    """
    try:
        init_stepsize_value: float = float(init_stepsize)
    except (TypeError, ValueError) as exc:
        raise TypeError("`init_stepsize` must be a float.") from exc
    if init_stepsize_value == 0:
        raise ValueError("`init_stepsize` must be non-zero.")

    if direction != 0.0:
        signed_init_stepsize: float = np.copysign(
            abs(init_stepsize_value), direction
        )
        return signed_init_stepsize
    return init_stepsize_value


def _validate_t_eval(
    t_eval: Optional[ArrayLike],
    t0: float,
    tf: float,
    direction: float,
) -> Tuple[Optional[NDArray[np.float64]], bool]:
    """Validate `t_eval` and return normalized representation.

    Parameters
    ----------
    t_eval : array_like or None
        Times at which to store the computed solution.
    t0, tf : float
        Integration interval bounds.
    direction : float
        Integration direction (-1.0, 0.0, or 1.0).

    Returns
    -------
    t_eval_array : ndarray or None
        Array of evaluation times if provided.
    t_eval_empty : bool
        Whether an empty array was supplied.

    Raises
    ------
    ValueError
        If `t_eval` is not 1-D, is out of bounds, or is not strictly
        monotonic in the integration direction.
    """
    t_eval_array: Optional[NDArray[np.float64]] = None
    t_eval_empty: bool = False
    if t_eval is not None:
        t_eval_array: NDArray[np.float64] = np.asarray(t_eval, dtype=np.float64)
        if t_eval_array.ndim != 1:
            raise ValueError("`t_eval` must be 1-dimensional.")
        if t_eval_array.size == 0:
            t_eval_empty: bool = True
        elif direction != 0.0:
            t_eval_min: float = min(t0, tf)
            t_eval_max: float = max(t0, tf)
            if np.any(t_eval_array < t_eval_min) or np.any(t_eval_array > t_eval_max):
                raise ValueError("`t_eval` must be within `t_span`.")
            diffs: NDArray[np.float64] = np.diff(t_eval_array)
            if direction > 0 and np.any(diffs <= 0):
                raise ValueError("`t_eval` must be strictly increasing.")
            if direction < 0 and np.any(diffs >= 0):
                raise ValueError("`t_eval` must be strictly decreasing.")
    return t_eval_array, t_eval_empty


def _validate_events(
    events: Optional[Union[Callable[..., Any], Sequence[Callable[..., Any]]]],
) -> None:
    """Validate events and raise for unsupported usage.

    Parameters
    ----------
    events : callable or sequence of callables or None
        Event functions.

    Raises
    ------
    TypeError
        If an event is not callable.
    NotImplementedError
        Always raised when events are provided.
    """
    if events is None:
        return
    if callable(events):
        events_iter: Sequence[Callable[..., Any]] = (events,)
    else:
        events_iter: Sequence[Callable[..., Any]] = events
    try:
        for event in events_iter:
            if not callable(event):
                raise TypeError("Each event must be callable.")
            if hasattr(event, "direction"):
                float(event.direction)
    except TypeError as exc:
        raise TypeError("Each event must be callable.") from exc
    raise NotImplementedError(
        "events are not yet supported for solve_ivp_osc"
    )


def _validate_args(args: Optional[Tuple[Any, ...]]) -> None:
    """Validate args and raise for unsupported usage.

    Parameters
    ----------
    args : tuple or None
        Extra arguments to be passed to `omega_fun` and `gamma_fun`.

    Raises
    ------
    TypeError
        If `args` cannot be unpacked as a tuple.
    NotImplementedError
        Always raised when args are provided.
    """
    if args is None:
        return
    try:
        args_list: list[Any] = [*(args)]
    except TypeError as exc:
        suggestion_tuple: str = (
            "Supplied 'args' cannot be unpacked. Please supply `args`"
            f" as a tuple (e.g. `args=({args},)`)"
        )
        raise TypeError(suggestion_tuple) from exc
    raise NotImplementedError(
        "args are not yet supported for solve_ivp_osc"
    )


def _validate_bool_flags(dense_output: bool, vectorized: bool) -> Tuple[bool, bool]:
    """Validate boolean flags.

    Parameters
    ----------
    dense_output, vectorized : bool
        Boolean flags.

    Returns
    -------
    dense_output, vectorized : bool
        Validated boolean flags.

    Raises
    ------
    TypeError
        If either flag is not boolean.
    """
    if not isinstance(dense_output, (bool, np.bool_)):
        raise TypeError("`dense_output` must be bool.")
    if not isinstance(vectorized, (bool, np.bool_)):
        raise TypeError("`vectorized` must be bool.")
    return bool(dense_output), bool(vectorized)


def _validate_solver_options(
    options: Mapping[str, Any],
) -> Tuple[bool, int, int, int, int]:
    """Validate solver options and apply defaults.

    Parameters
    ----------
    options : mapping
        Solver options dictionary.

    Returns
    -------
    hard_stop : bool
        Whether to stop on solver warnings.
    nini, nmax, n, p : int
        Chebyshev collocation parameters.

    Raises
    ------
    TypeError
        If option values cannot be converted to the expected types.
    ValueError
        If option values are non-positive.
    """
    hard_stop: bool = options.get('hard_stop', False)
    if not isinstance(hard_stop, (bool, np.bool_)):
        raise TypeError("`hard_stop` must be bool.")

    nini: int = options.get('nini', 16)
    nmax: int = options.get('nmax', 32)
    user_set_n: bool = 'n' in options
    user_set_p: bool = 'p' in options
    n: int = options.get('n', 32)
    p: int = options.get('p', 32)
    if user_set_n and not user_set_p:
        p: int = n
    elif user_set_p and not user_set_n:
        n: int = p

    for name, value in [('nini', nini), ('nmax', nmax), ('n', n), ('p', p)]:
        try:
            value_int: int = int(value)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"`{name}` must be an integer.") from exc
        if value_int <= 0:
            raise ValueError(f"`{name}` must be positive.")

    return hard_stop, nini, nmax, n, p


@dataclass(frozen=True)
class _ValidatedInputs:
    """Container for validated inputs used by `solve_ivp_osc`."""

    t0: float
    tf: float
    direction: float
    y0: NDArray[np.complex128]
    rtol: float
    atol: float
    eps: float
    epsilon_h: float
    init_stepsize: float
    t_eval_array: Optional[NDArray[np.float64]]
    t_eval_empty: bool
    hard_stop: bool
    nini: int
    nmax: int
    n: int
    p: int

def _validate_tolerances(rtol: float, atol: float) -> Tuple[float, float]:
    """Validate and coerce tolerance values to positive floats.

    Parameters
    ----------
    rtol, atol : float
        Relative and absolute tolerances.

    Returns
    -------
    rtol, atol : float
        Validated positive tolerances.

    Raises
    ------
    TypeError
        If either tolerance cannot be converted to float.
    ValueError
        If either tolerance is non-positive.
    """
    try:
        rtol: float = float(rtol)
    except (TypeError, ValueError) as exc:
        raise TypeError("`rtol` must be a positive float.") from exc

    try:
        atol: float = float(atol)
    except (TypeError, ValueError) as exc:
        raise TypeError("`atol` must be a positive float.") from exc

    if not (rtol > 0):
        raise ValueError("`rtol` must be positive.")
    if not (atol > 0):
        raise ValueError("`atol` must be positive.")

    return rtol, atol


def _validate_callable(name: str, fun: Callable[..., Any]) -> None:
    """Ensure a parameter is callable.

    Parameters
    ----------
    name : str
        Parameter name for error reporting.
    fun : callable
        Object expected to be callable.

    Raises
    ------
    TypeError
        If `fun` is not callable.
    """
    if not callable(fun):
        raise TypeError(f"`{name}` must be callable.")


class RiccatiDenseOutput(DenseOutput):
    """Dense output using riccati solver evaluation at requested points."""

    def __init__(
        self,
        t_old: float,
        t: float,
        y0: NDArray[np.complex128],
        omega_fun: Callable[[ArrayLike], ArrayLike],
        gamma_fun: Callable[[ArrayLike], ArrayLike],
        eps: float,
        epsilon_h: float,
        init_stepsize: float,
        hard_stop: bool,
        nini: int,
        nmax: int,
        n: int,
        p: int,
    ) -> None:
        """Store solver configuration for dense output evaluation.

        Parameters
        ----------
        t_old, t : float
            Start and end of the dense-output interval.
        y0 : ndarray, shape (2,), dtype complex128
            Initial conditions at `t_old` as ``[y, y']``.
        omega_fun, gamma_fun : callable
            Frequency and friction callables.
        eps : float
            Relative tolerance.
        epsilon_h : float
            Stepsize selection tolerance.
        init_stepsize : float
            Initial stepsize.
        hard_stop : bool
            Whether to stop on solver warnings.
        nini, nmax, n, p : int
            Chebyshev collocation parameters.
        """
        super().__init__(t_old, t)
        self._y0: NDArray[np.complex128] = y0
        self._omega_fun: Callable[[ArrayLike], ArrayLike] = omega_fun
        self._gamma_fun: Callable[[ArrayLike], ArrayLike] = gamma_fun
        self._eps: float = eps
        self._epsilon_h: float = epsilon_h
        self._init_stepsize: float = init_stepsize
        self._hard_stop: bool = hard_stop
        self._nini: int = nini
        self._nmax: int = nmax
        self._n: int = n
        self._p: int = p
        self._direction: float = (
            -1.0 if t < t_old else 1.0 if t > t_old else 0.0
        )

    def _evaluate(
        self,
        t: ArrayLike,
        return_derivative: bool = False,
    ) -> NDArray[np.complex128]:
        """Evaluate y(t) or y'(t) at requested points.

        Parameters
        ----------
        t : float or array_like with shape (n_points,)
            Points at which to evaluate the solution.
        return_derivative : bool, optional
            If True, return ``y'(t)`` instead of ``y(t)``.

        Returns
        -------
        y : ndarray, shape (1,) or (1, n_points), dtype complex128
            Evaluated solution (or derivative) values.

        Raises
        ------
        Exception
            If the underlying Riccati solver fails for the requested points.
        """
        t_eval: NDArray[np.float64] = np.asarray(t, dtype=np.float64)
        t_eval_flat: NDArray[np.float64] = t_eval.ravel()
        if t_eval_flat.size == 0:
            return np.empty((1, 0), dtype=np.complex128)

        y_out: NDArray[np.complex128] = np.empty(
            t_eval_flat.shape, dtype=np.complex128
        )
        t0: float = self.t_old
        y_at_t0: np.complex128 = self._y0[1] if return_derivative else self._y0[0]

        mask_eq: NDArray[np.bool_] = t_eval_flat == t0
        if np.any(mask_eq):
            y_out[mask_eq] = y_at_t0

        def _solve_for(mask: NDArray[np.bool_], direction: float) -> None:
            if not np.any(mask):
                return
            t_segment: NDArray[np.float64] = t_eval_flat[mask]
            if direction > 0:
                order: NDArray[np.intp] = np.argsort(t_segment)
            else:
                order: NDArray[np.intp] = np.argsort(-t_segment)
            reverse: NDArray[np.intp] = np.empty_like(order)
            reverse[order] = np.arange(order.size)
            t_sorted: NDArray[np.float64] = t_segment[order]

            xf: float = float(t_sorted[-1])
            init_stepsize: float = np.copysign(abs(self._init_stepsize), direction)

            riccati_eval: Tuple[
                NDArray[np.float64],
                NDArray[np.complex128],
                NDArray[np.complex128],
                NDArray[np.int_],
                NDArray[np.float64],
                NDArray[np.int_],
                NDArray[np.complex128],
                NDArray[np.complex128],
            ] = _ric._riccati_solve_default(
                xi=t0,
                xf=xf,
                y0_0=self._y0[0],
                y0_1=self._y0[1],
                omega_fun=self._omega_fun,
                gamma_fun=self._gamma_fun,
                eps=self._eps,
                epsilon_h=self._epsilon_h,
                init_stepsize=init_stepsize,
                t_eval=t_sorted,
                hard_stop=self._hard_stop,
                nini=self._nini,
                nmax=self._nmax,
                n=self._n,
                p=self._p,
            )
            y_eval: NDArray[np.complex128] = riccati_eval[6]
            ydot_eval: NDArray[np.complex128] = riccati_eval[7]

            eval_array: NDArray[np.complex128] = (
                ydot_eval if return_derivative else y_eval
            )
            y_unsorted: NDArray[np.complex128] = np.asarray(eval_array)[reverse]
            y_out[mask] = y_unsorted

        mask_left: NDArray[np.bool_] = t_eval_flat < t0
        mask_right: NDArray[np.bool_] = t_eval_flat > t0
        _solve_for(mask_left, direction=-1.0)
        _solve_for(mask_right, direction=1.0)

        if t.ndim == 0:
            return np.array([y_out[0]])
        return y_out[np.newaxis, :]

    def _call_impl(self, t: ArrayLike) -> NDArray[np.complex128]:
        """Evaluate the dense solution at points t.

        Parameters
        ----------
        t : float or array_like with shape (n_points,)
            Points at which to evaluate the solution.

        Returns
        -------
        y : ndarray, shape (1,) or (1, n_points), dtype complex128
            Solution values at `t`.

        Raises
        ------
        Exception
            If the underlying Riccati solver fails for the requested points.
        """
        return self._evaluate(t, return_derivative=False)

    def eval_derivative(self, t: ArrayLike) -> NDArray[np.complex128]:
        """Evaluate y'(t) at points t.

        Parameters
        ----------
        t : float or array_like with shape (n_points,)
            Points at which to evaluate the derivative.

        Returns
        -------
        ydot : ndarray, shape (1,) or (1, n_points), dtype complex128
            Derivative values at `t`.

        Raises
        ------
        Exception
            If the underlying Riccati solver fails for the requested points.
        """
        return self._evaluate(t, return_derivative=True)


def solve_ivp_osc(
    omega_fun: Callable[[ArrayLike], ArrayLike],
    gamma_fun: Callable[[ArrayLike], ArrayLike],
    t_span: Sequence[float],
    y0: ArrayLike,
    *,
    t_eval: Optional[ArrayLike] = None,
    dense_output: bool = False,
    rtol: float = 1e-3,
    atol: float = 1e-6,
    epsilon_h: Optional[float] = None,
    init_stepsize: float = 0.01,
    events: Optional[Union[Callable[..., Any], Sequence[Callable[..., Any]]]] = None,
    vectorized: bool = False,
    args: Optional[Tuple[Any, ...]] = None,
    **options: Any,
) -> OdeResult:
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

        - t : ndarray, shape (n_points,), dtype float
            Time points.
        - y : ndarray, shape (n_points,), dtype complex
            Solution values at `t`.
        - status : int
            Reason for algorithm termination (0 = success).
        - message : str
            Verbal description of status.
        - success : bool
            True if solver reached end of interval.
        - t_events : list
            Empty (events not yet supported).
        - y_events : list
            Empty (events not yet supported).
        - nfev : int
            Number of omega_fun/gamma_fun evaluations.
        - njev : int
            Always 0 (not applicable).
        - nlu : int
            Always 0 (not applicable).
        - sol : callable or None
            Continuous solution if dense_output=True, otherwise None.
        - extra : dict
            Riccati diagnostics and helpers. Includes `successes`, `phases`,
            `steptypes`, `ydot`, and `sol_ydot` (callable) when dense_output
            is requested.

    Raises
    ------
    TypeError
        If inputs have incorrect types (e.g., non-callable functions, non-float
        tolerances, non-boolean flags).
    ValueError
        If `t_span` or `t_eval` are invalid, tolerances are non-positive, or
        `y0` has the wrong shape.
    NotImplementedError
        If `events` or `args` are provided.

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

    # Validate scalar inputs and options.
    time_span: Tuple[float, float, float] = _validate_time_span(t_span)
    t0: float = time_span[0]
    tf: float = time_span[1]
    direction: float = time_span[2]

    y0_array: NDArray[np.complex128] = _validate_y0(y0)
    if y0_array.shape[0] != 2:
        raise ValueError("`y0` must have exactly 2 elements [y(t0), y'(t0)].")

    tolerances: Tuple[float, float, float] = _validate_tolerances_and_eps(rtol, atol)
    rtol_value: float = tolerances[0]
    atol_value: float = tolerances[1]
    eps: float = tolerances[2]

    epsilon_h_value: float = _validate_epsilon_h(epsilon_h, rtol_value)
    init_stepsize_value: float = _validate_init_stepsize(init_stepsize, direction)
    t_eval_tuple: Tuple[Optional[NDArray[np.float64]], bool] = _validate_t_eval(
        t_eval,
        t0,
        tf,
        direction,
    )
    t_eval_array: Optional[NDArray[np.float64]] = t_eval_tuple[0]
    t_eval_empty: bool = t_eval_tuple[1]

    _validate_events(events)
    _validate_args(args)
    bool_flags: Tuple[bool, bool] = _validate_bool_flags(dense_output, vectorized)
    dense_output: bool = bool_flags[0]
    _vectorized: bool = bool_flags[1]

    solver_opts: Tuple[bool, int, int, int, int] = _validate_solver_options(
        options
    )
    hard_stop: bool = solver_opts[0]
    nini: int = solver_opts[1]
    nmax: int = solver_opts[2]
    n: int = solver_opts[3]
    p: int = solver_opts[4]

    validated: _ValidatedInputs = _ValidatedInputs(
        t0=t0,
        tf=tf,
        direction=direction,
        y0=y0_array,
        rtol=rtol_value,
        atol=atol_value,
        eps=eps,
        epsilon_h=epsilon_h_value,
        init_stepsize=init_stepsize_value,
        t_eval_array=t_eval_array,
        t_eval_empty=t_eval_empty,
        hard_stop=hard_stop,
        nini=nini,
        nmax=nmax,
        n=n,
        p=p,
    )
    
    # Call the riccati solver
    try:
        riccati_result: Tuple[
            NDArray[np.float64],
            NDArray[np.complex128],
            NDArray[np.complex128],
            NDArray[np.int_],
            NDArray[np.float64],
            NDArray[np.int_],
            NDArray[np.complex128],
            NDArray[np.complex128],
        ] = _ric._riccati_solve_default(
            xi=validated.t0,
            xf=validated.tf,
            y0_0=validated.y0[0],
            y0_1=validated.y0[1],
            omega_fun=omega_fun,
            gamma_fun=gamma_fun,
            eps=validated.eps,
            epsilon_h=validated.epsilon_h,
            init_stepsize=validated.init_stepsize,
            t_eval=(
                validated.t_eval_array
                if (validated.t_eval_array is not None and not validated.t_eval_empty)
                else None
            ),
            hard_stop=validated.hard_stop,
            nini=validated.nini,
            nmax=validated.nmax,
            n=validated.n,
            p=validated.p,
        )
        t_steps: NDArray[np.float64] = riccati_result[0]
        y_steps: NDArray[np.complex128] = riccati_result[1]
        ydot_steps: NDArray[np.complex128] = riccati_result[2]
        success_out: NDArray[np.int_] = riccati_result[3]
        phase_out: NDArray[np.float64] = riccati_result[4]
        steptype_out: NDArray[np.int_] = riccati_result[5]
        y_eval: NDArray[np.complex128] = riccati_result[6]
        ydot_eval: NDArray[np.complex128] = riccati_result[7]
    except Exception as e:
        # Build failure result
        return OdeResult(
            t=np.array([validated.t0]),
            y=validated.y0[0:1],
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
        status: int = 0
        message: str = (
            "The solver successfully reached the end of the integration interval."
        )
        success: bool = True
    else:
        status: int = -1
        message: str = "The solver did not reach the end of the integration interval."
        success: bool = False

    if validated.t_eval_empty:
        t_out: NDArray[np.float64] = validated.t_eval_array
        y_out: NDArray[np.complex128] = np.empty((0,), dtype=np.complex128)
        ydot_out: NDArray[np.complex128] = np.empty((0,), dtype=np.complex128)
    elif validated.t_eval_array is not None and y_eval.size:
        t_out: NDArray[np.float64] = validated.t_eval_array
        y_out: NDArray[np.complex128] = y_eval
        ydot_out: NDArray[np.complex128] = ydot_eval
    else:
        t_out: NDArray[np.float64] = t_steps
        y_out: NDArray[np.complex128] = y_steps
        ydot_out: NDArray[np.complex128] = ydot_steps

    sol: Optional[RiccatiDenseOutput] = None
    if dense_output:
        sol: RiccatiDenseOutput = RiccatiDenseOutput(
            validated.t0,
            validated.tf,
            validated.y0,
            omega_fun,
            gamma_fun,
            validated.eps,
            validated.epsilon_h,
            validated.init_stepsize,
            validated.hard_stop,
            validated.nini,
            validated.nmax,
            validated.n,
            validated.p,
        )

    # Build OdeResult
    # Note: y_out contains the solution values (not [y, y'])
    result: OdeResult = OdeResult(
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
    result.successes: NDArray[np.int_] = success_out
    result.phases: NDArray[np.float64] = phase_out
    result.steptypes: NDArray[np.int_] = steptype_out
    result.ydot: NDArray[np.complex128] = ydot_out
    result.extra: dict = {
        "successes": success_out,
        "phases": phase_out,
        "steptypes": steptype_out,
        "ydot": ydot_out,
    }
    if sol is not None:
        result.extra["sol_ydot"] = sol.eval_derivative

    return result
