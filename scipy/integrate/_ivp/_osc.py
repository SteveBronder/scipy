import inspect

import numpy as np

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
        y0 = np.array(y0, dtype=np.result_type(y0), copy=False)
    except TypeError as exc:
        raise TypeError("`y0` must be array_like.") from exc

    if y0.ndim != 1:
        raise ValueError("`y0` must be 1-dimensional.")

    if y0.dtype not in (np.float64, np.complex128):
        raise TypeError("`y0` must have dtype float64 or complex128.")

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
    """Solve an initial value problem for oscillatory systems.

    This function is a specialized variant of :func:`solve_ivp` for
    oscillatory problems. It is intended to integrate second-order
    oscillatory systems using a Riccati-based formulation.

    The API mirrors :func:`solve_ivp` with two leading callables,
    ``omega_fun`` and ``gamma_fun``, which describe the local frequency
    and damping/friction of the underlying oscillatory system. All other
    parameters follow the semantics of :func:`solve_ivp`.

    Parameters
    ----------
    omega_fun, gamma_fun : callable
        Callables describing the (possibly time- and state-dependent)
        oscillation frequency and damping. The exact calling convention
        will be documented when the solver implementation is added.
    t_span : 2-member sequence of float
        Interval of integration ``(t0, tf)``. Both entries must be
        finite, real numbers.
    y0 : array_like, shape (n,)
        Initial value of the state. For complex-valued problems pass a
        complex initial state. This stub currently requires ``float64``
        or ``complex128`` input.
    t_eval : array_like or None, optional
        Times at which to store the computed solution. Semantics mirror
        :func:`solve_ivp`. Currently accepted but ignored by the stub.
    dense_output : bool, optional
        Whether to construct a continuous solution. Currently accepted
        but ignored by the stub.
    rtol, atol : float, optional
        Relative and absolute tolerances. Must be positive scalars and
        follow the same conventions as :func:`solve_ivp`.
    events : callable or list of callables, optional
        Event functions with the same semantics as in :func:`solve_ivp`.
        Currently accepted but not used.
    vectorized : bool, optional
        Whether the user functions support vectorized evaluation.
        Accepted but not used in this stub.
    args : tuple, optional
        Additional arguments passed to user callables. Accepted but not
        used in this stub.
    **options
        Additional keyword options reserved for future extensions.

    Returns
    -------
    OdeResult
        Object with the same shape and attributes as returned by
        :func:`solve_ivp`. For this stub, successful validation always
        ends with :class:`NotImplementedError`.

    Notes
    -----
    This is a Phase 1 stub implementation which only performs basic
    argument validation and then raises :class:`NotImplementedError`.
    The actual solver logic will be provided in later phases of the
    riccati migration.
    """
    _validate_callable("omega_fun", omega_fun)
    _validate_callable("gamma_fun", gamma_fun)
    _validate_t_span(t_span)
    _validate_y0(y0)
    _validate_tolerances(rtol, atol)

    if t_eval is not None:
        t_eval = np.asarray(t_eval)
        if t_eval.ndim != 1:
            raise ValueError("`t_eval` must be 1-dimensional.")

    if args is not None:
        try:
            _ = [*(args)]
        except TypeError as exc:
            suggestion_tuple = (
                "Supplied 'args' cannot be unpacked. Please supply `args`"
                f" as a tuple (e.g. `args=({args},)`)"
            )
            raise TypeError(suggestion_tuple) from exc

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

    if not isinstance(dense_output, (bool, np.bool_)):
        raise TypeError("`dense_output` must be bool.")

    if not isinstance(vectorized, (bool, np.bool_)):
        raise TypeError("`vectorized` must be bool.")

    # Call dummy extension to verify it's callable
    _ = _ric._dummy_riccati(1)

    raise NotImplementedError("solve_ivp_osc is not implemented yet")

