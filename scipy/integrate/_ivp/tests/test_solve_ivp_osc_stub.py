import numpy as np
import pytest

from scipy.integrate import solve_ivp_osc
from scipy.integrate._ivp.ivp import OdeResult


def _omega(x):
    if np.isscalar(x):
        return 1.0
    return np.ones_like(x)


def _gamma(x):
    if np.isscalar(x):
        return 0.0
    return np.zeros_like(x)


def test_import_from_public_namespace():
    # Ensure the function is exposed at the public scipy.integrate level.
    assert callable(solve_ivp_osc)


def test_invalid_omega_gamma_fun():
    t_span = (0.0, 1.0)
    y0 = [1.0, 0.0]

    with pytest.raises(TypeError, match="`omega_fun` must be callable."):
        solve_ivp_osc(None, _gamma, t_span, y0)

    with pytest.raises(TypeError, match="`gamma_fun` must be callable."):
        solve_ivp_osc(_omega, None, t_span, y0)


def test_invalid_t_span_and_y0():
    y0 = [1.0, 0.0]

    with pytest.raises(ValueError, match="`t_span` must be a 2-element sequence."):
        solve_ivp_osc(_omega, _gamma, (0.0,), y0)

    with pytest.raises(ValueError, match="Values in `t_span` must be real numbers."):
        solve_ivp_osc(_omega, _gamma, ("a", "b"), y0)

    with pytest.raises(ValueError, match="`y0` must be 1-dimensional."):
        solve_ivp_osc(_omega, _gamma, (0.0, 1.0), [[1.0, 0.0]])

    with pytest.raises(ValueError, match="`y0` must have exactly 2 elements"):
        solve_ivp_osc(_omega, _gamma, (0.0, 1.0), [1.0])


def test_invalid_tolerances():
    t_span = (0.0, 1.0)
    y0 = [1.0, 0.0]

    with pytest.raises(ValueError, match="`rtol` must be positive."):
        solve_ivp_osc(_omega, _gamma, t_span, y0, rtol=0.0)

    with pytest.raises(ValueError, match="`atol` must be positive."):
        solve_ivp_osc(_omega, _gamma, t_span, y0, atol=0.0)


def test_valid_arguments_return_ode_result():
    t_span = (0.0, 0.1)
    y0 = np.array([1.0, 0.0], dtype=np.float64)
    t_eval = np.linspace(0.0, 0.1, 3)

    result = solve_ivp_osc(
        _omega,
        _gamma,
        t_span,
        y0,
        t_eval=t_eval,
        dense_output=False,
        events=None,
        vectorized=False,
        args=None,
    )

    assert isinstance(result, OdeResult)
