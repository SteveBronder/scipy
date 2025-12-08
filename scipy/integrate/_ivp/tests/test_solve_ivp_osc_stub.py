import numpy as np
import pytest

from scipy.integrate import solve_ivp_osc


def test_import_from_public_namespace():
    # Ensure the function is exposed at the public scipy.integrate level.
    assert callable(solve_ivp_osc)


def test_invalid_omega_gamma_fun():
    t_span = (0.0, 1.0)
    y0 = [1.0, 0.0]

    with pytest.raises(TypeError, match="`omega_fun` must be callable."):
        solve_ivp_osc(None, lambda t, y: 0.0, t_span, y0)

    with pytest.raises(TypeError, match="`gamma_fun` must be callable."):
        solve_ivp_osc(lambda t, y: 0.0, None, t_span, y0)


def test_invalid_t_span_and_y0():
    def omega(t, y):
        return 1.0

    def gamma(t, y):
        return 0.0

    y0 = [1.0, 0.0]

    with pytest.raises(ValueError, match="`t_span` must be a 2-element sequence."):
        solve_ivp_osc(omega, gamma, (0.0,), y0)

    with pytest.raises(ValueError, match="Values in `t_span` must be real numbers."):
        solve_ivp_osc(omega, gamma, ("a", "b"), y0)

    with pytest.raises(ValueError, match="`y0` must be 1-dimensional."):
        solve_ivp_osc(omega, gamma, (0.0, 1.0), [[1.0, 0.0]])

    with pytest.raises(TypeError, match="`y0` must have dtype float64 or complex128."):
        solve_ivp_osc(omega, gamma, (0.0, 1.0), np.array([1], dtype=np.int64))


def test_invalid_tolerances():
    def omega(t, y):
        return 1.0

    def gamma(t, y):
        return 0.0

    t_span = (0.0, 1.0)
    y0 = [1.0, 0.0]

    with pytest.raises(ValueError, match="`rtol` must be positive."):
        solve_ivp_osc(omega, gamma, t_span, y0, rtol=0.0)

    with pytest.raises(ValueError, match="`atol` must be positive."):
        solve_ivp_osc(omega, gamma, t_span, y0, atol=0.0)


def test_valid_arguments_raise_not_implemented():
    def omega(t, y):
        return 1.0

    def gamma(t, y):
        return 0.0

    t_span = (0.0, 1.0)
    y0 = np.array([1.0, 0.0], dtype=np.float64)
    t_eval = np.linspace(0.0, 1.0, 5)

    with pytest.raises(NotImplementedError, match="solve_ivp_osc is not implemented yet"):
        solve_ivp_osc(
            omega,
            gamma,
            t_span,
            y0,
            t_eval=t_eval,
            dense_output=False,
            events=None,
            vectorized=False,
            args=(),
        )

