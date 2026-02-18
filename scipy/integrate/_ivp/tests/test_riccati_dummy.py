"""
Tests for the _riccati extension module (Phase 2 dummy implementation).
"""
import pytest
from scipy.integrate._ivp import _riccati


class TestRiccatiDummy:
    """Test suite for the dummy _riccati extension."""

    def test_import(self):
        """Test that _riccati module can be imported."""
        # If we got here, the import at the top succeeded
        assert _riccati is not None

    def test_dummy_riccati_function(self):
        """Test that _dummy_riccati returns its input."""
        assert _riccati._dummy_riccati(5) == 5
        assert _riccati._dummy_riccati(0) == 0
        assert _riccati._dummy_riccati(-3) == -3
        assert _riccati._dummy_riccati(100) == 100
