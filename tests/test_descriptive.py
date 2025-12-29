import pytest
import numpy as np

from mlpeople.statistic.descriptive import (
    get_mean,
    get_median,
    get_variance,
    get_std,
    get_covariance,
    get_corrcoef,
)

# Sample data for testing
sample_1 = [1, 2, 3, 4, 5]
sample_2 = [2, 4, 6, 8, 10]

# Tolerance for floating point comparisons
TOL = 1e-8


# -------------------------------
# Test get_mean
# -------------------------------
@pytest.mark.parametrize("use_python", [True, False])
def test_get_mean(use_python):
    result = get_mean(sample_1, use_python=use_python)
    expected = np.mean(sample_1)
    assert abs(result - expected) < TOL


# -------------------------------
# Test get_median
# -------------------------------
@pytest.mark.parametrize("use_python", [True, False])
def test_get_median(use_python):
    result = get_median(sample_1, use_python=use_python)
    expected = np.median(sample_1)
    assert abs(result - expected) < TOL


# -------------------------------
# Test get_variance
# -------------------------------
@pytest.mark.parametrize("use_python", [True, False])
def test_get_variance(use_python):
    result = get_variance(sample_1, ddof=1, use_python=use_python)
    expected = np.var(sample_1, ddof=1)
    assert abs(result - expected) < TOL


# -------------------------------
# Test get_std
# -------------------------------
@pytest.mark.parametrize("use_python", [True, False])
def test_get_std(use_python):
    result = get_std(sample_1, ddof=1, use_python=use_python)
    expected = np.std(sample_1, ddof=1)
    assert abs(result - expected) < TOL


# -------------------------------
# Test get_covariance
# -------------------------------
@pytest.mark.parametrize("use_python", [True, False])
def test_get_covariance(use_python):
    result = get_covariance(sample_1, sample_2, ddof=1, use_python=use_python)
    expected = np.cov(sample_1, sample_2, ddof=1)
    if use_python:
        # Python version returns a single float
        expected_value = expected[0, 1]
        assert abs(result - expected_value) < TOL
    else:
        # NumPy returns 2x2 array
        assert np.allclose(result, expected, atol=TOL)


# -------------------------------
# Test get_corrcoef
# -------------------------------
@pytest.mark.parametrize("use_python", [True, False])
def test_get_corrcoef(use_python):
    result = get_corrcoef(sample_1, sample_2, use_python=use_python)
    expected = np.corrcoef(sample_1, sample_2)
    if use_python:
        # Python version returns a single float
        expected_value = expected[0, 1]
        assert abs(result - expected_value) < TOL
    else:
        # NumPy returns 2x2 array
        assert np.allclose(result, expected, atol=TOL)
