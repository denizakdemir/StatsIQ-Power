import numpy as np
from statsiq_power.modules.regression_power import (
    linear_regression_power,
    multiple_regression_power,
    logistic_regression_power,
    sample_size_for_regression,
    sample_size_for_logistic_regression,
)

def test_linear_regression_power():
    # Test with known values
    power = linear_regression_power(r2=0.3, n=100, k=1, alpha=0.05)
    assert 0.5 < power < 1.0
    
    # Test with different sample sizes
    power_small = linear_regression_power(r2=0.3, n=50, k=1, alpha=0.05)
    power_large = linear_regression_power(r2=0.3, n=150, k=1, alpha=0.05)
    assert power_small < power_large  # Larger sample size should increase power

def test_multiple_regression_power():
    # Test with known values
    power = multiple_regression_power(r2=0.3, n=100, k=3, alpha=0.05)
    assert 0.5 < power < 1.0
    
    # Test with different numbers of predictors
    power_k2 = multiple_regression_power(r2=0.3, n=100, k=2, alpha=0.05)
    power_k4 = multiple_regression_power(r2=0.3, n=100, k=4, alpha=0.05)
    assert power_k2 > power_k4  # More predictors should reduce power

def test_logistic_regression_power():
    # Test with known values
    power = logistic_regression_power(odds_ratio=2.0, p0=0.3, n=100, alpha=0.05)
    assert 0.5 < power < 1.0
    
    # Test with different odds ratios
    power_small = logistic_regression_power(odds_ratio=1.5, p0=0.3, n=100, alpha=0.05)
    power_large = logistic_regression_power(odds_ratio=3.0, p0=0.3, n=100, alpha=0.05)
    assert power_small < power_large  # Larger odds ratio should increase power

def test_sample_size_for_regression():
    # Test with known values
    n = sample_size_for_regression(r2=0.3, power=0.8, k=1, alpha=0.05)
    assert n > 0
    power = linear_regression_power(r2=0.3, n=n, k=1, alpha=0.05)
    assert abs(power - 0.8) < 0.01
    
    # Test with different numbers of predictors
    n_k1 = sample_size_for_regression(r2=0.3, power=0.8, k=1, alpha=0.05)
    n_k3 = sample_size_for_regression(r2=0.3, power=0.8, k=3, alpha=0.05)
    assert n_k1 < n_k3  # More predictors should require larger sample size

def test_sample_size_for_logistic_regression():
    # Test with known values
    n = sample_size_for_logistic_regression(odds_ratio=2.0, p0=0.3, power=0.8, alpha=0.05)
    assert n > 0
    power = logistic_regression_power(odds_ratio=2.0, p0=0.3, n=n, alpha=0.05)
    assert abs(power - 0.8) < 0.01
    
    # Test with different baseline probabilities
    n_p01 = sample_size_for_logistic_regression(odds_ratio=2.0, p0=0.1, power=0.8, alpha=0.05)
    n_p02 = sample_size_for_logistic_regression(odds_ratio=2.0, p0=0.5, power=0.8, alpha=0.05)
    assert n_p01 > n_p02  # Extreme baseline probabilities should require larger sample size

def test_edge_cases():
    # Test with very small R-squared
    power = linear_regression_power(r2=0.01, n=1000, k=1, alpha=0.05)
    assert 0 < power < 0.5
    
    # Test with very large R-squared
    power = linear_regression_power(r2=0.8, n=20, k=1, alpha=0.05)
    assert power > 0.8
    
    # Test with very small sample size
    power = linear_regression_power(r2=0.3, n=10, k=1, alpha=0.05)
    assert 0 < power < 0.5
    
    # Test with very large sample size
    power = linear_regression_power(r2=0.1, n=1000, k=1, alpha=0.05)
    assert power > 0.8 