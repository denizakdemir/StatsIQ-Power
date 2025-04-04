import numpy as np
from statsiq_power.modules.correlation_power import (
    pearson_correlation_power,
    spearman_correlation_power,
    kendall_correlation_power,
    partial_correlation_power,
    sample_size_for_correlation,
)

def test_pearson_correlation_power():
    # Test with known values
    power = pearson_correlation_power(r=0.3, n=100, alpha=0.05)
    assert 0.5 < power < 1.0
    
    # Test with different alternatives
    power_two_sided = pearson_correlation_power(r=0.3, n=100, alpha=0.05, alternative='two-sided')
    power_greater = pearson_correlation_power(r=0.3, n=100, alpha=0.05, alternative='greater')
    power_less = pearson_correlation_power(r=0.3, n=100, alpha=0.05, alternative='less')
    
    assert power_two_sided < power_greater  # Two-sided should have lower power
    assert power_greater > power_less  # Greater should have higher power for positive correlation

def test_spearman_correlation_power():
    # Test with known values
    power = spearman_correlation_power(r=0.3, n=100, alpha=0.05)
    assert 0.5 < power < 1.0
    
    # Test with different alternatives
    power_two_sided = spearman_correlation_power(r=0.3, n=100, alpha=0.05, alternative='two-sided')
    power_greater = spearman_correlation_power(r=0.3, n=100, alpha=0.05, alternative='greater')
    power_less = spearman_correlation_power(r=0.3, n=100, alpha=0.05, alternative='less')
    
    assert power_two_sided < power_greater  # Two-sided should have lower power
    assert power_greater > power_less  # Greater should have higher power for positive correlation

def test_kendall_correlation_power():
    # Test with known values
    power = kendall_correlation_power(tau=0.3, n=100, alpha=0.05)
    assert 0.5 < power < 1.0
    
    # Test with different alternatives
    power_two_sided = kendall_correlation_power(tau=0.3, n=100, alpha=0.05, alternative='two-sided')
    power_greater = kendall_correlation_power(tau=0.3, n=100, alpha=0.05, alternative='greater')
    power_less = kendall_correlation_power(tau=0.3, n=100, alpha=0.05, alternative='less')
    
    assert power_two_sided < power_greater  # Two-sided should have lower power
    assert power_greater > power_less  # Greater should have higher power for positive correlation

def test_partial_correlation_power():
    # Test with known values
    power = partial_correlation_power(r=0.3, n=100, k=2, alpha=0.05)
    assert 0.5 < power < 1.0
    
    # Test with different numbers of control variables
    power_k1 = partial_correlation_power(r=0.3, n=100, k=1, alpha=0.05)
    power_k3 = partial_correlation_power(r=0.3, n=100, k=3, alpha=0.05)
    assert power_k1 > power_k3  # More control variables should reduce power

def test_sample_size_for_correlation():
    # Test with known values
    n = sample_size_for_correlation(r=0.3, power=0.8, alpha=0.05)
    assert n > 0
    power = pearson_correlation_power(r=0.3, n=n, alpha=0.05)
    assert abs(power - 0.8) < 0.01
    
    # Test with different alternatives
    n_two_sided = sample_size_for_correlation(r=0.3, power=0.8, alpha=0.05, alternative='two-sided')
    n_greater = sample_size_for_correlation(r=0.3, power=0.8, alpha=0.05, alternative='greater')
    assert n_two_sided > n_greater  # Two-sided should require larger sample size

def test_edge_cases():
    # Test with very small correlation
    power = pearson_correlation_power(r=0.01, n=1000, alpha=0.05)
    assert 0 < power < 0.5
    
    # Test with very large correlation
    power = pearson_correlation_power(r=0.8, n=20, alpha=0.05)
    assert power > 0.8
    
    # Test with very small sample size
    power = pearson_correlation_power(r=0.3, n=10, alpha=0.05)
    assert 0 < power < 0.5
    
    # Test with very large sample size
    power = pearson_correlation_power(r=0.1, n=1000, alpha=0.05)
    assert power > 0.8 