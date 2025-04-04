import numpy as np
from statsiq_power.modules.t_test_power import (
    one_sample_t_test_power,
    independent_t_test_power,
    paired_t_test_power,
    sample_size_for_t_test,
)

def test_one_sample_t_test_power():
    # Test with known values
    power = one_sample_t_test_power(effect_size=0.5, n=30, alpha=0.05)
    assert 0.5 < power < 1.0  # Power should be reasonable
    
    # Test with different alternatives
    power_two_sided = one_sample_t_test_power(effect_size=0.5, n=30, alpha=0.05, alternative='two-sided')
    power_greater = one_sample_t_test_power(effect_size=0.5, n=30, alpha=0.05, alternative='greater')
    power_less = one_sample_t_test_power(effect_size=0.5, n=30, alpha=0.05, alternative='less')
    
    assert power_two_sided > power_greater  # Two-sided should have lower power
    assert power_greater > power_less  # Greater should have higher power for positive effect

def test_independent_t_test_power():
    # Test with equal sample sizes
    power = independent_t_test_power(effect_size=0.5, n1=30, n2=30, alpha=0.05)
    assert 0.5 < power < 1.0
    
    # Test with unequal sample sizes
    power_unequal = independent_t_test_power(effect_size=0.5, n1=30, n2=20, alpha=0.05)
    assert 0.5 < power_unequal < 1.0
    
    # Test with different alternatives
    power_two_sided = independent_t_test_power(effect_size=0.5, n1=30, n2=30, alpha=0.05, alternative='two-sided')
    power_greater = independent_t_test_power(effect_size=0.5, n1=30, n2=30, alpha=0.05, alternative='greater')
    power_less = independent_t_test_power(effect_size=0.5, n1=30, n2=30, alpha=0.05, alternative='less')
    
    assert power_two_sided > power_greater
    assert power_greater > power_less

def test_paired_t_test_power():
    # Test with known values
    power = paired_t_test_power(effect_size=0.5, n=30, alpha=0.05)
    assert 0.5 < power < 1.0
    
    # Test with different alternatives
    power_two_sided = paired_t_test_power(effect_size=0.5, n=30, alpha=0.05, alternative='two-sided')
    power_greater = paired_t_test_power(effect_size=0.5, n=30, alpha=0.05, alternative='greater')
    power_less = paired_t_test_power(effect_size=0.5, n=30, alpha=0.05, alternative='less')
    
    assert power_two_sided > power_greater
    assert power_greater > power_less

def test_sample_size_for_t_test():
    # Test one-sample t-test
    n = sample_size_for_t_test(effect_size=0.5, power=0.8, alpha=0.05, test_type='one-sample')
    assert n > 0
    power = one_sample_t_test_power(effect_size=0.5, n=n, alpha=0.05)
    assert abs(power - 0.8) < 0.01  # Power should be close to target
    
    # Test independent t-test
    n1, n2 = sample_size_for_t_test(effect_size=0.5, power=0.8, alpha=0.05, test_type='independent')
    assert n1 > 0 and n2 > 0
    power = independent_t_test_power(effect_size=0.5, n1=n1, n2=n2, alpha=0.05)
    assert abs(power - 0.8) < 0.01
    
    # Test paired t-test
    n = sample_size_for_t_test(effect_size=0.5, power=0.8, alpha=0.05, test_type='paired')
    assert n > 0
    power = paired_t_test_power(effect_size=0.5, n=n, alpha=0.05)
    assert abs(power - 0.8) < 0.01

def test_edge_cases():
    # Test with very small effect size
    power = one_sample_t_test_power(effect_size=0.01, n=1000, alpha=0.05)
    assert 0 < power < 0.5  # Power should be low
    
    # Test with very large effect size
    power = one_sample_t_test_power(effect_size=2.0, n=10, alpha=0.05)
    assert power > 0.8  # Power should be high
    
    # Test with very small sample size
    power = one_sample_t_test_power(effect_size=0.5, n=5, alpha=0.05)
    assert 0 < power < 0.5  # Power should be low
    
    # Test with very large sample size
    power = one_sample_t_test_power(effect_size=0.1, n=1000, alpha=0.05)
    assert power > 0.8  # Power should be high 