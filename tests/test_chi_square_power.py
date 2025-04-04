import numpy as np
from statsiq_power.modules.chi_square_power import (
    chi_square_goodness_of_fit_power,
    chi_square_independence_power,
    chi_square_homogeneity_power,
    sample_size_for_chi_square,
    fisher_exact_power,
)

def test_chi_square_goodness_of_fit_power():
    # Test with known values
    expected_proportions = np.array([0.25, 0.25, 0.25, 0.25])
    power = chi_square_goodness_of_fit_power(expected_proportions, n=100, alpha=0.05)
    assert 0.5 < power < 1.0
    
    # Test with different sample sizes
    power_small = chi_square_goodness_of_fit_power(expected_proportions, n=50, alpha=0.05)
    power_large = chi_square_goodness_of_fit_power(expected_proportions, n=200, alpha=0.05)
    assert power_small < power_large  # Larger sample size should increase power

def test_chi_square_independence_power():
    # Test with 2x2 contingency table
    observed = np.array([[50, 30], [20, 50]])
    expected = np.array([[40, 40], [30, 60]])
    power = chi_square_independence_power(observed, expected, alpha=0.05)
    assert 0.5 < power < 1.0
    
    # Test with 3x3 contingency table
    observed = np.array([[30, 20, 10], [20, 30, 20], [10, 20, 30]])
    expected = np.array([[25, 25, 10], [20, 30, 20], [15, 15, 30]])
    power = chi_square_independence_power(observed, expected, alpha=0.05)
    assert 0.5 < power < 1.0

def test_chi_square_homogeneity_power():
    # Test with 2x2 contingency table
    observed = np.array([[50, 30], [20, 50]])
    expected = np.array([[40, 40], [30, 60]])
    power = chi_square_homogeneity_power(observed, expected, alpha=0.05)
    assert 0.5 < power < 1.0
    
    # Test with 3x3 contingency table
    observed = np.array([[30, 20, 10], [20, 30, 20], [10, 20, 30]])
    expected = np.array([[25, 25, 10], [20, 30, 20], [15, 15, 30]])
    power = chi_square_homogeneity_power(observed, expected, alpha=0.05)
    assert 0.5 < power < 1.0

def test_sample_size_for_chi_square():
    # Test with known values
    n = sample_size_for_chi_square(effect_size=0.3, power=0.8, df=1, alpha=0.05)
    assert n > 0
    
    # Test with different degrees of freedom
    n_df1 = sample_size_for_chi_square(effect_size=0.3, power=0.8, df=1, alpha=0.05)
    n_df4 = sample_size_for_chi_square(effect_size=0.3, power=0.8, df=4, alpha=0.05)
    assert n_df1 < n_df4  # More degrees of freedom should require larger sample size

def test_fisher_exact_power():
    # Test with known values
    proportions = np.array([0.3, 0.7])
    power = fisher_exact_power(proportions, n_per_group=50, alpha=0.05)
    assert 0.5 < power < 1.0
    
    # Test with different sample sizes
    power_small = fisher_exact_power(proportions, n_per_group=30, alpha=0.05)
    power_large = fisher_exact_power(proportions, n_per_group=70, alpha=0.05)
    assert power_small < power_large  # Larger sample size should increase power

def test_edge_cases():
    # Test with very small effect size
    expected_proportions = np.array([0.25, 0.25, 0.25, 0.25])
    power = chi_square_goodness_of_fit_power(expected_proportions, n=1000, alpha=0.05)
    assert 0 < power < 0.5
    
    # Test with very large effect size
    observed = np.array([[80, 20], [20, 80]])
    expected = np.array([[50, 50], [50, 50]])
    power = chi_square_independence_power(observed, expected, alpha=0.05)
    assert power > 0.8
    
    # Test with very small sample size
    expected_proportions = np.array([0.25, 0.25, 0.25, 0.25])
    power = chi_square_goodness_of_fit_power(expected_proportions, n=20, alpha=0.05)
    assert 0 < power < 0.5
    
    # Test with very large sample size
    observed = np.array([[50, 30], [20, 50]])
    expected = np.array([[45, 35], [25, 45]])
    power = chi_square_independence_power(observed, expected, alpha=0.05)
    assert power > 0.8 