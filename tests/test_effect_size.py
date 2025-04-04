import numpy as np
from statsiq_power.utils.effect_size import (
    cohens_d,
    hedges_g,
    cohens_f,
    cohens_w,
    eta_squared,
    partial_eta_squared,
    odds_ratio,
    risk_ratio,
)

def test_cohens_d():
    # Test with known values
    d = cohens_d(mean1=10, mean2=8, sd1=2, sd2=2)
    assert abs(d - 1.0) < 0.01  # Should be 1.0 for these values
    
    # Test with different standard deviations
    d_unequal = cohens_d(mean1=10, mean2=8, sd1=2, sd2=3)
    assert abs(d_unequal - 0.8) < 0.01  # Should be 0.8 for these values
    
    # Test with sample sizes (for Hedges' g correction)
    d_with_n = cohens_d(mean1=10, mean2=8, sd1=2, sd2=2, n1=30, n2=30)
    assert abs(d_with_n - 1.0) < 0.01

def test_hedges_g():
    # Test with known values
    g = hedges_g(d=1.0, n1=30, n2=30)
    assert g < 1.0  # Hedges' g should be smaller than Cohen's d
    
    # Test with different sample sizes
    g_small = hedges_g(d=1.0, n1=10, n2=10)
    g_large = hedges_g(d=1.0, n1=100, n2=100)
    assert g_small < g_large  # Larger sample size should make g closer to d

def test_cohens_f():
    # Test with known values
    means = np.array([10, 12, 14])
    sds = np.array([2, 2, 2])
    ns = np.array([30, 30, 30])
    f = cohens_f(means, sds, ns)
    assert f > 0  # Should be positive
    
    # Test with equal means (should be 0)
    means_equal = np.array([10, 10, 10])
    f_equal = cohens_f(means_equal, sds, ns)
    assert abs(f_equal) < 0.01

def test_cohens_w():
    # Test with known values
    observed = np.array([50, 30, 20])
    expected = np.array([33.33, 33.33, 33.33])
    w = cohens_w(observed, expected)
    assert w > 0  # Should be positive
    
    # Test with equal observed and expected (should be 0)
    w_equal = cohens_w(observed, observed)
    assert abs(w_equal) < 0.01

def test_eta_squared():
    # Test with known values
    ss_effect = 100
    ss_total = 200
    eta2 = eta_squared(ss_effect, ss_total)
    assert abs(eta2 - 0.5) < 0.01  # Should be 0.5 for these values
    
    # Test with zero effect (should be 0)
    eta2_zero = eta_squared(0, ss_total)
    assert abs(eta2_zero) < 0.01

def test_partial_eta_squared():
    # Test with known values
    ss_effect = 100
    ss_error = 100
    p_eta2 = partial_eta_squared(ss_effect, ss_error)
    assert abs(p_eta2 - 0.5) < 0.01  # Should be 0.5 for these values
    
    # Test with zero effect (should be 0)
    p_eta2_zero = partial_eta_squared(0, ss_error)
    assert abs(p_eta2_zero) < 0.01

def test_odds_ratio():
    # Test with known values
    or_value = odds_ratio(50, 30, 20, 50)
    assert or_value > 1  # Should be greater than 1 for these values
    
    # Test with equal proportions (should be 1)
    or_equal = odds_ratio(50, 50, 50, 50)
    assert abs(or_equal - 1.0) < 0.01

def test_risk_ratio():
    # Test with known values
    rr = risk_ratio(50, 30, 20, 50)
    assert rr > 1  # Should be greater than 1 for these values
    
    # Test with equal proportions (should be 1)
    rr_equal = risk_ratio(50, 50, 50, 50)
    assert abs(rr_equal - 1.0) < 0.01

def test_edge_cases():
    # Test with very small values
    d_small = cohens_d(mean1=10.01, mean2=10, sd1=2, sd2=2)
    assert 0 < d_small < 0.1
    
    # Test with very large values
    d_large = cohens_d(mean1=20, mean2=10, sd1=2, sd2=2)
    assert d_large > 2.0
    
    # Test with zero standard deviation
    d_zero_sd = cohens_d(mean1=10, mean2=8, sd1=0.0001, sd2=2)
    assert d_zero_sd > 10.0  # Should be very large
    
    # Test with extreme proportions
    rr_extreme = risk_ratio(1, 99, 50, 50)
    assert rr_extreme < 0.1  # Should be very small 