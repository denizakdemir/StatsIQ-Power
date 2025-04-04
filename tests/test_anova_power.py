import numpy as np
from statsiq_power.modules.anova_power import (
    one_way_anova_power,
    factorial_anova_power,
    repeated_measures_anova_power,
    mixed_anova_power,
    sample_size_for_anova,
)

def test_one_way_anova_power():
    # Test with known values
    power = one_way_anova_power(effect_size=0.3, n_per_group=20, k=3, alpha=0.05)
    assert 0.5 < power < 1.0
    
    # Test with different numbers of groups
    power_k2 = one_way_anova_power(effect_size=0.3, n_per_group=20, k=2, alpha=0.05)
    power_k4 = one_way_anova_power(effect_size=0.3, n_per_group=20, k=4, alpha=0.05)
    assert power_k2 > power_k4  # More groups should require more power

def test_factorial_anova_power():
    # Test with 2x2 design
    effect_sizes = {
        'factor_1': 0.3,
        'factor_2': 0.3,
        'interaction_1_2': 0.3
    }
    power = factorial_anova_power(effect_sizes, n_per_cell=20, factors=[2, 2], alpha=0.05)
    assert all(0.5 < p < 1.0 for p in power.values())
    
    # Test with 2x3 design
    effect_sizes = {
        'factor_1': 0.3,
        'factor_2': 0.3,
        'interaction_1_2': 0.3
    }
    power = factorial_anova_power(effect_sizes, n_per_cell=20, factors=[2, 3], alpha=0.05)
    assert all(0.5 < p < 1.0 for p in power.values())

def test_repeated_measures_anova_power():
    # Test with known values
    power = repeated_measures_anova_power(effect_size=0.3, n=20, k=3, alpha=0.05)
    assert 0.5 < power < 1.0
    
    # Test with different epsilon values
    power_eps1 = repeated_measures_anova_power(effect_size=0.3, n=20, k=3, alpha=0.05, epsilon=1.0)
    power_eps5 = repeated_measures_anova_power(effect_size=0.3, n=20, k=3, alpha=0.05, epsilon=0.5)
    assert power_eps1 > power_eps5  # Lower epsilon should reduce power

def test_mixed_anova_power():
    # Test with known values
    power = mixed_anova_power(
        between_effect_size=0.3,
        within_effect_size=0.3,
        n_per_group=20,
        k=3,
        num_groups=2,
        alpha=0.05
    )
    assert all(0.5 < p < 1.0 for p in power.values())
    
    # Test with different epsilon values
    power_eps1 = mixed_anova_power(
        between_effect_size=0.3,
        within_effect_size=0.3,
        n_per_group=20,
        k=3,
        num_groups=2,
        alpha=0.05,
        epsilon=1.0
    )
    power_eps5 = mixed_anova_power(
        between_effect_size=0.3,
        within_effect_size=0.3,
        n_per_group=20,
        k=3,
        num_groups=2,
        alpha=0.05,
        epsilon=0.5
    )
    assert power_eps1['within'] > power_eps5['within']
    assert power_eps1['interaction'] > power_eps5['interaction']

def test_sample_size_for_anova():
    # Test one-way ANOVA
    n = sample_size_for_anova(effect_size=0.3, power=0.8, k=3, alpha=0.05, design='one-way')
    assert n > 0
    power = one_way_anova_power(effect_size=0.3, n_per_group=n, k=3, alpha=0.05)
    assert abs(power - 0.8) < 0.01
    
    # Test repeated measures ANOVA
    n = sample_size_for_anova(effect_size=0.3, power=0.8, k=3, alpha=0.05, design='repeated')
    assert n > 0
    power = repeated_measures_anova_power(effect_size=0.3, n=n, k=3, alpha=0.05)
    assert abs(power - 0.8) < 0.01

def test_edge_cases():
    # Test with very small effect size
    power = one_way_anova_power(effect_size=0.01, n_per_group=1000, k=3, alpha=0.05)
    assert 0 < power < 0.5
    
    # Test with very large effect size
    power = one_way_anova_power(effect_size=1.0, n_per_group=10, k=3, alpha=0.05)
    assert power > 0.8
    
    # Test with very small sample size
    power = one_way_anova_power(effect_size=0.3, n_per_group=5, k=3, alpha=0.05)
    assert 0 < power < 0.5
    
    # Test with very large sample size
    power = one_way_anova_power(effect_size=0.1, n_per_group=1000, k=3, alpha=0.05)
    assert power > 0.8 