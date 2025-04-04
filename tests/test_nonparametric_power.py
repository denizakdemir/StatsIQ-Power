import pytest
import numpy as np
from statsiq_power.modules.nonparametric_power import (
    mann_whitney_power,
    wilcoxon_power,
    kruskal_wallis_power,
    friedman_power
)

def test_mann_whitney_power():
    """Test Mann-Whitney U test power calculation."""
    # Test with medium effect size
    power = mann_whitney_power(effect_size=0.5, n1=30, n2=30, alpha=0.05)
    assert 0.5 < power < 1.0
    
    # Test with small effect size
    power = mann_whitney_power(effect_size=0.2, n1=30, n2=30, alpha=0.05)
    assert 0.1 < power < 0.5
    
    # Test with large effect size
    power = mann_whitney_power(effect_size=0.8, n1=30, n2=30, alpha=0.05)
    assert 0.8 < power < 1.0
    
    # Test with unequal sample sizes
    power = mann_whitney_power(effect_size=0.5, n1=20, n2=40, alpha=0.05)
    assert 0.5 < power < 1.0
    
    # Test with different alpha
    power1 = mann_whitney_power(effect_size=0.5, n1=30, n2=30, alpha=0.01)
    power2 = mann_whitney_power(effect_size=0.5, n1=30, n2=30, alpha=0.05)
    assert power1 < power2  # Lower alpha should result in lower power

def test_wilcoxon_power():
    """Test Wilcoxon signed-rank test power calculation."""
    # Test with medium effect size
    power = wilcoxon_power(effect_size=0.5, n=30, alpha=0.05)
    assert 0.5 < power < 1.0
    
    # Test with small effect size
    power = wilcoxon_power(effect_size=0.2, n=30, alpha=0.05)
    assert 0.1 < power < 0.5
    
    # Test with large effect size
    power = wilcoxon_power(effect_size=0.8, n=30, alpha=0.05)
    assert 0.8 < power < 1.0
    
    # Test with different alpha
    power1 = wilcoxon_power(effect_size=0.5, n=30, alpha=0.01)
    power2 = wilcoxon_power(effect_size=0.5, n=30, alpha=0.05)
    assert power1 < power2  # Lower alpha should result in lower power

def test_kruskal_wallis_power():
    """Test Kruskal-Wallis test power calculation."""
    # Test with medium effect size
    power = kruskal_wallis_power(effect_size=0.5, n_per_group=30, k=3, alpha=0.05)
    assert 0.5 < power < 1.0
    
    # Test with small effect size
    power = kruskal_wallis_power(effect_size=0.2, n_per_group=30, k=3, alpha=0.05)
    assert 0.1 < power < 0.5
    
    # Test with large effect size
    power = kruskal_wallis_power(effect_size=0.8, n_per_group=30, k=3, alpha=0.05)
    assert 0.8 < power < 1.0
    
    # Test with more groups
    power1 = kruskal_wallis_power(effect_size=0.5, n_per_group=30, k=3, alpha=0.05)
    power2 = kruskal_wallis_power(effect_size=0.5, n_per_group=30, k=5, alpha=0.05)
    assert power1 > power2  # More groups should result in lower power
    
    # Test with different alpha
    power1 = kruskal_wallis_power(effect_size=0.5, n_per_group=30, k=3, alpha=0.01)
    power2 = kruskal_wallis_power(effect_size=0.5, n_per_group=30, k=3, alpha=0.05)
    assert power1 < power2  # Lower alpha should result in lower power

def test_friedman_power():
    """Test Friedman test power calculation."""
    # Test with medium effect size
    power = friedman_power(effect_size=0.5, n=30, k=3, alpha=0.05)
    assert 0.5 < power < 1.0
    
    # Test with small effect size
    power = friedman_power(effect_size=0.2, n=30, k=3, alpha=0.05)
    assert 0.1 < power < 0.5
    
    # Test with large effect size
    power = friedman_power(effect_size=0.8, n=30, k=3, alpha=0.05)
    assert 0.8 < power < 1.0
    
    # Test with more conditions
    power1 = friedman_power(effect_size=0.5, n=30, k=3, alpha=0.05)
    power2 = friedman_power(effect_size=0.5, n=30, k=5, alpha=0.05)
    assert power1 > power2  # More conditions should result in lower power
    
    # Test with different alpha
    power1 = friedman_power(effect_size=0.5, n=30, k=3, alpha=0.01)
    power2 = friedman_power(effect_size=0.5, n=30, k=3, alpha=0.05)
    assert power1 < power2  # Lower alpha should result in lower power

def test_edge_cases():
    """Test edge cases for non-parametric power calculations."""
    # Test with very small sample sizes
    power = mann_whitney_power(effect_size=0.5, n1=5, n2=5, alpha=0.05)
    assert 0 < power < 1.0
    
    # Test with very large effect size
    power = wilcoxon_power(effect_size=2.0, n=30, alpha=0.05)
    assert power > 0.9
    
    # Test with very small effect size
    power = kruskal_wallis_power(effect_size=0.05, n_per_group=30, k=3, alpha=0.05)
    assert power < 0.2
    
    # Test with many groups/conditions
    power = friedman_power(effect_size=0.5, n=30, k=10, alpha=0.05)
    assert 0 < power < 1.0 