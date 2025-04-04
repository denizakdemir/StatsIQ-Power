import pytest
import numpy as np
from statsiq_power.modules.roc_power import (
    roc_power,
    roc_auc_power,
    roc_curve_power
)

def test_roc_power():
    """Test ROC curve power calculation."""
    # Test with medium effect size (AUC = 0.75)
    power = roc_power(
        auc=0.75,
        n_cases=50,
        n_controls=50,
        alpha=0.05
    )
    assert 0.5 < power < 1.0
    
    # Test with small effect size (AUC = 0.6)
    power = roc_power(
        auc=0.6,
        n_cases=50,
        n_controls=50,
        alpha=0.05
    )
    assert 0.1 < power < 0.5
    
    # Test with large effect size (AUC = 0.9)
    power = roc_power(
        auc=0.9,
        n_cases=50,
        n_controls=50,
        alpha=0.05
    )
    assert power > 0.9

def test_roc_auc_power():
    """Test ROC AUC power calculation."""
    # Test with medium effect size (AUC = 0.75)
    power = roc_auc_power(
        auc=0.75,
        n_cases=50,
        n_controls=50,
        alpha=0.05
    )
    assert 0.5 < power < 1.0
    
    # Test with small effect size (AUC = 0.6)
    power = roc_auc_power(
        auc=0.6,
        n_cases=50,
        n_controls=50,
        alpha=0.05
    )
    assert 0.1 < power < 0.5
    
    # Test with large effect size (AUC = 0.9)
    power = roc_auc_power(
        auc=0.9,
        n_cases=50,
        n_controls=50,
        alpha=0.05
    )
    assert power > 0.9

def test_roc_curve_power():
    """Test ROC curve power calculation with specific points."""
    # Test with medium effect size
    power = roc_curve_power(
        sensitivity=0.8,
        specificity=0.7,
        n_cases=50,
        n_controls=50,
        alpha=0.05
    )
    assert 0.5 < power < 1.0
    
    # Test with small effect size
    power = roc_curve_power(
        sensitivity=0.6,
        specificity=0.6,
        n_cases=50,
        n_controls=50,
        alpha=0.05
    )
    assert 0.1 < power < 0.5
    
    # Test with large effect size
    power = roc_curve_power(
        sensitivity=0.9,
        specificity=0.9,
        n_cases=50,
        n_controls=50,
        alpha=0.05
    )
    assert power > 0.9

def test_sample_size_scaling():
    """Test that power increases with sample size."""
    # Test with medium effect size
    power1 = roc_power(
        auc=0.75,
        n_cases=30,
        n_controls=30,
        alpha=0.05
    )
    power2 = roc_power(
        auc=0.75,
        n_cases=100,
        n_controls=100,
        alpha=0.05
    )
    assert power1 < power2

def test_effect_size_scaling():
    """Test that power increases with effect size."""
    # Test with small sample size
    power1 = roc_power(
        auc=0.6,
        n_cases=30,
        n_controls=30,
        alpha=0.05
    )
    power2 = roc_power(
        auc=0.8,
        n_cases=30,
        n_controls=30,
        alpha=0.05
    )
    assert power1 < power2

def test_edge_cases():
    """Test edge cases for ROC power calculations."""
    # Test with very small sample size
    power = roc_power(
        auc=0.75,
        n_cases=10,
        n_controls=10,
        alpha=0.05
    )
    assert 0 < power < 1.0
    
    # Test with very large effect size
    power = roc_power(
        auc=0.95,
        n_cases=20,
        n_controls=20,
        alpha=0.05
    )
    assert power > 0.9
    
    # Test with balanced vs unbalanced samples
    power1 = roc_power(
        auc=0.75,
        n_cases=50,
        n_controls=50,
        alpha=0.05
    )
    power2 = roc_power(
        auc=0.75,
        n_cases=30,
        n_controls=70,
        alpha=0.05
    )
    # Unbalanced samples should have lower power
    assert power1 > power2 