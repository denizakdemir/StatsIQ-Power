import pytest
import numpy as np
from statsiq_power.modules.multivariate_power import (
    manova_power,
    discriminant_power
)

def test_manova_power():
    """Test MANOVA power calculation."""
    # Test with medium effect size
    power = manova_power(
        effect_size=0.5,
        n_per_group=30,
        n_groups=3,
        n_dependent=2,
        alpha=0.05
    )
    assert 0.5 < power < 1.0
    
    # Test with small effect size
    power = manova_power(
        effect_size=0.2,
        n_per_group=30,
        n_groups=3,
        n_dependent=2,
        alpha=0.05
    )
    assert 0.1 < power < 0.5
    
    # Test with large effect size
    power = manova_power(
        effect_size=0.8,
        n_per_group=30,
        n_groups=3,
        n_dependent=2,
        alpha=0.05
    )
    assert 0.8 < power < 1.0
    
    # Test with more dependent variables
    power1 = manova_power(
        effect_size=0.5,
        n_per_group=30,
        n_groups=3,
        n_dependent=2,
        alpha=0.05
    )
    power2 = manova_power(
        effect_size=0.5,
        n_per_group=30,
        n_groups=3,
        n_dependent=4,
        alpha=0.05
    )
    assert power1 > power2  # More dependent variables should result in lower power
    
    # Test with different alpha
    power1 = manova_power(
        effect_size=0.5,
        n_per_group=30,
        n_groups=3,
        n_dependent=2,
        alpha=0.01
    )
    power2 = manova_power(
        effect_size=0.5,
        n_per_group=30,
        n_groups=3,
        n_dependent=2,
        alpha=0.05
    )
    assert power1 < power2  # Lower alpha should result in lower power

def test_discriminant_power():
    """Test discriminant analysis power calculation."""
    # Test with medium effect size
    power = discriminant_power(
        effect_size=0.5,
        n_per_group=30,
        n_groups=3,
        n_predictors=4,
        alpha=0.05
    )
    assert 0.5 < power < 1.0
    
    # Test with small effect size
    power = discriminant_power(
        effect_size=0.2,
        n_per_group=30,
        n_groups=3,
        n_predictors=4,
        alpha=0.05
    )
    assert 0.1 < power < 0.5
    
    # Test with large effect size
    power = discriminant_power(
        effect_size=0.8,
        n_per_group=30,
        n_groups=3,
        n_predictors=4,
        alpha=0.05
    )
    assert 0.8 < power < 1.0
    
    # Test with more predictors
    power1 = discriminant_power(
        effect_size=0.5,
        n_per_group=30,
        n_groups=3,
        n_predictors=4,
        alpha=0.05
    )
    power2 = discriminant_power(
        effect_size=0.5,
        n_per_group=30,
        n_groups=3,
        n_predictors=8,
        alpha=0.05
    )
    assert power1 > power2  # More predictors should result in lower power
    
    # Test with different alpha
    power1 = discriminant_power(
        effect_size=0.5,
        n_per_group=30,
        n_groups=3,
        n_predictors=4,
        alpha=0.01
    )
    power2 = discriminant_power(
        effect_size=0.5,
        n_per_group=30,
        n_groups=3,
        n_predictors=4,
        alpha=0.05
    )
    assert power1 < power2  # Lower alpha should result in lower power

def test_edge_cases():
    """Test edge cases for multivariate power calculations."""
    # Test with very small sample sizes
    power = manova_power(
        effect_size=0.5,
        n_per_group=10,
        n_groups=3,
        n_dependent=2,
        alpha=0.05
    )
    assert 0 < power < 1.0
    
    # Test with very large effect size
    power = discriminant_power(
        effect_size=2.0,
        n_per_group=30,
        n_groups=3,
        n_predictors=4,
        alpha=0.05
    )
    assert power > 0.9
    
    # Test with very small effect size
    power = manova_power(
        effect_size=0.05,
        n_per_group=30,
        n_groups=3,
        n_dependent=2,
        alpha=0.05
    )
    assert power < 0.2
    
    # Test with many groups
    power = discriminant_power(
        effect_size=0.5,
        n_per_group=30,
        n_groups=10,
        n_predictors=4,
        alpha=0.05
    )
    assert 0 < power < 1.0 