import pytest
import numpy as np
from statsiq_power.modules.mixed_power import (
    linear_mixed_power,
    glmm_power
)

def test_linear_mixed_power():
    """Test linear mixed effects model power calculation."""
    # Test with medium effect size
    power = linear_mixed_power(
        effect_size=0.5,
        n_subjects=30,
        n_observations=5,
        n_groups=2,
        icc=0.3,
        alpha=0.05
    )
    assert 0.5 < power < 1.0
    
    # Test with small effect size
    power = linear_mixed_power(
        effect_size=0.2,
        n_subjects=30,
        n_observations=5,
        n_groups=2,
        icc=0.3,
        alpha=0.05
    )
    assert 0.1 < power < 0.5
    
    # Test with large effect size
    power = linear_mixed_power(
        effect_size=0.8,
        n_subjects=30,
        n_observations=5,
        n_groups=2,
        icc=0.3,
        alpha=0.05
    )
    assert 0.8 < power < 1.0
    
    # Test with higher ICC
    power1 = linear_mixed_power(
        effect_size=0.5,
        n_subjects=30,
        n_observations=5,
        n_groups=2,
        icc=0.3,
        alpha=0.05
    )
    power2 = linear_mixed_power(
        effect_size=0.5,
        n_subjects=30,
        n_observations=5,
        n_groups=2,
        icc=0.6,
        alpha=0.05
    )
    assert power1 > power2  # Higher ICC should result in lower power
    
    # Test with more observations per subject
    power1 = linear_mixed_power(
        effect_size=0.5,
        n_subjects=30,
        n_observations=5,
        n_groups=2,
        icc=0.3,
        alpha=0.05
    )
    power2 = linear_mixed_power(
        effect_size=0.5,
        n_subjects=30,
        n_observations=10,
        n_groups=2,
        icc=0.3,
        alpha=0.05
    )
    assert power1 < power2  # More observations should result in higher power
    
    # Test with different alpha
    power1 = linear_mixed_power(
        effect_size=0.5,
        n_subjects=30,
        n_observations=5,
        n_groups=2,
        icc=0.3,
        alpha=0.01
    )
    power2 = linear_mixed_power(
        effect_size=0.5,
        n_subjects=30,
        n_observations=5,
        n_groups=2,
        icc=0.3,
        alpha=0.05
    )
    assert power1 < power2  # Lower alpha should result in lower power

def test_glmm_power():
    """Test generalized linear mixed model power calculation."""
    # Test with medium effect size
    power = glmm_power(
        effect_size=0.5,
        n_subjects=30,
        n_observations=5,
        n_groups=2,
        icc=0.3,
        alpha=0.05
    )
    assert 0.5 < power < 1.0
    
    # Test with small effect size
    power = glmm_power(
        effect_size=0.2,
        n_subjects=30,
        n_observations=5,
        n_groups=2,
        icc=0.3,
        alpha=0.05
    )
    assert 0.1 < power < 0.5
    
    # Test with large effect size
    power = glmm_power(
        effect_size=0.8,
        n_subjects=30,
        n_observations=5,
        n_groups=2,
        icc=0.3,
        alpha=0.05
    )
    assert 0.8 < power < 1.0
    
    # Test with higher ICC
    power1 = glmm_power(
        effect_size=0.5,
        n_subjects=30,
        n_observations=5,
        n_groups=2,
        icc=0.3,
        alpha=0.05
    )
    power2 = glmm_power(
        effect_size=0.5,
        n_subjects=30,
        n_observations=5,
        n_groups=2,
        icc=0.6,
        alpha=0.05
    )
    assert power1 > power2  # Higher ICC should result in lower power
    
    # Test with more observations per subject
    power1 = glmm_power(
        effect_size=0.5,
        n_subjects=30,
        n_observations=5,
        n_groups=2,
        icc=0.3,
        alpha=0.05
    )
    power2 = glmm_power(
        effect_size=0.5,
        n_subjects=30,
        n_observations=10,
        n_groups=2,
        icc=0.3,
        alpha=0.05
    )
    assert power1 < power2  # More observations should result in higher power
    
    # Test with different alpha
    power1 = glmm_power(
        effect_size=0.5,
        n_subjects=30,
        n_observations=5,
        n_groups=2,
        icc=0.3,
        alpha=0.01
    )
    power2 = glmm_power(
        effect_size=0.5,
        n_subjects=30,
        n_observations=5,
        n_groups=2,
        icc=0.3,
        alpha=0.05
    )
    assert power1 < power2  # Lower alpha should result in lower power

def test_edge_cases():
    """Test edge cases for mixed models power calculations."""
    # Test with very small sample sizes
    power = linear_mixed_power(
        effect_size=0.5,
        n_subjects=10,
        n_observations=3,
        n_groups=2,
        icc=0.3,
        alpha=0.05
    )
    assert 0 < power < 1.0
    
    # Test with very large effect size
    power = glmm_power(
        effect_size=2.0,
        n_subjects=30,
        n_observations=5,
        n_groups=2,
        icc=0.3,
        alpha=0.05
    )
    assert power > 0.9
    
    # Test with very small effect size
    power = linear_mixed_power(
        effect_size=0.05,
        n_subjects=30,
        n_observations=5,
        n_groups=2,
        icc=0.3,
        alpha=0.05
    )
    assert power < 0.2
    
    # Test with very high ICC
    power = glmm_power(
        effect_size=0.5,
        n_subjects=30,
        n_observations=5,
        n_groups=2,
        icc=0.9,
        alpha=0.05
    )
    assert 0 < power < 1.0 