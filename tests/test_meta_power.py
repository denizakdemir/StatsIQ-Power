import pytest
import numpy as np
from statsiq_power.modules.meta_power import (
    fixed_effects_power,
    random_effects_power
)

def test_fixed_effects_power():
    """Test fixed effects meta-analysis power calculation."""
    # Test with medium effect size
    power = fixed_effects_power(
        effect_size=0.5,
        n_studies=10,
        avg_sample_size=50,
        alpha=0.05
    )
    assert 0.5 < power < 1.0
    
    # Test with small effect size
    power = fixed_effects_power(
        effect_size=0.2,
        n_studies=10,
        avg_sample_size=50,
        alpha=0.05
    )
    assert 0.1 < power < 0.5
    
    # Test with large effect size
    power = fixed_effects_power(
        effect_size=0.8,
        n_studies=10,
        avg_sample_size=50,
        alpha=0.05
    )
    assert 0.8 < power < 1.0
    
    # Test with more studies
    power1 = fixed_effects_power(
        effect_size=0.5,
        n_studies=5,
        avg_sample_size=50,
        alpha=0.05
    )
    power2 = fixed_effects_power(
        effect_size=0.5,
        n_studies=20,
        avg_sample_size=50,
        alpha=0.05
    )
    assert power1 < power2  # More studies should result in higher power
    
    # Test with larger sample sizes
    power1 = fixed_effects_power(
        effect_size=0.5,
        n_studies=10,
        avg_sample_size=30,
        alpha=0.05
    )
    power2 = fixed_effects_power(
        effect_size=0.5,
        n_studies=10,
        avg_sample_size=100,
        alpha=0.05
    )
    assert power1 < power2  # Larger sample sizes should result in higher power

def test_random_effects_power():
    """Test random effects meta-analysis power calculation."""
    # Test with medium effect size
    power = random_effects_power(
        effect_size=0.5,
        n_studies=10,
        avg_sample_size=50,
        heterogeneity=0.2,
        alpha=0.05
    )
    assert 0.5 < power < 1.0
    
    # Test with small effect size
    power = random_effects_power(
        effect_size=0.2,
        n_studies=10,
        avg_sample_size=50,
        heterogeneity=0.2,
        alpha=0.05
    )
    assert 0.1 < power < 0.5
    
    # Test with large effect size
    power = random_effects_power(
        effect_size=0.8,
        n_studies=10,
        avg_sample_size=50,
        heterogeneity=0.2,
        alpha=0.05
    )
    assert 0.8 < power < 1.0
    
    # Test with more studies
    power1 = random_effects_power(
        effect_size=0.5,
        n_studies=5,
        avg_sample_size=50,
        heterogeneity=0.2,
        alpha=0.05
    )
    power2 = random_effects_power(
        effect_size=0.5,
        n_studies=20,
        avg_sample_size=50,
        heterogeneity=0.2,
        alpha=0.05
    )
    assert power1 < power2  # More studies should result in higher power
    
    # Test with larger sample sizes
    power1 = random_effects_power(
        effect_size=0.5,
        n_studies=10,
        avg_sample_size=30,
        heterogeneity=0.2,
        alpha=0.05
    )
    power2 = random_effects_power(
        effect_size=0.5,
        n_studies=10,
        avg_sample_size=100,
        heterogeneity=0.2,
        alpha=0.05
    )
    assert power1 < power2  # Larger sample sizes should result in higher power
    
    # Test with different heterogeneity levels
    power1 = random_effects_power(
        effect_size=0.5,
        n_studies=10,
        avg_sample_size=50,
        heterogeneity=0.1,
        alpha=0.05
    )
    power2 = random_effects_power(
        effect_size=0.5,
        n_studies=10,
        avg_sample_size=50,
        heterogeneity=0.5,
        alpha=0.05
    )
    assert power1 > power2  # Lower heterogeneity should result in higher power

def test_edge_cases():
    """Test edge cases for meta-analysis power calculations."""
    # Test with very small sample size
    power = fixed_effects_power(
        effect_size=0.5,
        n_studies=3,
        avg_sample_size=10,
        alpha=0.05
    )
    assert 0 < power < 1.0
    
    # Test with very large effect size
    power = random_effects_power(
        effect_size=2.0,
        n_studies=5,
        avg_sample_size=30,
        heterogeneity=0.3,
        alpha=0.05
    )
    assert power > 0.9
    
    # Test with very small effect size
    power = fixed_effects_power(
        effect_size=0.05,
        n_studies=10,
        avg_sample_size=100,
        alpha=0.05
    )
    assert power < 0.2
    
    # Test with high heterogeneity
    power = random_effects_power(
        effect_size=0.5,
        n_studies=5,
        avg_sample_size=50,
        heterogeneity=0.8,
        alpha=0.05
    )
    assert 0 < power < 1.0 