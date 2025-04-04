import pytest
import numpy as np
from statsiq_power.modules.timeseries_power import (
    arima_power,
    intervention_power
)

def test_arima_power():
    """Test ARIMA model power calculation."""
    # Test with medium effect size
    power = arima_power(
        effect_size=0.5,
        n_observations=100,
        ar_order=1,
        ma_order=1,
        alpha=0.05
    )
    assert 0.5 < power < 1.0
    
    # Test with small effect size
    power = arima_power(
        effect_size=0.2,
        n_observations=100,
        ar_order=1,
        ma_order=1,
        alpha=0.05
    )
    assert 0.1 < power < 0.5
    
    # Test with large effect size
    power = arima_power(
        effect_size=0.8,
        n_observations=100,
        ar_order=1,
        ma_order=1,
        alpha=0.05
    )
    assert 0.8 < power < 1.0
    
    # Test with more observations
    power1 = arima_power(
        effect_size=0.5,
        n_observations=50,
        ar_order=1,
        ma_order=1,
        alpha=0.05
    )
    power2 = arima_power(
        effect_size=0.5,
        n_observations=100,
        ar_order=1,
        ma_order=1,
        alpha=0.05
    )
    assert power1 < power2  # More observations should result in higher power
    
    # Test with higher model complexity
    power1 = arima_power(
        effect_size=0.5,
        n_observations=100,
        ar_order=1,
        ma_order=1,
        alpha=0.05
    )
    power2 = arima_power(
        effect_size=0.5,
        n_observations=100,
        ar_order=2,
        ma_order=2,
        alpha=0.05
    )
    assert power1 > power2  # Higher complexity should result in lower power

def test_intervention_power():
    """Test intervention analysis power calculation."""
    # Test with medium effect size
    power = intervention_power(
        effect_size=0.5,
        n_pre=50,
        n_post=50,
        ar_order=1,
        alpha=0.05
    )
    assert 0.5 < power < 1.0
    
    # Test with small effect size
    power = intervention_power(
        effect_size=0.2,
        n_pre=50,
        n_post=50,
        ar_order=1,
        alpha=0.05
    )
    assert 0.1 < power < 0.5
    
    # Test with large effect size
    power = intervention_power(
        effect_size=0.8,
        n_pre=50,
        n_post=50,
        ar_order=1,
        alpha=0.05
    )
    assert 0.8 < power < 1.0
    
    # Test with more post-intervention observations
    power1 = intervention_power(
        effect_size=0.5,
        n_pre=50,
        n_post=50,
        ar_order=1,
        alpha=0.05
    )
    power2 = intervention_power(
        effect_size=0.5,
        n_pre=50,
        n_post=100,
        ar_order=1,
        alpha=0.05
    )
    assert power1 < power2  # More post-intervention observations should increase power
    
    # Test with higher model complexity
    power1 = intervention_power(
        effect_size=0.5,
        n_pre=50,
        n_post=50,
        ar_order=1,
        alpha=0.05
    )
    power2 = intervention_power(
        effect_size=0.5,
        n_pre=50,
        n_post=50,
        ar_order=2,
        alpha=0.05
    )
    assert power1 > power2  # Higher complexity should result in lower power

def test_edge_cases():
    """Test edge cases for time series power calculations."""
    # Test with very small sample size
    power = arima_power(
        effect_size=0.5,
        n_observations=20,
        ar_order=1,
        ma_order=1,
        alpha=0.05
    )
    assert 0 < power < 1.0
    
    # Test with very large effect size
    power = intervention_power(
        effect_size=2.0,
        n_pre=50,
        n_post=50,
        ar_order=1,
        alpha=0.05
    )
    assert power > 0.9
    
    # Test with very small effect size
    power = arima_power(
        effect_size=0.05,
        n_observations=100,
        ar_order=1,
        ma_order=1,
        alpha=0.05
    )
    assert power < 0.2
    
    # Test with high model complexity
    power = arima_power(
        effect_size=0.5,
        n_observations=100,
        ar_order=3,
        ma_order=3,
        alpha=0.05
    )
    assert 0 < power < 1.0 