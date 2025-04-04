import pytest
import numpy as np
from statsiq_power.modules.survival_power import (
    log_rank_power,
    cox_power
)

def test_log_rank_power():
    """Test log-rank test power calculation."""
    # Test with medium hazard ratio
    power = log_rank_power(
        hazard_ratio=1.5,
        n_per_group=50,
        followup_time=12,
        alpha=0.05
    )
    assert 0.5 < power < 1.0
    
    # Test with small hazard ratio
    power = log_rank_power(
        hazard_ratio=1.2,
        n_per_group=50,
        followup_time=12,
        alpha=0.05
    )
    assert 0.1 < power < 0.5
    
    # Test with large hazard ratio
    power = log_rank_power(
        hazard_ratio=2.0,
        n_per_group=50,
        followup_time=12,
        alpha=0.05
    )
    assert 0.8 < power < 1.0
    
    # Test with longer follow-up time
    power1 = log_rank_power(
        hazard_ratio=1.5,
        n_per_group=50,
        followup_time=12,
        alpha=0.05
    )
    power2 = log_rank_power(
        hazard_ratio=1.5,
        n_per_group=50,
        followup_time=24,
        alpha=0.05
    )
    assert power1 < power2  # Longer follow-up should result in higher power
    
    # Test with different alpha
    power1 = log_rank_power(
        hazard_ratio=1.5,
        n_per_group=50,
        followup_time=12,
        alpha=0.01
    )
    power2 = log_rank_power(
        hazard_ratio=1.5,
        n_per_group=50,
        followup_time=12,
        alpha=0.05
    )
    assert power1 < power2  # Lower alpha should result in lower power

def test_cox_power():
    """Test Cox proportional hazards power calculation."""
    # Test with medium hazard ratio
    power = cox_power(
        hazard_ratio=1.5,
        n_per_group=50,
        followup_time=12,
        alpha=0.05
    )
    assert 0.5 < power < 1.0
    
    # Test with small hazard ratio
    power = cox_power(
        hazard_ratio=1.2,
        n_per_group=50,
        followup_time=12,
        alpha=0.05
    )
    assert 0.1 < power < 0.5
    
    # Test with large hazard ratio
    power = cox_power(
        hazard_ratio=2.0,
        n_per_group=50,
        followup_time=12,
        alpha=0.05
    )
    assert 0.8 < power < 1.0
    
    # Test with covariates
    power1 = cox_power(
        hazard_ratio=1.5,
        n_per_group=50,
        followup_time=12,
        alpha=0.05
    )
    power2 = cox_power(
        hazard_ratio=1.5,
        n_per_group=50,
        followup_time=12,
        n_covariates=3,
        alpha=0.05
    )
    assert power1 > power2  # More covariates should result in lower power
    
    # Test with different alpha
    power1 = cox_power(
        hazard_ratio=1.5,
        n_per_group=50,
        followup_time=12,
        alpha=0.01
    )
    power2 = cox_power(
        hazard_ratio=1.5,
        n_per_group=50,
        followup_time=12,
        alpha=0.05
    )
    assert power1 < power2  # Lower alpha should result in lower power

def test_edge_cases():
    """Test edge cases for survival power calculations."""
    # Test with very small sample sizes
    power = log_rank_power(
        hazard_ratio=1.5,
        n_per_group=10,
        followup_time=12,
        alpha=0.05
    )
    assert 0 < power < 1.0
    
    # Test with very large hazard ratio
    power = cox_power(
        hazard_ratio=5.0,
        n_per_group=50,
        followup_time=12,
        alpha=0.05
    )
    assert power > 0.9
    
    # Test with very small hazard ratio
    power = log_rank_power(
        hazard_ratio=1.05,
        n_per_group=50,
        followup_time=12,
        alpha=0.05
    )
    assert power < 0.2
    
    # Test with many covariates
    power = cox_power(
        hazard_ratio=1.5,
        n_per_group=50,
        followup_time=12,
        n_covariates=10,
        alpha=0.05
    )
    assert 0 < power < 1.0 