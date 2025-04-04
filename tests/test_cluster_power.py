import pytest
import numpy as np
from statsiq_power.modules.cluster_power import (
    cluster_power,
    cluster_continuous_power,
    cluster_binary_power
)

def test_cluster_continuous_power():
    """Test power calculation for cluster randomized trials with continuous outcome."""
    # Test with medium effect size (d = 0.5)
    power = cluster_continuous_power(
        effect_size=0.5,
        n_clusters=20,
        cluster_size=30,
        icc=0.05,  # Intraclass correlation coefficient
        alpha=0.05
    )
    assert 0.5 < power < 1.0
    
    # Test with small effect size (d = 0.2)
    power = cluster_continuous_power(
        effect_size=0.2,
        n_clusters=20,
        cluster_size=30,
        icc=0.05,
        alpha=0.05
    )
    assert 0.1 < power < 0.5
    
    # Test with large effect size (d = 0.8)
    power = cluster_continuous_power(
        effect_size=0.8,
        n_clusters=20,
        cluster_size=30,
        icc=0.05,
        alpha=0.05
    )
    assert power > 0.9

def test_cluster_binary_power():
    """Test power calculation for cluster randomized trials with binary outcome."""
    # Test with medium effect (risk ratio = 1.5)
    power = cluster_binary_power(
        p1=0.3,  # Proportion in control group
        p2=0.45,  # Proportion in intervention group
        n_clusters=20,
        cluster_size=30,
        icc=0.05,
        alpha=0.05
    )
    assert 0.5 < power < 1.0
    
    # Test with small effect (risk ratio = 1.2)
    power = cluster_binary_power(
        p1=0.3,
        p2=0.36,
        n_clusters=20,
        cluster_size=30,
        icc=0.05,
        alpha=0.05
    )
    assert 0.1 < power < 0.5
    
    # Test with large effect (risk ratio = 2.0)
    power = cluster_binary_power(
        p1=0.3,
        p2=0.6,
        n_clusters=20,
        cluster_size=30,
        icc=0.05,
        alpha=0.05
    )
    assert power > 0.9

def test_icc_effect():
    """Test that power decreases with increasing ICC."""
    power1 = cluster_continuous_power(
        effect_size=0.5,
        n_clusters=20,
        cluster_size=30,
        icc=0.05,
        alpha=0.05
    )
    power2 = cluster_continuous_power(
        effect_size=0.5,
        n_clusters=20,
        cluster_size=30,
        icc=0.20,
        alpha=0.05
    )
    assert power1 > power2

def test_cluster_size_effect():
    """Test the effect of cluster size on power."""
    # Test that power increases with cluster size (when ICC is low)
    power1 = cluster_continuous_power(
        effect_size=0.5,
        n_clusters=20,
        cluster_size=20,
        icc=0.05,
        alpha=0.05
    )
    power2 = cluster_continuous_power(
        effect_size=0.5,
        n_clusters=20,
        cluster_size=40,
        icc=0.05,
        alpha=0.05
    )
    assert power1 < power2
    
    # Test that increasing cluster size has diminishing returns with high ICC
    power_diff_low_icc = (
        cluster_continuous_power(
            effect_size=0.5,
            n_clusters=20,
            cluster_size=40,
            icc=0.05,
            alpha=0.05
        ) -
        cluster_continuous_power(
            effect_size=0.5,
            n_clusters=20,
            cluster_size=20,
            icc=0.05,
            alpha=0.05
        )
    )
    
    power_diff_high_icc = (
        cluster_continuous_power(
            effect_size=0.5,
            n_clusters=20,
            cluster_size=40,
            icc=0.20,
            alpha=0.05
        ) -
        cluster_continuous_power(
            effect_size=0.5,
            n_clusters=20,
            cluster_size=20,
            icc=0.20,
            alpha=0.05
        )
    )
    
    assert power_diff_low_icc > power_diff_high_icc

def test_n_clusters_effect():
    """Test that power increases with number of clusters."""
    power1 = cluster_continuous_power(
        effect_size=0.5,
        n_clusters=15,
        cluster_size=30,
        icc=0.05,
        alpha=0.05
    )
    power2 = cluster_continuous_power(
        effect_size=0.5,
        n_clusters=30,
        cluster_size=30,
        icc=0.05,
        alpha=0.05
    )
    assert power1 < power2

def test_edge_cases():
    """Test edge cases for cluster randomized trial power calculations."""
    # Test with very small ICC
    power = cluster_continuous_power(
        effect_size=0.5,
        n_clusters=20,
        cluster_size=30,
        icc=0.01,
        alpha=0.05
    )
    assert 0 < power < 1.0
    
    # Test with very high ICC
    power = cluster_continuous_power(
        effect_size=0.5,
        n_clusters=20,
        cluster_size=30,
        icc=0.5,
        alpha=0.05
    )
    assert 0 < power < 1.0
    
    # Test with unequal cluster sizes
    power = cluster_power(
        effect_size=0.5,
        n_clusters_1=15,
        n_clusters_2=25,
        cluster_size_1=30,
        cluster_size_2=20,
        icc=0.05,
        outcome_type="continuous",
        alpha=0.05
    )
    assert 0 < power < 1.0 