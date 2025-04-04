import numpy as np
import matplotlib.pyplot as plt
from statsiq_power.utils.visualization import (
    power_curve,
    sample_size_curve,
    power_analysis_summary,
)

def test_power_curve():
    # Test with t-test
    effect_sizes = np.linspace(0.1, 1.0, 5)
    sample_sizes = np.linspace(10, 100, 5)
    fig = power_curve(effect_sizes, sample_sizes, test_type='t-test')
    assert isinstance(fig, plt.Figure)
    plt.close(fig)
    
    # Test with ANOVA
    fig = power_curve(effect_sizes, sample_sizes, test_type='anova', k=3)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)
    
    # Test with correlation
    fig = power_curve(effect_sizes, sample_sizes, test_type='correlation')
    assert isinstance(fig, plt.Figure)
    plt.close(fig)
    
    # Test with chi-square
    fig = power_curve(effect_sizes, sample_sizes, test_type='chi-square', df=1)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

def test_sample_size_curve():
    # Test with t-test
    effect_sizes = np.linspace(0.1, 1.0, 5)
    powers = [0.7, 0.8, 0.9]
    fig = sample_size_curve(effect_sizes, powers, test_type='t-test')
    assert isinstance(fig, plt.Figure)
    plt.close(fig)
    
    # Test with ANOVA
    fig = sample_size_curve(effect_sizes, powers, test_type='anova', k=3)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)
    
    # Test with correlation
    fig = sample_size_curve(effect_sizes, powers, test_type='correlation')
    assert isinstance(fig, plt.Figure)
    plt.close(fig)
    
    # Test with chi-square
    fig = sample_size_curve(effect_sizes, powers, test_type='chi-square', df=1)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

def test_power_analysis_summary():
    # Test with t-test
    fig = power_analysis_summary(effect_size=0.5, sample_size=30, test_type='t-test')
    assert isinstance(fig, plt.Figure)
    plt.close(fig)
    
    # Test with ANOVA
    fig = power_analysis_summary(effect_size=0.3, sample_size=30, test_type='anova', k=3)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)
    
    # Test with correlation
    fig = power_analysis_summary(effect_size=0.3, sample_size=30, test_type='correlation')
    assert isinstance(fig, plt.Figure)
    plt.close(fig)
    
    # Test with chi-square
    fig = power_analysis_summary(effect_size=0.3, sample_size=30, test_type='chi-square', df=1)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

def test_edge_cases():
    # Test with very small effect size
    effect_sizes = np.linspace(0.01, 0.1, 5)
    sample_sizes = np.linspace(10, 100, 5)
    fig = power_curve(effect_sizes, sample_sizes, test_type='t-test')
    assert isinstance(fig, plt.Figure)
    plt.close(fig)
    
    # Test with very large effect size
    effect_sizes = np.linspace(0.8, 2.0, 5)
    fig = power_curve(effect_sizes, sample_sizes, test_type='t-test')
    assert isinstance(fig, plt.Figure)
    plt.close(fig)
    
    # Test with very small sample size
    effect_sizes = np.linspace(0.1, 1.0, 5)
    sample_sizes = np.linspace(5, 20, 5)
    fig = power_curve(effect_sizes, sample_sizes, test_type='t-test')
    assert isinstance(fig, plt.Figure)
    plt.close(fig)
    
    # Test with very large sample size
    sample_sizes = np.linspace(100, 1000, 5)
    fig = power_curve(effect_sizes, sample_sizes, test_type='t-test')
    assert isinstance(fig, plt.Figure)
    plt.close(fig) 