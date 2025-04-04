import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from ..modules.t_test_power import one_sample_t_test_power, independent_t_test_power, paired_t_test_power
from ..modules.anova_power import one_way_anova_power, factorial_anova_power, repeated_measures_anova_power
from ..modules.chi_square_power import chi_square_goodness_of_fit_power, chi_square_independence_power
from ..modules.correlation_power import pearson_correlation_power, spearman_correlation_power
from ..modules.regression_power import linear_regression_power, logistic_regression_power

def power_curve(effect_sizes, sample_sizes, test_type='t-test', alpha=0.05, **kwargs):
    """
    Create a power curve visualization.
    
    Parameters:
    -----------
    effect_sizes : array-like
        Array of effect sizes
    sample_sizes : array-like
        Array of sample sizes
    test_type : str, optional
        Type of statistical test
    alpha : float, optional
        Significance level
    **kwargs : dict
        Additional arguments for specific test types
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object containing the power curve
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create meshgrid for contour plot
    X, Y = np.meshgrid(sample_sizes, effect_sizes)
    Z = np.zeros_like(X)
    
    # Calculate power for each combination
    for i, effect_size in enumerate(effect_sizes):
        for j, n in enumerate(sample_sizes):
            if test_type == 't-test':
                Z[i, j] = independent_t_test_power(effect_size, n, alpha=alpha)
            elif test_type == 'anova':
                k = kwargs.get('k', 2)
                Z[i, j] = one_way_anova_power(effect_size, n, k, alpha)
            elif test_type == 'chi-square':
                df = kwargs.get('df', 1)
                Z[i, j] = chi_square_goodness_of_fit_power(np.array([1/df]*df), n, alpha)
            elif test_type == 'correlation':
                Z[i, j] = pearson_correlation_power(effect_size, n, alpha)
            elif test_type == 'regression':
                k = kwargs.get('k', 1)
                Z[i, j] = linear_regression_power(effect_size, n, k, alpha)
    
    # Create contour plot
    contour = ax.contourf(X, Y, Z, levels=np.linspace(0, 1, 11), cmap='RdYlBu_r')
    plt.colorbar(contour, ax=ax, label='Power')
    
    # Add labels and title
    ax.set_xlabel('Sample Size')
    ax.set_ylabel('Effect Size')
    ax.set_title(f'Power Curve for {test_type.replace("-", " ").title()} Test')
    
    return fig

def sample_size_curve(effect_sizes, powers, test_type='t-test', alpha=0.05, **kwargs):
    """
    Create a sample size curve visualization.
    
    Parameters:
    -----------
    effect_sizes : array-like
        Array of effect sizes
    powers : array-like
        Array of desired powers
    test_type : str, optional
        Type of statistical test
    alpha : float, optional
        Significance level
    **kwargs : dict
        Additional arguments for specific test types
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object containing the sample size curve
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculate sample sizes for each combination
    for power in powers:
        sample_sizes = []
        for effect_size in effect_sizes:
            if test_type == 't-test':
                n = independent_t_test_power(effect_size, 100, alpha=alpha)
                while n < 1000 and independent_t_test_power(effect_size, n, alpha=alpha) < power:
                    n += 1
            elif test_type == 'anova':
                k = kwargs.get('k', 2)
                n = one_way_anova_power(effect_size, 100, k, alpha)
                while n < 1000 and one_way_anova_power(effect_size, n, k, alpha) < power:
                    n += 1
            elif test_type == 'chi-square':
                df = kwargs.get('df', 1)
                n = chi_square_goodness_of_fit_power(np.array([1/df]*df), 100, alpha)
                while n < 1000 and chi_square_goodness_of_fit_power(np.array([1/df]*df), n, alpha) < power:
                    n += 1
            elif test_type == 'correlation':
                n = pearson_correlation_power(effect_size, 100, alpha)
                while n < 1000 and pearson_correlation_power(effect_size, n, alpha) < power:
                    n += 1
            elif test_type == 'regression':
                k = kwargs.get('k', 1)
                n = linear_regression_power(effect_size, 100, k, alpha)
                while n < 1000 and linear_regression_power(effect_size, n, k, alpha) < power:
                    n += 1
            sample_sizes.append(n)
        
        ax.plot(effect_sizes, sample_sizes, label=f'Power = {power:.2f}')
    
    # Add labels and title
    ax.set_xlabel('Effect Size')
    ax.set_ylabel('Required Sample Size')
    ax.set_title(f'Sample Size Curve for {test_type.replace("-", " ").title()} Test')
    ax.legend()
    
    return fig

def power_analysis_summary(effect_size, sample_size, test_type='t-test', alpha=0.05, **kwargs):
    """
    Create a power analysis summary visualization.
    
    Parameters:
    -----------
    effect_size : float
        Effect size
    sample_size : int
        Sample size
    test_type : str, optional
        Type of statistical test
    alpha : float, optional
        Significance level
    **kwargs : dict
        Additional arguments for specific test types
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object containing the power analysis summary
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Calculate power
    if test_type == 't-test':
        power = independent_t_test_power(effect_size, sample_size, alpha=alpha)
    elif test_type == 'anova':
        k = kwargs.get('k', 2)
        power = one_way_anova_power(effect_size, sample_size, k, alpha)
    elif test_type == 'chi-square':
        df = kwargs.get('df', 1)
        power = chi_square_goodness_of_fit_power(np.array([1/df]*df), sample_size, alpha)
    elif test_type == 'correlation':
        power = pearson_correlation_power(effect_size, sample_size, alpha)
    elif test_type == 'regression':
        k = kwargs.get('k', 1)
        power = linear_regression_power(effect_size, sample_size, k, alpha)
    
    # Create power gauge
    ax1.pie([power, 1-power], colors=['#2ecc71', '#e74c3c'], 
            labels=['Power', 'Type II Error'],
            autopct='%1.1f%%')
    ax1.set_title('Power Analysis')
    
    # Create effect size bar
    ax2.bar(['Effect Size'], [effect_size], color='#3498db')
    ax2.set_ylim(0, max(effect_size * 1.2, 1))
    ax2.set_title('Effect Size')
    
    # Add overall title
    fig.suptitle(f'Power Analysis Summary for {test_type.replace("-", " ").title()} Test')
    
    return fig 