import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def power_curve(effect_sizes, sample_sizes, alpha=0.05, test_type='t-test', **kwargs):
    """
    Create a power curve showing the relationship between effect size and power
    for different sample sizes.
    
    Parameters
    ----------
    effect_sizes : array-like
        Array of effect sizes to plot
    sample_sizes : array-like
        Array of sample sizes to plot
    alpha : float, optional
        Significance level (default: 0.05)
    test_type : str, optional
        Type of statistical test (default: 't-test')
    **kwargs : dict
        Additional arguments passed to the power calculation function
    
    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the power curve
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculate power for each combination of effect size and sample size
    power_matrix = np.zeros((len(effect_sizes), len(sample_sizes)))
    for i, effect_size in enumerate(effect_sizes):
        for j, n in enumerate(sample_sizes):
            if test_type == 't-test':
                from ..modules.t_test_power import one_sample_t_test_power
                power_matrix[i, j] = one_sample_t_test_power(effect_size, n, alpha, **kwargs)
            elif test_type == 'anova':
                from ..modules.anova_power import one_way_anova_power
                power_matrix[i, j] = one_way_anova_power(effect_size, n, kwargs.get('k', 2), alpha)
            elif test_type == 'correlation':
                from ..modules.correlation_power import pearson_correlation_power
                power_matrix[i, j] = pearson_correlation_power(effect_size, n, alpha, **kwargs)
            elif test_type == 'chi-square':
                from ..modules.chi_square_power import chi_square_goodness_of_fit_power
                power_matrix[i, j] = chi_square_goodness_of_fit_power(
                    np.array([1/len(effect_sizes)] * len(effect_sizes)), n, alpha)
    
    # Create contour plot
    contour = ax.contourf(sample_sizes, effect_sizes, power_matrix, 
                         levels=np.linspace(0, 1, 11), cmap='RdYlBu_r')
    plt.colorbar(contour, ax=ax, label='Power')
    
    # Add contour lines
    cs = ax.contour(sample_sizes, effect_sizes, power_matrix, 
                   levels=[0.5, 0.8, 0.9], colors='black', linestyles='--')
    ax.clabel(cs, inline=True, fmt='%.1f')
    
    ax.set_xlabel('Sample Size')
    ax.set_ylabel('Effect Size')
    ax.set_title(f'Power Curve for {test_type.replace("-", " ").title()}')
    
    return fig

def sample_size_curve(effect_sizes, powers, alpha=0.05, test_type='t-test', **kwargs):
    """
    Create a curve showing the relationship between effect size and required sample size
    for different power levels.
    
    Parameters
    ----------
    effect_sizes : array-like
        Array of effect sizes to plot
    powers : array-like
        Array of power levels to plot
    alpha : float, optional
        Significance level (default: 0.05)
    test_type : str, optional
        Type of statistical test (default: 't-test')
    **kwargs : dict
        Additional arguments passed to the sample size calculation function
    
    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the sample size curve
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculate required sample size for each combination of effect size and power
    for power in powers:
        sample_sizes = []
        for effect_size in effect_sizes:
            if test_type == 't-test':
                from ..modules.t_test_power import sample_size_for_t_test
                n = sample_size_for_t_test(effect_size, power, alpha, **kwargs)
            elif test_type == 'anova':
                from ..modules.anova_power import sample_size_for_anova
                n = sample_size_for_anova(effect_size, power, kwargs.get('k', 2), alpha)
            elif test_type == 'correlation':
                from ..modules.correlation_power import sample_size_for_correlation
                n = sample_size_for_correlation(effect_size, power, alpha, **kwargs)
            elif test_type == 'chi-square':
                from ..modules.chi_square_power import sample_size_for_chi_square
                n = sample_size_for_chi_square(effect_size, power, kwargs.get('df', 1), alpha)
            sample_sizes.append(n)
        
        ax.plot(effect_sizes, sample_sizes, label=f'Power = {power:.1f}')
    
    ax.set_xlabel('Effect Size')
    ax.set_ylabel('Required Sample Size')
    ax.set_title(f'Sample Size Curve for {test_type.replace("-", " ").title()}')
    ax.legend()
    ax.grid(True)
    
    return fig

def power_analysis_summary(effect_size, sample_size, alpha=0.05, test_type='t-test', **kwargs):
    """
    Create a summary plot showing power analysis results.
    
    Parameters
    ----------
    effect_size : float
        Effect size
    sample_size : int
        Sample size
    alpha : float, optional
        Significance level (default: 0.05)
    test_type : str, optional
        Type of statistical test (default: 't-test')
    **kwargs : dict
        Additional arguments passed to the power calculation function
    
    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the power analysis summary
    """
    fig = plt.figure(figsize=(12, 8))
    gs = GridSpec(2, 2, figure=fig)
    
    # Calculate power
    if test_type == 't-test':
        from ..modules.t_test_power import one_sample_t_test_power
        power = one_sample_t_test_power(effect_size, sample_size, alpha, **kwargs)
    elif test_type == 'anova':
        from ..modules.anova_power import one_way_anova_power
        power = one_way_anova_power(effect_size, sample_size, kwargs.get('k', 2), alpha)
    elif test_type == 'correlation':
        from ..modules.correlation_power import pearson_correlation_power
        power = pearson_correlation_power(effect_size, sample_size, alpha, **kwargs)
    elif test_type == 'chi-square':
        from ..modules.chi_square_power import chi_square_goodness_of_fit_power
        power = chi_square_goodness_of_fit_power(
            np.array([1/len(effect_size)] * len(effect_size)), sample_size, alpha)
    
    # Power gauge
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.pie([power, 1-power], colors=['green', 'lightgray'],
            labels=['Power', ''], autopct='%1.1f%%')
    ax1.set_title('Statistical Power')
    
    # Effect size visualization
    ax2 = fig.add_subplot(gs[0, 1])
    if test_type in ['t-test', 'anova']:
        # Normal distribution plot
        x = np.linspace(-4, 4, 1000)
        y1 = stats.norm.pdf(x, 0, 1)
        y2 = stats.norm.pdf(x, effect_size, 1)
        ax2.plot(x, y1, label='Null')
        ax2.plot(x, y2, label='Alternative')
        ax2.fill_between(x, y1, alpha=0.3)
        ax2.fill_between(x, y2, alpha=0.3)
        ax2.set_title('Effect Size Distribution')
        ax2.legend()
    elif test_type == 'correlation':
        # Scatter plot example
        x = np.linspace(-1, 1, 100)
        y = effect_size * x + np.random.normal(0, 0.1, 100)
        ax2.scatter(x, y, alpha=0.5)
        ax2.set_title('Correlation Example')
    elif test_type == 'chi-square':
        # Bar plot example
        categories = ['A', 'B', 'C', 'D']
        expected = np.ones(4) / 4
        observed = expected + effect_size * np.random.normal(0, 0.1, 4)
        ax2.bar(categories, observed)
        ax2.set_title('Chi-square Example')
    
    # Sample size vs. power curve
    ax3 = fig.add_subplot(gs[1, :])
    n_range = np.linspace(max(2, sample_size//2), sample_size*2, 100)
    powers = []
    for n in n_range:
        if test_type == 't-test':
            powers.append(one_sample_t_test_power(effect_size, n, alpha, **kwargs))
        elif test_type == 'anova':
            powers.append(one_way_anova_power(effect_size, n, kwargs.get('k', 2), alpha))
        elif test_type == 'correlation':
            powers.append(pearson_correlation_power(effect_size, n, alpha, **kwargs))
        elif test_type == 'chi-square':
            powers.append(chi_square_goodness_of_fit_power(
                np.array([1/len(effect_size)] * len(effect_size)), n, alpha))
    
    ax3.plot(n_range, powers)
    ax3.axvline(x=sample_size, color='r', linestyle='--', label='Current Sample Size')
    ax3.axhline(y=power, color='g', linestyle='--', label='Current Power')
    ax3.set_xlabel('Sample Size')
    ax3.set_ylabel('Power')
    ax3.set_title('Sample Size vs. Power')
    ax3.legend()
    ax3.grid(True)
    
    plt.tight_layout()
    return fig 