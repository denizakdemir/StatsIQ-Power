import numpy as np
from scipy import stats
from ..utils.effect_size import cohens_f

def one_way_anova_power(effect_size, n_per_group, k, alpha=0.05):
    """
    Calculate power for a one-way ANOVA.
    
    Parameters
    ----------
    effect_size : float
        Cohen's f effect size
    n_per_group : int
        Sample size per group
    k : int
        Number of groups
    alpha : float, optional
        Significance level (default: 0.05)
    
    Returns
    -------
    float
        Statistical power (1 - beta)
    """
    # Calculate degrees of freedom
    df1 = k - 1  # between groups
    df2 = k * (n_per_group - 1)  # within groups
    
    # Calculate non-centrality parameter
    ncp = effect_size**2 * n_per_group * k
    
    # Calculate critical value
    critical_value = stats.f.ppf(1 - alpha, df1, df2)
    
    # Calculate power
    power = 1 - stats.f.cdf(critical_value, df1, df2, loc=ncp)
    
    return power

def factorial_anova_power(effect_sizes, n_per_cell, factors, alpha=0.05):
    """
    Calculate power for a factorial ANOVA.
    
    Parameters
    ----------
    effect_sizes : dict
        Dictionary of effect sizes (Cohen's f) for each factor and interaction
    n_per_cell : int
        Sample size per cell
    factors : list
        List of factor levels (e.g., [2, 3] for a 2x3 design)
    alpha : float, optional
        Significance level (default: 0.05)
    
    Returns
    -------
    dict
        Dictionary of power values for each effect
    """
    # Calculate total number of cells
    total_cells = np.prod(factors)
    
    # Calculate degrees of freedom for each effect
    df_effects = {}
    for i, factor in enumerate(factors):
        df_effects[f'factor_{i+1}'] = factor - 1
    
    # Calculate interaction degrees of freedom
    for i in range(len(factors)):
        for j in range(i+1, len(factors)):
            df_effects[f'interaction_{i+1}_{j+1}'] = (factors[i] - 1) * (factors[j] - 1)
    
    # Calculate error degrees of freedom
    df_error = total_cells * (n_per_cell - 1)
    
    # Calculate power for each effect
    power = {}
    for effect, effect_size in effect_sizes.items():
        df1 = df_effects[effect]
        ncp = effect_size**2 * n_per_cell * total_cells
        
        critical_value = stats.f.ppf(1 - alpha, df1, df_error)
        power[effect] = 1 - stats.f.cdf(critical_value, df1, df_error, loc=ncp)
    
    return power

def repeated_measures_anova_power(effect_size, n, k, alpha=0.05, epsilon=1.0):
    """
    Calculate power for a repeated measures ANOVA.
    
    Parameters
    ----------
    effect_size : float
        Cohen's f effect size
    n : int
        Number of subjects
    k : int
        Number of repeated measures
    alpha : float, optional
        Significance level (default: 0.05)
    epsilon : float, optional
        Greenhouse-Geisser epsilon correction (default: 1.0)
    
    Returns
    -------
    float
        Statistical power (1 - beta)
    """
    # Calculate degrees of freedom
    df1 = (k - 1) * epsilon  # between measures
    df2 = (n - 1) * (k - 1) * epsilon  # error
    
    # Calculate non-centrality parameter
    ncp = effect_size**2 * n * k
    
    # Calculate critical value
    critical_value = stats.f.ppf(1 - alpha, df1, df2)
    
    # Calculate power
    power = 1 - stats.f.cdf(critical_value, df1, df2, loc=ncp)
    
    return power

def mixed_anova_power(between_effect_size, within_effect_size, n_per_group, k, 
                     num_groups, alpha=0.05, epsilon=1.0):
    """
    Calculate power for a mixed-design ANOVA.
    
    Parameters
    ----------
    between_effect_size : float
        Cohen's f effect size for between-subjects factor
    within_effect_size : float
        Cohen's f effect size for within-subjects factor
    n_per_group : int
        Sample size per group
    k : int
        Number of repeated measures
    num_groups : int
        Number of groups in the between-subjects factor
    alpha : float, optional
        Significance level (default: 0.05)
    epsilon : float, optional
        Greenhouse-Geisser epsilon correction (default: 1.0)
    
    Returns
    -------
    dict
        Dictionary of power values for between-subjects, within-subjects,
        and interaction effects
    """
    # Calculate degrees of freedom
    df_between = num_groups - 1
    df_within = (k - 1) * epsilon
    df_interaction = (num_groups - 1) * (k - 1) * epsilon
    df_error_between = num_groups * (n_per_group - 1)
    df_error_within = df_error_between * (k - 1) * epsilon
    
    # Calculate non-centrality parameters
    ncp_between = between_effect_size**2 * n_per_group * num_groups
    ncp_within = within_effect_size**2 * n_per_group * num_groups * k
    ncp_interaction = (between_effect_size * within_effect_size)**2 * n_per_group * num_groups * k
    
    # Calculate power for each effect
    power = {}
    
    # Between-subjects effect
    critical_value = stats.f.ppf(1 - alpha, df_between, df_error_between)
    power['between'] = 1 - stats.f.cdf(critical_value, df_between, df_error_between, loc=ncp_between)
    
    # Within-subjects effect
    critical_value = stats.f.ppf(1 - alpha, df_within, df_error_within)
    power['within'] = 1 - stats.f.cdf(critical_value, df_within, df_error_within, loc=ncp_within)
    
    # Interaction effect
    critical_value = stats.f.ppf(1 - alpha, df_interaction, df_error_within)
    power['interaction'] = 1 - stats.f.cdf(critical_value, df_interaction, df_error_within, loc=ncp_interaction)
    
    return power

def sample_size_for_anova(effect_size, power=0.8, k=2, alpha=0.05, design='one-way'):
    """
    Calculate required sample size for ANOVA to achieve desired power.
    
    Parameters
    ----------
    effect_size : float
        Cohen's f effect size
    power : float, optional
        Desired statistical power (default: 0.8)
    k : int, optional
        Number of groups/levels (default: 2)
    alpha : float, optional
        Significance level (default: 0.05)
    design : str, optional
        ANOVA design: 'one-way', 'factorial', 'repeated', or 'mixed' (default: 'one-way')
    
    Returns
    -------
    int
        Required sample size per group/cell
    """
    def power_at_n(n):
        if design == 'one-way':
            return one_way_anova_power(effect_size, n, k, alpha)
        elif design == 'repeated':
            return repeated_measures_anova_power(effect_size, n, k, alpha)
        else:
            raise ValueError("Sample size calculation for this design not implemented yet")
    
    # Binary search for sample size
    n_min, n_max = 2, 1000
    while power_at_n(n_max) < power:
        n_max *= 2
    
    while n_max - n_min > 1:
        n_mid = (n_min + n_max) // 2
        if power_at_n(n_mid) < power:
            n_min = n_mid
        else:
            n_max = n_mid
    
    return n_max 