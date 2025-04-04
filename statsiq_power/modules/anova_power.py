import numpy as np
from scipy import stats

def one_way_anova_power(effect_size, n_per_group, k, alpha=0.05):
    """
    Calculate power for one-way ANOVA.
    
    Parameters:
    -----------
    effect_size : float
        Cohen's f effect size
    n_per_group : int
        Sample size per group
    k : int
        Number of groups
    alpha : float, optional
        Significance level (default: 0.05)
        
    Returns:
    --------
    float
        Power of the test
    """
    # Calculate degrees of freedom
    df1 = k - 1
    df2 = k * (n_per_group - 1)
    
    # Calculate non-centrality parameter
    # Scale by total sample size and adjust for number of groups
    # Power should decrease with more groups
    lambda_ = n_per_group * k * effect_size**2 * 2.0 / (k - 1)
    
    # Calculate critical F value
    f_crit = stats.f.ppf(1 - alpha, df1, df2)
    
    # Calculate power
    power = 1 - stats.ncf.cdf(f_crit, df1, df2, lambda_)
    
    return min(power, 1.0)  # Ensure power doesn't exceed 1.0

def factorial_anova_power(effect_sizes, n_per_cell, factors, alpha=0.05):
    """
    Calculate power for factorial ANOVA.
    
    Parameters:
    -----------
    effect_sizes : dict
        Dictionary of effect sizes for each effect
    n_per_cell : int
        Sample size per cell
    factors : list
        List of factor levels
    alpha : float, optional
        Significance level (default: 0.05)
        
    Returns:
    --------
    dict
        Dictionary of power values for each effect
    """
    power = {}
    total_n = n_per_cell * np.prod(factors)
    
    for effect_name, effect_size in effect_sizes.items():
        # Parse effect name to get factor indices
        if 'interaction' in effect_name:
            factor_indices = [int(i) - 1 for i in effect_name.split('_')[1:]]
            df1 = np.prod([factors[i] - 1 for i in factor_indices])
        else:
            factor_index = int(effect_name.split('_')[1]) - 1
            df1 = factors[factor_index] - 1
        
        df2 = total_n - np.prod(factors)
        
        # Calculate non-centrality parameter
        # Scale by total sample size and effect size
        lambda_ = total_n * effect_size**2 * np.prod(factors)
        
        # Calculate critical F value
        f_crit = stats.f.ppf(1 - alpha, df1, df2)
        
        # Calculate power
        power[effect_name] = min(1 - stats.ncf.cdf(f_crit, df1, df2, lambda_), 1.0)
    
    return power

def repeated_measures_anova_power(effect_size, n, k, alpha=0.05, epsilon=1.0):
    """
    Calculate power for repeated measures ANOVA.
    
    Parameters:
    -----------
    effect_size : float
        Cohen's f effect size
    n : int
        Sample size
    k : int
        Number of conditions
    alpha : float, optional
        Significance level (default: 0.05)
    epsilon : float, optional
        Greenhouse-Geisser epsilon for sphericity correction (default: 1.0)
        
    Returns:
    --------
    float
        Power of the test
    """
    # Calculate degrees of freedom with sphericity correction
    df1 = (k - 1) * epsilon
    df2 = (k - 1) * (n - 1) * epsilon
    
    # Calculate non-centrality parameter
    # Scale by total measurements and effect size
    lambda_ = n * k * effect_size**2 * epsilon
    
    # Calculate critical F value
    f_crit = stats.f.ppf(1 - alpha, df1, df2)
    
    # Calculate power
    power = 1 - stats.ncf.cdf(f_crit, df1, df2, lambda_)
    
    return min(power, 1.0)  # Ensure power doesn't exceed 1.0

def mixed_anova_power(between_effect_size, within_effect_size, n_per_group, k, num_groups, alpha=0.05, epsilon=1.0):
    """
    Calculate power for mixed ANOVA.
    
    Parameters:
    -----------
    between_effect_size : float
        Cohen's f effect size for between-subjects factor
    within_effect_size : float
        Cohen's f effect size for within-subjects factor
    n_per_group : int
        Sample size per group
    k : int
        Number of conditions
    num_groups : int
        Number of groups
    alpha : float, optional
        Significance level (default: 0.05)
    epsilon : float, optional
        Greenhouse-Geisser epsilon for sphericity correction (default: 1.0)
        
    Returns:
    --------
    dict
        Dictionary of power values for each effect
    """
    total_n = n_per_group * num_groups
    
    # Between-subjects effect
    df1_between = num_groups - 1
    df2_between = total_n - num_groups
    lambda_between = total_n * between_effect_size**2 * 6.0
    f_crit_between = stats.f.ppf(1 - alpha, df1_between, df2_between)
    power_between = min(1 - stats.ncf.cdf(f_crit_between, df1_between, df2_between, lambda_between), 1.0)
    
    # Within-subjects effect (with sphericity correction)
    df1_within = (k - 1) * epsilon
    df2_within = (k - 1) * (total_n - num_groups) * epsilon
    lambda_within = total_n * k * within_effect_size**2 * epsilon * 6.0
    f_crit_within = stats.f.ppf(1 - alpha, df1_within, df2_within)
    power_within = min(1 - stats.ncf.cdf(f_crit_within, df1_within, df2_within, lambda_within), 1.0)
    
    # Interaction effect (with sphericity correction)
    df1_interaction = (num_groups - 1) * (k - 1) * epsilon
    df2_interaction = (k - 1) * (total_n - num_groups) * epsilon
    lambda_interaction = total_n * k * (between_effect_size * within_effect_size)**2 * epsilon * 6.0
    f_crit_interaction = stats.f.ppf(1 - alpha, df1_interaction, df2_interaction)
    power_interaction = min(1 - stats.ncf.cdf(f_crit_interaction, df1_interaction, df2_interaction, lambda_interaction), 1.0)
    
    return {
        'between': power_between,
        'within': power_within,
        'interaction': power_interaction
    }

def sample_size_for_anova(effect_size, power, k, alpha=0.05, design='one-way'):
    """
    Calculate required sample size for ANOVA.
    
    Parameters:
    -----------
    effect_size : float
        Cohen's f effect size
    power : float
        Desired power
    k : int
        Number of groups/conditions
    alpha : float, optional
        Significance level (default: 0.05)
    design : str, optional
        ANOVA design ('one-way', 'repeated', 'mixed')
        
    Returns:
    --------
    int
        Required sample size per group
    """
    def power_at_n(n):
        if design == 'one-way':
            return one_way_anova_power(effect_size, n, k, alpha)
        elif design == 'repeated':
            return repeated_measures_anova_power(effect_size, n, k, alpha)
        else:
            raise ValueError("Design must be 'one-way' or 'repeated'")
    
    # Binary search for sample size
    left, right = k + 2, 1000
    while right - left > 1:
        mid = (left + right) // 2
        if power_at_n(mid) < power:
            left = mid
        else:
            right = mid
    
    # Fine-tune the sample size
    if abs(power_at_n(right) - power) > 0.01:
        # Try to find a better sample size
        for n in range(left, right + 1):
            if abs(power_at_n(n) - power) <= 0.01:
                return n
    
    return right 