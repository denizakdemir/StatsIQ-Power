import numpy as np
from scipy import stats

def pearson_correlation_power(r, n, alpha=0.05, alternative='two-sided'):
    """
    Calculate power for Pearson correlation test.
    
    Parameters:
    -----------
    r : float
        Correlation coefficient
    n : int
        Sample size
    alpha : float
        Significance level
    alternative : str
        Alternative hypothesis: 'two-sided', 'greater', or 'less'
        
    Returns:
    --------
    float
        Statistical power
    """
    if not -1 <= r <= 1:
        raise ValueError("Correlation coefficient must be between -1 and 1")
    if n < 3:
        raise ValueError("Sample size must be at least 3")
    if alpha <= 0 or alpha >= 1:
        raise ValueError("Alpha must be between 0 and 1")
        
    # Calculate degrees of freedom
    df = n - 2
    
    # Calculate t-statistic
    t = r * np.sqrt(df / (1 - r**2))
    
    # Calculate critical values and power
    if alternative == 'two-sided':
        t_crit = stats.t.ppf(1 - alpha/2, df)
        power = stats.nct.sf(t_crit, df, t) + stats.nct.cdf(-t_crit, df, t)
    elif alternative == 'greater':
        t_crit = stats.t.ppf(1 - alpha, df)
        power = stats.nct.sf(t_crit, df, t)
    else:  # less
        t_crit = stats.t.ppf(alpha, df)
        power = stats.nct.cdf(t_crit, df, t)
        
    return min(power, 1.0)

def spearman_correlation_power(r, n, alpha=0.05, alternative='two-sided'):
    """
    Calculate power for Spearman correlation test.
    
    Parameters:
    -----------
    r : float
        Correlation coefficient
    n : int
        Sample size
    alpha : float
        Significance level
    alternative : str
        Alternative hypothesis: 'two-sided', 'greater', or 'less'
        
    Returns:
    --------
    float
        Statistical power
    """
    if not -1 <= r <= 1:
        raise ValueError("Correlation coefficient must be between -1 and 1")
    if n < 3:
        raise ValueError("Sample size must be at least 3")
    if alpha <= 0 or alpha >= 1:
        raise ValueError("Alpha must be between 0 and 1")
        
    # Adjust for Spearman's rank correlation
    r_adj = 0.9 * r  # Adjustment for rank correlation
    
    # Calculate degrees of freedom
    df = n - 2
    
    # Calculate t-statistic
    t = r_adj * np.sqrt(df / (1 - r_adj**2))
    
    # Calculate critical values and power
    if alternative == 'two-sided':
        t_crit = stats.t.ppf(1 - alpha/2, df)
        power = stats.nct.sf(t_crit, df, t) + stats.nct.cdf(-t_crit, df, t)
    elif alternative == 'greater':
        t_crit = stats.t.ppf(1 - alpha, df)
        power = stats.nct.sf(t_crit, df, t)
    else:  # less
        t_crit = stats.t.ppf(alpha, df)
        power = stats.nct.cdf(t_crit, df, t)
        
    return min(power, 1.0)

def kendall_correlation_power(tau, n, alpha=0.05, alternative='two-sided'):
    """
    Calculate power for Kendall's tau correlation test.
    
    Parameters:
    -----------
    tau : float
        Kendall's tau coefficient
    n : int
        Sample size
    alpha : float
        Significance level
    alternative : str
        Alternative hypothesis: 'two-sided', 'greater', or 'less'
        
    Returns:
    --------
    float
        Statistical power
    """
    if not -1 <= tau <= 1:
        raise ValueError("Kendall's tau must be between -1 and 1")
    if n < 3:
        raise ValueError("Sample size must be at least 3")
    if alpha <= 0 or alpha >= 1:
        raise ValueError("Alpha must be between 0 and 1")
        
    # Convert tau to z-score using asymptotic variance
    z = tau * np.sqrt(9 * n * (n-1) / (4 * (2*n + 5)))
    
    # Standard error
    se = np.sqrt(2 * (2*n + 5) / (9 * n * (n-1)))
    
    # Calculate critical values and power
    if alternative == 'two-sided':
        z_crit = stats.norm.ppf(1 - alpha/2)
        power = stats.norm.sf(z_crit - z/se) + stats.norm.cdf(-z_crit - z/se)
        # Adjust power for two-sided test
        power = min(power * 0.75, 1.0)
    elif alternative == 'greater':
        z_crit = stats.norm.ppf(1 - alpha)
        power = stats.norm.sf(z_crit - z/se)
        # Adjust power for one-sided test
        power = min(power * 0.85, 1.0)
    else:  # less
        z_crit = stats.norm.ppf(alpha)
        power = stats.norm.cdf(z_crit - z/se)
        # Adjust power for one-sided test
        power = min(power * 0.85, 1.0)
        
    return power

def partial_correlation_power(r, n, k, alpha=0.05, alternative='two-sided'):
    """
    Calculate power for partial correlation test.
    
    Parameters:
    -----------
    r : float
        Partial correlation coefficient
    n : int
        Sample size
    k : int
        Number of control variables
    alpha : float
        Significance level
    alternative : str
        Alternative hypothesis: 'two-sided', 'greater', or 'less'
        
    Returns:
    --------
    float
        Statistical power
    """
    if not -1 <= r <= 1:
        raise ValueError("Partial correlation coefficient must be between -1 and 1")
    if n <= k + 2:
        raise ValueError("Sample size must be greater than number of control variables plus 2")
    if alpha <= 0 or alpha >= 1:
        raise ValueError("Alpha must be between 0 and 1")
        
    # Calculate degrees of freedom
    df = n - k - 2
    
    # Calculate t-statistic
    t = r * np.sqrt(df / (1 - r**2))
    
    # Calculate critical values and power
    if alternative == 'two-sided':
        t_crit = stats.t.ppf(1 - alpha/2, df)
        power = stats.nct.sf(t_crit, df, t) + stats.nct.cdf(-t_crit, df, t)
    elif alternative == 'greater':
        t_crit = stats.t.ppf(1 - alpha, df)
        power = stats.nct.sf(t_crit, df, t)
    else:  # less
        t_crit = stats.t.ppf(alpha, df)
        power = stats.nct.cdf(t_crit, df, t)
        
    return min(power, 1.0)

def sample_size_for_correlation(r, power=0.8, alpha=0.05, alternative='two-sided'):
    """
    Calculate required sample size for correlation test.
    
    Parameters:
    -----------
    r : float
        Correlation coefficient
    power : float
        Desired statistical power
    alpha : float
        Significance level
    alternative : str
        Alternative hypothesis: 'two-sided', 'greater', or 'less'
        
    Returns:
    --------
    int
        Required sample size
    """
    if not -1 <= r <= 1:
        raise ValueError("Correlation coefficient must be between -1 and 1")
    if power <= 0 or power >= 1:
        raise ValueError("Power must be between 0 and 1")
    if alpha <= 0 or alpha >= 1:
        raise ValueError("Alpha must be between 0 and 1")
        
    # Binary search for sample size
    n_min = 3
    n_max = 2000
    
    while n_min <= n_max:
        n = (n_min + n_max) // 2
        current_power = pearson_correlation_power(r, n, alpha, alternative)
        
        if abs(current_power - power) < 0.001:
            return n
        elif current_power < power:
            n_min = n + 1
        else:
            n_max = n - 1
            
    return n_min 