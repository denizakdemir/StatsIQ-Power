import numpy as np
from scipy import stats

def pearson_correlation_power(r, n, alpha=0.05, alternative='two-sided'):
    """
    Calculate power for a Pearson correlation test.
    
    Parameters
    ----------
    r : float
        Expected correlation coefficient
    n : int
        Sample size
    alpha : float, optional
        Significance level (default: 0.05)
    alternative : str, optional
        Alternative hypothesis: 'two-sided' (default), 'greater', or 'less'
    
    Returns
    -------
    float
        Statistical power (1 - beta)
    """
    # Fisher's z transformation
    z = 0.5 * np.log((1 + r) / (1 - r))
    
    # Standard error of z
    se = 1 / np.sqrt(n - 3)
    
    # Calculate critical value
    if alternative == 'two-sided':
        critical_value = stats.norm.ppf(1 - alpha/2) * se
        power = 1 - stats.norm.cdf(critical_value, loc=z, scale=se) + \
                stats.norm.cdf(-critical_value, loc=z, scale=se)
    elif alternative == 'greater':
        critical_value = stats.norm.ppf(1 - alpha) * se
        power = 1 - stats.norm.cdf(critical_value, loc=z, scale=se)
    else:  # 'less'
        critical_value = stats.norm.ppf(alpha) * se
        power = stats.norm.cdf(critical_value, loc=z, scale=se)
    
    return power

def spearman_correlation_power(r, n, alpha=0.05, alternative='two-sided'):
    """
    Calculate power for a Spearman correlation test.
    
    Parameters
    ----------
    r : float
        Expected correlation coefficient
    n : int
        Sample size
    alpha : float, optional
        Significance level (default: 0.05)
    alternative : str, optional
        Alternative hypothesis: 'two-sided' (default), 'greater', or 'less'
    
    Returns
    -------
    float
        Statistical power (1 - beta)
    """
    # Convert Spearman correlation to Pearson correlation
    # This is an approximation based on the relationship between
    # Spearman and Pearson correlations
    r_pearson = 2 * np.sin(r * np.pi / 6)
    
    return pearson_correlation_power(r_pearson, n, alpha, alternative)

def kendall_correlation_power(tau, n, alpha=0.05, alternative='two-sided'):
    """
    Calculate power for a Kendall correlation test.
    
    Parameters
    ----------
    tau : float
        Expected Kendall's tau coefficient
    n : int
        Sample size
    alpha : float, optional
        Significance level (default: 0.05)
    alternative : str, optional
        Alternative hypothesis: 'two-sided' (default), 'greater', or 'less'
    
    Returns
    -------
    float
        Statistical power (1 - beta)
    """
    # Convert Kendall's tau to Pearson correlation
    # This is an approximation based on the relationship between
    # Kendall's tau and Pearson correlations
    r_pearson = np.sin(tau * np.pi / 2)
    
    return pearson_correlation_power(r_pearson, n, alpha, alternative)

def partial_correlation_power(r, n, k, alpha=0.05, alternative='two-sided'):
    """
    Calculate power for a partial correlation test.
    
    Parameters
    ----------
    r : float
        Expected partial correlation coefficient
    n : int
        Sample size
    k : int
        Number of control variables
    alpha : float, optional
        Significance level (default: 0.05)
    alternative : str, optional
        Alternative hypothesis: 'two-sided' (default), 'greater', or 'less'
    
    Returns
    -------
    float
        Statistical power (1 - beta)
    """
    # Fisher's z transformation
    z = 0.5 * np.log((1 + r) / (1 - r))
    
    # Standard error of z for partial correlation
    se = 1 / np.sqrt(n - k - 3)
    
    # Calculate critical value
    if alternative == 'two-sided':
        critical_value = stats.norm.ppf(1 - alpha/2) * se
        power = 1 - stats.norm.cdf(critical_value, loc=z, scale=se) + \
                stats.norm.cdf(-critical_value, loc=z, scale=se)
    elif alternative == 'greater':
        critical_value = stats.norm.ppf(1 - alpha) * se
        power = 1 - stats.norm.cdf(critical_value, loc=z, scale=se)
    else:  # 'less'
        critical_value = stats.norm.ppf(alpha) * se
        power = stats.norm.cdf(critical_value, loc=z, scale=se)
    
    return power

def sample_size_for_correlation(r, power=0.8, alpha=0.05, alternative='two-sided'):
    """
    Calculate required sample size for a correlation test to achieve desired power.
    
    Parameters
    ----------
    r : float
        Expected correlation coefficient
    power : float, optional
        Desired statistical power (default: 0.8)
    alpha : float, optional
        Significance level (default: 0.05)
    alternative : str, optional
        Alternative hypothesis: 'two-sided' (default), 'greater', or 'less'
    
    Returns
    -------
    int
        Required sample size
    """
    def power_at_n(n):
        return pearson_correlation_power(r, n, alpha, alternative)
    
    # Binary search for sample size
    n_min, n_max = 4, 1000  # Minimum n is 4 for correlation
    while power_at_n(n_max) < power:
        n_max *= 2
    
    while n_max - n_min > 1:
        n_mid = (n_min + n_max) // 2
        if power_at_n(n_mid) < power:
            n_min = n_mid
        else:
            n_max = n_mid
    
    return n_max 