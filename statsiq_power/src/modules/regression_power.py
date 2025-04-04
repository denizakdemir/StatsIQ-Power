import numpy as np
from scipy import stats

def linear_regression_power(r2, n, k, alpha=0.05):
    """
    Calculate power for a linear regression model.
    
    Parameters
    ----------
    r2 : float
        Expected R-squared value
    n : int
        Sample size
    k : int
        Number of predictors
    alpha : float, optional
        Significance level (default: 0.05)
    
    Returns
    -------
    float
        Statistical power (1 - beta)
    """
    # Calculate degrees of freedom
    df1 = k  # numerator degrees of freedom
    df2 = n - k - 1  # denominator degrees of freedom
    
    # Calculate non-centrality parameter
    ncp = (n * r2) / (1 - r2)
    
    # Calculate critical value
    critical_value = stats.f.ppf(1 - alpha, df1, df2)
    
    # Calculate power
    power = 1 - stats.f.cdf(critical_value, df1, df2, loc=ncp)
    
    return power

def multiple_regression_power(r2, n, k, alpha=0.05):
    """
    Calculate power for a multiple regression model.
    
    Parameters
    ----------
    r2 : float
        Expected R-squared value
    n : int
        Sample size
    k : int
        Number of predictors
    alpha : float, optional
        Significance level (default: 0.05)
    
    Returns
    -------
    float
        Statistical power (1 - beta)
    """
    return linear_regression_power(r2, n, k, alpha)

def logistic_regression_power(odds_ratio, p0, n, alpha=0.05):
    """
    Calculate power for a logistic regression model.
    
    Parameters
    ----------
    odds_ratio : float
        Expected odds ratio
    p0 : float
        Baseline probability of success
    n : int
        Sample size
    alpha : float, optional
        Significance level (default: 0.05)
    
    Returns
    -------
    float
        Statistical power (1 - beta)
    """
    # Convert odds ratio to probability
    p1 = (odds_ratio * p0) / (1 - p0 + odds_ratio * p0)
    
    # Calculate standard error
    se = np.sqrt((p0 * (1 - p0) + p1 * (1 - p1)) / n)
    
    # Calculate effect size
    effect_size = np.log(odds_ratio)
    
    # Calculate critical value
    critical_value = stats.norm.ppf(1 - alpha/2) * se
    
    # Calculate power
    power = 1 - stats.norm.cdf(critical_value, loc=effect_size, scale=se) + \
            stats.norm.cdf(-critical_value, loc=effect_size, scale=se)
    
    return power

def sample_size_for_regression(r2, power=0.8, k=1, alpha=0.05):
    """
    Calculate required sample size for a regression model to achieve desired power.
    
    Parameters
    ----------
    r2 : float
        Expected R-squared value
    power : float, optional
        Desired statistical power (default: 0.8)
    k : int, optional
        Number of predictors (default: 1)
    alpha : float, optional
        Significance level (default: 0.05)
    
    Returns
    -------
    int
        Required sample size
    """
    def power_at_n(n):
        return linear_regression_power(r2, n, k, alpha)
    
    # Binary search for sample size
    n_min, n_max = k + 2, 1000  # Minimum n is k + 2 for regression
    while power_at_n(n_max) < power:
        n_max *= 2
    
    while n_max - n_min > 1:
        n_mid = (n_min + n_max) // 2
        if power_at_n(n_mid) < power:
            n_min = n_mid
        else:
            n_max = n_mid
    
    return n_max

def sample_size_for_logistic_regression(odds_ratio, p0, power=0.8, alpha=0.05):
    """
    Calculate required sample size for a logistic regression model to achieve desired power.
    
    Parameters
    ----------
    odds_ratio : float
        Expected odds ratio
    p0 : float
        Baseline probability of success
    power : float, optional
        Desired statistical power (default: 0.8)
    alpha : float, optional
        Significance level (default: 0.05)
    
    Returns
    -------
    int
        Required sample size
    """
    def power_at_n(n):
        return logistic_regression_power(odds_ratio, p0, n, alpha)
    
    # Binary search for sample size
    n_min, n_max = 10, 1000  # Minimum n is 10 for logistic regression
    while power_at_n(n_max) < power:
        n_max *= 2
    
    while n_max - n_min > 1:
        n_mid = (n_min + n_max) // 2
        if power_at_n(n_mid) < power:
            n_min = n_mid
        else:
            n_max = n_mid
    
    return n_max 