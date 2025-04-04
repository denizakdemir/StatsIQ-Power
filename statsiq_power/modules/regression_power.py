import numpy as np
from scipy import stats

def linear_regression_power(r2, n, k=1, alpha=0.05):
    """
    Calculate power for linear regression.
    
    Parameters:
    -----------
    r2 : float
        Expected R-squared value
    n : int
        Sample size
    k : int, optional
        Number of predictors (default: 1)
    alpha : float, optional
        Significance level (default: 0.05)
        
    Returns:
    --------
    float
        Power of the test
    """
    # Calculate non-centrality parameter
    ncp = n * r2 / (1 - r2)
    
    # Calculate degrees of freedom
    df1 = k
    df2 = n - k - 1
    
    # Calculate critical F value
    f_crit = stats.f.ppf(1 - alpha, df1, df2)
    
    # Calculate power
    power = 1 - stats.ncf.cdf(f_crit, df1, df2, ncp)
    
    return min(power, 1.0)  # Ensure power doesn't exceed 1.0

def multiple_regression_power(r2, n, k, alpha=0.05):
    """
    Calculate power for multiple regression.
    
    Parameters:
    -----------
    r2 : float
        Expected R-squared value
    n : int
        Sample size
    k : int
        Number of predictors
    alpha : float, optional
        Significance level (default: 0.05)
        
    Returns:
    --------
    float
        Power of the test
    """
    return linear_regression_power(r2, n, k, alpha)

def logistic_regression_power(odds_ratio, p0, n, k=1, alpha=0.05):
    """
    Calculate power for logistic regression.
    
    Parameters:
    -----------
    odds_ratio : float
        Expected odds ratio
    p0 : float
        Probability of success in control group
    n : int
        Sample size
    k : int, optional
        Number of predictors (default: 1)
    alpha : float, optional
        Significance level (default: 0.05)
        
    Returns:
    --------
    float
        Power of the test
    """
    # Calculate probabilities
    p1 = (odds_ratio * p0) / (1 + (odds_ratio - 1) * p0)
    
    # Calculate standard error
    se = np.sqrt(1/(n*p0*(1-p0)) + 1/(n*p1*(1-p1)))
    
    # Calculate effect size
    effect_size = np.log(odds_ratio) / se
    
    # Calculate critical value
    z_crit = stats.norm.ppf(1 - alpha/2)
    
    # Calculate power
    power = 1 - stats.norm.cdf(z_crit - effect_size) + stats.norm.cdf(-z_crit - effect_size)
    
    return min(power, 1.0)  # Ensure power doesn't exceed 1.0

def sample_size_for_regression(r2, power, k=1, alpha=0.05):
    """
    Calculate required sample size for linear regression.
    
    Parameters:
    -----------
    r2 : float
        Expected R-squared value
    power : float
        Desired power
    k : int, optional
        Number of predictors (default: 1)
    alpha : float, optional
        Significance level (default: 0.05)
        
    Returns:
    --------
    int
        Required sample size
    """
    def power_at_n(n):
        return linear_regression_power(r2, n, k, alpha)
    
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

def sample_size_for_logistic_regression(odds_ratio, p0, power, k=1, alpha=0.05):
    """
    Calculate required sample size for logistic regression.
    
    Parameters:
    -----------
    odds_ratio : float
        Expected odds ratio
    p0 : float
        Probability of success in control group
    power : float
        Desired power
    k : int, optional
        Number of predictors (default: 1)
    alpha : float, optional
        Significance level (default: 0.05)
        
    Returns:
    --------
    int
        Required sample size
    """
    def power_at_n(n):
        return logistic_regression_power(odds_ratio, p0, n, k, alpha)
    
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