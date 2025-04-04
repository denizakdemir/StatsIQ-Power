import numpy as np
from scipy import stats

def one_sample_t_test_power(effect_size, n, alpha=0.05, alternative='two-sided'):
    """
    Calculate power for one-sample t-test.
    
    Parameters:
    -----------
    effect_size : float
        Cohen's d effect size
    n : int
        Sample size
    alpha : float, optional
        Significance level (default: 0.05)
    alternative : str, optional
        Alternative hypothesis ('two-sided', 'greater', 'less')
        
    Returns:
    --------
    float
        Power of the test
    """
    # Calculate non-centrality parameter
    ncp = effect_size * np.sqrt(n)
    
    # Calculate degrees of freedom
    df = n - 1
    
    if alternative == 'two-sided':
        # Calculate critical t values
        t_crit = stats.t.ppf(1 - alpha/2, df)
        # Calculate power
        power = 1 - stats.nct.cdf(t_crit, df, ncp) + stats.nct.cdf(-t_crit, df, ncp)
    elif alternative == 'greater':
        # Calculate critical t value
        t_crit = stats.t.ppf(1 - alpha, df)
        # Calculate power
        power = 1 - stats.nct.cdf(t_crit, df, ncp)
    else:  # less
        # Calculate critical t value
        t_crit = stats.t.ppf(alpha, df)
        # Calculate power
        power = stats.nct.cdf(t_crit, df, ncp)
    
    return min(power, 1.0)  # Ensure power doesn't exceed 1.0

def independent_t_test_power(effect_size, n1, n2=None, alpha=0.05, alternative='two-sided'):
    """
    Calculate power for independent samples t-test.
    
    Parameters:
    -----------
    effect_size : float
        Cohen's d effect size
    n1 : int
        Sample size of first group
    n2 : int, optional
        Sample size of second group (default: None, equal to n1)
    alpha : float, optional
        Significance level (default: 0.05)
    alternative : str, optional
        Alternative hypothesis ('two-sided', 'greater', 'less')
        
    Returns:
    --------
    float
        Power of the test
    """
    if n2 is None:
        n2 = n1
    
    # Calculate non-centrality parameter
    ncp = effect_size * np.sqrt((n1 * n2) / (n1 + n2))
    
    # Calculate degrees of freedom
    df = n1 + n2 - 2
    
    if alternative == 'two-sided':
        # Calculate critical t values
        t_crit = stats.t.ppf(1 - alpha/2, df)
        # Calculate power
        power = 1 - stats.nct.cdf(t_crit, df, ncp) + stats.nct.cdf(-t_crit, df, ncp)
    elif alternative == 'greater':
        # Calculate critical t value
        t_crit = stats.t.ppf(1 - alpha, df)
        # Calculate power
        power = 1 - stats.nct.cdf(t_crit, df, ncp)
    else:  # less
        # Calculate critical t value
        t_crit = stats.t.ppf(alpha, df)
        # Calculate power
        power = stats.nct.cdf(t_crit, df, ncp)
    
    return min(power, 1.0)  # Ensure power doesn't exceed 1.0

def paired_t_test_power(effect_size, n, alpha=0.05, alternative='two-sided'):
    """
    Calculate power for paired samples t-test.
    
    Parameters:
    -----------
    effect_size : float
        Cohen's d effect size
    n : int
        Sample size
    alpha : float, optional
        Significance level (default: 0.05)
    alternative : str, optional
        Alternative hypothesis ('two-sided', 'greater', 'less')
        
    Returns:
    --------
    float
        Power of the test
    """
    # Calculate non-centrality parameter
    ncp = effect_size * np.sqrt(n)
    
    # Calculate degrees of freedom
    df = n - 1
    
    if alternative == 'two-sided':
        # Calculate critical t values
        t_crit = stats.t.ppf(1 - alpha/2, df)
        # Calculate power
        power = 1 - stats.nct.cdf(t_crit, df, ncp) + stats.nct.cdf(-t_crit, df, ncp)
    elif alternative == 'greater':
        # Calculate critical t value
        t_crit = stats.t.ppf(1 - alpha, df)
        # Calculate power
        power = 1 - stats.nct.cdf(t_crit, df, ncp)
    else:  # less
        # Calculate critical t value
        t_crit = stats.t.ppf(alpha, df)
        # Calculate power
        power = stats.nct.cdf(t_crit, df, ncp)
    
    return min(power, 1.0)  # Ensure power doesn't exceed 1.0

def sample_size_for_t_test(effect_size, power, alpha=0.05, test_type='independent', alternative='two-sided'):
    """
    Calculate required sample size for t-test.
    
    Parameters:
    -----------
    effect_size : float
        Cohen's d effect size
    power : float
        Desired power
    alpha : float, optional
        Significance level (default: 0.05)
    test_type : str, optional
        Type of t-test ('one-sample', 'independent', 'paired')
    alternative : str, optional
        Alternative hypothesis ('two-sided', 'greater', 'less')
        
    Returns:
    --------
    int or tuple
        Required sample size(s)
    """
    def power_at_n(n):
        if test_type == 'one-sample':
            return one_sample_t_test_power(effect_size, n, alpha, alternative)
        elif test_type == 'independent':
            return independent_t_test_power(effect_size, n, n, alpha, alternative)
        else:  # paired
            return paired_t_test_power(effect_size, n, alpha, alternative)
    
    # Binary search for sample size
    left, right = 2, 1000
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
                right = n
                break
    
    if test_type == 'independent':
        return right, right  # Return tuple for independent t-test
    return right 