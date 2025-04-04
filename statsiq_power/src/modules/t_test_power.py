import numpy as np
from scipy import stats

def one_sample_t_test_power(effect_size, n, alpha=0.05, alternative='two-sided'):
    """
    Calculate power for a one-sample t-test.
    
    Parameters
    ----------
    effect_size : float
        Cohen's d effect size
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
    # Calculate non-centrality parameter
    ncp = effect_size * np.sqrt(n)
    
    # Calculate critical value
    if alternative == 'two-sided':
        critical_value = stats.t.ppf(1 - alpha/2, df=n-1)
        power = 1 - stats.t.cdf(critical_value, df=n-1, loc=ncp) + \
                stats.t.cdf(-critical_value, df=n-1, loc=ncp)
    elif alternative == 'greater':
        critical_value = stats.t.ppf(1 - alpha, df=n-1)
        power = 1 - stats.t.cdf(critical_value, df=n-1, loc=ncp)
    else:  # 'less'
        critical_value = stats.t.ppf(alpha, df=n-1)
        power = stats.t.cdf(critical_value, df=n-1, loc=ncp)
    
    return power

def independent_t_test_power(effect_size, n1, n2=None, alpha=0.05, alternative='two-sided'):
    """
    Calculate power for an independent samples t-test.
    
    Parameters
    ----------
    effect_size : float
        Cohen's d effect size
    n1 : int
        Sample size of first group
    n2 : int, optional
        Sample size of second group (default: equal to n1)
    alpha : float, optional
        Significance level (default: 0.05)
    alternative : str, optional
        Alternative hypothesis: 'two-sided' (default), 'greater', or 'less'
    
    Returns
    -------
    float
        Statistical power (1 - beta)
    """
    if n2 is None:
        n2 = n1
    
    # Calculate non-centrality parameter
    ncp = effect_size * np.sqrt((n1 * n2) / (n1 + n2))
    
    # Calculate degrees of freedom
    df = n1 + n2 - 2
    
    # Calculate critical value
    if alternative == 'two-sided':
        critical_value = stats.t.ppf(1 - alpha/2, df=df)
        power = 1 - stats.t.cdf(critical_value, df=df, loc=ncp) + \
                stats.t.cdf(-critical_value, df=df, loc=ncp)
    elif alternative == 'greater':
        critical_value = stats.t.ppf(1 - alpha, df=df)
        power = 1 - stats.t.cdf(critical_value, df=df, loc=ncp)
    else:  # 'less'
        critical_value = stats.t.ppf(alpha, df=df)
        power = stats.t.cdf(critical_value, df=df, loc=ncp)
    
    return power

def paired_t_test_power(effect_size, n, alpha=0.05, alternative='two-sided'):
    """
    Calculate power for a paired samples t-test.
    
    Parameters
    ----------
    effect_size : float
        Cohen's d effect size
    n : int
        Sample size (number of pairs)
    alpha : float, optional
        Significance level (default: 0.05)
    alternative : str, optional
        Alternative hypothesis: 'two-sided' (default), 'greater', or 'less'
    
    Returns
    -------
    float
        Statistical power (1 - beta)
    """
    # Calculate non-centrality parameter
    ncp = effect_size * np.sqrt(n)
    
    # Calculate degrees of freedom
    df = n - 1
    
    # Calculate critical value
    if alternative == 'two-sided':
        critical_value = stats.t.ppf(1 - alpha/2, df=df)
        power = 1 - stats.t.cdf(critical_value, df=df, loc=ncp) + \
                stats.t.cdf(-critical_value, df=df, loc=ncp)
    elif alternative == 'greater':
        critical_value = stats.t.ppf(1 - alpha, df=df)
        power = 1 - stats.t.cdf(critical_value, df=df, loc=ncp)
    else:  # 'less'
        critical_value = stats.t.ppf(alpha, df=df)
        power = stats.t.cdf(critical_value, df=df, loc=ncp)
    
    return power

def sample_size_for_t_test(effect_size, power=0.8, alpha=0.05, test_type='independent', 
                          n_ratio=1.0, alternative='two-sided'):
    """
    Calculate required sample size for a t-test to achieve desired power.
    
    Parameters
    ----------
    effect_size : float
        Cohen's d effect size
    power : float, optional
        Desired statistical power (default: 0.8)
    alpha : float, optional
        Significance level (default: 0.05)
    test_type : str, optional
        Type of t-test: 'one-sample', 'independent', or 'paired' (default: 'independent')
    n_ratio : float, optional
        Ratio of sample sizes (n2/n1) for independent t-test (default: 1.0)
    alternative : str, optional
        Alternative hypothesis: 'two-sided' (default), 'greater', or 'less'
    
    Returns
    -------
    tuple
        Required sample size(s). For independent t-test, returns (n1, n2).
        For other tests, returns single sample size.
    """
    def power_at_n(n):
        if test_type == 'one-sample':
            return one_sample_t_test_power(effect_size, n, alpha, alternative)
        elif test_type == 'independent':
            n2 = int(n * n_ratio)
            return independent_t_test_power(effect_size, n, n2, alpha, alternative)
        else:  # paired
            return paired_t_test_power(effect_size, n, alpha, alternative)
    
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
    
    if test_type == 'independent':
        return n_max, int(n_max * n_ratio)
    return n_max 