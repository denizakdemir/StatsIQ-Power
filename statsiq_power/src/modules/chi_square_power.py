import numpy as np
from scipy import stats
from ..utils.effect_size import cohens_w

def chi_square_goodness_of_fit_power(expected_proportions, n, alpha=0.05):
    """
    Calculate power for a chi-square goodness of fit test.
    
    Parameters
    ----------
    expected_proportions : array-like
        Expected proportions under the null hypothesis
    n : int
        Sample size
    alpha : float, optional
        Significance level (default: 0.05)
    
    Returns
    -------
    float
        Statistical power (1 - beta)
    """
    # Calculate degrees of freedom
    df = len(expected_proportions) - 1
    
    # Calculate critical value
    critical_value = stats.chi2.ppf(1 - alpha, df)
    
    # Calculate non-centrality parameter
    ncp = n * np.sum((expected_proportions - 1/len(expected_proportions))**2 / 
                     (1/len(expected_proportions)))
    
    # Calculate power
    power = 1 - stats.chi2.cdf(critical_value, df, loc=ncp)
    
    return power

def chi_square_independence_power(observed, expected, alpha=0.05):
    """
    Calculate power for a chi-square test of independence.
    
    Parameters
    ----------
    observed : array-like
        Observed frequencies
    expected : array-like
        Expected frequencies under independence
    alpha : float, optional
        Significance level (default: 0.05)
    
    Returns
    -------
    float
        Statistical power (1 - beta)
    """
    # Calculate degrees of freedom
    df = (observed.shape[0] - 1) * (observed.shape[1] - 1)
    
    # Calculate critical value
    critical_value = stats.chi2.ppf(1 - alpha, df)
    
    # Calculate non-centrality parameter
    ncp = np.sum((observed - expected)**2 / expected)
    
    # Calculate power
    power = 1 - stats.chi2.cdf(critical_value, df, loc=ncp)
    
    return power

def chi_square_homogeneity_power(observed, expected, alpha=0.05):
    """
    Calculate power for a chi-square test of homogeneity.
    
    Parameters
    ----------
    observed : array-like
        Observed frequencies
    expected : array-like
        Expected frequencies under homogeneity
    alpha : float, optional
        Significance level (default: 0.05)
    
    Returns
    -------
    float
        Statistical power (1 - beta)
    """
    # Calculate degrees of freedom
    df = (observed.shape[0] - 1) * (observed.shape[1] - 1)
    
    # Calculate critical value
    critical_value = stats.chi2.ppf(1 - alpha, df)
    
    # Calculate non-centrality parameter
    ncp = np.sum((observed - expected)**2 / expected)
    
    # Calculate power
    power = 1 - stats.chi2.cdf(critical_value, df, loc=ncp)
    
    return power

def sample_size_for_chi_square(effect_size, power=0.8, df=1, alpha=0.05):
    """
    Calculate required sample size for a chi-square test to achieve desired power.
    
    Parameters
    ----------
    effect_size : float
        Cohen's w effect size
    power : float, optional
        Desired statistical power (default: 0.8)
    df : int, optional
        Degrees of freedom (default: 1)
    alpha : float, optional
        Significance level (default: 0.05)
    
    Returns
    -------
    int
        Required sample size
    """
    def power_at_n(n):
        # Calculate critical value
        critical_value = stats.chi2.ppf(1 - alpha, df)
        
        # Calculate non-centrality parameter
        ncp = n * effect_size**2
        
        # Calculate power
        return 1 - stats.chi2.cdf(critical_value, df, loc=ncp)
    
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

def fisher_exact_power(proportions, n_per_group, alpha=0.05):
    """
    Calculate power for Fisher's exact test.
    
    Parameters
    ----------
    proportions : array-like
        Expected proportions for each group
    n_per_group : int
        Sample size per group
    alpha : float, optional
        Significance level (default: 0.05)
    
    Returns
    -------
    float
        Statistical power (1 - beta)
    """
    # This is an approximation since Fisher's exact test doesn't have
    # a simple power calculation formula
    
    # Calculate expected frequencies
    expected = np.array(proportions) * n_per_group
    
    # Calculate chi-square statistic
    chi2 = np.sum((expected - n_per_group/len(proportions))**2 / 
                  (n_per_group/len(proportions)))
    
    # Calculate p-value
    p_value = 1 - stats.chi2.cdf(chi2, len(proportions) - 1)
    
    # Power is 1 - beta, where beta is the probability of Type II error
    # We approximate this using the chi-square distribution
    return 1 - p_value 