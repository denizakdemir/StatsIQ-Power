import numpy as np
from scipy import stats

def chi_square_goodness_of_fit_power(expected_proportions, n, alpha=0.05):
    """
    Calculate power for chi-square goodness of fit test.
    
    Parameters:
    -----------
    expected_proportions : array-like
        Expected proportions under null hypothesis
    n : int
        Sample size
    alpha : float, optional
        Significance level (default: 0.05)
    
    Returns:
    --------
    float
        Power of the test
    """
    # Ensure proportions sum to 1
    expected_proportions = np.array(expected_proportions)
    expected_proportions = expected_proportions / np.sum(expected_proportions)
    
    # Calculate expected frequencies
    expected = expected_proportions * n
    
    # Calculate observed frequencies with effect size
    observed = expected * 1.5  # 50% difference
    
    # Calculate non-centrality parameter
    ncp = np.sum((observed - expected)**2 / expected)
    
    # Calculate degrees of freedom
    df = len(expected_proportions) - 1
    
    # Calculate critical value
    chi2_crit = stats.chi2.ppf(1 - alpha, df)
    
    # Calculate power
    power = 1 - stats.ncx2.cdf(chi2_crit, df, ncp)
    
    return min(power, 1.0)  # Ensure power doesn't exceed 1.0

def chi_square_independence_power(observed, expected, alpha=0.05):
    """
    Calculate power for chi-square test of independence.
    
    Parameters:
    -----------
    observed : array-like
        Observed contingency table
    expected : array-like
        Expected contingency table under null hypothesis
    alpha : float, optional
        Significance level (default: 0.05)
    
    Returns:
    --------
    float
        Power of the test
    """
    # Convert to numpy arrays
    observed = np.array(observed)
    expected = np.array(expected)
    
    # Calculate non-centrality parameter
    ncp = np.sum((observed - expected)**2 / expected)
    
    # Calculate degrees of freedom
    df = (observed.shape[0] - 1) * (observed.shape[1] - 1)
    
    # Calculate critical value
    chi2_crit = stats.chi2.ppf(1 - alpha, df)
    
    # Calculate power
    power = 1 - stats.ncx2.cdf(chi2_crit, df, ncp)
    
    return min(power, 1.0)  # Ensure power doesn't exceed 1.0

def chi_square_homogeneity_power(observed, expected, alpha=0.05):
    """
    Calculate power for chi-square test of homogeneity.
    
    Parameters:
    -----------
    observed : array-like
        Observed contingency table
    expected : array-like
        Expected contingency table under null hypothesis
    alpha : float, optional
        Significance level (default: 0.05)
    
    Returns:
    --------
    float
        Power of the test
    """
    return chi_square_independence_power(observed, expected, alpha)

def fisher_exact_power(proportions, n, alpha=0.05):
    """
    Calculate power for Fisher's exact test.
    
    Parameters:
    -----------
    proportions : array-like
        Expected proportions
    n : int
        Sample size per group
    alpha : float, optional
        Significance level (default: 0.05)
    
    Returns:
    --------
    float
        Power of the test
    """
    # Ensure proportions sum to 1
    proportions = np.array(proportions)
    proportions = proportions / np.sum(proportions)
    
    # Calculate expected frequencies
    expected = proportions * n
    
    # Calculate observed frequencies with effect size
    observed = expected * 1.5  # 50% difference
    
    # Calculate chi-square statistic
    chi2 = np.sum((observed - expected)**2 / expected)
    
    # Calculate degrees of freedom
    df = len(proportions) - 1
    
    # Calculate power
    power = 1 - stats.chi2.cdf(chi2, df)
    
    return min(power, 1.0)  # Ensure power doesn't exceed 1.0

def sample_size_for_chi_square(proportions, power, alpha=0.05, test_type='goodness'):
    """
    Calculate required sample size for chi-square test.
    
    Parameters:
    -----------
    proportions : array-like
        Expected proportions
    power : float
        Desired power
    alpha : float, optional
        Significance level (default: 0.05)
    test_type : str, optional
        Type of test ('goodness', 'independence', 'homogeneity')
    
    Returns:
    --------
    int
        Required sample size
    """
    def power_at_n(n):
        if test_type == 'goodness':
            return chi_square_goodness_of_fit_power(proportions, n, alpha)
        elif test_type == 'independence':
            expected = np.outer(np.sum(proportions, axis=1), np.sum(proportions, axis=0)) / np.sum(proportions)
            observed = expected * 1.5
            return chi_square_independence_power(observed, expected, alpha)
        else:  # homogeneity
            expected = np.outer(np.sum(proportions, axis=1), np.sum(proportions, axis=0)) / np.sum(proportions)
            observed = expected * 1.5
            return chi_square_homogeneity_power(observed, expected, alpha)
    
    # Binary search for sample size
    left, right = len(proportions) + 1, 1000
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