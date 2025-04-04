import numpy as np
from scipy import stats

def cohens_d(mean1, mean2, sd1, sd2=None, n1=None, n2=None):
    """
    Calculate Cohen's d effect size.
    
    Parameters
    ----------
    mean1 : float
        Mean of first group
    mean2 : float
        Mean of second group
    sd1 : float
        Standard deviation of first group
    sd2 : float, optional
        Standard deviation of second group. If None, assumes equal variances.
    n1 : int, optional
        Sample size of first group (only needed for Hedges' g correction)
    n2 : int, optional
        Sample size of second group (only needed for Hedges' g correction)
    
    Returns
    -------
    float
        Cohen's d effect size
    """
    if sd2 is None:
        # Pooled standard deviation
        sd = sd1
    else:
        if n1 is not None and n2 is not None:
            # Pooled standard deviation
            sd = np.sqrt(((n1 - 1) * sd1**2 + (n2 - 1) * sd2**2) / (n1 + n2 - 2))
        else:
            # Simple average of standard deviations
            sd = (sd1 + sd2) / 2
    
    return (mean1 - mean2) / sd

def hedges_g(d, n1, n2):
    """
    Convert Cohen's d to Hedges' g (unbiased estimator).
    
    Parameters
    ----------
    d : float
        Cohen's d effect size
    n1 : int
        Sample size of first group
    n2 : int
        Sample size of second group
    
    Returns
    -------
    float
        Hedges' g effect size
    """
    df = n1 + n2 - 2
    correction = np.sqrt((df - 2) / df)
    return d * correction

def cohens_f(means, sds, ns):
    """
    Calculate Cohen's f effect size for ANOVA.
    
    Parameters
    ----------
    means : array-like
        Array of group means
    sds : array-like
        Array of group standard deviations
    ns : array-like
        Array of group sample sizes
    
    Returns
    -------
    float
        Cohen's f effect size
    """
    # Calculate grand mean
    grand_mean = np.sum(means * ns) / np.sum(ns)
    
    # Calculate between-groups variance
    between_var = np.sum(ns * (means - grand_mean)**2) / (len(means) - 1)
    
    # Calculate within-groups variance (pooled)
    within_var = np.sum((ns - 1) * sds**2) / (np.sum(ns) - len(means))
    
    return np.sqrt(between_var / within_var)

def cohens_w(observed, expected):
    """
    Calculate Cohen's w effect size for chi-square tests.
    
    Parameters
    ----------
    observed : array-like
        Observed frequencies
    expected : array-like
        Expected frequencies
    
    Returns
    -------
    float
        Cohen's w effect size
    """
    return np.sqrt(np.sum((observed - expected)**2 / expected))

def eta_squared(ss_effect, ss_total):
    """
    Calculate eta-squared effect size.
    
    Parameters
    ----------
    ss_effect : float
        Sum of squares for the effect
    ss_total : float
        Total sum of squares
    
    Returns
    -------
    float
        Eta-squared effect size
    """
    return ss_effect / ss_total

def partial_eta_squared(ss_effect, ss_error):
    """
    Calculate partial eta-squared effect size.
    
    Parameters
    ----------
    ss_effect : float
        Sum of squares for the effect
    ss_error : float
        Sum of squares for the error
    
    Returns
    -------
    float
        Partial eta-squared effect size
    """
    return ss_effect / (ss_effect + ss_error)

def odds_ratio(a, b, c, d):
    """
    Calculate odds ratio from a 2x2 contingency table.
    
    Parameters
    ----------
    a : int
        Count in cell (1,1)
    b : int
        Count in cell (1,2)
    c : int
        Count in cell (2,1)
    d : int
        Count in cell (2,2)
    
    Returns
    -------
    float
        Odds ratio
    """
    return (a * d) / (b * c)

def risk_ratio(a, b, c, d):
    """
    Calculate risk ratio from a 2x2 contingency table.
    
    Parameters
    ----------
    a : int
        Count in cell (1,1)
    b : int
        Count in cell (1,2)
    c : int
        Count in cell (2,1)
    d : int
        Count in cell (2,2)
    
    Returns
    -------
    float
        Risk ratio
    """
    risk1 = a / (a + b)
    risk2 = c / (c + d)
    return risk1 / risk2 