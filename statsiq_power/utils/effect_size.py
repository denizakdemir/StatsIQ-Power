import numpy as np
from scipy import stats

def cohens_d(mean1, mean2, sd1, sd2=None, n1=None, n2=None):
    """
    Calculate Cohen's d effect size.
    
    Parameters
    ----------
    mean1 : float
        Mean of group 1
    mean2 : float
        Mean of group 2
    sd1 : float
        Standard deviation of group 1
    sd2 : float, optional
        Standard deviation of group 2. If None, assumed equal to sd1
    n1 : int, optional
        Sample size of group 1
    n2 : int, optional
        Sample size of group 2
        
    Returns
    -------
    float
        Cohen's d effect size
    """
    if sd2 is None:
        sd2 = sd1
    
    if n1 is not None and n2 is not None:
        # Pooled standard deviation
        sd = np.sqrt(((n1 - 1) * sd1**2 + (n2 - 1) * sd2**2) / (n1 + n2 - 2))
    else:
        # Simple average of standard deviations
        sd = (sd1 + sd2) / 2
    
    return (mean1 - mean2) / sd

def hedges_g(d, n1, n2):
    """
    Calculate Hedges' g effect size.
    
    Parameters
    ----------
    d : float
        Cohen's d effect size
    n1 : int
        Sample size of group 1
    n2 : int
        Sample size of group 2
        
    Returns
    -------
    float
        Hedges' g effect size
    """
    df = n1 + n2 - 2
    correction = 1 - 3 / (4 * df - 1)
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
    means = np.array(means)
    sds = np.array(sds)
    ns = np.array(ns)
    
    # Calculate grand mean
    grand_mean = np.sum(means * ns) / np.sum(ns)
    
    # Calculate between-groups variance
    between_var = np.sum(ns * (means - grand_mean)**2) / np.sum(ns)
    
    # Calculate within-groups variance
    within_var = np.sum((ns - 1) * sds**2) / np.sum(ns - 1)
    
    # Calculate Cohen's f
    f = np.sqrt(between_var / within_var)
    
    return f

def cohens_w(observed, expected):
    """
    Calculate Cohen's w effect size for chi-square tests.
    
    Parameters
    ----------
    observed : array-like
        Array of observed frequencies
    expected : array-like
        Array of expected frequencies
        
    Returns
    -------
    float
        Cohen's w effect size
    """
    observed = np.array(observed)
    expected = np.array(expected)
    
    # Calculate chi-square statistic
    chi2 = np.sum((observed - expected)**2 / expected)
    
    # Calculate Cohen's w
    w = np.sqrt(chi2 / np.sum(observed))
    
    return w

def eta_squared(ss_effect, ss_total):
    """
    Calculate eta squared effect size.
    
    Parameters
    ----------
    ss_effect : float
        Sum of squares for the effect
    ss_total : float
        Total sum of squares
        
    Returns
    -------
    float
        Eta squared effect size
    """
    return ss_effect / ss_total

def partial_eta_squared(ss_effect, ss_error):
    """
    Calculate partial eta squared effect size.
    
    Parameters
    ----------
    ss_effect : float
        Sum of squares for the effect
    ss_error : float
        Sum of squares for the error
        
    Returns
    -------
    float
        Partial eta squared effect size
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
    p1 = a / (a + b)
    p2 = c / (c + d)
    return p1 / p2 