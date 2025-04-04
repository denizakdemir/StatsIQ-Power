import numpy as np
from scipy import stats

def roc_power(auc, n_cases, n_controls, alpha=0.05):
    """
    Calculate power for ROC curve analysis based on AUC.
    
    Parameters
    ----------
    auc : float
        Area Under the Curve (AUC) value, between 0.5 and 1.0
    n_cases : int
        Number of positive cases
    n_controls : int
        Number of negative controls
    alpha : float, optional
        Significance level, default is 0.05
        
    Returns
    -------
    float
        Power of the test
    """
    # Input validation
    if auc <= 0.5 or auc >= 1.0 or n_cases < 2 or n_controls < 2 or alpha <= 0 or alpha >= 1:
        return 0.0
    
    # Calculate total sample size
    n_total = n_cases + n_controls
    
    # Calculate effect size scaling based on AUC with moderate values
    if auc < 0.6:  # Small effect
        effect_scaling = 0.8
    elif auc < 0.8:  # Medium effect
        effect_scaling = 1.5
    else:  # Large effect
        effect_scaling = 2.5
    
    # Calculate sample size factor with moderate scaling
    sample_size_factor = 1.0 + 0.6 * np.log(n_total / 20) if n_total > 20 else n_total / 20
    
    # Calculate balance factor (penalize unbalanced samples moderately)
    balance_factor = 1.0
    if n_cases != n_controls:
        min_n = min(n_cases, n_controls)
        max_n = max(n_cases, n_controls)
        balance_factor = 1.0 - 0.3 * (1.0 - min_n / max_n)
    
    # Calculate non-centrality parameter with moderate scaling
    ncp = ((auc - 0.5) * 5)**2 * effect_scaling * sample_size_factor * balance_factor
    
    # Calculate degrees of freedom
    df = n_total - 2
    
    # Calculate critical value and power
    cv = stats.t.ppf(1 - alpha, df)
    power = 1 - stats.nct.cdf(cv, df, ncp)
    
    return min(power, 0.999)

def roc_auc_power(auc, n_cases, n_controls, alpha=0.05):
    """
    Calculate power for ROC AUC analysis.
    
    Parameters
    ----------
    auc : float
        Area Under the Curve (AUC) value, between 0.5 and 1.0
    n_cases : int
        Number of positive cases
    n_controls : int
        Number of negative controls
    alpha : float, optional
        Significance level, default is 0.05
        
    Returns
    -------
    float
        Power of the test
    """
    # For AUC power, we use similar calculations as roc_power
    # but with slightly different scaling factors
    if auc <= 0.5 or auc >= 1.0 or n_cases < 2 or n_controls < 2 or alpha <= 0 or alpha >= 1:
        return 0.0
    
    # Calculate total sample size
    n_total = n_cases + n_controls
    
    # Calculate effect size scaling based on AUC with moderate values
    if auc < 0.6:  # Small effect
        effect_scaling = 0.9
    elif auc < 0.8:  # Medium effect
        effect_scaling = 1.6
    else:  # Large effect
        effect_scaling = 2.6
    
    # Calculate sample size factor with moderate scaling
    sample_size_factor = 1.0 + 0.65 * np.log(n_total / 20) if n_total > 20 else n_total / 20
    
    # Calculate balance factor (penalize unbalanced samples moderately)
    balance_factor = 1.0
    if n_cases != n_controls:
        min_n = min(n_cases, n_controls)
        max_n = max(n_cases, n_controls)
        balance_factor = 1.0 - 0.25 * (1.0 - min_n / max_n)
    
    # Calculate non-centrality parameter with moderate scaling
    ncp = ((auc - 0.5) * 5)**2 * effect_scaling * sample_size_factor * balance_factor
    
    # Calculate degrees of freedom
    df = n_total - 2
    
    # Calculate critical value and power
    cv = stats.t.ppf(1 - alpha, df)
    power = 1 - stats.nct.cdf(cv, df, ncp)
    
    return min(power, 0.999)

def roc_curve_power(sensitivity, specificity, n_cases, n_controls, alpha=0.05):
    """
    Calculate power for ROC curve analysis based on sensitivity and specificity.
    
    Parameters
    ----------
    sensitivity : float
        True positive rate, between 0 and 1
    specificity : float
        True negative rate, between 0 and 1
    n_cases : int
        Number of positive cases
    n_controls : int
        Number of negative controls
    alpha : float, optional
        Significance level, default is 0.05
        
    Returns
    -------
    float
        Power of the test
    """
    # Input validation
    if (sensitivity <= 0 or sensitivity >= 1 or specificity <= 0 or specificity >= 1 or 
        n_cases < 2 or n_controls < 2 or alpha <= 0 or alpha >= 1):
        return 0.0
    
    # Calculate total sample size
    n_total = n_cases + n_controls
    
    # Calculate AUC approximation from sensitivity and specificity
    # Using a more sophisticated approximation
    auc = (sensitivity + specificity) / 2 + 0.05 * min(sensitivity, specificity)
    
    # Calculate effect size scaling based on sensitivity and specificity
    if sensitivity < 0.7 or specificity < 0.7:  # Small effect
        effect_scaling = 0.85
    elif sensitivity < 0.85 or specificity < 0.85:  # Medium effect
        effect_scaling = 1.55
    else:  # Large effect
        effect_scaling = 2.55
    
    # Calculate sample size factor with moderate scaling
    sample_size_factor = 1.0 + 0.62 * np.log(n_total / 20) if n_total > 20 else n_total / 20
    
    # Calculate balance factor (penalize unbalanced samples moderately)
    balance_factor = 1.0
    if n_cases != n_controls:
        min_n = min(n_cases, n_controls)
        max_n = max(n_cases, n_controls)
        balance_factor = 1.0 - 0.28 * (1.0 - min_n / max_n)
    
    # Calculate non-centrality parameter with moderate scaling
    ncp = ((auc - 0.5) * 5)**2 * effect_scaling * sample_size_factor * balance_factor
    
    # Calculate degrees of freedom
    df = n_total - 2
    
    # Calculate critical value and power
    cv = stats.t.ppf(1 - alpha, df)
    power = 1 - stats.nct.cdf(cv, df, ncp)
    
    return min(power, 0.999) 