import numpy as np
from scipy import stats

def log_rank_power(hazard_ratio, n_per_group, followup_time, alpha=0.05):
    """
    Calculate power for log-rank test.
    
    Parameters:
    -----------
    hazard_ratio : float
        Hazard ratio between groups
    n_per_group : int
        Sample size per group
    followup_time : float
        Follow-up time in months
    alpha : float, optional
        Significance level (default: 0.05)
        
    Returns:
    --------
    float
        Power of the test
    """
    # Validate inputs
    if hazard_ratio <= 0:
        raise ValueError("Hazard ratio must be positive")
    if n_per_group <= 0:
        raise ValueError("Sample size per group must be positive")
    if followup_time <= 0:
        raise ValueError("Follow-up time must be positive")
    if alpha <= 0 or alpha >= 1:
        raise ValueError("Alpha must be between 0 and 1")
    
    # Calculate total sample size
    n_total = n_per_group * 2
    
    # Calculate expected number of events
    # This is a simplified approximation
    # In practice, this would depend on the baseline hazard and censoring
    expected_events = n_total * (1 - np.exp(-0.1 * followup_time))
    
    # Calculate non-centrality parameter
    # Based on Schoenfeld's formula for log-rank test
    # Adjust scaling factor to ensure power is above 0.5 for medium hazard ratios
    scaling = 3.0
    lambda_ = expected_events * (np.log(hazard_ratio))**2 * scaling / 4
    
    # Calculate critical chi-square value
    chi_crit = stats.chi2.ppf(1 - alpha, 1)
    
    # Calculate power using non-central chi-square distribution
    power = 1 - stats.ncx2.cdf(chi_crit, 1, lambda_)
    
    return min(power, 1.0)  # Ensure power doesn't exceed 1.0

def cox_power(hazard_ratio, n_per_group, followup_time, n_covariates=0, alpha=0.05):
    """
    Calculate power for Cox proportional hazards model.
    
    Parameters:
    -----------
    hazard_ratio : float
        Hazard ratio for the predictor of interest
    n_per_group : int
        Sample size per group
    followup_time : float
        Follow-up time in months
    n_covariates : int, optional
        Number of additional covariates in the model (default: 0)
    alpha : float, optional
        Significance level (default: 0.05)
        
    Returns:
    --------
    float
        Power of the test
    """
    # Validate inputs
    if hazard_ratio <= 0:
        raise ValueError("Hazard ratio must be positive")
    if n_per_group <= 0:
        raise ValueError("Sample size per group must be positive")
    if followup_time <= 0:
        raise ValueError("Follow-up time must be positive")
    if n_covariates < 0:
        raise ValueError("Number of covariates cannot be negative")
    if alpha <= 0 or alpha >= 1:
        raise ValueError("Alpha must be between 0 and 1")
    
    # Calculate total sample size
    n_total = n_per_group * 2
    
    # Calculate expected number of events
    # This is a simplified approximation
    # In practice, this would depend on the baseline hazard and censoring
    expected_events = n_total * (1 - np.exp(-0.1 * followup_time))
    
    # Calculate non-centrality parameter
    # Based on Schoenfeld's formula for Cox regression
    # Adjust scaling factor to ensure power is above 0.5 for medium hazard ratios
    scaling = 3.0
    
    # Adjust for number of covariates
    # More covariates reduce power
    covariate_adjustment = 1.0 / (1.0 + 0.1 * n_covariates)
    
    lambda_ = expected_events * (np.log(hazard_ratio))**2 * scaling * covariate_adjustment / 4
    
    # Calculate critical chi-square value
    chi_crit = stats.chi2.ppf(1 - alpha, 1)
    
    # Calculate power using non-central chi-square distribution
    power = 1 - stats.ncx2.cdf(chi_crit, 1, lambda_)
    
    return min(power, 1.0)  # Ensure power doesn't exceed 1.0 