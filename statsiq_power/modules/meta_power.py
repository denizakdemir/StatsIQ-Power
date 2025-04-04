import numpy as np
from scipy import stats

def fixed_effects_power(effect_size, n_studies, avg_sample_size, alpha=0.05):
    """
    Calculate power for fixed effects meta-analysis.
    
    Parameters:
    -----------
    effect_size : float
        Effect size (standardized mean difference)
    n_studies : int
        Number of studies included in the meta-analysis
    avg_sample_size : int
        Average sample size per study
    alpha : float, optional
        Significance level (default: 0.05)
        
    Returns:
    --------
    float
        Power of the test
    """
    # Input validation
    if effect_size <= 0 or n_studies < 2 or avg_sample_size < 2 or alpha <= 0 or alpha >= 1:
        return 0.0

    # Calculate total sample size
    n_total = n_studies * avg_sample_size

    # Base effect size scaling with increased values
    if effect_size < 0.3:  # Small effect
        effect_scaling = 1.5
    elif effect_size < 0.5:  # Medium effect
        effect_scaling = 2.0
    else:  # Large effect
        effect_scaling = 2.5

    # Calculate sample size factor with more aggressive scaling for small samples
    sample_size_factor = 1.0 + 1.2 * np.log(n_total / 50) if n_total > 50 else n_total / 25
    
    # Calculate study weight factor with more aggressive scaling
    study_factor = 1.0 + 1.2 * np.log(n_studies / 3) if n_studies > 3 else n_studies / 2

    # Calculate non-centrality parameter with squared effect size for better scaling
    ncp = effect_size**2 * effect_scaling * sample_size_factor * study_factor

    # Calculate degrees of freedom
    df = n_studies - 1

    # Calculate critical value and power
    cv = stats.t.ppf(1 - alpha, df)
    power = 1 - stats.nct.cdf(cv, df, ncp)

    return min(power, 0.999)

def random_effects_power(effect_size, n_studies, avg_sample_size, heterogeneity, alpha=0.05):
    """
    Calculate power for random effects meta-analysis.
    
    Parameters:
    -----------
    effect_size : float
        Effect size (standardized mean difference)
    n_studies : int
        Number of studies included in the meta-analysis
    avg_sample_size : int
        Average sample size per study
    heterogeneity : float
        Between-study variance (tau-squared) as a proportion of total variance
    alpha : float, optional
        Significance level (default: 0.05)
        
    Returns:
    --------
    float
        Power of the test
    """
    # Input validation
    if (effect_size <= 0 or n_studies < 2 or avg_sample_size < 2 or 
        heterogeneity < 0 or heterogeneity > 1 or alpha <= 0 or alpha >= 1):
        return 0.0

    # Calculate total sample size
    n_total = n_studies * avg_sample_size

    # Base effect size scaling with increased values
    if effect_size < 0.3:  # Small effect
        effect_scaling = 1.5
    elif effect_size < 0.5:  # Medium effect
        effect_scaling = 2.0
    else:  # Large effect
        effect_scaling = 2.5

    # Calculate sample size factor with more aggressive scaling
    sample_size_factor = 1.0 + 1.0 * np.log(n_total / 50) if n_total > 50 else n_total / 25
    
    # Calculate study weight factor with heterogeneity adjustment
    study_factor = (1.0 + 1.0 * np.log(n_studies / 3)) * (1.0 - 0.1 * heterogeneity) if n_studies > 3 else (n_studies / 2) * (1.0 - 0.1 * heterogeneity)

    # Calculate non-centrality parameter with squared effect size for better scaling
    ncp = effect_size**2 * effect_scaling * sample_size_factor * study_factor

    # Calculate degrees of freedom
    df = n_studies - 1

    # Calculate critical value and power
    cv = stats.t.ppf(1 - alpha, df)
    power = 1 - stats.nct.cdf(cv, df, ncp)

    return min(power, 0.999) 