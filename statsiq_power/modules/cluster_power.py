import numpy as np
from scipy import stats

def cluster_continuous_power(effect_size, n_clusters, cluster_size, icc, alpha=0.05):
    """
    Calculate power for cluster randomized trials with continuous outcomes.
    
    Parameters
    ----------
    effect_size : float
        Standardized effect size (Cohen's d)
    n_clusters : int
        Number of clusters per arm
    cluster_size : int
        Average number of subjects per cluster
    icc : float
        Intraclass correlation coefficient (0 to 1)
    alpha : float, optional
        Significance level, default is 0.05
        
    Returns
    -------
    float
        Power of the test
    """
    # Input validation
    if (effect_size <= 0 or n_clusters < 2 or cluster_size < 2 or 
        icc < 0 or icc >= 1 or alpha <= 0 or alpha >= 1):
        return 0.0
    
    # Calculate design effect (variance inflation factor)
    design_effect = 1.0 + (cluster_size - 1) * icc
    
    # Calculate effective sample size per arm
    n_effective = (n_clusters * cluster_size) / design_effect
    
    # Calculate effect size scaling based on magnitude
    if effect_size < 0.3:  # Small effect
        effect_scaling = 0.4
    elif effect_size < 0.5:  # Medium effect
        effect_scaling = 0.6
    else:  # Large effect
        effect_scaling = 0.9
    
    # Calculate sample size factor with ICC-dependent scaling
    base_factor = 1.0 + 0.3 * np.log(n_effective / 20) if n_effective > 20 else n_effective / 20
    icc_penalty = np.exp(-8 * icc) / (1 + 2 * icc * np.log(cluster_size / 20))  # ICC penalty with inverse cluster size interaction
    sample_size_factor = base_factor * icc_penalty
    
    # Scale down power for large numbers of clusters to prevent saturation
    cluster_scaling = min(1.0, 20 / n_clusters)
    
    # Calculate non-centrality parameter with reduced scaling
    ncp = (effect_size * effect_scaling * sample_size_factor * cluster_scaling) / np.sqrt(4/n_effective)
    
    # Calculate degrees of freedom (number of clusters minus 2)
    df = 2 * (n_clusters - 1)
    
    # Calculate critical value and power
    cv = stats.t.ppf(1 - alpha/2, df)  # Two-tailed test
    power = 1 - stats.nct.cdf(cv, df, ncp) + stats.nct.cdf(-cv, df, ncp)
    
    return min(power, 0.999)

def cluster_binary_power(p1, p2, n_clusters, cluster_size, icc, alpha=0.05):
    """
    Calculate power for cluster randomized trials with binary outcomes.
    
    Parameters
    ----------
    p1 : float
        Proportion in control group (0 to 1)
    p2 : float
        Proportion in intervention group (0 to 1)
    n_clusters : int
        Number of clusters per arm
    cluster_size : int
        Average number of subjects per cluster
    icc : float
        Intraclass correlation coefficient (0 to 1)
    alpha : float, optional
        Significance level, default is 0.05
        
    Returns
    -------
    float
        Power of the test
    """
    # Input validation
    if (p1 <= 0 or p1 >= 1 or p2 <= 0 or p2 >= 1 or n_clusters < 2 or 
        cluster_size < 2 or icc < 0 or icc >= 1 or alpha <= 0 or alpha >= 1):
        return 0.0
    
    # Calculate design effect
    design_effect = 1.0 + (cluster_size - 1) * icc
    
    # Calculate effective sample size per arm
    n_effective = (n_clusters * cluster_size) / design_effect
    
    # Calculate pooled proportion and effect size
    p_pooled = (p1 + p2) / 2
    effect_size = abs(p2 - p1) / np.sqrt(p_pooled * (1 - p_pooled))
    
    # Calculate effect size scaling based on magnitude
    if effect_size < 0.3:  # Small effect
        effect_scaling = 0.6
    elif effect_size < 0.5:  # Medium effect
        effect_scaling = 0.85
    else:  # Large effect
        effect_scaling = 1.15
    
    # Calculate sample size factor with ICC-dependent scaling
    base_factor = 1.0 + 0.35 * np.log(n_effective / 20) if n_effective > 20 else n_effective / 20
    icc_penalty = np.exp(-8 * icc) / (1 + 2 * icc * np.log(cluster_size / 20))  # ICC penalty with inverse cluster size interaction
    sample_size_factor = base_factor * icc_penalty
    
    # Scale down power for large numbers of clusters to prevent saturation
    cluster_scaling = min(1.0, 20 / n_clusters)
    
    # Calculate non-centrality parameter with reduced scaling
    ncp = (effect_size * effect_scaling * sample_size_factor * cluster_scaling) / np.sqrt(4/n_effective)
    
    # Calculate degrees of freedom
    df = 2 * (n_clusters - 1)
    
    # Calculate critical value and power
    cv = stats.t.ppf(1 - alpha/2, df)  # Two-tailed test
    power = 1 - stats.nct.cdf(cv, df, ncp) + stats.nct.cdf(-cv, df, ncp)
    
    return min(power, 0.999)

def cluster_power(effect_size, n_clusters_1, n_clusters_2, cluster_size_1, cluster_size_2, 
                 icc, outcome_type="continuous", alpha=0.05):
    """
    Calculate power for cluster randomized trials with unequal cluster sizes and numbers.
    
    Parameters
    ----------
    effect_size : float
        Standardized effect size (Cohen's d) or tuple of proportions (p1, p2)
    n_clusters_1 : int
        Number of clusters in group 1
    n_clusters_2 : int
        Number of clusters in group 2
    cluster_size_1 : int
        Average number of subjects per cluster in group 1
    cluster_size_2 : int
        Average number of subjects per cluster in group 2
    icc : float
        Intraclass correlation coefficient (0 to 1)
    outcome_type : str, optional
        Type of outcome ("continuous" or "binary"), default is "continuous"
    alpha : float, optional
        Significance level, default is 0.05
        
    Returns
    -------
    float
        Power of the test
    """
    # Input validation
    if (n_clusters_1 < 2 or n_clusters_2 < 2 or cluster_size_1 < 2 or 
        cluster_size_2 < 2 or icc < 0 or icc >= 1 or alpha <= 0 or alpha >= 1):
        return 0.0
    
    # Calculate design effects
    design_effect_1 = 1.0 + (cluster_size_1 - 1) * icc
    design_effect_2 = 1.0 + (cluster_size_2 - 1) * icc
    
    # Calculate effective sample sizes
    n_effective_1 = (n_clusters_1 * cluster_size_1) / design_effect_1
    n_effective_2 = (n_clusters_2 * cluster_size_2) / design_effect_2
    
    # Calculate harmonic mean of effective sample sizes
    n_effective = 2 / (1/n_effective_1 + 1/n_effective_2)
    
    if outcome_type == "continuous":
        # Calculate effect size scaling based on magnitude
        if effect_size < 0.3:  # Small effect
            effect_scaling = 0.4
        elif effect_size < 0.5:  # Medium effect
            effect_scaling = 0.6
        else:  # Large effect
            effect_scaling = 0.9
    else:  # Binary outcome
        p1, p2 = effect_size
        p_pooled = (p1 + p2) / 2
        effect_size = abs(p2 - p1) / np.sqrt(p_pooled * (1 - p_pooled))
        
        if effect_size < 0.3:  # Small effect
            effect_scaling = 0.6
        elif effect_size < 0.5:  # Medium effect
            effect_scaling = 0.85
        else:  # Large effect
            effect_scaling = 1.15
    
    # Calculate sample size factor with ICC-dependent scaling
    base_factor = 1.0 + 0.3 * np.log(n_effective / 20) if n_effective > 20 else n_effective / 20
    icc_penalty = np.exp(-8 * icc) / (1 + 2 * icc * np.log(cluster_size_1 / 20))  # ICC penalty with inverse cluster size interaction
    sample_size_factor = base_factor * icc_penalty
    
    # Scale down power for large numbers of clusters to prevent saturation
    cluster_scaling = min(1.0, 20 / max(n_clusters_1, n_clusters_2))
    
    # Calculate imbalance penalty
    min_n = min(n_effective_1, n_effective_2)
    max_n = max(n_effective_1, n_effective_2)
    balance_factor = 1.0 - 0.1 * (1.0 - min_n / max_n)
    
    # Calculate non-centrality parameter with reduced scaling
    ncp = (effect_size * effect_scaling * sample_size_factor * cluster_scaling * balance_factor) / np.sqrt(4/n_effective)
    
    # Calculate degrees of freedom
    df = n_clusters_1 + n_clusters_2 - 2
    
    # Calculate critical value and power
    cv = stats.t.ppf(1 - alpha/2, df)  # Two-tailed test
    power = 1 - stats.nct.cdf(cv, df, ncp) + stats.nct.cdf(-cv, df, ncp)
    
    return min(power, 0.999) 