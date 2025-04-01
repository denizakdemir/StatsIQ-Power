"""
Utility functions for calculating various effect size measures.
"""

import numpy as np
import warnings

def calculate_cohens_d_one_sample(sample_mean, pop_mean, sample_stddev):
    """
    Calculates Cohen's d for a one-sample t-test.

    d = (sample_mean - pop_mean) / sample_stddev

    Args:
        sample_mean (float): The mean of the sample.
        pop_mean (float): The population mean under the null hypothesis.
        sample_stddev (float): The standard deviation of the sample.

    Returns:
        float: Cohen's d.
    """
    if sample_stddev <= 0:
        raise ValueError("Sample standard deviation must be positive.")
    return (sample_mean - pop_mean) / sample_stddev

def calculate_cohens_d_independent(mean1, stddev1, n1, mean2, stddev2, n2, pooled=True):
    """
    Calculates Cohen's d for an independent two-sample t-test.

    By default, uses the pooled standard deviation:
    pooled_sd = sqrt(((n1 - 1) * sd1^2 + (n2 - 1) * sd2^2) / (n1 + n2 - 2))
    d = (mean1 - mean2) / pooled_sd

    If pooled=False, uses the standard deviation of the second group (often the control group)
    or sometimes the average standard deviation (not implemented here). Using sd2 is common.
    d = (mean1 - mean2) / sd2

    Args:
        mean1 (float): Mean of the first group.
        stddev1 (float): Standard deviation of the first group.
        n1 (int): Sample size of the first group.
        mean2 (float): Mean of the second group.
        stddev2 (float): Standard deviation of the second group.
        n2 (int): Sample size of the second group.
        pooled (bool): If True (default), use pooled standard deviation.
                       If False, use standard deviation of the second group (stddev2).

    Returns:
        float: Cohen's d.
    """
    if stddev1 <= 0 or stddev2 <= 0:
        raise ValueError("Standard deviations must be positive.")
    if n1 < 1 or n2 < 1:
         raise ValueError("Sample sizes must be at least 1.") # Need n1+n2 > 2 for pooled

    diff_means = mean1 - mean2

    if pooled:
        if n1 + n2 <= 2:
             raise ValueError("Total sample size must be greater than 2 for pooled standard deviation.")
        # Calculate pooled standard deviation
        pooled_var = ((n1 - 1) * stddev1**2 + (n2 - 1) * stddev2**2) / (n1 + n2 - 2)
        if pooled_var <= 0:
             # Should not happen if stddevs are positive, but as safety check
             warnings.warn("Calculated pooled variance is non-positive. Check inputs.")
             return np.nan # Or raise error?
        pooled_sd = np.sqrt(pooled_var)
        if pooled_sd == 0: return np.inf * np.sign(diff_means) # Avoid division by zero
        cohens_d = diff_means / pooled_sd
    else:
        # Use stddev2 as the denominator (common alternative)
        if stddev2 == 0: return np.inf * np.sign(diff_means)
        cohens_d = diff_means / stddev2

    return cohens_d

def calculate_cohens_d_paired(mean_diff, stddev_diff):
    """
    Calculates Cohen's d for a paired samples t-test.

    d = mean_difference / stddev_difference

    Args:
        mean_diff (float): The mean of the differences between paired observations.
        stddev_diff (float): The standard deviation of the differences.

    Returns:
        float: Cohen's d (often denoted dz).
    """
    if stddev_diff <= 0:
        raise ValueError("Standard deviation of differences must be positive.")
    return mean_diff / stddev_diff


def calculate_hedges_g(mean1, stddev1, n1, mean2, stddev2, n2):
    """
    Calculates Hedges' g for an independent two-sample t-test.

    Hedges' g is a correction to Cohen's d for small sample bias.
    g = d * J
    where J is the correction factor, approximated by J ≈ 1 - (3 / (4*df - 1))
    and df = n1 + n2 - 2.

    Args:
        mean1 (float): Mean of the first group.
        stddev1 (float): Standard deviation of the first group.
        n1 (int): Sample size of the first group.
        mean2 (float): Mean of the second group.
        stddev2 (float): Standard deviation of the second group.
        n2 (int): Sample size of the second group.

    Returns:
        float: Hedges' g.
    """
    # Calculate Cohen's d first (using pooled SD)
    cohens_d = calculate_cohens_d_independent(mean1, stddev1, n1, mean2, stddev2, n2, pooled=True)

    # Calculate degrees of freedom
    df = n1 + n2 - 2
    if df <= 1: # Correction factor approximation requires df > 1/4
         warnings.warn(f"Degrees of freedom ({df}) is too low for reliable Hedges' g correction factor. Returning uncorrected Cohen's d.")
         return cohens_d

    # Calculate correction factor J (approximation)
    correction_factor_j = 1 - (3 / (4 * df - 1))

    # Apply correction
    hedges_g = cohens_d * correction_factor_j
    return hedges_g


def calculate_cohens_h(prop1, prop2):
    """
    Calculates Cohen's h effect size for proportions.

    h = 2 * (arcsin(sqrt(prop1)) - arcsin(sqrt(prop2)))

    Args:
        prop1 (float): First proportion (between 0 and 1).
        prop2 (float): Second proportion (between 0 and 1).

    Returns:
        float: Cohen's h.
    """
    if not (0 <= prop1 <= 1 and 0 <= prop2 <= 1):
        raise ValueError("Proportions must be between 0 and 1.")

    # Ensure inputs are not exactly 0 or 1 for arcsin domain if needed,
    # although numpy handles endpoints.
    # Use np.clip just in case for robustness, though arcsin handles 0 and 1.
    # prop1_clipped = np.clip(prop1, 1e-9, 1 - 1e-9)
    # prop2_clipped = np.clip(prop2, 1e-9, 1 - 1e-9)

    h = 2 * (np.arcsin(np.sqrt(prop1)) - np.arcsin(np.sqrt(prop2)))
    return h

# --- Effect Sizes for ANOVA ---

def calculate_cohens_f_from_eta_sq(eta_squared):
    """
    Calculates Cohen's f from eta-squared (η²).

    f = sqrt(η² / (1 - η²))
    Used for ANOVA power analysis.

    Args:
        eta_squared (float): Eta-squared value (proportion of variance explained by groups).
                             Must be between 0 and 1 (exclusive of 1).

    Returns:
        float: Cohen's f.
    """
    if not (0 <= eta_squared < 1):
        raise ValueError("Eta-squared must be between 0 (inclusive) and 1 (exclusive).")
    if eta_squared == 0:
        return 0.0
    return np.sqrt(eta_squared / (1 - eta_squared))

# --- Effect Sizes for Chi-Square ---

def calculate_cohens_w(p_observed, p_expected):
    """
    Calculates Cohen's w effect size for Chi-Square tests (GoF or Independence).

    w = sqrt( SUM[ (P_observed_i - P_expected_i)^2 / P_expected_i ] )
    where P are proportions/probabilities summing to 1.

    Args:
        p_observed (list or np.array): Array of observed proportions under H1. Must sum to 1.
        p_expected (list or np.array): Array of expected proportions under H0. Must sum to 1.
                                       Must have the same length as p_observed.

    Returns:
        float: Cohen's w.
    """
    p_obs = np.asarray(p_observed)
    p_exp = np.asarray(p_expected)

    if not np.isclose(np.sum(p_obs), 1.0):
        raise ValueError(f"Observed proportions must sum to 1 (sums to {np.sum(p_obs):.4f}).")
    if not np.isclose(np.sum(p_exp), 1.0):
        raise ValueError(f"Expected proportions must sum to 1 (sums to {np.sum(p_exp):.4f}).")
    if len(p_obs) != len(p_exp):
        raise ValueError("Observed and expected proportion arrays must have the same length.")
    if np.any(p_exp <= 0):
        raise ValueError("Expected proportions must all be positive.")
    if np.any(p_obs < 0) or np.any(p_exp < 0):
         raise ValueError("Proportions cannot be negative.") # Already covered by sum=1 and p_exp>0

    w_squared = np.sum(((p_obs - p_exp)**2) / p_exp)
    return np.sqrt(w_squared)


# --- Effect Sizes for 2x2 Tables ---

def calculate_odds_ratio(a, b, c, d):
    """
    Calculates the Odds Ratio (OR) from a 2x2 contingency table.

    Table format:
        Outcome=1 | Outcome=0
    Group=1 |    a    |    b
    Group=0 |    c    |    d

    OR = (a*d) / (b*c)

    Args:
        a (int): Count in cell [Group 1, Outcome 1].
        b (int): Count in cell [Group 1, Outcome 0].
        c (int): Count in cell [Group 0, Outcome 1].
        d (int): Count in cell [Group 0, Outcome 0].

    Returns:
        float: Odds Ratio. Returns np.inf if b or c is 0, np.nan if division by zero occurs elsewhere.
    """
    if any(count < 0 for count in [a, b, c, d]):
        raise ValueError("Counts (a, b, c, d) cannot be negative.")

    if b == 0 or c == 0:
        # Handle division by zero in the denominator (b*c)
        warnings.warn("Division by zero (b or c is 0). Odds Ratio is undefined or infinite.")
        # OR is infinite if a or d is non-zero, otherwise undefined (0/0)
        if a > 0 or d > 0:
            return np.inf
        else:
            return np.nan # Case where b=c=0, potentially a=d=0 too

    try:
        odds_ratio = (a * d) / (b * c)
        return odds_ratio
    except ZeroDivisionError:
         # This case should be caught above, but as safety
         warnings.warn("Unexpected division by zero calculating Odds Ratio.")
         return np.nan


def calculate_risk_ratio(a, b, c, d):
    """
    Calculates the Risk Ratio (RR) or Relative Risk from a 2x2 contingency table.

    Table format:
        Outcome=1 | Outcome=0
    Group=1 |    a    |    b
    Group=0 |    c    |    d

    RR = Risk_Group1 / Risk_Group0 = [a / (a+b)] / [c / (c+d)]

    Args:
        a (int): Count in cell [Group 1, Outcome 1].
        b (int): Count in cell [Group 1, Outcome 0].
        c (int): Count in cell [Group 0, Outcome 1].
        d (int): Count in cell [Group 0, Outcome 0].

    Returns:
        float: Risk Ratio. Returns np.inf or np.nan if division by zero occurs.
    """
    if any(count < 0 for count in [a, b, c, d]):
        raise ValueError("Counts (a, b, c, d) cannot be negative.")

    risk_group1_denom = a + b
    risk_group0_denom = c + d

    if risk_group1_denom == 0 or risk_group0_denom == 0:
         warnings.warn("Cannot calculate Risk Ratio: total count in one or both groups is zero.")
         return np.nan

    risk_group1 = a / risk_group1_denom
    risk_group0 = c / risk_group0_denom

    if risk_group0 == 0:
        # Handle division by zero (Risk in Group 0 is zero)
        warnings.warn("Division by zero (Risk in Group 0 is 0). Risk Ratio is undefined or infinite.")
        if risk_group1 > 0:
            return np.inf
        else:
            return np.nan # Case where a=0 and c=0

    risk_ratio = risk_group1 / risk_group0
    return risk_ratio


# --- Placeholders for other effect sizes ---

def calculate_cohens_f_anova(*args, **kwargs):
     """Placeholder for calculating Cohen's f for ANOVA (e.g., from means/SDs)."""
     # This is more complex, requiring calculation of SS_between and SS_within/total
     raise NotImplementedError("Cohen's f calculation for ANOVA from means/SDs not yet implemented.")

def calculate_r_from_d(cohens_d, n1, n2):
     """
     Converts Cohen's d (from independent samples) to Pearson's correlation coefficient r.

     Uses the formula: r = d / sqrt(d^2 + a)
     where a = (n1 + n2)^2 / (n1 * n2)  (for unequal n)
           a = 4                         (if n1 = n2)

     Args:
         cohens_d (float): Cohen's d value.
         n1 (int): Sample size of the first group.
         n2 (int): Sample size of the second group.

     Returns:
         float: Pearson's r.
     """
     if n1 < 1 or n2 < 1:
         raise ValueError("Sample sizes n1 and n2 must be at least 1.")

     # Calculate 'a' term
     if n1 == n2:
         a = 4.0
     else:
         a = (n1 + n2)**2 / (n1 * n2)

     # Calculate r
     r = cohens_d / np.sqrt(cohens_d**2 + a)
     return r

def calculate_d_from_r(r, n1, n2):
     """
     Converts Pearson's correlation coefficient r to Cohen's d (for independent samples).

     Uses the formula: d = r * sqrt(a) / sqrt(1 - r^2)
     where a = (n1 + n2)^2 / (n1 * n2)  (for unequal n)
           a = 4                         (if n1 = n2)

     Args:
         r (float): Pearson's correlation coefficient r (between -1 and 1).
         n1 (int): Sample size of the first group.
         n2 (int): Sample size of the second group.

     Returns:
         float: Cohen's d.
     """
     if not (-1 < r < 1):
         # If r = +/- 1, d is infinite
         raise ValueError("Correlation coefficient r must be between -1 and 1 (exclusive).")
     if n1 < 1 or n2 < 1:
         raise ValueError("Sample sizes n1 and n2 must be at least 1.")

     # Calculate 'a' term
     if n1 == n2:
         a = 4.0
     else:
         a = (n1 + n2)**2 / (n1 * n2)

     # Calculate d
     d = r * np.sqrt(a) / np.sqrt(1 - r**2)
     return d
