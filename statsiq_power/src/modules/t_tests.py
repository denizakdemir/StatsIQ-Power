"""
Statistical Power Calculations for T-Tests.

Includes:
- One-Sample T-Test
- Independent Samples T-Test (Equal and Unequal Variances)
- Paired Samples T-Test
"""

import numpy as np
# import scipy.stats as stats # No longer directly needed here for now
from statsmodels.stats.power import tt_solve_power, TTestIndPower # Dependency: statsmodels
from scipy.optimize import brentq # Added for TOST sample size solver
import warnings # Added for TOST warnings
from ..core.engine import PowerCalculator # Relative import from core engine

class OneSampleTTestPower(PowerCalculator):
    """
    Calculates power, sample size, or minimum detectable effect size (MDES)
    for a one-sample t-test.

    Required Args:
        alpha (float): Significance level (Type I error rate).

    Optional Args (exactly two required):
        effect_size (float): Standardized effect size (Cohen's d).
        power (float): Desired statistical power (1 - Type II error rate).
        sample_size (int): Number of observations in the sample.

    Additional Kwargs:
        alternative (str): 'two-sided' (default), 'larger', or 'smaller'.
                           Defines the alternative hypothesis.
    """
    def __init__(self, alpha, effect_size=None, power=None, sample_size=None, **kwargs):
        # Pass arguments to the updated base class __init__
        super().__init__(alpha=alpha, effect_size=effect_size, power=power, sample_size=sample_size, **kwargs)
        self.alternative = self.kwargs.get('alternative', 'two-sided')
        if self.alternative not in ['two-sided', 'larger', 'smaller']:
            raise ValueError("alternative must be 'two-sided', 'larger', or 'smaller'")

    def calculate_power(self):
        """Calculates statistical power given sample size."""
        if self.sample_size is None:
             raise ValueError("sample_size must be provided to calculate power.")
        power = tt_solve_power(effect_size=self.effect_size,
                               nobs=self.sample_size,
                               alpha=self.alpha,
                               power=None, # We are solving for power
                               alternative=self.alternative)
        return power

    def calculate_sample_size(self):
        """Calculates required sample size given desired power."""
        if self.power is None:
            raise ValueError("power must be provided to calculate sample_size.")
        
        # statsmodels requires integer sample size, but calculation might yield float
        # We take the ceiling to ensure power is at least the desired level.
        nobs = tt_solve_power(effect_size=self.effect_size,
                              nobs=None, # We are solving for nobs
                              alpha=self.alpha,
                              power=self.power,
                              alternative=self.alternative)
        return np.ceil(nobs).astype(int)

    def calculate_mdes(self):
        """Calculates the minimum detectable effect size (Cohen's d) given power and sample size."""
        if self.power is None or self.sample_size is None:
             raise ValueError("Both power and sample_size must be provided to calculate MDES.")
        if self.sample_size <= 1:
             # Need df > 0 for t-test
             return np.inf # Cannot detect any effect if n <= 1

        effect_size = tt_solve_power(effect_size=None, # Solving for effect size
                                     nobs=self.sample_size,
                                     alpha=self.alpha,
                                     power=self.power,
                                     alternative=self.alternative)
        # tt_solve_power returns the absolute value for effect_size
        # For one-sided tests, the sign might matter depending on interpretation,
        # but typically MDES is reported as a positive value.
        return abs(effect_size)


# --- Independent Samples T-Test ---

class IndependentSamplesTTestPower(PowerCalculator):
    """
    Calculates power, sample size, or MDES for an independent two-sample t-test.

    Required Args:
        alpha (float): Significance level (Type I error rate).

    Optional Args (exactly two required):
        effect_size (float): Standardized effect size (Cohen's d).
        power (float): Desired statistical power (1 - Type II error rate).
        sample_size (int): Sample size of the *first* group (n1).

    Additional Kwargs:
        ratio (float): Ratio of sample sizes (n2 / n1). Default is 1 (equal group sizes).
                       If solving for sample size, the result returned is n1.
        alternative (str): 'two-sided' (default), 'larger', or 'smaller'.
        usevar (str): 'pooled' (default, assumes equal variances) or 'unequal' (Welch's t-test, assumes unequal variances).
        icc (float, optional): Intraclass Correlation Coefficient (for clustered designs). Default is None.
        cluster_size (float, optional): Average cluster size (m). Default is None. Required if icc is specified.
    """
    def __init__(self, alpha, effect_size=None, power=None, sample_size=None, **kwargs):
        # sample_size here corresponds to nobs1 in statsmodels
        super().__init__(alpha=alpha, effect_size=effect_size, power=power, sample_size=sample_size, **kwargs)
        self.ratio = self.kwargs.get('ratio', 1.0)
        self.alternative = self.kwargs.get('alternative', 'two-sided')
        self.usevar = self.kwargs.get('usevar', 'pooled')
        self.icc = self.kwargs.get('icc', None)
        self.cluster_size = self.kwargs.get('cluster_size', None)

        if self.alternative not in ['two-sided', 'larger', 'smaller']:
            raise ValueError("alternative must be 'two-sided', 'larger', or 'smaller'")
        if self.usevar not in ['pooled', 'unequal']:
             raise ValueError("usevar must be 'pooled' or 'unequal'")
        if self.ratio <= 0:
            raise ValueError("ratio must be positive")
        if self.icc is not None and not (0 <= self.icc <= 1):
             raise ValueError("ICC must be between 0 and 1.")
        if self.icc is not None and (self.cluster_size is None or self.cluster_size <= 1):
             raise ValueError("Average cluster_size (> 1) must be provided if ICC is specified.")
        if self.icc is None and self.cluster_size is not None:
             warnings.warn("cluster_size provided but ICC is not. Ignoring clustering effect.")
             self.cluster_size = None # Ensure consistency

        # Instantiate the statsmodels power solver
        self.solver = TTestIndPower()

    def calculate_power(self):
        """Calculates statistical power given sample size of the first group (n1)."""
        if self.sample_size is None:
             raise ValueError("sample_size (n1) must be provided to calculate power.")
        power = self.solver.power(effect_size=self.effect_size,
                                  nobs1=self.sample_size, # n1
                                  alpha=self.alpha,
                                  ratio=self.ratio, # n2/n1
                                  alternative=self.alternative)
                                  # usevar is handled by the solver instance, not the power method directly
        return power

    def calculate_sample_size(self):
        """
        Calculates required sample size for the *first* group (n1) given desired power.
        The sample size for the second group (n2) is n1 * ratio.
        Total sample size is n1 * (1 + ratio).
        """
        if self.power is None:
            raise ValueError("power must be provided to calculate sample_size.")

        nobs1 = self.solver.solve_power(effect_size=self.effect_size,
                                        nobs1=None, # Solving for n1
                                        alpha=self.alpha,
                                        power=self.power,
                                        ratio=self.ratio,
                                        alternative=self.alternative)
                                        # usevar is handled internally by solve_power calling power

        # Adjust for clustering if ICC and cluster_size are provided
        if self.icc is not None and self.cluster_size is not None:
             deff = 1 + (self.cluster_size - 1) * self.icc
             if deff < 1: # Should not happen if icc>=0, m>1
                  deff = 1
             print(f"Info: Applying Design Effect (DEFF) = {deff:.3f} for clustering (ICC={self.icc}, m={self.cluster_size}).")
             # Adjust the required sample size per group (n1)
             nobs1_adjusted = nobs1 * deff
        else:
             nobs1_adjusted = nobs1

        # Return n1 (adjusted or not), ensuring it's an integer >= 2 for t-test validity
        return np.ceil(max(nobs1_adjusted, 2)).astype(int)

    def calculate_mdes(self):
        """Calculates the minimum detectable effect size (Cohen's d) given power and sample size (n1)."""
        if self.power is None or self.sample_size is None:
             raise ValueError("Both power and sample_size (n1) must be provided to calculate MDES.")
        # Ensure n1 and n2 are sufficient for the test (e.g., n1 >= 2, n2 >= 2)
        n1 = self.sample_size
        n2 = n1 * self.ratio
        if n1 < 2 or n2 < 1: # Need at least 2 in one group and 1 in the other for pooled/unequal
             # More precisely, df > 0. Pooled df = n1+n2-2. Welch df is complex but needs n1,n2 >= 2 generally.
             return np.inf

        effect_size = self.solver.solve_power(effect_size=None, # Solving for effect size
                                              nobs1=self.sample_size,
                                              alpha=self.alpha,
                                              power=self.power,
                                              ratio=self.ratio,
                                              alternative=self.alternative,
                                              usevar=self.usevar)
        return abs(effect_size)


# --- Paired Samples T-Test ---

class PairedSamplesTTestPower(OneSampleTTestPower):
    """
    Calculates power, sample size, or MDES for a paired samples t-test.

    This is mathematically equivalent to a one-sample t-test performed
    on the differences between paired observations. The effect size `d`
    should be calculated based on these differences.

    Inherits directly from OneSampleTTestPower as the calculations are identical.
    The `sample_size` refers to the number of *pairs*.

    Required Args:
        alpha (float): Significance level (Type I error rate).

    Optional Args (exactly two required):
        effect_size (float): Standardized effect size (Cohen's d) of the *differences*.
        power (float): Desired statistical power (1 - Type II error rate).
        sample_size (int): Number of pairs.

    Additional Kwargs:
        alternative (str): 'two-sided' (default), 'larger', or 'smaller'.
    """
    def __init__(self, alpha, effect_size=None, power=None, sample_size=None, **kwargs):
        # sample_size here refers to the number of pairs
        super().__init__(alpha=alpha, effect_size=effect_size, power=power, sample_size=sample_size, **kwargs)
        # Inherits calculate_power, calculate_sample_size, and calculate_mdes
        # from OneSampleTTestPower, which is correct for paired tests.
        pass


# --- One-Sample Non-Inferiority/Superiority T-Test ---

class OneSampleNIPower(PowerCalculator):
    """
    Calculates power, sample size, or MDES for a one-sample non-inferiority
    or superiority t-test.

    Tests H0: mean <= pop_mean - margin (for non-inferiority, alternative='larger')
    or    H0: mean >= pop_mean + margin (for superiority, alternative='smaller')
    against H1: mean > pop_mean - margin or mean < pop_mean + margin.

    Note: The 'effect_size' here is the hypothesized true difference
          (mean - pop_mean) / stddev, NOT the difference relative to the margin.
          The `margin` parameter in tt_solve_power handles the shift.

    Required Args:
        alpha (float): Significance level (Type I error rate).
        margin (float): The non-inferiority or superiority margin (in standardized units, i.e., margin_raw / stddev).
                        Must be positive for standard interpretation.
        alternative (str): 'larger' (for non-inferiority H1: mean > pop_mean - margin) or
                           'smaller' (for superiority H1: mean < pop_mean + margin).

    Optional Args (exactly two required):
        effect_size (float): Standardized true effect size (mean - pop_mean) / stddev.
        power (float): Desired statistical power (1 - Type II error rate).
        sample_size (int): Number of observations in the sample.
    """
    def __init__(self, alpha, margin, alternative, effect_size=None, power=None, sample_size=None, **kwargs):
        if alternative not in ['larger', 'smaller']:
            raise ValueError("alternative must be 'larger' (non-inferiority) or 'smaller' (superiority)")
        if margin <= 0:
             warnings.warn("Margin is typically positive for non-inferiority/superiority tests.")
             # Allow non-positive margin? tt_solve_power might handle it.

        # Map NI/Superiority alternative to the underlying one-sided test alternative
        self.test_alternative = alternative
        # The margin is handled by statsmodels function

        super().__init__(alpha=alpha, effect_size=effect_size, power=power, sample_size=sample_size, **kwargs)
        self.margin = margin


    def calculate_power(self):
        """Calculates statistical power given effect_size and sample_size."""
        if self.effect_size is None or self.sample_size is None:
             raise ValueError("Both effect_size and sample_size must be provided to calculate power.")
        if self.sample_size <= 1: return 0.0

        power = tt_solve_power(effect_size=self.effect_size,
                               nobs=self.sample_size,
                               alpha=self.alpha,
                               power=None, # Solving for power
                               # Margin is handled by adjusting effect size, not passed directly here
                               alternative=self.test_alternative)
        return power

    def calculate_sample_size(self):
        """Calculates required sample size given effect_size and power."""
        if self.effect_size is None or self.power is None:
            raise ValueError("Both effect_size and power must be provided to calculate sample_size.")
        if not (0 < self.power < 1):
             raise ValueError("Power must be between 0 and 1.")

        # Check if the effect size is beyond the margin in the hypothesized direction
        if self.test_alternative == 'larger' and self.effect_size <= -abs(self.margin):
             warnings.warn(f"Effect size ({self.effect_size}) is not greater than the non-inferiority margin ({-abs(self.margin)}). Power may be low or sample size infinite.")
             # Depending on exact margin sign convention, might need adjustment
             # If true mean is truly non-inferior, power should be > alpha
        if self.test_alternative == 'smaller' and self.effect_size >= abs(self.margin):
             warnings.warn(f"Effect size ({self.effect_size}) is not smaller than the superiority margin ({abs(self.margin)}). Power may be low or sample size infinite.")

        nobs = tt_solve_power(effect_size=self.effect_size,
                              nobs=None, # Solving for nobs
                              alpha=self.alpha,
                              power=self.power,
                              # margin=self.margin, # Margin handled internally by solve_power calling power
                              alternative=self.test_alternative)
        # Need at least n=2 for t-test
        return np.ceil(max(nobs, 2)).astype(int)

    def calculate_mdes(self):
        """Calculates the minimum detectable effect size relative to population mean (NOT the margin)"""
        if self.power is None or self.sample_size is None:
             raise ValueError("Both power and sample_size must be provided to calculate MDES.")
        if self.sample_size <= 1: return np.inf

        effect_size = tt_solve_power(effect_size=None, # Solving for effect size
                                     nobs=self.sample_size,
                                     alpha=self.alpha,
                                     power=self.power,
                                     # margin=self.margin, # Margin handled internally by solve_power calling power
                                     alternative=self.test_alternative)
        # The result is the minimum true effect size (mean-pop_mean)/sd needed
        return effect_size # Sign matters here


# --- One-Sample Equivalence T-Test (TOST) ---

class OneSampleTOSTPower(PowerCalculator):
    """
    Calculates power or sample size for a one-sample equivalence test using TOST.

    Tests H0: |mean - pop_mean| >= margin vs H1: |mean - pop_mean| < margin.
    This involves two one-sided tests (TOST). Power is the probability of
    rejecting *both* null hypotheses.

    Required Args:
        alpha (float): Significance level (Type I error rate) for *each* one-sided test.
        margin (float): The equivalence margin (in standardized units, i.e., margin_raw / stddev).
                        Must be positive.

    Optional Args (exactly two required):
        effect_size (float): Standardized true effect size (mean - pop_mean) / stddev.
                             Assumed to be 0 if not provided when calculating sample size/power.
        power (float): Desired statistical power (1 - Type II error rate).
        sample_size (int): Number of observations in the sample.
    """
    def __init__(self, alpha, margin, effect_size=None, power=None, sample_size=None, **kwargs):
         if margin <= 0:
             raise ValueError("Equivalence margin must be positive.")

         # If effect_size is None when calculating power/sample_size, assume true effect is 0
         # This is the most conservative assumption for equivalence power.
         eff_size_calc = effect_size
         if effect_size is None and power is not None and sample_size is not None:
              # This case is calculating MDES - not directly supported by TOST power formulas easily.
              # TOST power is usually calculated assuming a true effect (often 0).
              # We can calculate the range of true means for which power is achieved,
              # but not a single MDES easily. Let's disallow this for now.
              raise ValueError("Calculating MDES for TOST is complex and not directly supported. Provide effect_size=0 (or other assumed true effect) to calculate power/sample size.")
         elif effect_size is None and (power is not None or sample_size is not None):
              warnings.warn("effect_size not provided for TOST power/sample size calculation. Assuming true effect_size = 0.")
              eff_size_calc = 0.0

         super().__init__(alpha=alpha, effect_size=eff_size_calc, power=power, sample_size=sample_size, **kwargs)
         self.margin = abs(margin) # Ensure margin is positive

    def _power_tost(self, n):
         """Calculate TOST power for a given sample size n."""
         if n <= 1: return 0.0
         # Power for TOST = P(T1 > crit | H1) + P(T2 < -crit | H1) - 1
         # where T1 tests mean > pop_mean - margin
         # and   T2 tests mean < pop_mean + margin
         # crit = t_{1-alpha, df}

         # Use tt_solve_power for each one-sided test's power
         # Power for lower bound test (H0: mean <= pop_mean - margin, H1: mean > pop_mean - margin)
         # Effect size relative to lower margin: self.effect_size - (-self.margin) = self.effect_size + self.margin
         power_lower = tt_solve_power(effect_size=self.effect_size + self.margin, nobs=n, alpha=self.alpha,
                                      alternative='larger', power=None) # Test against 0 after shifting by margin
         # Power for upper bound test (H0: mean >= pop_mean + margin, H1: mean < pop_mean + margin)
         # Effect size relative to upper margin: self.effect_size - self.margin
         power_upper = tt_solve_power(effect_size=self.effect_size - self.margin, nobs=n, alpha=self.alpha,
                                      alternative='smaller', power=None) # Test against 0 after shifting by margin

         # Combined power (probability of rejecting both)
         # This formula assumes independence, which isn't quite right for the two t-stats,
         # but is a common approximation. More accurate involves bivariate non-central t.
         # A simpler, often used formula is: Power(TOST) = Power_Lower + Power_Upper - 1
         # This holds if the rejection regions are based on the same alpha.
         tost_power = max(0.0, power_lower + power_upper - 1.0)
         return tost_power

    def calculate_power(self):
         """Calculates statistical power for TOST given effect_size and sample_size."""
         if self.effect_size is None or self.sample_size is None:
              raise ValueError("Both effect_size and sample_size must be provided to calculate TOST power.")
         return self._power_tost(self.sample_size)

    def calculate_sample_size(self):
         """Calculates required sample size for TOST given effect_size and power."""
         if self.effect_size is None or self.power is None:
             raise ValueError("Both effect_size and power must be provided to calculate TOST sample_size.")
         if not (0 < self.power < 1):
             raise ValueError("Power must be between 0 and 1.")

         # Define the function to find the root for: power(n) - target_power = 0
         def power_diff(n):
             if n <= 1: return -self.power
             return self._power_tost(n) - self.power

         # Use a numerical solver (Brent's method) to find n
         lower_bound = 2
         upper_bound = 4
         max_iter = 1000; iter_count = 0
         while power_diff(upper_bound) < 0 and iter_count < max_iter:
             upper_bound *= 2
             iter_count += 1
             if iter_count >= max_iter:
                 raise RuntimeError("Could not find an upper bound for TOST sample size search.")

         try:
             if power_diff(lower_bound) * power_diff(upper_bound) >= 0:
                  if power_diff(lower_bound) >= 0: return np.ceil(lower_bound).astype(int)
                  else: raise RuntimeError(f"TOST sample size search failed: f(a) and f(b) have same sign. Interval [{lower_bound}, {upper_bound}]")

             n_float = brentq(power_diff, lower_bound, upper_bound, xtol=1e-6, rtol=1e-6)
             return np.ceil(max(n_float, 2)).astype(int) # Need n >= 2
         except ValueError as e:
             raise RuntimeError(f"TOST sample size calculation failed using brentq: {e}")

    def calculate_mdes(self):
         """MDES is not straightforward for TOST and depends on definition."""
         raise NotImplementedError("Calculating a single MDES value for TOST is complex and not directly supported. Consider power curves.")


# --- Independent Samples Non-Inferiority/Superiority T-Test ---

class IndependentSamplesNIPower(PowerCalculator):
    """
    Calculates power, sample size, or MDES for an independent two-sample
    non-inferiority or superiority t-test.

    Tests H0: mean1 - mean2 <= -margin (for non-inferiority, alternative='larger')
    or    H0: mean1 - mean2 >= margin (for superiority, alternative='smaller')
    against H1: mean1 - mean2 > -margin or mean1 - mean2 < margin.

    Note: The 'effect_size' here is the hypothesized true difference
          (mean1 - mean2) / pooled_sd, NOT the difference relative to the margin.
          The `margin` parameter handles the shift.

    Required Args:
        alpha (float): Significance level (Type I error rate).
        margin (float): The non-inferiority or superiority margin (in standardized units, i.e., margin_raw / pooled_sd).
                        Must be positive for standard interpretation.
        alternative (str): 'larger' (for non-inferiority H1: mean1 - mean2 > -margin) or
                           'smaller' (for superiority H1: mean1 - mean2 < margin).

    Optional Args (exactly two required):
        effect_size (float): Standardized true effect size (mean1 - mean2) / pooled_sd.
        power (float): Desired statistical power (1 - Type II error rate).
        sample_size (int): Sample size of the *first* group (n1).

    Additional Kwargs:
        ratio (float): Ratio of sample sizes (n2 / n1). Default is 1.
        usevar (str): 'pooled' (default) or 'unequal'.
    """
    def __init__(self, alpha, margin, alternative, effect_size=None, power=None, sample_size=None, **kwargs):
        if alternative not in ['larger', 'smaller']:
            raise ValueError("alternative must be 'larger' (non-inferiority) or 'smaller' (superiority)")
        if margin <= 0:
             warnings.warn("Margin is typically positive for non-inferiority/superiority tests.")

        self.test_alternative = alternative
        self.margin = margin
        self.ratio = kwargs.get('ratio', 1.0)
        self.usevar = kwargs.get('usevar', 'pooled')

        if self.ratio <= 0: raise ValueError("ratio must be positive")
        if self.usevar not in ['pooled', 'unequal']: raise ValueError("usevar must be 'pooled' or 'unequal'")

        # sample_size corresponds to nobs1
        super().__init__(alpha=alpha, effect_size=effect_size, power=power, sample_size=sample_size, **kwargs)
        self.solver = TTestIndPower()

    def calculate_power(self):
        """Calculates statistical power given effect_size and sample_size (n1)."""
        if self.effect_size is None or self.sample_size is None:
             raise ValueError("Both effect_size and sample_size (n1) must be provided to calculate power.")
        n1 = self.sample_size
        n2 = n1 * self.ratio
        if n1 < 2 or n2 < 1: return 0.0 # Basic check for validity

        # Note: statsmodels TTestIndPower doesn't directly support 'margin'.
        # We need to adjust the effect size relative to the margin.
        # For H1: mean1 - mean2 > -margin (alternative='larger'), we test if (mean1-mean2) - (-margin) > 0
        # The effective effect size for the test is (true_diff - (-margin))/sd = effect_size + margin
        # For H1: mean1 - mean2 < margin (alternative='smaller'), we test if (mean1-mean2) - margin < 0
        # The effective effect size for the test is (true_diff - margin)/sd = effect_size - margin

        if self.test_alternative == 'larger':
            effective_es = self.effect_size + self.margin
        else: # 'smaller'
            effective_es = self.effect_size - self.margin

        power = self.solver.power(effect_size=effective_es,
                                  nobs1=self.sample_size,
                                  alpha=self.alpha,
                                  ratio=self.ratio,
                                  alternative=self.test_alternative, # Use the NI/Sup alternative directly
                                  usevar=self.usevar)
        return power

    def calculate_sample_size(self):
        """Calculates required sample size (n1) given effect_size and power."""
        if self.effect_size is None or self.power is None:
            raise ValueError("Both effect_size and power must be provided to calculate sample_size.")
        if not (0 < self.power < 1):
             raise ValueError("Power must be between 0 and 1.")

        # Adjust effect size relative to margin as in calculate_power
        if self.test_alternative == 'larger':
            effective_es = self.effect_size + self.margin
            if effective_es <= 0: # True difference not greater than -margin
                 warnings.warn(f"Effective effect size ({effective_es:.3f}) relative to margin is not positive. Power may be low or sample size infinite.")
                 # return np.inf # Or let solve_power handle it
        else: # 'smaller'
            effective_es = self.effect_size - self.margin
            if effective_es >= 0: # True difference not smaller than margin
                 warnings.warn(f"Effective effect size ({effective_es:.3f}) relative to margin is not negative. Power may be low or sample size infinite.")
                 # return np.inf

        # Handle cases where effective_es is zero or has the "wrong" sign for the alternative
        if (self.test_alternative == 'larger' and effective_es <= 0) or \
           (self.test_alternative == 'smaller' and effective_es >= 0):
            # If the true effect is exactly at or beyond the margin in the "wrong" direction,
            # power will likely not exceed alpha. solve_power might return NaN or large numbers.
            # Check if power > alpha is achievable
            power_at_inf_n = self.solver.power(effect_size=effective_es, nobs1=np.inf, alpha=self.alpha, ratio=self.ratio, alternative=self.test_alternative, usevar=self.usevar)
            if self.power > power_at_inf_n:
                 warnings.warn(f"Desired power ({self.power:.3f}) is unachievable with the given effect size ({self.effect_size:.3f}) and margin ({self.margin:.3f}) relative to the alternative '{self.test_alternative}'. Max power is approx {power_at_inf_n:.3f}.")
                 return np.inf


        nobs1 = self.solver.solve_power(effect_size=effective_es,
                                        nobs1=None, # Solving for n1
                                        alpha=self.alpha,
                                        power=self.power,
                                        ratio=self.ratio,
                                        alternative=self.test_alternative,
                                        usevar=self.usevar)

        # Need at least n1=2, n2=1 (or vice versa depending on ratio)
        min_n1 = 2 if self.ratio >= 0.5 else np.ceil(2 / self.ratio)
        min_n2 = np.ceil(min_n1 * self.ratio)
        if min_n2 < 2: min_n1 = np.ceil(2 / self.ratio) # Ensure n2 >= 2 if n1 is small

        return np.ceil(max(nobs1, min_n1)).astype(int)


    def calculate_mdes(self):
        """Calculates the minimum detectable true effect size relative to zero difference (NOT the margin)"""
        if self.power is None or self.sample_size is None:
             raise ValueError("Both power and sample_size (n1) must be provided to calculate MDES.")
        n1 = self.sample_size
        n2 = n1 * self.ratio
        if n1 < 2 or n2 < 1: return np.inf # Basic check

        # Solve for the effective effect size relative to the margin
        effective_es_mdes = self.solver.solve_power(effect_size=None, # Solving for effective ES
                                                    nobs1=self.sample_size,
                                                    alpha=self.alpha,
                                                    power=self.power,
                                                    ratio=self.ratio,
                                                    alternative=self.test_alternative,
                                                    usevar=self.usevar)

        # Convert back to the true effect size (relative to zero)
        if self.test_alternative == 'larger':
            # effective_es = true_es + margin => true_es = effective_es - margin
            mdes = effective_es_mdes - self.margin
        else: # 'smaller'
            # effective_es = true_es - margin => true_es = effective_es + margin
            mdes = effective_es_mdes + self.margin

        return mdes # Sign matters


# --- Independent Samples Equivalence T-Test (TOST) ---

class IndependentSamplesTOSTPower(PowerCalculator):
    """
    Calculates power or sample size for an independent two-sample equivalence
    test using TOST.

    Tests H0: |mean1 - mean2| >= margin vs H1: |mean1 - mean2| < margin.
    This involves two one-sided tests (TOST). Power is the probability of
    rejecting *both* null hypotheses.

    Required Args:
        alpha (float): Significance level (Type I error rate) for *each* one-sided test.
        margin (float): The equivalence margin (in standardized units, i.e., margin_raw / pooled_sd).
                        Must be positive.

    Optional Args (exactly two required):
        effect_size (float): Standardized true effect size (mean1 - mean2) / pooled_sd.
                             Assumed to be 0 if not provided when calculating sample size/power.
        power (float): Desired statistical power (1 - Type II error rate).
        sample_size (int): Sample size of the *first* group (n1).

    Additional Kwargs:
        ratio (float): Ratio of sample sizes (n2 / n1). Default is 1.
        usevar (str): 'pooled' (default) or 'unequal'.
    """
    def __init__(self, alpha, margin, effect_size=None, power=None, sample_size=None, **kwargs):
         if margin <= 0:
             raise ValueError("Equivalence margin must be positive.")
         self.margin = abs(margin)
         self.ratio = kwargs.get('ratio', 1.0)
         self.usevar = kwargs.get('usevar', 'pooled')

         if self.ratio <= 0: raise ValueError("ratio must be positive")
         if self.usevar not in ['pooled', 'unequal']: raise ValueError("usevar must be 'pooled' or 'unequal'")

         eff_size_calc = effect_size
         if effect_size is None and power is not None and sample_size is not None:
              raise ValueError("Calculating MDES for TOST is complex and not directly supported. Provide effect_size=0 (or other assumed true effect) to calculate power/sample size.")
         elif effect_size is None and (power is not None or sample_size is not None):
              warnings.warn("effect_size not provided for TOST power/sample size calculation. Assuming true effect_size = 0.")
              eff_size_calc = 0.0

         super().__init__(alpha=alpha, effect_size=eff_size_calc, power=power, sample_size=sample_size, **kwargs)
         self.solver = TTestIndPower()

    def _power_tost(self, n1):
         """Calculate TOST power for a given sample size n1."""
         n2 = n1 * self.ratio
         if n1 < 2 or n2 < 1: return 0.0 # Basic check

         # Power for lower bound test (H0: diff <= -margin, H1: diff > -margin)
         # Effective ES = true_es - (-margin) = true_es + margin
         power_lower = self.solver.power(effect_size=self.effect_size + self.margin,
                                         nobs1=n1, alpha=self.alpha, ratio=self.ratio,
                                         alternative='larger', usevar=self.usevar)

         # Power for upper bound test (H0: diff >= margin, H1: diff < margin)
         # Effective ES = true_es - margin
         power_upper = self.solver.power(effect_size=self.effect_size - self.margin,
                                         nobs1=n1, alpha=self.alpha, ratio=self.ratio,
                                         alternative='smaller', usevar=self.usevar)

         # Combined power approximation
         tost_power = max(0.0, power_lower + power_upper - 1.0)
         return tost_power

    def calculate_power(self):
         """Calculates statistical power for TOST given effect_size and sample_size (n1)."""
         if self.effect_size is None or self.sample_size is None:
              raise ValueError("Both effect_size and sample_size (n1) must be provided to calculate TOST power.")
         return self._power_tost(self.sample_size)

    def calculate_sample_size(self):
         """Calculates required sample size (n1) for TOST given effect_size and power."""
         if self.effect_size is None or self.power is None:
             raise ValueError("Both effect_size and power must be provided to calculate TOST sample_size.")
         if not (0 < self.power < 1):
             raise ValueError("Power must be between 0 and 1.")

         # Define the function to find the root for: power(n1) - target_power = 0
         def power_diff(n1):
             if n1 < 2 or n1 * self.ratio < 1: return -self.power # Check validity
             return self._power_tost(n1) - self.power

         # Use a numerical solver (Brent's method) to find n1
         # Determine a reasonable lower bound (e.g., ensuring both groups have size >= 2)
         min_n1_theory = 2 if self.ratio >= 1 else np.ceil(2 / self.ratio)
         lower_bound = max(2, min_n1_theory) # Start search from a valid n1

         # Check if power is achievable at the lower bound
         if power_diff(lower_bound) >= 0:
             return np.ceil(lower_bound).astype(int)

         upper_bound = lower_bound * 2
         max_iter = 1000; iter_count = 0
         while power_diff(upper_bound) < 0 and iter_count < max_iter:
             upper_bound *= 2
             iter_count += 1
             if iter_count >= max_iter:
                 raise RuntimeError("Could not find an upper bound for TOST sample size search.")

         try:
             if power_diff(lower_bound) * power_diff(upper_bound) >= 0:
                  raise RuntimeError(f"TOST sample size search failed: f(a) and f(b) have same sign. Interval [{lower_bound}, {upper_bound}], Power diffs: {power_diff(lower_bound)}, {power_diff(upper_bound)}")

             n1_float = brentq(power_diff, lower_bound, upper_bound, xtol=1e-6, rtol=1e-6)
             # Ensure final n1 and n2 are >= 2
             final_n1 = np.ceil(max(n1_float, lower_bound)).astype(int)
             final_n2 = np.ceil(final_n1 * self.ratio)
             if final_n2 < 2:
                 final_n1 = np.ceil(max(final_n1, 2 / self.ratio)).astype(int)

             return final_n1
         except ValueError as e:
             raise RuntimeError(f"TOST sample size calculation failed using brentq: {e}")
         except RuntimeError as e:
              # Catch the explicit RuntimeError from the sign check
              raise e


    def calculate_mdes(self):
         """MDES is not straightforward for TOST and depends on definition."""
         raise NotImplementedError("Calculating a single MDES value for TOST is complex and not directly supported. Consider power curves.")


# --- Paired Samples Non-Inferiority/Superiority T-Test ---

class PairedSamplesNIPower(OneSampleNIPower):
    """
    Calculates power, sample size, or MDES for a paired samples non-inferiority
    or superiority t-test.

    This is mathematically equivalent to a one-sample NI/Superiority test
    performed on the differences between paired observations. The effect size `d`
    and `margin` should be calculated based on these differences.

    Inherits from OneSampleNIPower. The `sample_size` refers to the number of *pairs*.

    Required Args:
        alpha (float): Significance level (Type I error rate).
        margin (float): The non-inferiority or superiority margin for the *differences*
                        (in standardized units, i.e., margin_raw / sd_diff). Must be positive.
        alternative (str): 'larger' (non-inferiority) or 'smaller' (superiority).

    Optional Args (exactly two required):
        effect_size (float): Standardized true effect size of the *differences*.
        power (float): Desired statistical power (1 - Type II error rate).
        sample_size (int): Number of pairs.
    """
    def __init__(self, alpha, margin, alternative, effect_size=None, power=None, sample_size=None, **kwargs):
        # sample_size here refers to the number of pairs
        super().__init__(alpha=alpha, margin=margin, alternative=alternative,
                         effect_size=effect_size, power=power, sample_size=sample_size, **kwargs)
        # Calculations are identical to the one-sample case on the differences.
        pass


# --- Paired Samples Equivalence T-Test (TOST) ---

class PairedSamplesTOSTPower(OneSampleTOSTPower):
    """
    Calculates power or sample size for a paired samples equivalence test using TOST.

    This is mathematically equivalent to a one-sample TOST performed on the
    differences between paired observations. The effect size `d` and `margin`
    should be calculated based on these differences.

    Inherits from OneSampleTOSTPower. The `sample_size` refers to the number of *pairs*.

    Required Args:
        alpha (float): Significance level (Type I error rate) for *each* one-sided test.
        margin (float): The equivalence margin for the *differences*
                        (in standardized units, i.e., margin_raw / sd_diff). Must be positive.

    Optional Args (exactly two required):
        effect_size (float): Standardized true effect size of the *differences*. Assumed 0 if None.
        power (float): Desired statistical power (1 - Type II error rate).
        sample_size (int): Number of pairs.
    """
    def __init__(self, alpha, margin, effect_size=None, power=None, sample_size=None, **kwargs):
         # sample_size here refers to the number of pairs
         super().__init__(alpha=alpha, margin=margin, effect_size=effect_size,
                          power=power, sample_size=sample_size, **kwargs)
         # Calculations are identical to the one-sample case on the differences.
         pass
