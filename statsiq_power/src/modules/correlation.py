"""
Statistical Power Calculations for Correlation Analyses.

Includes:
- Pearson Correlation
- Spearman/Kendall Rank Correlations (Placeholder)
- Partial/Semi-partial Correlation (Placeholder)
"""

import numpy as np
import scipy.stats as stats
from scipy.optimize import brentq

from ..core.engine import PowerCalculator # Relative import from core engine

class PearsonCorrelationPower(PowerCalculator):
    """
    Calculates power, sample size, or MDES for testing Pearson correlation coefficient (rho != 0).

    Uses the non-central t-distribution.

    Required Args:
        alpha (float): Significance level (Type I error rate).

    Optional Args (exactly two required):
        effect_size (float): Hypothesized population correlation coefficient (rho).
                             Must be between -1 and 1 (exclusive of -1, 1).
        power (float): Desired statistical power (1 - Type II error rate).
        sample_size (int): Number of pairs of observations (N).

    Additional Kwargs:
        alternative (str): 'two-sided' (default), 'larger' (rho > 0), or 'smaller' (rho < 0).
    """
    def __init__(self, alpha, effect_size=None, power=None, sample_size=None, **kwargs):
        # Note: effect_size is rho
        if effect_size is not None and not (-1 < effect_size < 1):
            raise ValueError("effect_size (rho) must be between -1 and 1 (exclusive).")
        # Adjust sample_size check later based on calculation needs (n > 2)
        super().__init__(alpha=alpha, effect_size=effect_size, power=power, sample_size=sample_size, **kwargs)
        self.alternative = self.kwargs.get('alternative', 'two-sided')
        if self.alternative not in ['two-sided', 'larger', 'smaller']:
            raise ValueError("alternative must be 'two-sided', 'larger', or 'smaller'")
        if self.effect_size == 0 and self.power is not None:
             print("Warning: Effect size (rho) is 0. Sample size will be infinite for power > alpha.")


    def _calculate_power_from_n(self, n):
        """Helper function to calculate power for a given sample size n."""
        if n <= 2: # Need df = n - 2 > 0
            return 0.0
        
        df = n - 2
        rho = self.effect_size

        # Calculate critical t-value(s) from the central t-distribution (H0: rho=0)
        if self.alternative == 'two-sided':
            crit_lower = stats.t.ppf(self.alpha / 2, df)
            crit_upper = stats.t.ppf(1 - self.alpha / 2, df)
        elif self.alternative == 'larger': # H1: rho > 0
            crit_lower = -np.inf
            crit_upper = stats.t.ppf(1 - self.alpha, df)
        else: # alternative == 'smaller', H1: rho < 0
            crit_lower = stats.t.ppf(self.alpha, df)
            crit_upper = np.inf

        # Calculate non-centrality parameter (delta) for the non-central t-distribution (H1: rho != 0)
        # delta = rho * sqrt(n / (1 - rho^2)) - This is for testing rho = rho_0, not rho = 0.
        # For testing rho = 0, the test statistic t = r * sqrt(df / (1 - r^2))
        # Under H1, t follows a non-central t distribution with df=n-2 and nc parameter:
        # nc = rho * sqrt(n) / sqrt(1 - rho^2) -- This seems incorrect based on some sources.
        # Let's use the approximation via Fisher's z: z = 0.5 * ln((1+r)/(1-r)), Var(z) = 1/(n-3)
        # Test statistic Z = z * sqrt(n-3) ~ N(zeta * sqrt(n-3), 1), where zeta = 0.5 * ln((1+rho)/(1-rho))
        # This uses Normal distribution, not t.

        # Let's try the direct non-central t approach as described in some stats texts/software (e.g., G*Power):
        # The test statistic t = r * sqrt(n-2) / sqrt(1-r^2)
        # Under H1, this t follows a non-central t distribution with df = n-2
        # and non-centrality parameter nc = rho * sqrt(n-1) -- No, this seems off too.
        # Let's stick to the common formula used in power calculators:
        # nc = rho * sqrt(n) -- This is often used but might be an approximation or specific context.

        # Trying the formula derived from the relationship between t and F:
        # F = t^2, F ~ non-central F(df1=1, df2=n-2, nc_F = nc_t^2)
        # nc_F = n * rho^2 / (1 - rho^2) -- This relates to regression F-test nc.
        # So, nc_t = sqrt(n * rho^2 / (1 - rho^2)) = abs(rho) * sqrt(n / (1 - rho^2))
        # Let's use this nc. Sign should match rho.
        if rho == 0:
            nc = 0.0
        else:
            nc = rho * np.sqrt(n / (1 - rho**2))

        # Calculate power using the non-central t CDF
        power = 0.0
        if self.alternative == 'two-sided':
            power = stats.nct.cdf(crit_lower, df, nc) + (1 - stats.nct.cdf(crit_upper, df, nc))
        elif self.alternative == 'larger': # H1: rho > 0. Reject if t > crit_upper
             # Power = P(T > crit_upper | H1)
            power = 1 - stats.nct.cdf(crit_upper, df, nc)
        else: # alternative == 'smaller', H1: rho < 0. Reject if t < crit_lower
             # Power = P(T < crit_lower | H1)
            power = stats.nct.cdf(crit_lower, df, nc)

        return power


    def calculate_power(self):
        """Calculates statistical power given sample size."""
        if self.sample_size is None:
             raise ValueError("sample_size must be provided to calculate power.")
        if self.sample_size <= 2:
             return 0.0 # df must be > 0
        if self.effect_size == 0:
             # Power under H0 is the probability of Type I error
             return self.alpha if self.alternative == 'two-sided' else self.alpha

        return self._calculate_power_from_n(self.sample_size)

    def calculate_sample_size(self):
        """Calculates required sample size given desired power."""
        if self.power is None:
            raise ValueError("power must be provided to calculate sample_size.")
        if not (0 < self.power < 1):
            raise ValueError("Power must be between 0 and 1 (exclusive).")
        if self.effect_size == 0:
            return np.inf

        # Define the function to find the root for: power(n) - target_power = 0
        def power_diff(n):
             # Need n > 2 for df > 0. Start search slightly above 2.
            if n <= 2.1:
                return -self.power # Ensure search moves towards positive n
            return self._calculate_power_from_n(n) - self.power

        # Use a numerical solver (Brent's method) to find n
        lower_bound = 2.1 # Start just above 2
        upper_bound = 4 # Initial guess
        max_iter = 1000
        iter_count = 0
        # Increase upper_bound until power_diff is positive
        while power_diff(upper_bound) < 0 and iter_count < max_iter:
             # Increase more rapidly for correlation as sample size needs can be large
             upper_bound = max(upper_bound * 1.5, upper_bound + 10)
             iter_count += 1
             if iter_count >= max_iter:
                 raise RuntimeError("Could not find an upper bound for sample size search. Check parameters (effect size might be too small for desired power).")

        try:
            if power_diff(lower_bound) * power_diff(upper_bound) >= 0:
                 if power_diff(lower_bound) >= 0:
                     # Power target met or exceeded at the minimum possible df+epsilon
                     # Need n=3 minimum for df=1
                     return 3
                 else:
                     raise RuntimeError(f"Sample size search failed: f(a)={power_diff(lower_bound):.4f}, f(b)={power_diff(upper_bound):.4f}. Interval [{lower_bound}, {upper_bound}]")

            n_float = brentq(power_diff, lower_bound, upper_bound, xtol=1e-6, rtol=1e-6)
            # Need integer n >= 3
            return int(np.ceil(max(n_float, 3)))
        except ValueError as e:
            raise RuntimeError(f"Sample size calculation failed using brentq: {e}. Interval [{lower_bound}, {upper_bound}], f(a)={power_diff(lower_bound):.4f}, f(b)={power_diff(upper_bound):.4f}")

    def calculate_mdes(self):
        """Calculates the minimum detectable effect size (Pearson's rho) given power and sample size."""
        if self.power is None or self.sample_size is None:
             raise ValueError("Both power and sample_size must be provided to calculate MDES.")
        if self.sample_size <= 2: # Need df = n-2 > 0
             return np.inf

        # Define the function to find the root for: power(rho) - target_power = 0
        # Need to wrap _calculate_power_from_n to vary effect_size (rho)
        def power_diff_rho(rho):
            # Temporarily set self.effect_size for the helper function
            original_rho = self.effect_size
            self.effect_size = rho
            # Handle edge case rho near +/- 1 for nc calculation
            if abs(rho) >= 1.0:
                 power_val = 1.0 # Power is 1 at perfect correlation
            else:
                 power_val = self._calculate_power_from_n(self.sample_size)
            self.effect_size = original_rho # Restore original rho
            return power_val - self.power

        # Search for rho in [0, 1) as MDES is usually positive.
        lower_bound_rho = 1e-9 # Start near zero
        upper_bound_rho = 1.0 - 1e-9 # Just below 1

        try:
            # Check if power target is achievable even at max effect size
            if power_diff_rho(upper_bound_rho) < 0:
                 print(f"Warning: Target power ({self.power}) may not be achievable with N={self.sample_size} and alpha={self.alpha}. Max power ~ {self._calculate_power_from_n(self.sample_size):.4f} (at rho near 1)")
                 return np.inf # Cannot reach target power

            # Ensure signs are different for brentq
            if power_diff_rho(lower_bound_rho) * power_diff_rho(upper_bound_rho) >= 0:
                 if power_diff_rho(lower_bound_rho) >= 0:
                     return 0.0 # Detectable effect size is essentially 0
                 else:
                     # Should be caught by the check above, but just in case
                     return np.inf

            rho_float = brentq(power_diff_rho, lower_bound_rho, upper_bound_rho, xtol=1e-6, rtol=1e-6)
            return abs(rho_float) # Return positive MDES
        except ValueError as e:
             fa = power_diff_rho(lower_bound_rho)
             fb = power_diff_rho(upper_bound_rho)
             if fa * fb >= 0:
                  if fa >= 0 : return 0.0
                  else: return np.inf
             else:
                  raise RuntimeError(f"MDES calculation failed using brentq: {e}. Interval [{lower_bound_rho:.4f}, {upper_bound_rho:.4f}], f(a)={fa:.4f}, f(b)={fb:.4f}")


# --- Placeholder classes for other correlation types ---

class SpearmanKendallPower(PowerCalculator):
    """Placeholder for Spearman/Kendall Rank Correlation power calculations."""
    # Power analysis often relies on simulation or approximations based on ARE relative to Pearson.
    def calculate_power(self):
        raise NotImplementedError("Spearman/Kendall power calculation not yet implemented. Consider simulation or Pearson approximation.")

    def calculate_sample_size(self):
        raise NotImplementedError("Spearman/Kendall sample size calculation not yet implemented. Consider simulation or Pearson approximation.")

    def calculate_mdes(self):
        raise NotImplementedError("Spearman/Kendall MDES calculation not yet implemented. Consider simulation or Pearson approximation.")


class PartialCorrelationPower(PowerCalculator):
    """Placeholder for Partial/Semi-partial Correlation power calculations."""
    # Can sometimes be framed as regression problems (testing change in R^2) or require specific formulas/simulation.
    def calculate_power(self):
        raise NotImplementedError("Partial/Semi-partial correlation power calculation not yet implemented. Consider regression framework or simulation.")

    def calculate_sample_size(self):
        raise NotImplementedError("Partial/Semi-partial correlation sample size calculation not yet implemented. Consider regression framework or simulation.")

    def calculate_mdes(self):
        raise NotImplementedError("Partial/Semi-partial correlation MDES calculation not yet implemented. Consider regression framework or simulation.")


class IntraclassCorrelationPower(PowerCalculator):
    """Placeholder for Intraclass Correlation (ICC) power calculations."""
    # Power depends on the specific ICC form (e.g., consistency, agreement),
    # the study design (k raters/measurements, n subjects), and expected ICC values.
    # Often requires specialized formulas or simulation.
    def calculate_power(self):
        raise NotImplementedError("Intraclass Correlation (ICC) power calculation not yet implemented.")

    def calculate_sample_size(self):
        raise NotImplementedError("Intraclass Correlation (ICC) sample size calculation not yet implemented.")

    def calculate_mdes(self):
        raise NotImplementedError("Intraclass Correlation (ICC) MDES calculation not yet implemented.")
