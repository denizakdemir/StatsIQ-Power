"""
Statistical Power Calculations for Tests of Proportions.

Includes:
- One-Proportion Z-Test
- Two-Proportion Z-Test
- McNemar's Test (Placeholder)
- Fisher's Exact Test (Placeholder)
- Cochran's Q Test (Placeholder)
"""

import numpy as np
import scipy.stats as stats
from statsmodels.stats.proportion import proportion_effectsize
from statsmodels.stats.power import NormalIndPower # For two-sample z-test
from scipy.optimize import brentq # For solving sample size in one-sample

from ..core.engine import PowerCalculator # Relative import from core engine

# --- One-Proportion Z-Test ---

class OneProportionZTestPower(PowerCalculator):
    """
    Calculates power, sample size, or MDES for a one-sample proportion z-test.

    Uses Cohen's h as the effect size measure.

    Required Args:
        alpha (float): Significance level (Type I error rate).

    Optional Args (exactly two required):
        effect_size (float): Standardized effect size (Cohen's h).
                             Can be calculated using statsmodels.stats.proportion.proportion_effectsize(prop1, prop0).
        power (float): Desired statistical power (1 - Type II error rate).
        sample_size (int): Number of observations in the sample.

    Additional Kwargs:
        alternative (str): 'two-sided' (default), 'larger', or 'smaller'.
    """
    def __init__(self, alpha, effect_size=None, power=None, sample_size=None, **kwargs):
        # Note: effect_size here is Cohen's h
        super().__init__(alpha=alpha, effect_size=effect_size, power=power, sample_size=sample_size, **kwargs)
        self.alternative = self.kwargs.get('alternative', 'two-sided')
        if self.alternative not in ['two-sided', 'larger', 'smaller']:
            raise ValueError("alternative must be 'two-sided', 'larger', or 'smaller'")
        if self.effect_size == 0:
             # Power is alpha if effect size is 0, sample size is infinite
             print("Warning: Effect size is 0. Power calculation might be trivial or sample size infinite.")


    def _calculate_power_from_n(self, n):
        """Helper function to calculate power for a given n."""
        if n <= 0:
            return 0.0
        std_dev = 1.0 # For standardized effect size h
        crit_val_lower = stats.norm.ppf(self.alpha / 2) if self.alternative == 'two-sided' else stats.norm.ppf(self.alpha)
        crit_val_upper = stats.norm.ppf(1 - self.alpha / 2) if self.alternative == 'two-sided' else stats.norm.ppf(1 - self.alpha)
        
        non_centrality = self.effect_size * np.sqrt(n) / std_dev

        if self.alternative == 'two-sided':
            power = stats.norm.cdf(crit_val_lower - non_centrality) + (1 - stats.norm.cdf(crit_val_upper - non_centrality))
        elif self.alternative == 'larger':
             # H1: prop > prop0 => h > 0. Reject if Z > Z_crit
            power = 1 - stats.norm.cdf(crit_val_upper - non_centrality)
        elif self.alternative == 'smaller':
             # H1: prop < prop0 => h < 0. Reject if Z < Z_crit
            power = stats.norm.cdf(crit_val_lower - non_centrality)
        return power

    def calculate_power(self):
        """Calculates statistical power given sample size."""
        if self.sample_size is None:
             raise ValueError("sample_size must be provided to calculate power.")
        if self.sample_size <= 0:
             return 0.0 # Or raise error? Power is 0 for n=0.
        if self.effect_size == 0:
            return self.alpha if self.alternative == 'two-sided' else self.alpha / 2 # Power is alpha at H0

        return self._calculate_power_from_n(self.sample_size)

    def calculate_sample_size(self):
        """Calculates required sample size given desired power."""
        if self.power is None:
            raise ValueError("power must be provided to calculate sample_size.")
        if not (0 < self.power < 1):
            raise ValueError("Power must be between 0 and 1 (exclusive).")
        if self.effect_size == 0:
            return np.inf # Infinite sample size needed for non-zero power if effect is zero

        # Define the function to find the root for: power(n) - target_power = 0
        def power_diff(n):
            if n <= 1: # Need n > 0, start search slightly above 1
                return -self.power # Ensure search moves towards positive n
            return self._calculate_power_from_n(n) - self.power

        # Use a numerical solver (Brent's method) to find n
        # Need a reasonable search interval [a, b]
        # Start low (e.g., 2) and find an upper bound where power > target_power
        lower_bound = 2
        upper_bound = 2
        max_iter = 1000 # Safety break
        iter_count = 0
        # Increase upper_bound exponentially until power_diff is positive
        while power_diff(upper_bound) < 0 and iter_count < max_iter:
             upper_bound *= 2
             iter_count += 1
             if iter_count >= max_iter:
                 raise RuntimeError("Could not find an upper bound for sample size search. Check parameters.")

        try:
            # brentq requires the function values at the interval ends to have opposite signs
            if power_diff(lower_bound) * power_diff(upper_bound) >= 0:
                 # This might happen if power at lower_bound is already >= target_power
                 if power_diff(lower_bound) >= 0:
                     return np.ceil(lower_bound).astype(int)
                 else:
                     # Or if effect size is tiny / power target very high
                     raise RuntimeError(f"Sample size search failed: f(a)={power_diff(lower_bound):.4f}, f(b)={power_diff(upper_bound):.4f}. Interval [{lower_bound}, {upper_bound}]")

            n_float = brentq(power_diff, lower_bound, upper_bound, xtol=1e-6, rtol=1e-6)
            return np.ceil(n_float).astype(int)
        except ValueError as e:
            raise RuntimeError(f"Sample size calculation failed using brentq: {e}. Interval [{lower_bound}, {upper_bound}], f(a)={power_diff(lower_bound):.4f}, f(b)={power_diff(upper_bound):.4f}")

    def calculate_mdes(self):
        """Calculates the minimum detectable effect size (Cohen's h) given power and sample size."""
        if self.power is None or self.sample_size is None:
             raise ValueError("Both power and sample_size must be provided to calculate MDES.")
        if self.sample_size <= 0:
             return np.inf # Cannot detect effect with n <= 0

        # Define the function to find the root for: power(h) - target_power = 0
        # Need to wrap _calculate_power_from_n to vary effect_size (h)
        def power_diff_h(h):
            # Temporarily set self.effect_size for the helper function
            original_h = self.effect_size
            self.effect_size = h
            power_val = self._calculate_power_from_n(self.sample_size)
            self.effect_size = original_h # Restore original h
            return power_val - self.power

        # Search for h. Cohen's h ranges roughly from -pi to pi, but practically smaller.
        # Let's search in [0, pi] as MDES is usually reported positive.
        # Small effect h=0.2, medium h=0.5, large h=0.8
        lower_bound_h = 1e-9 # Start near zero
        upper_bound_h = np.pi # Theoretical max

        try:
            # Check if power target is achievable even at max effect size
            if power_diff_h(upper_bound_h) < 0:
                 # Power target might be too high for the given sample size/alpha
                 print(f"Warning: Target power ({self.power}) may not be achievable with N={self.sample_size} and alpha={self.alpha}. Max power ~ {self._calculate_power_from_n(self.sample_size):.4f} (at h=pi)")
                 # Still try to find the root, might yield pi
                 # Or return inf? Let's return the result from brentq if it converges.

            # Ensure signs are different for brentq
            if power_diff_h(lower_bound_h) * power_diff_h(upper_bound_h) >= 0:
                 # This happens if power at h=0 (which is alpha) >= target power
                 if power_diff_h(lower_bound_h) >= 0:
                     return 0.0 # Detectable effect size is essentially 0
                 else:
                     # This case implies power never reaches target, even at h=pi
                     # Let's return inf as the effect size needed is beyond limits
                     return np.inf


            h_float = brentq(power_diff_h, lower_bound_h, upper_bound_h, xtol=1e-6, rtol=1e-6)
            return abs(h_float) # Return positive MDES
        except ValueError as e:
             # Check if the error is due to interval signs
             fa = power_diff_h(lower_bound_h)
             fb = power_diff_h(upper_bound_h)
             if fa * fb >= 0:
                  if fa >= 0 : return 0.0 # Power target met at h=0
                  else: return np.inf # Power target never met
             else:
                  # Other brentq error
                  raise RuntimeError(f"MDES calculation failed using brentq: {e}. Interval [{lower_bound_h:.4f}, {upper_bound_h:.4f}], f(a)={fa:.4f}, f(b)={fb:.4f}")


# --- Two-Proportion Z-Test ---

class TwoProportionZTestPower(PowerCalculator):
    """
    Calculates power, sample size, or MDES for an independent two-sample proportion z-test.

    Uses Cohen's h as the effect size measure.

    Required Args:
        alpha (float): Significance level (Type I error rate).

    Optional Args (exactly two required):
        effect_size (float): Standardized effect size (Cohen's h).
                             Can be calculated using statsmodels.stats.proportion.proportion_effectsize(prop1, prop2).
        power (float): Desired statistical power (1 - Type II error rate).
        sample_size (int): Sample size of the *first* group (n1).

    Additional Kwargs:
        ratio (float): Ratio of sample sizes (n2 / n1). Default is 1 (equal group sizes).
        alternative (str): 'two-sided' (default), 'larger', or 'smaller'.
    """
    def __init__(self, alpha, effect_size=None, power=None, sample_size=None, **kwargs):
        # Note: effect_size is Cohen's h, sample_size is n1
        super().__init__(alpha=alpha, effect_size=effect_size, power=power, sample_size=sample_size, **kwargs)
        self.ratio = self.kwargs.get('ratio', 1.0)
        self.alternative = self.kwargs.get('alternative', 'two-sided')

        if self.alternative not in ['two-sided', 'larger', 'smaller']:
            raise ValueError("alternative must be 'two-sided', 'larger', or 'smaller'")
        if self.ratio <= 0:
            raise ValueError("ratio must be positive")

        # Instantiate the statsmodels power solver for Normal distribution (z-test)
        self.solver = NormalIndPower()

    def calculate_power(self):
        """Calculates statistical power given sample size of the first group (n1)."""
        if self.sample_size is None:
             raise ValueError("sample_size (n1) must be provided to calculate power.")
        if self.sample_size <= 0:
             return 0.0

        power = self.solver.power(effect_size=self.effect_size,
                                  nobs1=self.sample_size, # n1
                                  alpha=self.alpha,
                                  ratio=self.ratio, # n2/n1
                                  alternative=self.alternative)
        return power

    def calculate_sample_size(self):
        """
        Calculates required sample size for the *first* group (n1) given desired power.
        The sample size for the second group (n2) is n1 * ratio.
        Total sample size is n1 * (1 + ratio).
        """
        if self.power is None:
            raise ValueError("power must be provided to calculate sample_size.")
        if not (0 < self.power < 1):
            raise ValueError("Power must be between 0 and 1 (exclusive).")
        if self.effect_size == 0:
            return np.inf

        nobs1 = self.solver.solve_power(effect_size=self.effect_size,
                                        nobs1=None, # Solving for n1
                                        alpha=self.alpha,
                                        power=self.power,
                                        ratio=self.ratio,
                                        alternative=self.alternative)
        # Return n1, ensuring it's at least 1 (or maybe 2?)
        return np.ceil(max(nobs1, 1)).astype(int)

    def calculate_mdes(self):
        """Calculates the minimum detectable effect size (Cohen's h) given power and sample size (n1)."""
        if self.power is None or self.sample_size is None:
             raise ValueError("Both power and sample_size (n1) must be provided to calculate MDES.")
        if self.sample_size <= 0:
             return np.inf

        # Need n1 >= 1 and n2 >= 1 (implicitly handled by solver if ratio > 0)

        effect_size = self.solver.solve_power(effect_size=None, # Solving for effect size
                                              nobs1=self.sample_size,
                                              alpha=self.alpha,
                                              power=self.power,
                                              ratio=self.ratio,
                                              alternative=self.alternative)
        return abs(effect_size)


# --- McNemar's Test ---

class McNemarTestPower(PowerCalculator):
    """
    Calculates power, sample size (number of pairs), or MDES (difference in
    discordant proportions) for McNemar's test for paired proportions.

    Uses normal approximation methods. Requires specifying the discordant
    proportions under the alternative hypothesis (p01, p10).

    Required Args:
        alpha (float): Significance level (Type I error rate).

    Optional Args (requires p01 and p10 if calculating power/sample_size):
        p01 (float): Proportion of pairs changing from 0 to 1 under H1.
        p10 (float): Proportion of pairs changing from 1 to 0 under H1.
        power (float): Desired statistical power (1 - Type II error rate).
        sample_size (int): Total number of pairs.

    Additional Kwargs:
        alternative (str): 'two-sided' (default, H1: p01 != p10),
                           'larger' (H1: p01 > p10),
                           'smaller' (H1: p01 < p10).
        continuity_correction (bool): Apply continuity correction. Default is True.
                                      (Note: Power calculations often omit this for simplicity,
                                       leading to slightly overestimated power/underestimated N).

    Calculations based on formulas like those in Chow, Shao, Wang (2008) Sample Size Calculations in Clinical Research.
    """
    def __init__(self, alpha, p01=None, p10=None, power=None, sample_size=None, **kwargs):
        # We don't use a single 'effect_size' here, but rather p01 and p10.
        # Store them separately and calculate derived values as needed.
        # The base class expects one of effect_size, power, sample_size to be None.
        # We adapt this: if power or sample_size is None, we need p01 and p10.
        # If p01/p10 are None, we must be solving for MDES (requires power and sample_size).

        # Determine parameter_to_solve based on None values
        params = {'p01': p01, 'p10': p10, 'power': power, 'sample_size': sample_size}
        none_params = [k for k, v in params.items() if v is None]

        if len(none_params) == 0:
             raise ValueError("Exactly one of p01/p10 (as a pair for MDES), power, or sample_size must be None.")
        elif len(none_params) == 1:
             if none_params[0] == 'power': self.parameter_to_solve = 'power'
             elif none_params[0] == 'sample_size': self.parameter_to_solve = 'sample_size'
             else: raise ValueError("If only one parameter is None, it must be 'power' or 'sample_size'.")
             if p01 is None or p10 is None:
                  raise ValueError("p01 and p10 must be provided to calculate power or sample_size.")
        elif len(none_params) == 2 and 'p01' in none_params and 'p10' in none_params:
             self.parameter_to_solve = 'mdes' # Solving for the difference p01-p10 implicitly
             if power is None or sample_size is None:
                  raise ValueError("power and sample_size must be provided to calculate MDES (p01-p10 difference).")
        else:
             raise ValueError("Invalid combination of None parameters. Provide alpha and exactly two of: (p01, p10), power, sample_size.")


        # Initialize base class with dummy effect_size=0, will use p01/p10 directly
        super().__init__(alpha=alpha, effect_size=0, power=power, sample_size=sample_size, **kwargs)

        self.p01 = p01
        self.p10 = p10
        self.alternative = self.kwargs.get('alternative', 'two-sided')
        self.continuity_correction = self.kwargs.get('continuity_correction', True)

        if self.alternative not in ['two-sided', 'larger', 'smaller']:
            raise ValueError("alternative must be 'two-sided', 'larger', or 'smaller'")
        if self.p01 is not None and not (0 <= self.p01 <= 1): raise ValueError("p01 must be between 0 and 1.")
        if self.p10 is not None and not (0 <= self.p10 <= 1): raise ValueError("p10 must be between 0 and 1.")
        if self.p01 is not None and self.p10 is not None and (self.p01 + self.p10 > 1):
             raise ValueError("Sum of discordant proportions (p01 + p10) cannot exceed 1.")


    def _get_z_alpha(self):
        """Get the critical Z value for the specified alpha and alternative."""
        if self.alternative == 'two-sided':
            return stats.norm.ppf(1 - self.alpha / 2)
        else:
            return stats.norm.ppf(1 - self.alpha)

    def _get_z_beta(self):
        """Get the Z value for the specified power."""
        if self.power is None: return None # Should not happen if called correctly
        return stats.norm.ppf(self.power)

    def calculate_power(self):
        """Calculates statistical power given sample size (number of pairs), p01, and p10."""
        if self.sample_size is None or self.p01 is None or self.p10 is None:
             raise ValueError("sample_size, p01, and p10 must be provided to calculate power.")
        if self.sample_size <= 0: return 0.0

        n = self.sample_size
        p_disc = self.p01 + self.p10 # Total discordant proportion under H1
        diff = self.p01 - self.p10
        if p_disc == 0: return 1.0 if diff == 0 else 0.0 # Edge case: no discordant pairs

        z_alpha = self._get_z_alpha()
        # Continuity correction term (subtract 1/(2n) from |diff| or 1/n from |b-c|)
        cc = 1 / (2 * n) if self.continuity_correction else 0

        # Calculate Z_beta based on the formula rearranged for Z_beta
        # Simplified formula often uses p_disc_null = p_disc_alt = p01+p10
        # Z_test = (b - c) / sqrt(b + c)  or with cc: (|b-c|-1)/sqrt(b+c)
        # Z_test approx N( sqrt(n)*(p01-p10)/sqrt(p01+p10), 1 ) under H1? Let's verify.
        # Chow formula (3.3.4) implies Z_beta = [sqrt(n)|p01-p10| - Z_alpha * sqrt(p01+p10)] / sqrt(p01+p10) (ignoring cc)

        # Let's use the Z-score formulation under H1
        # Z = (p01_hat - p10_hat - E[diff|H0]) / SE
        # E[diff|H0] = 0
        # SE under H0 approx sqrt( (p01+p10)/n )
        # SE under H1 approx sqrt( (p01+p10)/n ) ? (often simplified this way)
        # Z_beta = P(Reject H0 | H1)
        # Reject if |Z_test| > Z_alpha/2 (two-sided)

        # Calculate the non-centrality parameter under H1
        # Non-centrality parameter lambda = E[Z_test | H1]
        # lambda = (E[p01_hat - p10_hat | H1] - 0) / SE_H0
        # lambda = (p01 - p10) / sqrt( (p01+p10)/n ) = sqrt(n) * (p01 - p10) / sqrt(p01 + p10)

        non_centrality = np.sqrt(n) * diff / np.sqrt(p_disc) if p_disc > 0 else 0
        # Adjust for continuity correction? Reduces the effective difference.
        # Effective diff = |p01 - p10| - cc/sqrt(p_disc/n) ? No, simpler adjustment:
        # Adjust non-centrality directly? Or adjust critical value?
        # Let's adjust the difference in the numerator for CC:
        adj_diff = abs(diff) - (cc / np.sqrt(p_disc)) if self.continuity_correction and p_disc > 0 else abs(diff)
        adj_non_centrality = np.sign(diff) * np.sqrt(n) * adj_diff / np.sqrt(p_disc) if p_disc > 0 else 0


        if self.alternative == 'two-sided':
            # Power = P(Z > z_alpha/2 - lambda) + P(Z < -z_alpha/2 - lambda) where Z ~ N(0,1)
            power = stats.norm.cdf(-z_alpha - adj_non_centrality) + (1 - stats.norm.cdf(z_alpha - adj_non_centrality))
        elif self.alternative == 'larger': # H1: p01 > p10 => diff > 0
            # Reject if Z > z_alpha
            power = 1 - stats.norm.cdf(z_alpha - adj_non_centrality)
        elif self.alternative == 'smaller': # H1: p01 < p10 => diff < 0
            # Reject if Z < -z_alpha
            power = stats.norm.cdf(-z_alpha - adj_non_centrality)

        return max(0.0, min(1.0, power)) # Ensure power is in [0, 1]


    def calculate_sample_size(self):
        """Calculates required sample size (number of pairs) given desired power, p01, and p10."""
        if self.power is None or self.p01 is None or self.p10 is None:
            raise ValueError("power, p01, and p10 must be provided to calculate sample_size.")
        if not (0 < self.power < 1):
            raise ValueError("Power must be between 0 and 1 (exclusive).")

        p_disc = self.p01 + self.p10
        diff = self.p01 - self.p10
        if diff == 0: return np.inf # Cannot achieve power > alpha if true difference is 0

        if p_disc == 0:
             # If no discordant pairs are expected, power is impossible unless diff is also 0
             return np.inf if diff != 0 else 1 # Or some minimum?

        z_alpha = self._get_z_alpha()
        z_beta = self._get_z_beta() # Note: z_beta for P(Type II error) = 1-power

        # Formula from Chow (3.3.5), adapted for one-sided/two-sided and cc
        # n = [(Z_alpha * sqrt(psi_0) + Z_beta * sqrt(psi_1))]^2 / delta^2
        # Here, psi_0 = psi_1 = p01+p10 (approx), delta = p01-p10
        # Continuity correction adds complexity. Let's use the formula without CC first.

        if self.alternative == 'two-sided':
            # Use Z_alpha/2
            term1 = stats.norm.ppf(1 - self.alpha / 2) * np.sqrt(p_disc)
        else:
            # Use Z_alpha
            term1 = stats.norm.ppf(1 - self.alpha) * np.sqrt(p_disc)

        term2 = z_beta * np.sqrt(p_disc) # Using p_disc under H1 as approximation

        numerator = (term1 + term2)**2
        denominator = diff**2

        n_no_cc = numerator / denominator

        # Add continuity correction adjustment (approximate)
        # n_cc = n_no_cc * [1 + sqrt(1 + 2*sqrt(p_disc)/(n_no_cc*|diff|))]^2 / 4  (From Chow, complex)
        # Simpler adjustment: n_cc = n_no_cc + 1 / (2*|diff|) (From SAS docs)
        # Let's try the simpler one first if cc is enabled
        if self.continuity_correction:
             n = n_no_cc + 1 / (2 * abs(diff))
        else:
             n = n_no_cc

        return np.ceil(max(n, 2)).astype(int) # Need at least 2 pairs


    def calculate_mdes(self):
        """
        Calculates the minimum detectable absolute difference |p01 - p10|
        given power and sample size. Assumes p01+p10 is known or estimated.
        Requires an estimate of the total discordant proportion p_disc = p01 + p10.
        """
        if self.power is None or self.sample_size is None:
             raise ValueError("Both power and sample_size must be provided to calculate MDES.")
        if self.sample_size <= 1: return np.inf

        # User must provide expected p_disc via kwargs
        p_disc = self.kwargs.get('p_disc', None)
        if p_disc is None:
             raise ValueError("Must provide expected total discordant proportion 'p_disc' (p01+p10) in kwargs to calculate MDES for McNemar test.")
        if not (0 < p_disc <= 1):
             raise ValueError("p_disc must be between 0 and 1.")

        n = self.sample_size
        z_alpha = self._get_z_alpha()
        z_beta = self._get_z_beta()

        # Rearrange the sample size formula (without CC first) to solve for |diff|
        # n = (z_alpha*sqrt(p_disc) + z_beta*sqrt(p_disc))^2 / diff^2
        # diff^2 = p_disc * (z_alpha + z_beta)^2 / n
        # |diff| = sqrt(p_disc / n) * (z_alpha + z_beta)

        if self.alternative == 'two-sided':
            z_a = stats.norm.ppf(1 - self.alpha / 2)
        else:
            z_a = stats.norm.ppf(1 - self.alpha)

        mdes_no_cc = np.sqrt(p_disc / n) * (z_a + z_beta)

        # Adjust for continuity correction (approximate inversion)
        # |diff_cc| approx |diff_no_cc| + 1 / (2*n*sqrt(p_disc)) ? No...
        # Let's solve iteratively or ignore CC for MDES for now, as inversion is tricky.
        if self.continuity_correction:
             # Could use a root-finding approach on the power function, solving for diff
             warnings.warn("MDES calculation with continuity correction for McNemar test is approximate or not implemented; returning value without correction.")
             # TODO: Implement iterative solver for MDES with CC if needed.
             pass

        return mdes_no_cc


class FishersExactTestPower(PowerCalculator):
    """Placeholder for Fisher's Exact Test power calculations."""
    # Power for Fisher's exact test is complex, often requires specialized packages or simulation.
    def calculate_power(self):
        raise NotImplementedError("Fisher's Exact Test power calculation not yet implemented.")

    def calculate_sample_size(self):
        raise NotImplementedError("Fisher's Exact Test sample size calculation not yet implemented.")

class CochransQTestPower(PowerCalculator):
    """Placeholder for Cochran's Q Test power calculations (multiple paired proportions)."""
    def calculate_power(self):
        raise NotImplementedError("Cochran's Q Test power calculation not yet implemented.")

    def calculate_sample_size(self):
        raise NotImplementedError("Cochran's Q Test sample size calculation not yet implemented.")
