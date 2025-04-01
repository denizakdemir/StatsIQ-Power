"""
Statistical Power Calculations for Analysis of Variance (ANOVA).

Includes:
- One-Way ANOVA
- Factorial ANOVA
- Repeated Measures ANOVA
- Mixed-Design ANOVA (Placeholder)
- ANCOVA (Placeholder)
"""

import numpy as np
# Need brentq for sample size solver in Factorial ANOVA (if restored)
from scipy.optimize import brentq
# Need ncf and f for Repeated Measures ANOVA calculation
from scipy.stats import ncf, f
import warnings # For warnings about assumptions
from statsmodels.stats.power import FTestAnovaPower, FTestPower # Dependency: statsmodels
from ..core.engine import PowerCalculator # Relative import from core engine

class OneWayANOVAPower(PowerCalculator):
    """
    Calculates power, sample size, or MDES for a one-way ANOVA.

    Required Args:
        alpha (float): Significance level (Type I error rate).
        k_groups (int): Number of groups being compared.

    Optional Args (exactly two required):
        effect_size (float): Standardized effect size (Cohen's f).
                             Note: statsmodels uses f, not f^2 or eta^2.
        power (float): Desired statistical power (1 - Type II error rate).
        sample_size (int): Total number of observations across all groups.
                           Assumes equal sample sizes per group (n = N/k).
                           statsmodels uses `nobs` for total sample size.

    Additional Kwargs:
        # None specific to one-way ANOVA power via statsmodels' FTestAnovaPower
    """
    def __init__(self, alpha, k_groups, effect_size=None, power=None, sample_size=None, **kwargs):
        super().__init__(alpha=alpha, effect_size=effect_size, power=power, sample_size=sample_size, **kwargs)
        self.k_groups = k_groups
        if not isinstance(k_groups, int) or k_groups < 2:
            raise ValueError("k_groups must be an integer >= 2.")

        # Instantiate the statsmodels power solver
        self.solver = FTestAnovaPower()

    def calculate_power(self):
        """Calculates statistical power given total sample size."""
        if self.sample_size is None:
             raise ValueError("sample_size (total N) must be provided to calculate power.")
        if self.sample_size < self.k_groups:
             raise ValueError(f"Total sample_size ({self.sample_size}) must be at least k_groups ({self.k_groups}).")

        power = self.solver.power(effect_size=self.effect_size,
                                  k_groups=self.k_groups,
                                  nobs=self.sample_size, # Total sample size
                                  alpha=self.alpha)
        return power

    def calculate_sample_size(self):
        """
        Calculates required *total* sample size given desired power.
        Assumes equal group sizes. Sample size per group is N / k_groups.
        """
        if self.power is None:
            raise ValueError("power must be provided to calculate sample_size.")

        nobs = self.solver.solve_power(effect_size=self.effect_size,
                                       k_groups=self.k_groups,
                                       nobs=None, # Solving for total N
                                       alpha=self.alpha,
                                       power=self.power)

        # Ensure total sample size is at least k_groups and is an integer multiple
        # of k_groups for equal sample sizes.
        min_nobs = self.k_groups
        # Handle potential NaN from solve_power if power is unachievable
        if nobs is None or np.isnan(nobs):
             warnings.warn(f"Could not solve for sample size. Power={self.power} may be unachievable with effect_size={self.effect_size}.")
             return np.inf
        calculated_nobs = np.ceil(max(nobs, min_nobs))

        # Adjust to be a multiple of k_groups
        remainder = calculated_nobs % self.k_groups
        if remainder != 0:
            adjusted_nobs = calculated_nobs + (self.k_groups - remainder)
        else:
            adjusted_nobs = calculated_nobs

        # Final check: ensure at least 2 per group if possible (though solver handles df)
        if adjusted_nobs / self.k_groups < 2:
             # This case is unlikely if power > alpha, but good to consider
             adjusted_nobs = 2 * self.k_groups # Minimum practical size

        return int(adjusted_nobs)

    def calculate_mdes(self):
        """Calculates the minimum detectable effect size (Cohen's f) given power and total sample size."""
        if self.power is None or self.sample_size is None:
             raise ValueError("Both power and sample_size must be provided to calculate MDES.")
        if self.sample_size < self.k_groups:
             # Cannot perform ANOVA if N < k
             return np.inf

        effect_size = self.solver.solve_power(effect_size=None, # Solving for effect size
                                              k_groups=self.k_groups,
                                              nobs=self.sample_size,
                                              alpha=self.alpha,
                                              power=self.power)
        # Handle potential NaN
        if effect_size is None or np.isnan(effect_size):
             warnings.warn(f"Could not solve for MDES. Check inputs (N={self.sample_size}, power={self.power}).")
             return np.inf
        return abs(effect_size)


# --- Mixed-Design ANOVA ---

class MixedANOVAPower(PowerCalculator):
    """Placeholder for Mixed-Design ANOVA calculations."""
    def calculate_power(self):
        raise NotImplementedError("Mixed-Design ANOVA power calculation not yet implemented. Consider simulation.")

    def calculate_sample_size(self):
        raise NotImplementedError("Mixed-Design ANOVA sample size calculation not yet implemented. Consider simulation.")

    def calculate_mdes(self):
        raise NotImplementedError("Mixed-Design ANOVA MDES calculation not yet implemented. Consider simulation.")


# --- ANCOVA ---

class ANCOVAPower(PowerCalculator):
    """
    Placeholder for Analysis of Covariance (ANCOVA) power calculations.

    Note: Power depends on the effect size adjusted for covariates,
          the number of covariates, and the correlation between covariates
          and the dependent variable. Often approximated using adjusted
          effect sizes in standard ANOVA power formulas or via simulation.
    """
    def calculate_power(self):
        raise NotImplementedError("ANCOVA power calculation not yet implemented. Consider simulation or adjusted effect size methods.")

    def calculate_sample_size(self):
        raise NotImplementedError("ANCOVA sample size calculation not yet implemented. Consider simulation or adjusted effect size methods.")

    def calculate_mdes(self):
        raise NotImplementedError("ANCOVA MDES calculation not yet implemented. Consider simulation or adjusted effect size methods.")


# --- Factorial ANOVA ---

class FactorialANOVAPower(PowerCalculator):
    """
    Calculates power, sample size, or MDES for a specific effect (main effect or interaction)
    in a factorial ANOVA design, approximated using the general F-test framework.

    Note: This requires specifying the numerator degrees of freedom for the effect of interest.
          Calculating denominator degrees of freedom depends on the full design (between/within factors).
          This implementation assumes a standard between-subjects factorial design for df_denom calculation.
          More complex designs (mixed, repeated measures components) may need different df_denom.
          WARNING: Sample size calculation for this class has shown instability. Use with caution.

    Required Args:
        alpha (float): Significance level (Type I error rate).
        df_num (int): Numerator degrees of freedom for the specific effect of interest.
                      (e.g., A levels - 1 for main effect A; (A-1)*(B-1) for AxB interaction).
        k_groups (int): Total number of cells (groups) in the factorial design
                        (e.g., levels_A * levels_B for a two-way design).

    Optional Args (exactly two required):
        effect_size (float): Standardized effect size (Cohen's f) for the effect of interest.
        power (float): Desired statistical power (1 - Type II error rate).
        sample_size (int): Total number of observations across all cells.
                           Assumes balanced design (equal n per cell).

    Additional Kwargs:
        # None specific currently, but could add options for df_denom calculation method.
    """
    def __init__(self, alpha, df_num, k_groups, effect_size=None, power=None, sample_size=None, **kwargs):
        super().__init__(alpha=alpha, effect_size=effect_size, power=power, sample_size=sample_size, **kwargs)
        self.df_num = df_num
        self.k_groups = k_groups # Total number of cells

        if not isinstance(df_num, int) or df_num < 1:
            raise ValueError("df_num (numerator degrees of freedom) must be an integer >= 1.")
        if not isinstance(k_groups, int) or k_groups < 2:
             raise ValueError("k_groups (total number of cells) must be an integer >= 2.")
        # Basic check: df_num should be less than k_groups for standard designs
        if df_num >= k_groups:
             warnings.warn(f"Numerator df ({df_num}) is >= total number of cells ({k_groups}). Check design specification.")

        # Use the general FTestPower solver
        self.solver = FTestPower()

    def _calculate_df_denom(self, total_n):
        """Calculate denominator df assuming standard between-subjects design."""
        # df_denom = N - k_groups (total sample size - total number of cells)
        if total_n is None or total_n < self.k_groups:
            return None # Cannot calculate if N is unknown or too small
        df_denom = total_n - self.k_groups
        if df_denom <= 0:
             return None # Denominator df must be positive
        return df_denom

    def calculate_power(self):
        """Calculates statistical power given total sample size."""
        if self.sample_size is None:
             raise ValueError("sample_size (total N) must be provided to calculate power.")

        df_denom = self._calculate_df_denom(self.sample_size)
        if df_denom is None or df_denom <= 0:
             warnings.warn(f"Cannot calculate power with total N={self.sample_size} and k_groups={self.k_groups} (results in df_denom <= 0). Returning 0 power.")
             return 0.0

        # Call using positional arguments due to potential statsmodels version/API quirk
        power = self.solver.power(self.effect_size, self.df_num, df_denom, self.alpha)
        # Handle potential NaN from statsmodels
        if power is None or np.isnan(power):
             warnings.warn(f"Statsmodels returned invalid power value (NaN). Check inputs: ES={self.effect_size}, N={self.sample_size}, alpha={self.alpha}, df_num={self.df_num}, df_den={df_denom}")
             return 0.0
        return power

    def calculate_sample_size(self):
        """
        Calculates required *total* sample size given desired power for the specified effect.
        Assumes balanced design (equal n per cell).
        Uses numerical root finding (brentq) on the power function.
        NOTE: This method has shown instability with brentq and solve_power.
              Consider using external tools (like G*Power) or simulation for complex ANOVA.
        """
        warnings.warn("Factorial ANOVA sample size calculation may be unstable. Verify results with other software (e.g., G*Power) or use simulation.")
        if self.power is None:
            raise ValueError("power must be provided to calculate sample_size.")
        if not (0 < self.power < 1):
             raise ValueError("Power must be between 0 and 1.")
        if self.effect_size is None or self.effect_size == 0:
             return np.inf # Cannot achieve power > alpha with zero effect size

        # We need to solve N such that power(effect_size, df_num, N - k_groups, alpha) = target_power
        # Use a numerical solver as df_den depends on N.

        def power_diff(n_total):
            # Need N > k_groups for df_den > 0.
            if n_total <= self.k_groups:
                return -self.power
            df_den = float(n_total - self.k_groups)
            # Ensure df_den is positive
            if df_den <= 1e-9: # Use small epsilon for float comparison
                return -self.power

            current_power = self.solver.power(self.effect_size, float(self.df_num), df_den, self.alpha)
            if np.isnan(current_power):
                # If power is NaN (e.g., df_den too small), treat as insufficient power
                return -self.power
            return current_power - self.power

        # Find bounds for the solver
        # Minimum N for df_den > 0 is k_groups + 1.
        # Practical minimum often requires at least 2 per cell, so N >= 2*k_groups.
        # Start search slightly above the theoretical minimum.
        lower_bound = float(self.k_groups + 1)

        # Check if power is already met at this theoretical lower bound
        power_at_lower = power_diff(lower_bound)
        if power_at_lower >= 0:
             # Adjust to be multiple of k_groups, ensuring at least 1 per cell
             n_per_cell = np.ceil(lower_bound / self.k_groups)
             return int(max(n_per_cell * self.k_groups, self.k_groups + 1))

        # Start upper bound search: Increase more rapidly initially
        upper_bound = lower_bound * 5 # Start with a larger jump
        max_iter = 50 # Limit iterations to prevent infinite loops
        iter_count = 0
        power_at_upper = power_diff(upper_bound)

        # Increase upper bound until power_diff becomes positive
        while power_at_upper < 0 and iter_count < max_iter:
            upper_bound *= 2 # Double the upper bound
            power_at_upper = power_diff(upper_bound)
            iter_count += 1
            # Safety break for extremely large N
            if upper_bound > 1e9: # Reduced safety break
                 warnings.warn("Upper bound search exceeded 1e9. Calculation may be unstable or require very large N.")
                 # Try to proceed but result might be unreliable
                 break

        if iter_count >= max_iter and power_at_upper < 0:
             raise RuntimeError("Could not find an upper bound for Factorial ANOVA sample size search where power >= target. Power may be unachievable or parameters are problematic.")

        # Check if signs are different before calling brentq
        if power_at_lower * power_at_upper >= 0:
             # If upper bound search hit limit but power is still low, this might trigger
             if upper_bound > 1e9:
                  warnings.warn(f"Factorial ANOVA sample size search failed sign check after hitting upper bound limit. Result may be unreliable. Interval [{lower_bound}, {upper_bound}]")
                  # Return infinity or a very large number as an indicator
                  return np.inf
             else:
                  # If bounds are reasonable but signs are same, power function might be non-monotonic or flat
                  warnings.warn(f"Factorial ANOVA sample size search failed sign check. Power function might be non-monotonic or flat in interval [{lower_bound}, {upper_bound}]. Result may be unreliable.")
                  # Attempt brentq anyway, but it might fail or give wrong result
                  # return np.inf # Safer to return inf

        try:
             n_float = brentq(power_diff, lower_bound, upper_bound, xtol=1e-6, rtol=1e-6)

             # Ensure balanced design: round up N to nearest multiple of k_groups
             n_per_cell_float = n_float / self.k_groups
             # Ensure at least 1 per cell
             n_per_cell = np.ceil(max(n_per_cell_float, 1.0))

             total_n = int(n_per_cell * self.k_groups)
             # Final check: ensure N is at least the theoretical minimum k_groups + 1
             return max(total_n, self.k_groups + 1)

        except ValueError as e:
             # Brentq might fail if the function is ill-behaved within the interval
             raise RuntimeError(f"Factorial ANOVA sample size calculation failed using brentq: {e}. Check interval [{lower_bound}, {upper_bound}] and power function behavior.")
        except RuntimeError as e:
             # Catch the explicit RuntimeError from the sign check or upper bound search
             raise e


    def calculate_mdes(self):
        """Calculates the minimum detectable effect size (Cohen's f) for the specified effect."""
        if self.power is None or self.sample_size is None:
             raise ValueError("Both power and sample_size must be provided to calculate MDES.")

        df_denom = self._calculate_df_denom(self.sample_size)
        if df_denom is None or df_denom <= 0:
             warnings.warn(f"Cannot calculate MDES with total N={self.sample_size} and k_groups={self.k_groups} (results in df_denom <= 0). Returning inf.")
             return np.inf

        effect_size = self.solver.solve_power(effect_size=None, # Solving for effect size
                                           df_num=self.df_num,
                                           df_den=df_denom,
                                           alpha=self.alpha,
                                           power=self.power)
        # Handle potential NaN return from solve_power
        if effect_size is None or np.isnan(effect_size):
             warnings.warn(f"Could not solve for MDES. Check input parameters (power={self.power}, N={self.sample_size}, etc.).")
             return np.inf
        return abs(effect_size)


# --- Repeated Measures ANOVA ---

class RepeatedMeasuresANOVAPower(PowerCalculator):
    """
    Calculates power, sample size (number of subjects), or MDES for the main
    within-subjects effect in a repeated measures ANOVA.

    Note: Assumes a simple one-way repeated measures design. Power for interactions
          in mixed designs is more complex. Uses the general F-test framework
          with direct calculation based on non-central F distribution.

    Required Args:
        alpha (float): Significance level (Type I error rate).
        n_measurements (int): Number of repeated measurements per subject (>= 2).

    Optional Args (exactly two required):
        effect_size (float): Standardized effect size (Cohen's f) for the within-subjects effect.
        power (float): Desired statistical power (1 - Type II error rate).
        sample_size (int): Number of subjects.

    Additional Kwargs:
        eps (float): Greenhouse-Geisser epsilon correction factor for sphericity violation (0 < eps <= 1).
                     Default is 1 (assumes sphericity). Lower values reduce power.
    """
    def __init__(self, alpha, n_measurements, effect_size=None, power=None, sample_size=None, **kwargs):
        super().__init__(alpha=alpha, effect_size=effect_size, power=power, sample_size=sample_size, **kwargs)
        self.n_measurements = n_measurements
        self.eps = kwargs.get('eps', 1.0)

        if not isinstance(n_measurements, int) or n_measurements < 2:
            raise ValueError("n_measurements must be an integer >= 2.")
        if not (0 < self.eps <= 1):
            raise ValueError("eps (sphericity correction) must be between 0 and 1.")
        if self.eps < 1:
             warnings.warn(f"Applying sphericity correction epsilon = {self.eps:.3f}. This reduces degrees of freedom and power.")

        # Note: sample_size in the base class corresponds to n_subjects here.
        # We don't instantiate statsmodels solver here, use direct calculation.

    def _calculate_dfs(self, n_subjects):
        """Calculate numerator and denominator df for the within-subjects effect."""
        if n_subjects is None or n_subjects < 2: # Need at least 2 subjects
            return None, None
        if self.n_measurements < 2:
             return None, None # Should be caught in init

        df_num = self.eps * (self.n_measurements - 1)
        df_den = self.eps * (n_subjects - 1) * (self.n_measurements - 1)

        # Ensure dfs are positive
        if df_num <= 0 or df_den <= 0:
            return None, None # Invalid degrees of freedom

        return df_num, df_den

    def calculate_power(self):
        """Calculates statistical power given number of subjects."""
        if self.sample_size is None: # sample_size is n_subjects
             raise ValueError("sample_size (number of subjects) must be provided to calculate power.")
        if self.effect_size is None:
             raise ValueError("effect_size must be provided to calculate power.")

        n_subjects = self.sample_size
        df_num, df_den = self._calculate_dfs(n_subjects)

        if df_num is None or df_den is None:
             warnings.warn(f"Cannot calculate power with n_subjects={n_subjects} and n_measurements={self.n_measurements} (results in invalid df). Returning 0 power.")
             return 0.0

        # Critical F value at the given alpha level from central F distribution
        F_crit = f.isf(self.alpha, df_num, df_den)
        if np.isnan(F_crit) or np.isinf(F_crit):
             warnings.warn(f"Could not determine critical F value. Check parameters (alpha={self.alpha}, df_num={df_num:.2f}, df_den={df_den:.2f}).")
             return 0.0 # Or handle as error

        # Compute the noncentrality parameter
        # Note: Original example used lambda = n * f^2. Let's verify this against common definitions.
        # Some sources use lambda = N * f^2 where N is total observations (n*k), others use n * f^2.
        # G*Power uses lambda = n * k * f^2 for RM ANOVA (within effect). Let's try that.
        # lambda_ = n_subjects * self.n_measurements * self.effect_size**2
        # Let's stick to the provided example's lambda = n * f^2 first.
        lambda_ = n_subjects * self.effect_size**2

        # Calculate power using non-central F distribution CDF
        power = 1.0 - ncf.cdf(F_crit, df_num, df_den, lambda_)

        # Handle potential NaN from ncf.cdf
        if power is None or np.isnan(power):
             warnings.warn(f"Calculation resulted in invalid power value (NaN). Check inputs: ES={self.effect_size}, N={n_subjects}, alpha={self.alpha}, df_num={df_num:.2f}, df_den={df_den:.2f}, lambda={lambda_:.2f}")
             return 0.0 # Or handle as error
        return power

    def calculate_sample_size(self):
        """
        Calculates required number of subjects given desired power for the within-subjects effect.
        Uses iterative search based on the direct power calculation.
        """
        if self.power is None:
            raise ValueError("power must be provided to calculate sample_size (number of subjects).")
        if not (0 < self.power < 1):
             raise ValueError("Power must be between 0 and 1.")
        if self.effect_size is None or self.effect_size == 0:
             return np.inf # Cannot achieve power > alpha with zero effect size

        # Use the internal power calculation method for the search
        def _power_for_n(n):
            # Temporarily set sample_size to n for calculation
            original_n = self.sample_size
            self.sample_size = n
            try:
                p = self.calculate_power()
            finally:
                self.sample_size = original_n # Restore original value
            return p

        # Iterative search for sample size
        n = 2 # Start with minimum subjects
        max_n = 10000 # Set a reasonable maximum to prevent infinite loops
        iter_count = 0

        while n <= max_n:
            current_power = _power_for_n(n)
            if current_power >= self.power:
                return n # Found the minimum n
            n += 1
            iter_count += 1
            # Add a check for excessive iterations just in case
            if iter_count > max_n * 1.1: # Allow some buffer
                 warnings.warn(f"Sample size search exceeded {max_n} iterations. Check parameters or implementation.")
                 break

        # If loop finishes without finding sufficient power
        raise ValueError(f"Required sample size exceeds max_n ({max_n}). Increase max_n or check parameters (power target might be unachievable).")


    def calculate_mdes(self):
        """Calculates the minimum detectable effect size (Cohen's f) for the within-subjects effect."""
        # This requires solving for effect_size in the power formula, which is complex
        # due to the non-central F distribution. Using statsmodels' solve_power is easier here.
        if self.power is None or self.sample_size is None: # sample_size is n_subjects
             raise ValueError("Both power and sample_size (number of subjects) must be provided to calculate MDES.")

        n_subjects = self.sample_size
        df_num, df_den = self._calculate_dfs(n_subjects)

        if df_num is None or df_den is None:
             warnings.warn(f"Cannot calculate MDES with n_subjects={n_subjects} and n_measurements={self.n_measurements} (results in invalid df). Returning inf.")
             return np.inf

        # Use statsmodels solver for MDES as it's simpler than inverting ncf
        f_solver = FTestPower()
        try:
            effect_size = f_solver.solve_power(effect_size=None, # Solving for effect size
                                               df_num=df_num,
                                               df_den=df_den,
                                               alpha=self.alpha,
                                               power=self.power)
        except Exception as e:
             warnings.warn(f"Statsmodels solve_power for MDES failed: {e}. Check parameters.")
             return np.inf

        # Handle potential NaN return from solve_power
        if effect_size is None or np.isnan(effect_size):
             warnings.warn(f"Could not solve for MDES. Check input parameters (power={self.power}, N={n_subjects}, etc.).")
             return np.inf
        return abs(effect_size)
