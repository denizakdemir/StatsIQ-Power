"""
Statistical Power Calculations for Survival Analysis Models.

Includes:
- Cox Proportional Hazards Model (Events-based)
- Log-rank Test (Placeholder)
"""

import numpy as np
import scipy.stats as stats
from scipy.optimize import brentq

from ..core.engine import PowerCalculator # Relative import from core engine

class CoxPHPower(PowerCalculator):
    """
    Calculates power, required number of events, or minimum detectable
    hazard ratio (MDHR) for a Cox Proportional Hazards model comparing two groups.

    Uses Schoenfeld's formula approximation based on the number of events.
    Note: This calculates the required number of *events*, not total sample size N.
          Converting events to N requires assumptions about event rates/censoring.

    Effect size is the Hazard Ratio (HR).

    Required Args:
        alpha (float): Significance level (Type I error rate).

    Optional Args (exactly two required):
        effect_size (float): Anticipated Hazard Ratio (HR). Must be > 0 and != 1.
                             HR < 1 indicates benefit for group 1, HR > 1 indicates benefit for group 2.
        power (float): Desired statistical power (1 - Type II error rate).
        num_events (int): Total number of events required across both groups.

    Additional Kwargs:
        prop_group1 (float): Proportion of subjects in group 1 (default: 0.5 for equal groups).
                             Must be between 0 and 1.
        alternative (str): 'two-sided' (default, H1: HR != 1),
                           'smaller' (H1: HR < 1), or 'larger' (H1: HR > 1).
    """
    def __init__(self, alpha, effect_size=None, power=None, num_events=None, **kwargs):
        # Note: effect_size is Hazard Ratio (HR)
        # We use num_events instead of sample_size for this specific model
        if effect_size is not None and (effect_size <= 0): # HR must be positive
             raise ValueError("effect_size (Hazard Ratio) must be positive.")
        if effect_size == 1:
             warnings.warn("Hazard Ratio is 1 (no effect). Power will be alpha, required events infinite.")

        # Map num_events to the base class sample_size slot for consistency during init validation
        super().__init__(alpha=alpha, effect_size=effect_size, power=power, sample_size=num_events, **kwargs)
        self.prop_group1 = self.kwargs.get('prop_group1', 0.5)
        self.alternative = self.kwargs.get('alternative', 'two-sided')

        if not (0 < self.prop_group1 < 1):
            raise ValueError("prop_group1 must be between 0 and 1 (exclusive).")
        if self.alternative not in ['two-sided', 'larger', 'smaller']:
            raise ValueError("alternative must be 'two-sided', 'larger', or 'smaller'")

        # Rename sample_size back to num_events internally for clarity AFTER base init
        self.num_events = self.sample_size
        self.sample_size = None # Avoid confusion with base class usage

        # Adjust alpha for one-sided tests in the formula
        self.alpha_adj = self.alpha if self.alternative != 'two-sided' else self.alpha / 2.0

    # Override the solve method to use num_events for validation
    def solve(self):
        """Determines which calculation to perform and executes it, using num_events."""
        if self.parameter_to_solve == 'power':
            # Base class uses sample_size, but we stored num_events there initially
            # and then moved it to self.num_events. Check self.num_events here.
            if self.effect_size is None or self.num_events is None:
                 raise ValueError("Both effect_size (HR) and num_events must be provided to calculate power.")
            return self.calculate_power()
        elif self.parameter_to_solve == 'sample_size': # This corresponds to num_events for this class
             if self.effect_size is None or self.power is None:
                 raise ValueError("Both effect_size (HR) and power must be provided to calculate num_events.")
             return self.calculate_sample_size() # This method calculates num_events
        elif self.parameter_to_solve == 'effect_size':
             if self.power is None or self.num_events is None:
                 # Use the correct parameter name in the error message
                 raise ValueError("Both power and num_events must be provided to calculate effect_size (MDHR).")
             return self.calculate_mdes()
        else:
             # Should not happen due to __init__ check
             raise ValueError(f"Invalid parameter to solve: {self.parameter_to_solve}")


    def _calculate_power_from_events(self, D):
        """Helper: Calculate power given number of events D."""
        if D is None or D <= 0: # Check D directly
            return self.alpha if self.alternative == 'two-sided' else self.alpha_adj
        if self.effect_size is None or self.effect_size == 1:
            return self.alpha if self.alternative == 'two-sided' else self.alpha_adj # Power is alpha if HR=1

        p1 = self.prop_group1
        p2 = 1.0 - p1
        hr = self.effect_size
        log_hr_sq = (np.log(hr))**2

        if log_hr_sq == 0: # Should be caught by HR=1 check, but safety
             return self.alpha if self.alternative == 'two-sided' else self.alpha_adj

        z_alpha = stats.norm.ppf(1.0 - self.alpha_adj)
        z_beta_term = np.sqrt(D * p1 * p2 * log_hr_sq)

        z_beta = z_beta_term - z_alpha

        # Power = Phi(Z_beta)
        power = stats.norm.cdf(z_beta)
        return power

    def calculate_power(self):
        """Calculates statistical power given the number of events."""
        if self.num_events is None:
             raise ValueError("num_events must be provided to calculate power.")
        return self._calculate_power_from_events(self.num_events)

    def calculate_sample_size(self): # Actually calculates num_events
        """Calculates the required number of events given power and hazard ratio."""
        if self.power is None or self.effect_size is None:
            raise ValueError("Both power and effect_size (Hazard Ratio) must be provided to calculate required events.")
        if not (0 < self.power < 1):
            raise ValueError("Power must be between 0 and 1.")
        if self.effect_size == 1:
             return np.inf # Infinite events needed if HR=1

        p1 = self.prop_group1
        p2 = 1.0 - p1
        hr = self.effect_size
        log_hr_sq = (np.log(hr))**2

        z_alpha = stats.norm.ppf(1.0 - self.alpha_adj)
        z_beta = stats.norm.ppf(self.power)

        # Schoenfeld's formula for number of events D
        num_events_float = ((z_alpha + z_beta)**2) / (p1 * p2 * log_hr_sq)

        return np.ceil(num_events_float).astype(int)

    def calculate_mdes(self): # Actually calculates MDHR
        """Calculates the minimum detectable hazard ratio (MDHR) given power and number of events."""
        if self.power is None or self.num_events is None:
             # Corrected error message text
             raise ValueError("Both power and num_events must be provided to calculate MDHR.")
        if self.num_events <= 0:
             return np.nan # Cannot detect effect with 0 events

        p1 = self.prop_group1
        p2 = 1.0 - p1
        D = self.num_events

        z_alpha = stats.norm.ppf(1.0 - self.alpha_adj)
        z_beta = stats.norm.ppf(self.power)

        # Rearrange Schoenfeld's formula to solve for log(HR)^2
        # log(HR)^2 = (Z_alpha + Z_beta)^2 / (D * p1 * p2)
        log_hr_sq = ((z_alpha + z_beta)**2) / (D * p1 * p2)
        abs_log_hr = np.sqrt(log_hr_sq)

        # MDHR is exp(abs_log_hr) or exp(-abs_log_hr)
        # Typically report the HR further from 1.0
        mdhr_greater_1 = np.exp(abs_log_hr)
        mdhr_smaller_1 = np.exp(-abs_log_hr)

        # Return the HR corresponding to the specified alternative, or both if two-sided?
        # Convention often reports the HR > 1 or < 1 depending on context.
        # Let's return the one further from 1.0.
        if self.alternative == 'larger':
             return mdhr_greater_1
        elif self.alternative == 'smaller':
             return mdhr_smaller_1
        else: # two-sided, return the one further from 1 (larger magnitude effect)
             return mdhr_greater_1 # Or return tuple (mdhr_smaller_1, mdhr_greater_1)? For now, return > 1.


# --- Log-rank Test (2 Groups) ---

class LogRankPower(CoxPHPower):
    """
    Calculates power, required number of events, or minimum detectable
    hazard ratio (MDHR) for a two-group Log-rank test.

    Under the proportional hazards assumption, the calculation is identical
    to the Cox model power calculation based on Schoenfeld's formula.
    This class inherits directly from CoxPHPower.

    Note: Calculates required number of *events*, not total sample size N.

    Effect size is the Hazard Ratio (HR).

    Required Args:
        alpha (float): Significance level (Type I error rate).

    Optional Args (exactly two required):
        effect_size (float): Anticipated Hazard Ratio (HR). Must be > 0 and != 1.
        power (float): Desired statistical power (1 - Type II error rate).
        num_events (int): Total number of events required across both groups.

    Additional Kwargs:
        prop_group1 (float): Proportion of subjects in group 1 (default: 0.5 for equal groups).
        alternative (str): 'two-sided' (default, H1: HR != 1),
                           'smaller' (H1: HR < 1), or 'larger' (H1: HR > 1).
    """
    def __init__(self, alpha, effect_size=None, power=None, num_events=None, **kwargs):
        # All calculations are identical to CoxPHPower under proportional hazards
        super().__init__(alpha=alpha, effect_size=effect_size, power=power, num_events=num_events, **kwargs)
        # No specific logic needed here, inherits all methods including the overridden solve.
        pass


# --- Kaplan-Meier Analysis ---

class KaplanMeierPower(PowerCalculator):
    """
    Placeholder for power calculations related to Kaplan-Meier survival estimates.

    Note: Power calculations might involve comparing survival probabilities
          at specific time points between groups, often using normal approximations
          based on Greenwood's formula for standard error, or simulation.
    """
    def calculate_power(self):
        raise NotImplementedError("Kaplan-Meier related power calculation (e.g., comparing survival at time t) not yet implemented.")

    def calculate_sample_size(self):
        raise NotImplementedError("Kaplan-Meier related sample size calculation not yet implemented.")

    def calculate_mdes(self):
        raise NotImplementedError("Kaplan-Meier related MDES calculation not yet implemented.")
