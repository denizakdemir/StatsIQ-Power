"""
Core Calculation Engine for Statistical Power

This module will house the primary functions for calculating:
1. Statistical Power
2. Required Sample Size
3. Minimum Detectable Effect Size (MDES)

It will utilize analytical formulas and potentially simulation methods.
"""

import numpy as np
import scipy.stats as stats

class PowerCalculator:
    """Base class for power calculations."""
    def __init__(self, alpha, effect_size=None, power=None, sample_size=None, **kwargs):
        # Ensure exactly one parameter is None (the one to be calculated)
        params = {'effect_size': effect_size, 'power': power, 'sample_size': sample_size}
        none_params = [k for k, v in params.items() if v is None]

        if len(none_params) != 1:
            raise ValueError(f"Exactly one of 'effect_size', 'power', or 'sample_size' must be None (to be calculated). Provided: {params}")

        self.parameter_to_solve = none_params[0]
        self.effect_size = effect_size
        self.alpha = alpha
        self.power = power
        self.sample_size = sample_size
        self.kwargs = kwargs # For additional test-specific parameters

    def calculate_power(self):
        """Calculates the statistical power given effect_size and sample_size."""
        raise NotImplementedError("Subclasses must implement calculate_power.")

    def calculate_sample_size(self):
        """Calculates the required sample size given effect_size and power."""
        raise NotImplementedError("Subclasses must implement calculate_sample_size.")

    def calculate_mdes(self):
        """Calculates the minimum detectable effect size given power and sample_size."""
        raise NotImplementedError("Subclasses must implement calculate_mdes.")

    def solve(self):
        """Determines which calculation to perform and executes it."""
        if self.parameter_to_solve == 'power':
            if self.effect_size is None or self.sample_size is None:
                 raise ValueError("Both effect_size and sample_size must be provided to calculate power.")
            return self.calculate_power()
        elif self.parameter_to_solve == 'sample_size':
             if self.effect_size is None or self.power is None:
                 raise ValueError("Both effect_size and power must be provided to calculate sample_size.")
             return self.calculate_sample_size()
        elif self.parameter_to_solve == 'effect_size':
             if self.power is None or self.sample_size is None:
                 raise ValueError("Both power and sample_size must be provided to calculate effect_size (MDES).")
             return self.calculate_mdes()
        else:
             # Should not happen due to __init__ check
             raise ValueError(f"Invalid parameter to solve: {self.parameter_to_solve}")

# Example structure - specific implementations will be in modules
# def solve_power(test_type, params):
#     # Factory function or similar logic to instantiate the correct calculator
#     pass
