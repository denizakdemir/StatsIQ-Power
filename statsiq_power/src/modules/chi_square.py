"""
Statistical Power Calculations for Chi-Square Tests.

Includes:
- Chi-Square Goodness-of-Fit Test
- Chi-Square Test of Independence / Homogeneity
"""

import numpy as np
# Only GofChisquarePower seems available directly in statsmodels.stats.power
from statsmodels.stats.power import GofChisquarePower
from ..core.engine import PowerCalculator # Relative import from core engine

class ChiSquareGofPower(PowerCalculator):
    """
    Calculates power, sample size, or MDES for a Chi-Square Goodness-of-Fit test.

    Required Args:
        alpha (float): Significance level (Type I error rate).
        n_bins (int): Number of categories or bins in the distribution.
                      This determines the degrees of freedom (df = n_bins - 1).

    Optional Args (exactly two required):
        effect_size (float): Standardized effect size (Cohen's w).
        power (float): Desired statistical power (1 - Type II error rate).
        sample_size (int): Total number of observations.

    Additional Kwargs:
        ddof (int): Delta degrees of freedom (adjustment to df = n_bins - 1 - ddof). Default is 0.
    """
    def __init__(self, alpha, n_bins, effect_size=None, power=None, sample_size=None, **kwargs):
        super().__init__(alpha=alpha, effect_size=effect_size, power=power, sample_size=sample_size, **kwargs)
        if not isinstance(n_bins, int) or n_bins < 2:
            raise ValueError("n_bins must be an integer >= 2.")
        self.n_bins = n_bins
        self.ddof = kwargs.get('ddof', 0) # Delta degrees of freedom (usually 0 for GoF)
        self.df = self.n_bins - 1 - self.ddof
        if self.df < 1:
             raise ValueError(f"Degrees of freedom ({self.df}) must be at least 1 (n_bins={n_bins}, ddof={self.ddof}).")

        # Instantiate the statsmodels power solver
        self.solver = GofChisquarePower()

    def calculate_power(self):
        """Calculates statistical power given total sample size."""
        if self.sample_size is None:
             raise ValueError("sample_size must be provided to calculate power.")
        if self.sample_size <= self.df:
             # Technically possible but power will be very low / nonsensical
             print(f"Warning: sample_size ({self.sample_size}) is less than or equal to degrees of freedom ({self.df}).")

        power = self.solver.power(effect_size=self.effect_size,
                                  nobs=self.sample_size,
                                  n_bins=self.n_bins,
                                  ddof=self.ddof,
                                  alpha=self.alpha)
        return power

    def calculate_sample_size(self):
        """Calculates required total sample size given desired power."""
        if self.power is None:
            raise ValueError("power must be provided to calculate sample_size.")

        nobs = self.solver.solve_power(effect_size=self.effect_size,
                                       nobs=None, # Solving for nobs
                                       n_bins=self.n_bins,
                                       # ddof is handled internally by solve_power
                                       alpha=self.alpha,
                                       power=self.power)
        # Ensure sample size is at least slightly larger than df for meaningful test
        min_nobs = self.df + 1
        return np.ceil(max(nobs, min_nobs)).astype(int)

    def calculate_mdes(self):
        """Calculates the minimum detectable effect size (Cohen's w) given power and sample size."""
        if self.power is None or self.sample_size is None:
             raise ValueError("Both power and sample_size must be provided to calculate MDES.")
        if self.sample_size <= self.df:
             # Cannot perform test if N <= df
             return np.inf

        effect_size = self.solver.solve_power(effect_size=None, # Solving for effect size
                                              nobs=self.sample_size,
                                              n_bins=self.n_bins,
                                              ddof=self.ddof,
                                              alpha=self.alpha,
                                              power=self.power)
        return abs(effect_size)


class ChiSquareIndPower(PowerCalculator):
    """
    Calculates power, sample size, or MDES for a Chi-Square test of independence
    or homogeneity in a contingency table.

    Uses the GofChisquarePower framework by setting n_bins = df + 1.

    Required Args:
        alpha (float): Significance level (Type I error rate).
        df (int): Degrees of freedom for the test.
                  For an RxC table, df = (R-1) * (C-1). Must be >= 1.

    Optional Args (exactly two required):
        effect_size (float): Standardized effect size (Cohen's w).
        power (float): Desired statistical power (1 - Type II error rate).
        sample_size (int): Total number of observations in the table.

    Additional Kwargs:
        # None specific to independence test power
    """
    def __init__(self, alpha, df, effect_size=None, power=None, sample_size=None, **kwargs):
        super().__init__(alpha=alpha, effect_size=effect_size, power=power, sample_size=sample_size, **kwargs)
        if not isinstance(df, int) or df < 1:
            raise ValueError("Degrees of freedom (df) must be an integer >= 1.")
        self.df = df
        self.n_bins_equiv = self.df + 1 # Equivalent n_bins for GoF framework
        self.ddof_equiv = 0 # ddof is 0 in this mapping

        # Instantiate the statsmodels power solver using GofChisquarePower
        self.solver = GofChisquarePower()

    def calculate_power(self):
        """Calculates statistical power given total sample size."""
        if self.sample_size is None:
             raise ValueError("sample_size must be provided to calculate power.")
        if self.sample_size <= self.df:
             # Check against df, not n_bins_equiv
             print(f"Warning: sample_size ({self.sample_size}) is less than or equal to degrees of freedom ({self.df}).")

        power = self.solver.power(effect_size=self.effect_size,
                                  nobs=self.sample_size,
                                  n_bins=self.n_bins_equiv, # Use equivalent n_bins
                                  ddof=self.ddof_equiv,     # Use equivalent ddof
                                  alpha=self.alpha)
        return power

    def calculate_sample_size(self):
        """Calculates required total sample size given desired power."""
        if self.power is None:
            raise ValueError("power must be provided to calculate sample_size.")

        nobs = self.solver.solve_power(effect_size=self.effect_size,
                                       nobs=None, # Solving for nobs
                                       n_bins=self.n_bins_equiv, # Use equivalent n_bins
                                       # ddof is handled internally by solve_power
                                       alpha=self.alpha,
                                       power=self.power)
        # Ensure sample size is at least slightly larger than df
        min_nobs = self.df + 1
        return np.ceil(max(nobs, min_nobs)).astype(int)

    def calculate_mdes(self):
        """Calculates the minimum detectable effect size (Cohen's w) given power and sample size."""
        if self.power is None or self.sample_size is None:
             raise ValueError("Both power and sample_size must be provided to calculate MDES.")
        if self.sample_size <= self.df:
             # Cannot perform test if N <= df
             return np.inf

        effect_size = self.solver.solve_power(effect_size=None, # Solving for effect size
                                              nobs=self.sample_size,
                                              n_bins=self.n_bins_equiv,
                                              ddof=self.ddof_equiv,
                                              alpha=self.alpha,
                                              power=self.power)
        return abs(effect_size)
