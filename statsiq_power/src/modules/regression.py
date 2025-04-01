"""
Statistical Power Calculations for Regression Analyses.

Includes:
- Simple Linear Regression
- Multiple Linear Regression (Placeholder)
- Logistic Regression (Placeholder)
"""

import numpy as np
import statsmodels.api as sm # For GLM fitting
from statsmodels.stats.power import FTestAnovaPower, FTestPower # Dependency: statsmodels
from scipy.stats import norm, bernoulli, poisson # For data generation
from scipy.optimize import brentq # For sample size search

from ..core.engine import PowerCalculator # Relative import from core engine
# Import simulation framework
try:
     from ..core.simulation import SimulationModel, SimulationPowerEstimator
except ImportError:
     from statsiq_power.src.core.simulation import SimulationModel, SimulationPowerEstimator


class SimpleLinearRegressionPower(PowerCalculator):
    """
    Calculates power, sample size, or MDES for a simple linear regression (testing slope != 0).

    Uses the F-test framework (df_num=1). The effect size is Cohen's f.
    Cohen's f can be calculated from the squared correlation coefficient R²:
    f = sqrt(R² / (1 - R²))

    Required Args:
        alpha (float): Significance level (Type I error rate).

    Optional Args (exactly two required):
        effect_size (float): Standardized effect size (Cohen's f).
        power (float): Desired statistical power (1 - Type II error rate).
        sample_size (int): Total number of observations (N).

    Additional Kwargs:
        # None specific to simple linear regression power via FTestAnovaPower
    """
    def __init__(self, alpha, effect_size=None, power=None, sample_size=None, **kwargs):
        # Note: effect_size is Cohen's f
        super().__init__(alpha=alpha, effect_size=effect_size, power=power, sample_size=sample_size, **kwargs)
        self.df_num = 1 # Numerator df for simple linear regression slope test
        self.k_groups = self.df_num + 1 # For FTestAnovaPower compatibility

        # Instantiate the statsmodels power solver
        self.solver = FTestAnovaPower()

    @classmethod
    def from_r_squared(cls, r_squared, alpha, power=None, sample_size=None, **kwargs):
        """
        Alternative constructor using R-squared instead of Cohen's f.

        Args:
            r_squared (float): The squared Pearson correlation coefficient (proportion of variance explained).
                               Must be between 0 and 1.
            alpha (float): Significance level.
            power (float, optional): Desired power.
            sample_size (int, optional): Sample size.
            **kwargs: Additional arguments for the main constructor.
        """
        if not (0 <= r_squared < 1):
             # R^2 = 1 implies perfect correlation, power is 1 (or undefined effect size f)
            raise ValueError("R-squared must be between 0 (inclusive) and 1 (exclusive).")
        if r_squared == 0:
            effect_size_f = 0.0
        else:
            effect_size_f = np.sqrt(r_squared / (1 - r_squared))

        return cls(effect_size=effect_size_f, alpha=alpha, power=power, sample_size=sample_size, **kwargs)


    def calculate_power(self):
        """Calculates statistical power given total sample size."""
        if self.sample_size is None:
             raise ValueError("sample_size must be provided to calculate power.")
        # Denominator df = N - df_num - 1 = N - 2
        if self.sample_size <= self.k_groups: # N <= 2
             raise ValueError(f"Sample size ({self.sample_size}) must be greater than {self.k_groups} for simple linear regression.")

        power = self.solver.power(effect_size=self.effect_size, # Cohen's f
                                  k_groups=self.k_groups, # df_num + 1
                                  nobs=self.sample_size, # Total sample size N
                                  alpha=self.alpha)
        return power

    def calculate_sample_size(self):
        """Calculates required total sample size given desired power."""
        if self.power is None:
            raise ValueError("power must be provided to calculate sample_size.")
        if not (0 < self.power < 1):
            raise ValueError("Power must be between 0 and 1 (exclusive).")
        if self.effect_size == 0:
            return np.inf

        nobs = self.solver.solve_power(effect_size=self.effect_size,
                                       k_groups=self.k_groups,
                                       nobs=None, # Solving for total N
                                       alpha=self.alpha,
                                       power=self.power)

        # Ensure sample size allows for df_den > 0 (i.e., N > 2)
        min_nobs = self.k_groups + 1 # Minimum N = 3
        return np.ceil(max(nobs, min_nobs)).astype(int)

    def calculate_mdes(self):
        """Calculates the minimum detectable effect size (Cohen's f) given power and sample size."""
        if self.power is None or self.sample_size is None:
             raise ValueError("Both power and sample_size must be provided to calculate MDES.")
        if self.sample_size <= self.k_groups: # N <= 2
             # Cannot perform test if N <= 2
             return np.inf

        effect_size = self.solver.solve_power(effect_size=None, # Solving for effect size
                                              k_groups=self.k_groups,
                                              nobs=self.sample_size,
                                              alpha=self.alpha,
                                              power=self.power)
        # MDES (Cohen's f) is typically positive
        return abs(effect_size)


# --- Multiple Linear Regression (Overall F-Test) ---

class MultipleLinearRegressionPower(PowerCalculator):
    """
    Calculates power, sample size, or MDES for the overall F-test in a
    multiple linear regression. This tests H0: All regression slopes are zero.

    Uses the F-test framework. The effect size is Cohen's f-squared (f²).
    f² = R² / (1 - R²) = (R²_full - R²_reduced) / (1 - R²_full)
    where R² is the coefficient of determination for the full model.

    Required Args:
        alpha (float): Significance level (Type I error rate).
        num_predictors (int): Number of predictor variables in the full model (p).

    Optional Args (exactly two required):
        effect_size (float): Standardized effect size (Cohen's f-squared, f²).
        power (float): Desired statistical power (1 - Type II error rate).
        sample_size (int): Total number of observations (N).

    Additional Kwargs:
        # None specific to multiple linear regression overall F-test power
    """
    def __init__(self, alpha, num_predictors, effect_size=None, power=None, sample_size=None, **kwargs):
        # Note: effect_size is f-squared (f²)
        if not isinstance(num_predictors, int) or num_predictors < 1:
            raise ValueError("num_predictors must be an integer >= 1.")

        super().__init__(alpha=alpha, effect_size=effect_size, power=power, sample_size=sample_size, **kwargs)
        self.num_predictors = num_predictors
        self.df_num = self.num_predictors # Numerator df = p
        # statsmodels FTestAnovaPower uses k_groups = df_num + 1
        self.k_groups_equiv = self.df_num + 1

        # Instantiate the statsmodels power solver
        self.solver = FTestAnovaPower()

    @classmethod
    def from_r_squared(cls, r_squared_full, num_predictors, alpha, power=None, sample_size=None, **kwargs):
        """
        Alternative constructor using R-squared of the full model.

        Args:
            r_squared_full (float): R-squared for the model including all predictors. Must be between 0 and 1.
            num_predictors (int): Number of predictors.
            alpha (float): Significance level.
            power (float, optional): Desired power.
            sample_size (int, optional): Sample size.
            **kwargs: Additional arguments for the main constructor.
        """
        if not (0 <= r_squared_full < 1):
             raise ValueError("R-squared must be between 0 (inclusive) and 1 (exclusive).")
        if r_squared_full == 0:
            effect_size_f2 = 0.0
        else:
            effect_size_f2 = r_squared_full / (1 - r_squared_full)

        # The class init expects f^2
        return cls(alpha=alpha, num_predictors=num_predictors, effect_size=effect_size_f2, power=power, sample_size=sample_size, **kwargs)


    def calculate_power(self):
        """Calculates statistical power given total sample size."""
        if self.sample_size is None:
             raise ValueError("sample_size must be provided to calculate power.")
        # Denominator df = N - p - 1
        if self.sample_size <= self.num_predictors + 1:
             raise ValueError(f"Sample size ({self.sample_size}) must be greater than num_predictors+1 ({self.num_predictors + 1}).")
        if self.effect_size is None: # effect_size is f^2
             raise ValueError("effect_size (f-squared) must be provided to calculate power.")

        # FTestAnovaPower expects Cohen's f, so convert f^2
        effect_size_f = np.sqrt(self.effect_size) if self.effect_size >= 0 else 0

        power = self.solver.power(effect_size=effect_size_f, # Pass f
                                  k_groups=self.k_groups_equiv, # df_num + 1
                                  nobs=self.sample_size, # Total sample size N
                                  alpha=self.alpha)
        return power

    def calculate_sample_size(self):
        """Calculates required total sample size given desired power."""
        if self.power is None:
            raise ValueError("power must be provided to calculate sample_size.")
        if self.effect_size is None: # effect_size is f^2
             raise ValueError("effect_size (f-squared) must be provided to calculate sample_size.")
        if not (0 < self.power < 1):
            raise ValueError("Power must be between 0 and 1 (exclusive).")
        if self.effect_size <= 0:
            return np.inf

        # Convert f^2 to f for the solver
        effect_size_f = np.sqrt(self.effect_size)

        nobs = self.solver.solve_power(effect_size=effect_size_f, # Pass f
                                       k_groups=self.k_groups_equiv,
                                       nobs=None, # Solving for total N
                                       alpha=self.alpha,
                                       power=self.power)

        # Ensure sample size allows for df_den > 0 (i.e., N > p + 1)
        min_nobs = self.num_predictors + 2
        return np.ceil(max(nobs, min_nobs)).astype(int)

    def calculate_mdes(self):
        """Calculates the minimum detectable effect size (Cohen's f-squared) given power and sample size."""
        if self.power is None or self.sample_size is None:
             raise ValueError("Both power and sample_size must be provided to calculate MDES.")
        if self.sample_size <= self.num_predictors + 1:
             return np.inf # Cannot perform test

        # Solve for Cohen's f first
        effect_size_f = self.solver.solve_power(effect_size=None, # Solving for f
                                                k_groups=self.k_groups_equiv,
                                                nobs=self.sample_size,
                                                alpha=self.alpha,
                                                power=self.power)
        # Return f-squared
        return effect_size_f**2


# --- Multiple Linear Regression (Specific Predictors / R-squared Change) ---

class MultipleLinearRegressionPredictorPower(PowerCalculator):
    """
    Calculates power, sample size, or MDES for testing a specific predictor
    or a set of predictors in multiple linear regression.

    This tests the change in R-squared when adding the predictor(s) of interest
    to a model that already contains other predictors. Uses the F-test framework.
    Effect size is Cohen's f-squared (f²) based on R-squared change:
    f² = (R²_full - R²_reduced) / (1 - R²_full)

    Required Args:
        alpha (float): Significance level (Type I error rate).
        num_predictors_tested (int): Number of predictors being tested (p_test).
        num_predictors_total (int): Total number of predictors in the full model (p_total).
                                     Must be >= num_predictors_tested.

    Optional Args (exactly two required):
        effect_size (float): Standardized effect size (Cohen's f-squared, f²).
        power (float): Desired statistical power (1 - Type II error rate).
        sample_size (int): Total number of observations (N).

    Additional Kwargs:
        # None specific currently
    """
    def __init__(self, alpha, num_predictors_tested, num_predictors_total, effect_size=None, power=None, sample_size=None, **kwargs):
        # Note: effect_size is f-squared (f²)
        if not isinstance(num_predictors_tested, int) or num_predictors_tested < 1:
            raise ValueError("num_predictors_tested must be an integer >= 1.")
        if not isinstance(num_predictors_total, int) or num_predictors_total < num_predictors_tested:
            raise ValueError("num_predictors_total must be an integer >= num_predictors_tested.")

        # Store f^2 as effect_size
        super().__init__(alpha=alpha, effect_size=effect_size, power=power, sample_size=sample_size, **kwargs)
        self.num_predictors_tested = num_predictors_tested
        self.num_predictors_total = num_predictors_total
        self.df_num = self.num_predictors_tested # Numerator df = p_test

        # Use the general FTestPower solver directly
        self.solver = FTestPower()

    @classmethod
    def from_rsquared_change(cls, rsquared_full, rsquared_reduced, num_predictors_tested, num_predictors_total, alpha, power=None, sample_size=None, **kwargs):
        """
        Alternative constructor using R-squared from full and reduced models.

        Args:
            rsquared_full (float): R-squared for the model including all predictors.
            rsquared_reduced (float): R-squared for the model excluding the tested predictors.
            num_predictors_tested (int): Number of predictors being tested.
            num_predictors_total (int): Total number of predictors in the full model.
            alpha (float): Significance level.
            power (float, optional): Desired power.
            sample_size (int, optional): Sample size.
            **kwargs: Additional arguments for the main constructor.
        """
        if not (0 <= rsquared_reduced <= rsquared_full < 1):
             raise ValueError("R-squared values must satisfy 0 <= R²_reduced <= R²_full < 1.")

        if rsquared_full == rsquared_reduced:
            effect_size_f2 = 0.0
        else:
            # Avoid division by zero if rsquared_full is very close to 1
            denom = 1 - rsquared_full
            if denom < 1e-9: # Use a small tolerance
                 effect_size_f2 = np.inf
            else:
                 effect_size_f2 = (rsquared_full - rsquared_reduced) / denom

        # The class init expects f^2
        return cls(alpha=alpha, num_predictors_tested=num_predictors_tested, num_predictors_total=num_predictors_total,
                   effect_size=effect_size_f2, power=power, sample_size=sample_size, **kwargs)

    def _calculate_df_denom(self, n):
        """Calculate denominator df = N - p_total - 1."""
        if n is None or n <= self.num_predictors_total + 1:
            return None
        return n - self.num_predictors_total - 1

    def calculate_power(self):
        """Calculates statistical power given total sample size."""
        if self.sample_size is None:
             raise ValueError("sample_size must be provided to calculate power.")
        if self.effect_size is None: # effect_size is f^2
             raise ValueError("effect_size (f-squared) must be provided to calculate power.")

        df_denom = self._calculate_df_denom(self.sample_size)
        if df_denom is None or df_denom <= 0:
             raise ValueError(f"Sample size ({self.sample_size}) must be greater than num_predictors_total+1 ({self.num_predictors_total + 1}).")

        # FTestPower expects Cohen's f, so convert f^2
        effect_size_f = np.sqrt(self.effect_size) if self.effect_size >= 0 else 0

        power = self.solver.power(effect_size=effect_size_f, # Pass f
                                  df_num=self.df_num,
                                  df_den=df_denom,
                                  alpha=self.alpha)
        return power

    def calculate_sample_size(self):
        """Calculates required total sample size given desired power."""
        if self.power is None:
            raise ValueError("power must be provided to calculate sample_size.")
        if self.effect_size is None: # effect_size is f^2
             raise ValueError("effect_size (f-squared) must be provided to calculate sample_size.")
        if not (0 < self.power < 1):
            raise ValueError("Power must be between 0 and 1 (exclusive).")
        if self.effect_size <= 0:
            return np.inf

        # Convert f^2 to f for the solver
        effect_size_f = np.sqrt(self.effect_size)

        # We need to solve N such that power(f, df_num, N - p_total - 1, alpha) = target_power
        # Use a numerical solver as df_den depends on N.
        def power_diff(n_total):
            df_den = self._calculate_df_denom(n_total)
            if df_den is None or df_den <= 0: return -self.power # Invalid N
            # Use positional arguments for power call
            current_power = self.solver.power(effect_size_f, float(self.df_num), float(df_den), self.alpha)
            if np.isnan(current_power): return -self.power # Handle NaN
            return current_power - self.power

        # Find bounds for the solver
        lower_bound = self.num_predictors_total + 2 # Minimum N for df_den > 0
        power_at_lower = power_diff(lower_bound)
        if power_at_lower >= 0:
             return int(lower_bound) # Power met at minimum N

        upper_bound = lower_bound * 2
        max_iter = 1000; iter_count = 0
        power_at_upper = power_diff(upper_bound)
        while power_at_upper < 0 and iter_count < max_iter:
            upper_bound = max(upper_bound * 1.5, upper_bound + 10) # Increase upper bound
            power_at_upper = power_diff(upper_bound)
            iter_count += 1
            if iter_count >= max_iter:
                raise RuntimeError("Could not find an upper bound for sample size search. Power may be unachievable.")

        try:
             if power_at_lower * power_at_upper >= 0:
                  raise RuntimeError(f"Sample size search failed: f(a) and f(b) have same sign. Interval [{lower_bound}, {upper_bound}]")

             n_float = brentq(power_diff, lower_bound, upper_bound, xtol=1e-6, rtol=1e-6)
             return np.ceil(max(n_float, lower_bound)).astype(int)
        except ValueError as e:
             raise RuntimeError(f"Sample size calculation failed using brentq: {e}")
        except RuntimeError as e:
             raise e

    def calculate_mdes(self):
        """Calculates the minimum detectable effect size (Cohen's f-squared) given power and sample size."""
        if self.power is None or self.sample_size is None:
             raise ValueError("Both power and sample_size must be provided to calculate MDES.")

        df_denom = self._calculate_df_denom(self.sample_size)
        if df_denom is None or df_denom <= 0:
             return np.inf # Cannot perform test

        # Solve for Cohen's f first
        effect_size_f = self.solver.solve_power(effect_size=None, # Solving for f
                                                df_num=self.df_num,
                                                df_den=df_denom,
                                                alpha=self.alpha,
                                                power=self.power)
        # Return f-squared
        return effect_size_f**2


# --- Logistic Regression Simulation Model ---

class LogisticRegressionSimulationModel(SimulationModel):
    """
    Simulation model for Logistic Regression with a single predictor.

    Generates data based on specified parameters and analyzes it using
    statsmodels Logit. Assumes a normally distributed predictor.
    """
    def __init__(self, odds_ratio: float, baseline_prob: float, predictor_mean: float = 0, predictor_sd: float = 1):
        """
        Args:
            odds_ratio (float): The odds ratio associated with a one-unit increase in the predictor.
            baseline_prob (float): The probability of the outcome when the predictor is zero.
            predictor_mean (float): Mean of the normally distributed predictor.
            predictor_sd (float): Standard deviation of the normally distributed predictor.
        """
        if odds_ratio <= 0:
            raise ValueError("Odds ratio must be positive.")
        if not (0 < baseline_prob < 1):
            raise ValueError("Baseline probability must be between 0 and 1.")
        if predictor_sd <= 0:
            raise ValueError("Predictor standard deviation must be positive.")

        self.odds_ratio = odds_ratio
        self.baseline_prob = baseline_prob
        self.predictor_mean = predictor_mean
        self.predictor_sd = predictor_sd

        # Calculate logistic regression coefficients (beta0, beta1)
        # log(odds) = beta0 + beta1*X
        # log(odds_at_X=0) = beta0
        # odds_at_X=0 = baseline_prob / (1 - baseline_prob)
        self.beta0 = np.log(baseline_prob / (1 - baseline_prob))

        # odds_at_X=1 = odds_at_X=0 * odds_ratio
        # log(odds_at_X=1) = beta0 + beta1
        # beta1 = log(odds_at_X=1) - beta0 = log(odds_at_X=0 * odds_ratio) - beta0
        # beta1 = log(odds_at_X=0) + log(odds_ratio) - beta0
        # beta1 = beta0 + log(odds_ratio) - beta0 = log(odds_ratio)
        self.beta1 = np.log(odds_ratio)

    def generate_data(self, sample_size):
        """
        Generates data for logistic regression.

        Args:
            sample_size (int): Total sample size (N).

        Returns:
            tuple: (y, X) where y is the binary outcome and X includes intercept and predictor.
        """
        if sample_size < 2: # Need at least 2 for regression
             raise ValueError("Sample size must be at least 2.")

        # Generate predictor variable X
        x_predictor = norm.rvs(loc=self.predictor_mean, scale=self.predictor_sd, size=sample_size)

        # Calculate linear combination (logit)
        logit = self.beta0 + self.beta1 * x_predictor

        # Convert logit to probability
        prob = 1 / (1 + np.exp(-logit))

        # Generate binary outcome y based on probability
        y = bernoulli.rvs(p=prob, size=sample_size)

        # Create design matrix (add intercept)
        X = sm.add_constant(x_predictor, prepend=True) # Adds column of 1s

        return y, X

    def analyze_data(self, data):
        """
        Performs logistic regression and returns the p-value for the predictor.

        Args:
            data (tuple): (y, X)

        Returns:
            float: p-value for the predictor coefficient (beta1). None if model fails.
        """
        y, X = data
        try:
            logit_model = sm.Logit(y, X)
            result = logit_model.fit(disp=0) # disp=0 suppresses convergence messages
            # Return p-value for the predictor (second coefficient, index 1)
            return result.pvalues[1]
        except Exception as e:
            # Handle potential errors (e.g., perfect separation, convergence failure)
            # print(f"Warning: Logistic regression failed. Error: {e}")
            return None


# --- Logistic Regression Power Class (using Simulation) ---

class LogisticRegressionPower(PowerCalculator):
    """
    Estimates power/sample size for Logistic Regression (single predictor) via simulation.
    Effect size is defined via Odds Ratio and baseline probability.
    """
    def __init__(self, alpha, odds_ratio=None, baseline_prob=None, power=None, sample_size=None, **kwargs):
         # Note: effect_size is not used directly, OR and baseline_prob define the effect
         # We store OR in the effect_size slot for convenience in the base class logic,
         # but it's not a standardized effect size in the same way as d, f, w, h.
         super().__init__(alpha=alpha, effect_size=odds_ratio, power=power, sample_size=sample_size, **kwargs)

         self.odds_ratio = odds_ratio
         self.baseline_prob = baseline_prob
         self.predictor_mean = self.kwargs.get('predictor_mean', 0)
         self.predictor_sd = self.kwargs.get('predictor_sd', 1)
         self.n_simulations = self.kwargs.get('n_simulations', 1000) # Default sims
         self.seed = self.kwargs.get('seed', None)
         self.n_jobs = self.kwargs.get('n_jobs', 1)

         # MDES (detectable odds ratio) calculation via simulation is complex
         if self.parameter_to_solve == 'effect_size':
              raise NotImplementedError("MDES (detectable odds ratio) calculation via simulation is not yet implemented for Logistic Regression.")

         # Create the simulation model instance
         if self.odds_ratio is None or self.baseline_prob is None:
              raise ValueError("Both odds_ratio and baseline_prob must be provided for Logistic Regression simulation.")
         self.sim_model = LogisticRegressionSimulationModel(
             odds_ratio=self.odds_ratio,
             baseline_prob=self.baseline_prob,
             predictor_mean=self.predictor_mean,
             predictor_sd=self.predictor_sd
         )

         # Create the simulation estimator
         self.estimator = SimulationPowerEstimator(
             simulation_model=self.sim_model,
             alpha=self.alpha,
             n_simulations=self.n_simulations,
             seed=self.seed,
             n_jobs=self.n_jobs
         )

    def calculate_power(self):
        """Estimates power using simulations."""
        if self.sample_size is None:
             raise ValueError("sample_size must be provided to estimate power.")
        # sample_size here is total N
        return self.estimator.estimate_power(sample_size=self.sample_size)

    def calculate_sample_size(self):
        """Estimates sample size using simulations (requires robust search)."""
        if self.power is None:
             raise ValueError("power must be provided to estimate sample_size.")
        try:
             search_range = self.kwargs.get('search_range', (10, 5000)) # Adjust default range
             tolerance = self.kwargs.get('tolerance', 10)
             max_iterations = self.kwargs.get('max_iterations', 15)
             return self.estimator.find_sample_size(
                 target_power=self.power,
                 search_range=search_range,
                 tolerance=tolerance,
                 max_iterations=max_iterations
             )
        except NotImplementedError as e:
             print(f"Error: {e}")
             raise NotImplementedError("Sample size search via simulation needs implementation in SimulationPowerEstimator.")

    def calculate_mdes(self):
         # Already checked in __init__
         raise NotImplementedError("MDES calculation via simulation is not yet implemented for Logistic Regression.")


# --- Poisson Regression Simulation Model ---

class PoissonRegressionSimulationModel(SimulationModel):
    """
    Simulation model for Poisson Regression with a single predictor.

    Generates count data based on specified parameters and analyzes it using
    statsmodels GLM with Poisson family and log link. Assumes a normally
    distributed predictor.
    """
    def __init__(self, rate_ratio: float, baseline_rate: float, predictor_mean: float = 0, predictor_sd: float = 1):
        """
        Args:
            rate_ratio (float): The rate ratio (exp(beta1)) associated with a one-unit increase in the predictor. Must be > 0.
            baseline_rate (float): The expected count (rate) when the predictor is zero. Must be > 0.
            predictor_mean (float): Mean of the normally distributed predictor.
            predictor_sd (float): Standard deviation of the normally distributed predictor.
        """
        if rate_ratio <= 0:
            raise ValueError("Rate ratio must be positive.")
        if baseline_rate <= 0:
            raise ValueError("Baseline rate must be positive.")
        if predictor_sd <= 0:
            raise ValueError("Predictor standard deviation must be positive.")

        self.rate_ratio = rate_ratio
        self.baseline_rate = baseline_rate
        self.predictor_mean = predictor_mean
        self.predictor_sd = predictor_sd

        # Calculate Poisson regression coefficients (beta0, beta1) for log link
        # log(mu) = beta0 + beta1*X
        # log(mu_at_X=0) = beta0 => beta0 = log(baseline_rate)
        self.beta0 = np.log(baseline_rate)

        # mu_at_X=1 = mu_at_X=0 * rate_ratio = baseline_rate * rate_ratio
        # log(mu_at_X=1) = beta0 + beta1
        # beta1 = log(mu_at_X=1) - beta0 = log(baseline_rate * rate_ratio) - log(baseline_rate)
        # beta1 = log(baseline_rate) + log(rate_ratio) - log(baseline_rate) = log(rate_ratio)
        self.beta1 = np.log(rate_ratio)

    def generate_data(self, sample_size):
        """
        Generates count data for Poisson regression.

        Args:
            sample_size (int): Total sample size (N).

        Returns:
            tuple: (y, X) where y is the count outcome and X includes intercept and predictor.
        """
        if sample_size < 2:
             raise ValueError("Sample size must be at least 2.")

        # Generate predictor variable X
        x_predictor = norm.rvs(loc=self.predictor_mean, scale=self.predictor_sd, size=sample_size)

        # Calculate linear combination (log rate)
        log_mu = self.beta0 + self.beta1 * x_predictor

        # Convert log rate to rate (mu)
        mu = np.exp(log_mu)

        # Generate count outcome y from Poisson distribution with rate mu
        y = poisson.rvs(mu=mu, size=sample_size)

        # Create design matrix (add intercept)
        X = sm.add_constant(x_predictor, prepend=True)

        return y, X

    def analyze_data(self, data):
        """
        Performs Poisson regression and returns the p-value for the predictor.

        Args:
            data (tuple): (y, X)

        Returns:
            float: p-value for the predictor coefficient (beta1). None if model fails.
        """
        y, X = data
        try:
            poisson_model = sm.GLM(y, X, family=sm.families.Poisson())
            result = poisson_model.fit(disp=0)
            # Return p-value for the predictor (second coefficient, index 1)
            return result.pvalues[1]
        except Exception as e:
            # Handle potential errors (e.g., convergence failure)
            # print(f"Warning: Poisson regression failed. Error: {e}")
            return None


# --- Poisson Regression Power Class (using Simulation) ---

class PoissonRegressionPower(PowerCalculator):
    """
    Estimates power/sample size for Poisson Regression (single predictor) via simulation.
    Effect size is defined via Rate Ratio and baseline rate.
    """
    def __init__(self, alpha, rate_ratio=None, baseline_rate=None, power=None, sample_size=None, **kwargs):
         # Store rate_ratio in effect_size slot for base class logic
         super().__init__(alpha=alpha, effect_size=rate_ratio, power=power, sample_size=sample_size, **kwargs)

         self.rate_ratio = rate_ratio
         self.baseline_rate = baseline_rate
         self.predictor_mean = self.kwargs.get('predictor_mean', 0)
         self.predictor_sd = self.kwargs.get('predictor_sd', 1)
         self.n_simulations = self.kwargs.get('n_simulations', 1000)
         self.seed = self.kwargs.get('seed', None)
         self.n_jobs = self.kwargs.get('n_jobs', 1)

         if self.parameter_to_solve == 'effect_size':
              raise NotImplementedError("MDES (detectable rate ratio) calculation via simulation is not yet implemented for Poisson Regression.")

         if self.rate_ratio is None or self.baseline_rate is None:
              raise ValueError("Both rate_ratio and baseline_rate must be provided for Poisson Regression simulation.")

         self.sim_model = PoissonRegressionSimulationModel(
             rate_ratio=self.rate_ratio,
             baseline_rate=self.baseline_rate,
             predictor_mean=self.predictor_mean,
             predictor_sd=self.predictor_sd
         )

         self.estimator = SimulationPowerEstimator(
             simulation_model=self.sim_model,
             alpha=self.alpha,
             n_simulations=self.n_simulations,
             seed=self.seed,
             n_jobs=self.n_jobs
         )

    def calculate_power(self):
        """Estimates power using simulations."""
        if self.sample_size is None:
             raise ValueError("sample_size must be provided to estimate power.")
        return self.estimator.estimate_power(sample_size=self.sample_size)

    def calculate_sample_size(self):
        """Estimates sample size using simulations."""
        if self.power is None:
             raise ValueError("power must be provided to estimate sample_size.")
        try:
             search_range = self.kwargs.get('search_range', (10, 5000))
             tolerance = self.kwargs.get('tolerance', 10)
             max_iterations = self.kwargs.get('max_iterations', 15)
             return self.estimator.find_sample_size(
                 target_power=self.power,
                 search_range=search_range,
                 tolerance=tolerance,
                 max_iterations=max_iterations
             )
        except NotImplementedError as e:
             print(f"Error: {e}")
             raise NotImplementedError("Sample size search via simulation needs implementation in SimulationPowerEstimator.") # Or implement here

    def calculate_mdes(self):
         raise NotImplementedError("MDES calculation via simulation is not yet implemented for Poisson Regression.")


# --- Placeholder classes for other regression types ---

class PolynomialRegressionPower(PowerCalculator):
    """Placeholder for Polynomial Regression power calculations."""
    # Power depends on the specific polynomial term being tested.
    # Can sometimes be framed as a multiple linear regression problem.
    def calculate_power(self):
        raise NotImplementedError("Polynomial Regression power calculation not yet implemented.")

    def calculate_sample_size(self):
        raise NotImplementedError("Polynomial Regression sample size calculation not yet implemented.")

    def calculate_mdes(self):
        raise NotImplementedError("Polynomial Regression MDES calculation not yet implemented.")


class MultinomialOrdinalLogisticRegressionPower(PowerCalculator):
    """Placeholder for Multinomial or Ordinal Logistic Regression power calculations."""
    # Power analysis is complex, often category-specific, and usually requires simulation.
    def calculate_power(self):
        raise NotImplementedError("Multinomial/Ordinal Logistic Regression power calculation not yet implemented. Requires simulation.")

    def calculate_sample_size(self):
        raise NotImplementedError("Multinomial/Ordinal Logistic Regression sample size calculation not yet implemented. Requires simulation.")

    def calculate_mdes(self):
        raise NotImplementedError("Multinomial/Ordinal Logistic Regression MDES calculation not yet implemented. Requires simulation.")


class NegativeBinomialRegressionPower(PowerCalculator):
    """Placeholder for Negative Binomial Regression power calculations."""
    # Similar to Poisson, but accounts for overdispersion. Usually requires simulation.
    def calculate_power(self):
        raise NotImplementedError("Negative Binomial Regression power calculation not yet implemented. Requires simulation.")

    def calculate_sample_size(self):
        raise NotImplementedError("Negative Binomial Regression sample size calculation not yet implemented. Requires simulation.")

    def calculate_mdes(self):
        raise NotImplementedError("Negative Binomial Regression MDES calculation not yet implemented. Requires simulation.")


# --- Placeholder classes for Advanced Modeling ---

class ModerationAnalysisPower(PowerCalculator):
    """Placeholder for Moderation Analysis power calculations."""
    # Power for detecting interaction effects. Often requires simulation or specialized tools.
    def calculate_power(self):
        raise NotImplementedError("Moderation Analysis power calculation not yet implemented.")

    def calculate_sample_size(self):
        raise NotImplementedError("Moderation Analysis sample size calculation not yet implemented.")

    def calculate_mdes(self):
        raise NotImplementedError("Moderation Analysis MDES calculation not yet implemented.")


class MediationAnalysisPower(PowerCalculator):
    """Placeholder for Mediation Analysis power calculations."""
    # Power for detecting indirect effects (a*b path). Complex, often requires simulation (e.g., Monte Carlo).
    def calculate_power(self):
        raise NotImplementedError("Mediation Analysis power calculation not yet implemented. Requires simulation.")

    def calculate_sample_size(self):
        raise NotImplementedError("Mediation Analysis sample size calculation not yet implemented. Requires simulation.")

    def calculate_mdes(self):
        raise NotImplementedError("Mediation Analysis MDES calculation not yet implemented. Requires simulation.")


class PathAnalysisSEMPower(PowerCalculator):
    """Placeholder for Path Analysis / Structural Equation Modeling (SEM) power calculations."""
    # Power for overall model fit or specific path coefficients. Very complex, typically requires simulation based on model specification.
    def calculate_power(self):
        raise NotImplementedError("Path Analysis/SEM power calculation not yet implemented. Requires simulation.")

    def calculate_sample_size(self):
        raise NotImplementedError("Path Analysis/SEM sample size calculation not yet implemented. Requires simulation.")

    def calculate_mdes(self):
        raise NotImplementedError("Path Analysis/SEM MDES calculation not yet implemented. Requires simulation.")
