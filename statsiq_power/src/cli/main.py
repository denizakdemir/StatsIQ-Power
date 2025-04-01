"""
StatsIQ-Power: Command-Line Interface
"""

import argparse
import sys
import numpy as np
import warnings # Added for warnings

# Use absolute imports assuming 'statsiq_power' is the top-level package accessible
try:
    from statsiq_power.src.modules.t_tests import (
        OneSampleTTestPower, IndependentSamplesTTestPower, PairedSamplesTTestPower,
        OneSampleNIPower, OneSampleTOSTPower, # Added NI and TOST
        # TODO: Add IndependentSamplesNIPower, IndependentSamplesTOSTPower, PairedSamplesNIPower, PairedSamplesTOSTPower
    )
    from statsiq_power.src.modules.anova import (
        OneWayANOVAPower # Removed FactorialANOVAPower, RepeatedMeasuresANOVAPower due to instability
        # TODO: Add MixedANOVAPower, ANCOVAPower placeholders?
    )
    from statsiq_power.src.modules.chi_square import ChiSquareGofPower, ChiSquareIndPower
    from statsiq_power.src.modules.proportions import (
        OneProportionZTestPower, TwoProportionZTestPower, McNemarTestPower # Added McNemar
        # TODO: Add FishersExactTestPower, CochransQTestPower placeholders?
    )
    from statsiq_power.src.modules.regression import (
        SimpleLinearRegressionPower, MultipleLinearRegressionPower, # Removed MultipleLinearRegressionPredictorPower due to instability
        LogisticRegressionPower, PoissonRegressionPower
        # TODO: Add PolynomialRegressionPower etc placeholders?
    )
    from statsiq_power.src.modules.correlation import PearsonCorrelationPower
    from statsiq_power.src.utils.effect_sizes import ( # Added effect size utils
        calculate_cohens_d_one_sample,
        calculate_cohens_d_independent,
        calculate_cohens_d_paired,
        calculate_hedges_g, # Added Hedges' g
        calculate_cohens_h,
        calculate_cohens_f_from_eta_sq,
        calculate_cohens_w,
        calculate_odds_ratio, # Added OR
        calculate_risk_ratio, # Added RR
        calculate_r_from_d, # Added d/r conversion
        calculate_d_from_r  # Added d/r conversion
    )
    from statsiq_power.src.modules.nonparametric import MannWhitneyUTestPower
    # from statsiq_power.src.modules.regression import LogisticRegressionPower, PoissonRegressionPower # Already imported above
    from statsiq_power.src.modules.survival import CoxPHPower, LogRankPower
    # Import other modules here using absolute paths as needed
except ImportError as e:
    print(f"Error: Could not import calculation modules using absolute paths: {e}", file=sys.stderr)
    print("Make sure you are running the command from the project root directory containing 'statsiq_power'", file=sys.stderr)
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="StatsIQ-Power: Statistical Power Calculator Suite (CLI)")
    subparsers = parser.add_subparsers(dest='test_type', help='Select the type of statistical test', required=True)

    # --- One-Sample T-Test Subparser ---
    parser_t_one = subparsers.add_parser('t_one_sample', help='Power/Sample Size/MDES for One-Sample T-Test')
    parser_t_one.add_argument('--solve-for', choices=['power', 'sample_size', 'effect_size'], required=True, help="Parameter to calculate")
    # Effect size group
    group_t_one_es = parser_t_one.add_mutually_exclusive_group()
    group_t_one_es.add_argument('--effect-size', type=float, help="Effect size (Cohen's d).")
    group_t_one_es.add_argument('--means-sd', type=float, nargs=3, metavar=('SAMPLE_MEAN', 'POP_MEAN', 'SAMPLE_SD'), help="Calculate Cohen's d from sample mean, population mean, and sample SD.")
    # Other args
    parser_t_one.add_argument('--alpha', type=float, required=True, help="Significance level (e.g., 0.05)")
    parser_t_one.add_argument('--power', type=float, help="Desired power (e.g., 0.80). Required unless solving for power.")
    parser_t_one.add_argument('--sample-size', type=int, help="Sample size (N). Required unless solving for sample_size.")
    parser_t_one.add_argument('--alternative', choices=['two-sided', 'larger', 'smaller'], default='two-sided', help="Alternative hypothesis")
    parser_t_one.set_defaults(func=handle_t_one_sample)

    # --- Independent Samples T-Test Subparser ---
    parser_t_ind = subparsers.add_parser('t_independent', help='Power/Sample Size/MDES for Independent Samples T-Test')
    parser_t_ind.add_argument('--solve-for', choices=['power', 'sample_size', 'effect_size'], required=True, help="Parameter to calculate")
    # Effect size group
    group_t_ind_es = parser_t_ind.add_mutually_exclusive_group()
    group_t_ind_es.add_argument('--effect-size', type=float, help="Effect size (Cohen's d).")
    group_t_ind_es.add_argument('--means-sds-ns', type=float, nargs=6, metavar=('M1', 'SD1', 'N1', 'M2', 'SD2', 'N2'), help="Calculate Cohen's d from means, SDs, and Ns of two groups.")
    # Other args
    parser_t_ind.add_argument('--alpha', type=float, required=True, help="Significance level (e.g., 0.05)")
    parser_t_ind.add_argument('--power', type=float, help="Desired power (e.g., 0.80). Required unless solving for power.")
    parser_t_ind.add_argument('--sample-size', type=int, help="Sample size of the *first* group (n1). Required unless solving for sample_size (Note: N1 from --means-sds-ns is used if provided when calculating d).")
    parser_t_ind.add_argument('--ratio', type=float, default=1.0, help="Ratio of sample sizes (n2 / n1). Default is 1.0. (Note: N2 from --means-sds-ns is used if provided when calculating d).")
    parser_t_ind.add_argument('--alternative', choices=['two-sided', 'larger', 'smaller'], default='two-sided', help="Alternative hypothesis")
    parser_t_ind.add_argument('--usevar', choices=['pooled', 'unequal'], default='pooled', help="Assume equal ('pooled') or unequal ('unequal') variances. Default is 'pooled'.")
    # Cluster options
    parser_t_ind.add_argument('--icc', type=float, help="Intraclass Correlation Coefficient (for clustered designs).")
    parser_t_ind.add_argument('--cluster-size', type=float, help="Average cluster size (m). Required if --icc is specified.")
    parser_t_ind.set_defaults(func=handle_t_independent)

    # --- Paired Samples T-Test Subparser ---
    parser_t_paired = subparsers.add_parser('t_paired', help='Power/Sample Size/MDES for Paired Samples T-Test')
    parser_t_paired.add_argument('--solve-for', choices=['power', 'sample_size', 'effect_size'], required=True, help="Parameter to calculate")
    # Effect size group
    group_t_paired_es = parser_t_paired.add_mutually_exclusive_group()
    group_t_paired_es.add_argument('--effect-size', type=float, help="Effect size (Cohen's d) of the differences.")
    group_t_paired_es.add_argument('--diff-mean-sd', type=float, nargs=2, metavar=('MEAN_DIFF', 'SD_DIFF'), help="Calculate Cohen's d from the mean and SD of the differences.")
    # Other args
    parser_t_paired.add_argument('--alpha', type=float, required=True, help="Significance level (e.g., 0.05)")
    parser_t_paired.add_argument('--power', type=float, help="Desired power (e.g., 0.80). Required unless solving for power.")
    parser_t_paired.add_argument('--sample-size', type=int, help="Sample size (number of pairs). Required unless solving for sample_size.")
    parser_t_paired.add_argument('--alternative', choices=['two-sided', 'larger', 'smaller'], default='two-sided', help="Alternative hypothesis")
    parser_t_paired.set_defaults(func=handle_t_paired)

    # --- One-Way ANOVA Subparser ---
    parser_anova_one = subparsers.add_parser('anova_one_way', help='Power/Sample Size/MDES for One-Way ANOVA')
    parser_anova_one.add_argument('--solve-for', choices=['power', 'sample_size', 'effect_size'], required=True, help="Parameter to calculate")
    # Effect size group
    group_anova_es = parser_anova_one.add_mutually_exclusive_group()
    group_anova_es.add_argument('--effect-size', type=float, help="Effect size (Cohen's f).")
    group_anova_es.add_argument('--eta-squared', type=float, help="Eta-squared (η²).")
    # Other args
    parser_anova_one.add_argument('--alpha', type=float, required=True, help="Significance level (e.g., 0.05)")
    parser_anova_one.add_argument('--k-groups', type=int, required=True, help="Number of groups (k)")
    parser_anova_one.add_argument('--power', type=float, help="Desired power (e.g., 0.80). Required unless solving for power.")
    parser_anova_one.add_argument('--sample-size', type=int, help="Total sample size (N) across all groups. Required unless solving for sample_size.")
    parser_anova_one.set_defaults(func=handle_anova_one_way)

    # --- Chi-Square Goodness-of-Fit Subparser ---
    parser_chisq_gof = subparsers.add_parser('chisq_gof', help='Power/Sample Size/MDES for Chi-Square Goodness-of-Fit Test')
    parser_chisq_gof.add_argument('--solve-for', choices=['power', 'sample_size', 'effect_size'], required=True, help="Parameter to calculate")
    # Effect size group
    group_chisq_gof_es = parser_chisq_gof.add_mutually_exclusive_group()
    group_chisq_gof_es.add_argument('--effect-size', type=float, help="Effect size (Cohen's w).")
    group_chisq_gof_es.add_argument('--props', type=float, nargs='+', metavar=('P_OBS_1 P_OBS_2 ... -- P_EXP_1 P_EXP_2 ...'), help="Calculate Cohen's w from observed and expected proportions (separate lists with '--').")
    # Other args
    parser_chisq_gof.add_argument('--alpha', type=float, required=True, help="Significance level (e.g., 0.05)")
    parser_chisq_gof.add_argument('--n-bins', type=int, help="Number of categories/bins. Required if using --effect-size. Inferred from --props if provided.")
    parser_chisq_gof.add_argument('--power', type=float, help="Desired power (e.g., 0.80). Required unless solving for power.")
    parser_chisq_gof.add_argument('--sample-size', type=int, help="Total sample size (N). Required unless solving for sample_size.")
    parser_chisq_gof.add_argument('--ddof', type=int, default=0, help="Delta degrees of freedom (adjustment to df). Default is 0.")
    parser_chisq_gof.set_defaults(func=handle_chisq_gof)

    # --- Chi-Square Test of Independence/Homogeneity Subparser ---
    # Keep simple for now, only direct Cohen's w input
    parser_chisq_ind = subparsers.add_parser('chisq_indep', help='Power/Sample Size/MDES for Chi-Square Test of Independence/Homogeneity')
    parser_chisq_ind.add_argument('--solve-for', choices=['power', 'sample_size', 'effect_size'], required=True, help="Parameter to calculate")
    parser_chisq_ind.add_argument('--effect-size', type=float, help="Effect size (Cohen's w). Required unless solving for effect_size.")
    parser_chisq_ind.add_argument('--alpha', type=float, required=True, help="Significance level (e.g., 0.05)")
    parser_chisq_ind.add_argument('--df', type=int, required=True, help="Degrees of freedom (e.g., (rows-1)*(cols-1))")
    parser_chisq_ind.add_argument('--power', type=float, help="Desired power (e.g., 0.80). Required unless solving for power.")
    parser_chisq_ind.add_argument('--sample-size', type=int, help="Total sample size (N). Required unless solving for sample_size.")
    parser_chisq_ind.set_defaults(func=handle_chisq_indep)

    # --- One-Proportion Z-Test Subparser ---
    parser_prop_one = subparsers.add_parser('prop_one_sample', help='Power/Sample Size/MDES for One-Proportion Z-Test')
    parser_prop_one.add_argument('--solve-for', choices=['power', 'sample_size', 'effect_size'], required=True, help="Parameter to calculate")
    # Effect size group
    group_prop_one_es = parser_prop_one.add_mutually_exclusive_group()
    group_prop_one_es.add_argument('--effect-size', type=float, help="Effect size (Cohen's h).")
    group_prop_one_es.add_argument('--props', type=float, nargs=2, metavar=('PROP1', 'PROP0'), help="Calculate Cohen's h from observed proportion (prop1) and null proportion (prop0).")
    # Other args
    parser_prop_one.add_argument('--alpha', type=float, required=True, help="Significance level (e.g., 0.05)")
    parser_prop_one.add_argument('--power', type=float, help="Desired power (e.g., 0.80). Required unless solving for power.")
    parser_prop_one.add_argument('--sample-size', type=int, help="Sample size (N). Required unless solving for sample_size.")
    parser_prop_one.add_argument('--alternative', choices=['two-sided', 'larger', 'smaller'], default='two-sided', help="Alternative hypothesis")
    parser_prop_one.set_defaults(func=handle_prop_one)

    # --- Two-Proportion Z-Test Subparser ---
    parser_prop_two = subparsers.add_parser('prop_two_sample', help='Power/Sample Size/MDES for Two-Proportion Z-Test (Independent)')
    parser_prop_two.add_argument('--solve-for', choices=['power', 'sample_size', 'effect_size'], required=True, help="Parameter to calculate")
    # Effect size group
    group_prop_two_es = parser_prop_two.add_mutually_exclusive_group()
    group_prop_two_es.add_argument('--effect-size', type=float, help="Effect size (Cohen's h).")
    group_prop_two_es.add_argument('--props', type=float, nargs=2, metavar=('PROP1', 'PROP2'), help="Calculate Cohen's h from the two proportions.")
    # Other args
    parser_prop_two.add_argument('--alpha', type=float, required=True, help="Significance level (e.g., 0.05)")
    parser_prop_two.add_argument('--power', type=float, help="Desired power (e.g., 0.80). Required unless solving for power.")
    parser_prop_two.add_argument('--sample-size', type=int, help="Sample size of the *first* group (n1). Required unless solving for sample_size.")
    parser_prop_two.add_argument('--ratio', type=float, default=1.0, help="Ratio of sample sizes (n2 / n1). Default is 1.0.")
    parser_prop_two.add_argument('--alternative', choices=['two-sided', 'larger', 'smaller'], default='two-sided', help="Alternative hypothesis")
    parser_prop_two.set_defaults(func=handle_prop_two)

    # --- McNemar Test Subparser ---
    parser_mcnemar = subparsers.add_parser('mcnemar', help="Power/Sample Size/MDES for McNemar's Test (Paired Proportions)")
    parser_mcnemar.add_argument('--solve-for', choices=['power', 'sample_size', 'effect_size'], required=True, help="Parameter to calculate (effect_size is diff in discordant props)")
    # Effect size defined by discordant proportions p01, p10
    parser_mcnemar.add_argument('--p01', type=float, help="Proportion changing 0->1. Required unless solving for effect_size.")
    parser_mcnemar.add_argument('--p10', type=float, help="Proportion changing 1->0. Required unless solving for effect_size.")
    parser_mcnemar.add_argument('--p-disc', type=float, help="Total discordant proportion (p01+p10). Required if solving for effect_size.")
    # Other args
    parser_mcnemar.add_argument('--alpha', type=float, required=True, help="Significance level (e.g., 0.05)")
    parser_mcnemar.add_argument('--power', type=float, help="Desired power (e.g., 0.80). Required unless solving for power.")
    parser_mcnemar.add_argument('--sample-size', type=int, help="Sample size (number of pairs). Required unless solving for sample_size.")
    parser_mcnemar.add_argument('--alternative', choices=['two-sided', 'larger', 'smaller'], default='two-sided', help="Alternative hypothesis (H1: p01 != p10, p01 > p10, or p01 < p10)")
    parser_mcnemar.add_argument('--no-continuity-correction', action='store_false', dest='continuity_correction', help="Disable continuity correction in calculations.")
    parser_mcnemar.set_defaults(func=handle_mcnemar)

    # --- Simple Linear Regression Subparser ---
    # Arguments already updated, just need to update handler call
    parser_linreg = subparsers.add_parser('linreg_simple', help='Power/Sample Size/MDES for Simple Linear Regression (Slope Test)')
    parser_linreg.add_argument('--solve-for', choices=['power', 'sample_size', 'effect_size', 'r_squared'], required=True, help="Parameter to calculate (effect_size is Cohen's f)")
    # Allow input as f or R^2, make optional depending on solve_for
    parser_linreg.add_argument('--effect-size', type=float, help="Effect size (Cohen's f). Required unless solving for effect_size/r_squared.")
    parser_linreg.add_argument('--r-squared', type=float, help="Squared correlation coefficient (R^2). Can be used instead of effect_size unless solving for effect_size/r_squared.")

    parser_linreg.add_argument('--alpha', type=float, required=True, help="Significance level (e.g., 0.05)")
    parser_linreg.add_argument('--power', type=float, help="Desired power (e.g., 0.80). Required unless solving for power.")
    parser_linreg.add_argument('--sample-size', type=int, help="Total sample size (N). Required unless solving for sample_size.")
    parser_linreg.set_defaults(func=handle_linreg_simple) # Handler needs update

    # --- Pearson Correlation Subparser ---
    # Arguments already updated, just need to update handler call
    parser_corr = subparsers.add_parser('corr_pearson', help='Power/Sample Size/MDES for Pearson Correlation Test (rho != 0)')
    parser_corr.add_argument('--solve-for', choices=['power', 'sample_size', 'effect_size'], required=True, help="Parameter to calculate")
    parser_corr.add_argument('--effect-size', type=float, help="Hypothesized population correlation (rho). Required unless solving for effect_size.")
    parser_corr.add_argument('--alpha', type=float, required=True, help="Significance level (e.g., 0.05)")
    parser_corr.add_argument('--power', type=float, help="Desired power (e.g., 0.80). Required unless solving for power.")
    parser_corr.add_argument('--sample-size', type=int, help="Sample size (number of pairs, N). Required unless solving for sample_size.")
    parser_corr.add_argument('--alternative', choices=['two-sided', 'larger', 'smaller'], default='two-sided', help="Alternative hypothesis (rho > 0 or rho < 0)")
    parser_corr.set_defaults(func=handle_corr_pearson) # Handler needs update

    # --- Multiple Linear Regression (Overall F-test) Subparser ---
    parser_mlr = subparsers.add_parser('linreg_multiple', help='Power/Sample Size/MDES for Multiple Linear Regression (Overall F-Test)')
    parser_mlr.add_argument('--solve-for', choices=['power', 'sample_size', 'effect_size', 'r_squared'], required=True, help="Parameter to calculate (effect_size is Cohen's f)")
    parser_mlr.add_argument('--num-predictors', type=int, required=True, help="Number of predictor variables (p)")
    # Allow input as f or R^2
    group_mlr_es = parser_mlr.add_mutually_exclusive_group()
    group_mlr_es.add_argument('--effect-size', type=float, help="Effect size (Cohen's f). Required unless solving for effect_size/r_squared.")
    group_mlr_es.add_argument('--r-squared', type=float, help="Squared multiple correlation coefficient (R^2) for the full model. Required unless solving for effect_size/r_squared.")

    parser_mlr.add_argument('--alpha', type=float, required=True, help="Significance level (e.g., 0.05)")
    parser_mlr.add_argument('--power', type=float, help="Desired power (e.g., 0.80). Required unless solving for power.")
    parser_mlr.add_argument('--sample-size', type=int, help="Total sample size (N). Required unless solving for sample_size.")
    parser_mlr.set_defaults(func=handle_linreg_multiple)

    # --- One-Sample Non-Inferiority/Superiority T-Test Subparser ---
    parser_t_one_ni = subparsers.add_parser('t_one_sample_ni', help='Power/Sample Size/MDES for One-Sample Non-Inferiority/Superiority T-Test')
    parser_t_one_ni.add_argument('--solve-for', choices=['power', 'sample_size', 'effect_size'], required=True, help="Parameter to calculate")
    parser_t_one_ni.add_argument('--effect-size', type=float, help="True effect size (Cohen's d). Required unless solving for effect_size.")
    parser_t_one_ni.add_argument('--alpha', type=float, required=True, help="Significance level (e.g., 0.05)")
    parser_t_one_ni.add_argument('--margin', type=float, required=True, help="Non-inferiority/superiority margin (standardized units, >0 typically)")
    parser_t_one_ni.add_argument('--power', type=float, help="Desired power (e.g., 0.80). Required unless solving for power.")
    parser_t_one_ni.add_argument('--sample-size', type=int, help="Sample size (N). Required unless solving for sample_size.")
    parser_t_one_ni.add_argument('--alternative', choices=['larger', 'smaller'], required=True, help="'larger' for non-inferiority, 'smaller' for superiority")
    parser_t_one_ni.set_defaults(func=handle_t_one_ni)

    # --- One-Sample Equivalence T-Test (TOST) Subparser ---
    parser_t_one_tost = subparsers.add_parser('t_one_sample_tost', help='Power/Sample Size for One-Sample Equivalence T-Test (TOST)')
    # Note: MDES calculation is not directly supported for TOST via this interface
    parser_t_one_tost.add_argument('--solve-for', choices=['power', 'sample_size'], required=True, help="Parameter to calculate (MDES not directly supported)")
    parser_t_one_tost.add_argument('--effect-size', type=float, default=0.0, help="Assumed true effect size (Cohen's d). Default is 0.0 (most conservative).")
    parser_t_one_tost.add_argument('--alpha', type=float, required=True, help="Significance level for each one-sided test (e.g., 0.05)")
    parser_t_one_tost.add_argument('--margin', type=float, required=True, help="Equivalence margin (standardized units, must be >0)")
    parser_t_one_tost.add_argument('--power', type=float, help="Desired power (e.g., 0.80). Required if solving for sample_size.")
    parser_t_one_tost.add_argument('--sample-size', type=int, help="Sample size (N). Required if solving for power.")
    parser_t_one_tost.set_defaults(func=handle_t_one_tost)

    # --- Mann-Whitney U Test (Simulation) Subparser ---
    parser_mw = subparsers.add_parser('mann_whitney', help='Power/Sample Size for Mann-Whitney U Test (via Simulation)')
    # Note: MDES not supported yet for simulation
    parser_mw.add_argument('--solve-for', choices=['power', 'sample_size'], required=True, help="Parameter to estimate via simulation (MDES not supported)")
    parser_mw.add_argument('--effect-size', type=float, help="Effect size (Cohen's d for underlying distributions). Required.")
    parser_mw.add_argument('--alpha', type=float, required=True, help="Significance level (e.g., 0.05)")
    parser_mw.add_argument('--power', type=float, help="Desired power (e.g., 0.80). Required if solving for sample_size.")
    parser_mw.add_argument('--sample-size', type=int, help="Sample size of the *first* group (n1). Required if solving for power.")
    parser_mw.add_argument('--ratio', type=float, default=1.0, help="Ratio of sample sizes (n2 / n1). Default is 1.0.")
    # Simulation specific args
    parser_mw.add_argument('--n-simulations', type=int, default=1000, help="Number of Monte Carlo simulations (default: 1000)")
    parser_mw.add_argument('--seed', type=int, help="Random seed for reproducibility (optional)")
    parser_mw.add_argument('--n-jobs', type=int, default=1, help="Number of CPU cores for parallel simulations (default: 1, use -1 for all)")
    parser_mw.set_defaults(func=handle_mann_whitney)

    # --- Logistic Regression (Simulation) Subparser ---
    parser_logit = subparsers.add_parser('logistic_regression', help='Power/Sample Size for Logistic Regression (Single Predictor, via Simulation)')
    # Note: MDES not supported yet for simulation
    parser_logit.add_argument('--solve-for', choices=['power', 'sample_size'], required=True, help="Parameter to estimate via simulation (MDES not supported)")
    # Effect size defined by OR and baseline prob
    parser_logit.add_argument('--odds-ratio', type=float, required=True, help="Odds ratio for a one-unit increase in the predictor.")
    parser_logit.add_argument('--baseline-prob', type=float, required=True, help="Baseline probability of outcome when predictor is 0.")
    # Predictor distribution params (optional, defaults to standard normal)
    parser_logit.add_argument('--predictor-mean', type=float, default=0.0, help="Mean of the (assumed normal) predictor variable (default: 0.0)")
    parser_logit.add_argument('--predictor-sd', type=float, default=1.0, help="SD of the (assumed normal) predictor variable (default: 1.0)")
    # Other standard args
    parser_logit.add_argument('--alpha', type=float, required=True, help="Significance level (e.g., 0.05)")
    parser_logit.add_argument('--power', type=float, help="Desired power (e.g., 0.80). Required if solving for sample_size.")
    parser_logit.add_argument('--sample-size', type=int, help="Total sample size (N). Required if solving for power.")
    # Simulation specific args
    parser_logit.add_argument('--n-simulations', type=int, default=1000, help="Number of Monte Carlo simulations (default: 1000)")
    parser_logit.add_argument('--seed', type=int, help="Random seed for reproducibility (optional)")
    parser_logit.add_argument('--n-jobs', type=int, default=1, help="Number of CPU cores for parallel simulations (default: 1, use -1 for all)")
    parser_logit.set_defaults(func=handle_logistic_regression)

    # --- Cox Proportional Hazards Subparser ---
    parser_cox = subparsers.add_parser('coxph', help='Power/Events/MDHR for Cox Proportional Hazards Model (2 Groups)')
    parser_cox.add_argument('--solve-for', choices=['power', 'num_events', 'effect_size'], required=True, help="Parameter to calculate (effect_size is Hazard Ratio)")
    parser_cox.add_argument('--effect-size', type=float, help="Hazard Ratio (HR). Required unless solving for effect_size.")
    parser_cox.add_argument('--alpha', type=float, required=True, help="Significance level (e.g., 0.05)")
    parser_cox.add_argument('--power', type=float, help="Desired power (e.g., 0.80). Required unless solving for power.")
    parser_cox.add_argument('--num-events', type=int, help="Total number of events. Required unless solving for num_events.")
    parser_cox.add_argument('--prop-group1', type=float, default=0.5, help="Proportion of subjects in group 1 (default: 0.5)")
    parser_cox.add_argument('--alternative', choices=['two-sided', 'larger', 'smaller'], default='two-sided', help="Alternative hypothesis (HR > 1 or HR < 1)")
    parser_cox.set_defaults(func=handle_coxph)

    # --- Log-rank Test Subparser ---
    parser_logrank = subparsers.add_parser('logrank', help='Power/Events/MDHR for Log-rank Test (2 Groups)')
    parser_logrank.add_argument('--solve-for', choices=['power', 'num_events', 'effect_size'], required=True, help="Parameter to calculate (effect_size is Hazard Ratio)")
    parser_logrank.add_argument('--effect-size', type=float, help="Hazard Ratio (HR). Required unless solving for effect_size.")
    parser_logrank.add_argument('--alpha', type=float, required=True, help="Significance level (e.g., 0.05)")
    parser_logrank.add_argument('--power', type=float, help="Desired power (e.g., 0.80). Required unless solving for power.")
    parser_logrank.add_argument('--num-events', type=int, help="Total number of events. Required unless solving for num_events.")
    parser_logrank.add_argument('--prop-group1', type=float, default=0.5, help="Proportion of subjects in group 1 (default: 0.5)")
    parser_logrank.add_argument('--alternative', choices=['two-sided', 'larger', 'smaller'], default='two-sided', help="Alternative hypothesis (HR > 1 or HR < 1)")
    parser_logrank.set_defaults(func=handle_logrank) # Use a new handler, although logic is similar to Cox

    # --- Poisson Regression (Simulation) Subparser ---
    parser_poisson = subparsers.add_parser('poisson_regression', help='Power/Sample Size for Poisson Regression (Single Predictor, via Simulation)')
    # Note: MDES not supported yet for simulation
    parser_poisson.add_argument('--solve-for', choices=['power', 'sample_size'], required=True, help="Parameter to estimate via simulation (MDES not supported)")
    # Effect size defined by Rate Ratio and baseline rate
    parser_poisson.add_argument('--rate-ratio', type=float, required=True, help="Rate ratio (exp(beta1)) for a one-unit increase in the predictor.")
    parser_poisson.add_argument('--baseline-rate', type=float, required=True, help="Baseline expected count (rate) when predictor is 0.")
    # Predictor distribution params (optional, defaults to standard normal)
    parser_poisson.add_argument('--predictor-mean', type=float, default=0.0, help="Mean of the (assumed normal) predictor variable (default: 0.0)")
    parser_poisson.add_argument('--predictor-sd', type=float, default=1.0, help="SD of the (assumed normal) predictor variable (default: 1.0)")
    # Other standard args
    parser_poisson.add_argument('--alpha', type=float, required=True, help="Significance level (e.g., 0.05)")
    parser_poisson.add_argument('--power', type=float, help="Desired power (e.g., 0.80). Required if solving for sample_size.")
    parser_poisson.add_argument('--sample-size', type=int, help="Total sample size (N). Required if solving for power.")
    # Simulation specific args
    parser_poisson.add_argument('--n-simulations', type=int, default=1000, help="Number of Monte Carlo simulations (default: 1000)")
    parser_poisson.add_argument('--seed', type=int, help="Random seed for reproducibility (optional)")
    parser_poisson.add_argument('--n-jobs', type=int, default=1, help="Number of CPU cores for parallel simulations (default: 1, use -1 for all)")
    parser_poisson.set_defaults(func=handle_poisson_regression)


    # --- Add subparsers for other tests here as implemented ---
    # ...

    args = parser.parse_args()

    # Validate required arguments based on solve_for
    # Handle specific effect size input validation within handlers
    # Validate required arguments based on solve_for
    # Handle specific effect size input validation within handlers
    # Exclude TOST from effect_size solving validation
    # Exclude linreg_simple/multiple as they have specific validation
    # Exclude logistic_regression/poisson_regression as they use specific params
    # Exclude mcnemar as it has specific validation
    # Exclude survival tests (coxph, logrank) as they have specific validation below
    if args.test_type not in ['linreg_simple', 'linreg_multiple', 't_one_sample_tost', 'logistic_regression', 'poisson_regression', 'mcnemar', 'coxph', 'logrank']:
        effect_size_provided = False
        # Check direct effect size arg
        if hasattr(args, 'effect_size') and args.effect_size is not None: effect_size_provided = True
        # Check alternative effect size inputs specific to test types
        if args.test_type == 't_one_sample' and hasattr(args, 'means_sd') and args.means_sd is not None: effect_size_provided = True
        if args.test_type == 't_independent' and hasattr(args, 'means_sds_ns') and args.means_sds_ns is not None: effect_size_provided = True
        if args.test_type == 't_paired' and hasattr(args, 'diff_mean_sd') and args.diff_mean_sd is not None: effect_size_provided = True
        if args.test_type == 'anova_one_way' and hasattr(args, 'eta_squared') and args.eta_squared is not None: effect_size_provided = True
        if args.test_type == 'chisq_gof' and hasattr(args, 'props') and args.props is not None: effect_size_provided = True
        if args.test_type in ['prop_one_sample', 'prop_two_sample'] and hasattr(args, 'props') and args.props is not None: effect_size_provided = True

        if args.solve_for == 'power':
             if not effect_size_provided: parser.error(f"[{args.test_type}] --effect-size (or alternative input like --means-sd, --props, etc.) is required when solving for power.")
             if args.sample_size is None: parser.error(f"[{args.test_type}] --sample-size is required when solving for power.")
        elif args.solve_for == 'sample_size':
             if not effect_size_provided: parser.error(f"[{args.test_type}] --effect-size (or alternative input) is required when solving for sample_size.")
             if args.power is None: parser.error(f"[{args.test_type}] --power is required when solving for sample_size.")
        elif args.solve_for == 'effect_size':
             # Specific check for TOST where MDES is not supported via CLI
             if args.test_type == 't_one_sample_tost':
                  parser.error(f"[{args.test_type}] Solving for effect_size (MDES) is not directly supported for TOST.")
             # General check (should not apply to survival now)
             if args.power is None or args.sample_size is None:
                 parser.error(f"[{args.test_type}] --power and --sample-size are required when solving for effect_size (MDES).")
             if effect_size_provided:
                  # Allow providing effect size even when solving for it, but warn
                  warnings.warn(f"[{args.test_type}] Effect size input provided but solving for effect_size (MDES). Input effect size will be ignored in calculation but shown for context if calculated.")
    # Validation for TOST (power/sample_size only)
    elif args.test_type == 't_one_sample_tost':
         if args.solve_for == 'power' and args.sample_size is None:
              parser.error(f"[{args.test_type}] --sample-size is required when solving for power.")
         if args.solve_for == 'sample_size' and args.power is None:
              parser.error(f"[{args.test_type}] --power is required when solving for sample_size.")
    # Validation for Mann-Whitney (power/sample_size only, effect_size required)
    elif args.test_type == 'mann_whitney':
         if args.effect_size is None:
              parser.error(f"[{args.test_type}] --effect-size is required for simulation.")
         if args.solve_for == 'power' and args.sample_size is None:
              parser.error(f"[{args.test_type}] --sample-size is required when solving for power.")
         if args.solve_for == 'sample_size' and args.power is None:
              parser.error(f"[{args.test_type}] --power is required when solving for sample_size.")
    # Validation for Logistic Regression (power/sample_size only, OR/baseline required)
    elif args.test_type == 'logistic_regression':
         if args.odds_ratio is None or args.baseline_prob is None:
              parser.error(f"[{args.test_type}] --odds-ratio and --baseline-prob are required for simulation.")
         if args.solve_for == 'power' and args.sample_size is None:
              parser.error(f"[{args.test_type}] --sample-size is required when solving for power.")
         if args.solve_for == 'sample_size' and args.power is None:
              parser.error(f"[{args.test_type}] --power is required when solving for sample_size.")
    # Validation for Poisson Regression (power/sample_size only, RR/baseline required)
    elif args.test_type == 'poisson_regression':
         if args.rate_ratio is None or args.baseline_rate is None:
              parser.error(f"[{args.test_type}] --rate-ratio and --baseline-rate are required for simulation.")
         if args.solve_for == 'power' and args.sample_size is None:
              parser.error(f"[{args.test_type}] --sample-size is required when solving for power.")
         if args.solve_for == 'sample_size' and args.power is None:
              parser.error(f"[{args.test_type}] --power is required when solving for sample_size.")
    # Validation for CoxPH / LogRank
    elif args.test_type in ['coxph', 'logrank']:
         if args.solve_for == 'power' and (args.effect_size is None or args.num_events is None):
             parser.error(f"[{args.test_type}] --effect-size (HR) and --num-events are required when solving for power.")
         if args.solve_for == 'num_events' and (args.effect_size is None or args.power is None):
             parser.error(f"[{args.test_type}] --effect-size (HR) and --power are required when solving for num_events.")
         if args.solve_for == 'effect_size' and (args.power is None or args.num_events is None):
             parser.error(f"[{args.test_type}] --power and --num-events are required when solving for effect_size (MDHR).")


    # Call the function associated with the chosen subparser
    args.func(args)


def handle_t_one_sample(args):
    """Handles the calculation for the one-sample t-test."""
    print("\n--- One-Sample T-Test Calculation ---")
    print(f"Solving for: {args.solve_for}")
    print(f"Parameters:")

    effect_size_d = args.effect_size
    # Calculate effect size if alternative inputs provided
    if args.means_sd is not None:
        m_samp, m_pop, sd_samp = args.means_sd
        print(f"  Input Sample Mean: {m_samp}")
        print(f"  Input Population Mean (H0): {m_pop}")
        print(f"  Input Sample SD: {sd_samp}")
        try:
            effect_size_d = calculate_cohens_d_one_sample(m_samp, m_pop, sd_samp)
            print(f"  (Calculated Cohen's d: {effect_size_d:.4f})")
        except ValueError as e:
             print(f"Error calculating Cohen's d: {e}", file=sys.stderr); sys.exit(1)

    if effect_size_d is not None: print(f"  Effect Size (Cohen's d): {effect_size_d}")
    print(f"  Alpha: {args.alpha}")
    if args.power is not None: print(f"  Power: {args.power}")
    if args.sample_size is not None: print(f"  Sample Size (N): {args.sample_size}")
    print(f"  Alternative Hypothesis: {args.alternative}")
    print("-" * 35)

    try:
        calculator = OneSampleTTestPower(
            alpha=args.alpha,
            effect_size=effect_size_d if args.solve_for != 'effect_size' else None,
            power=args.power if args.solve_for != 'power' else None,
            sample_size=args.sample_size if args.solve_for != 'sample_size' else None,
            alternative=args.alternative
        )
        result = calculator.solve()

        print(f"Result:")
        if args.solve_for == 'power':
            print(f"  Calculated Power: {result:.4f}")
        elif args.solve_for == 'sample_size':
            print(f"  Required Sample Size (N): {result}")
        elif args.solve_for == 'effect_size':
             if np.isinf(result):
                 print(f"  Minimum Detectable Effect Size (Cohen's d): Infinite (Sample size may be too small)")
             else:
                 print(f"  Minimum Detectable Effect Size (Cohen's d): {result:.4f}")

    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except NotImplementedError:
         print(f"Error: Calculation not implemented for these parameters.", file=sys.stderr)
         sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)

    print("-" * 35)


def handle_t_independent(args):
    """Handles the calculation for the independent samples t-test."""
    print("\n--- Independent Samples T-Test Calculation ---")
    print(f"Solving for: {args.solve_for}")
    print(f"Parameters:")

    effect_size_d = args.effect_size
    n1_input = args.sample_size # Sample size arg corresponds to n1
    ratio_input = args.ratio

    # Calculate effect size if alternative inputs provided
    if args.means_sds_ns is not None:
        m1, sd1, n1_calc, m2, sd2, n2_calc = args.means_sds_ns
        print(f"  Input Group 1: Mean={m1}, SD={sd1}, N={int(n1_calc)}")
        print(f"  Input Group 2: Mean={m2}, SD={sd2}, N={int(n2_calc)}")
        try:
            # Use pooled=True/False based on usevar argument
            pooled_calc = (args.usevar == 'pooled')
            effect_size_d = calculate_cohens_d_independent(m1, sd1, n1_calc, m2, sd2, n2_calc, pooled=pooled_calc)
            print(f"  (Calculated Cohen's d: {effect_size_d:.4f})")
            # If solving for power/MDES, use the Ns provided for calculation
            if args.solve_for != 'sample_size':
                 n1_input = int(n1_calc)
                 if n1_calc > 0: ratio_input = n2_calc / n1_calc
                 else: ratio_input = 1 # Avoid division by zero, default to 1
                 print(f"  (Using N1={n1_input}, Ratio={ratio_input:.2f} from inputs for calculation)")

        except ValueError as e:
             print(f"Error calculating Cohen's d: {e}", file=sys.stderr); sys.exit(1)

    if effect_size_d is not None: print(f"  Effect Size (Cohen's d): {effect_size_d}")
    print(f"  Alpha: {args.alpha}")
    if args.power is not None: print(f"  Power: {args.power}")
    # Use n1_input which might have been updated if means_sds_ns was provided
    if n1_input is not None: print(f"  Sample Size (n1): {n1_input}")
    # Use ratio_input which might have been updated
    print(f"  Ratio (n2/n1): {ratio_input}")
    print(f"  Alternative Hypothesis: {args.alternative}")
    print(f"  Variance Assumption: {args.usevar}")
    if args.icc is not None:
        print(f"  Clustering ICC: {args.icc}")
        if args.cluster_size is not None: print(f"  Avg Cluster Size (m): {args.cluster_size}")
    print("-" * 42) # Adjusted width

    try:
        calculator = IndependentSamplesTTestPower(
            alpha=args.alpha,
            effect_size=effect_size_d if args.solve_for != 'effect_size' else None,
            power=args.power if args.solve_for != 'power' else None,
            sample_size=n1_input if args.solve_for != 'sample_size' else None,
            ratio=ratio_input,
            alternative=args.alternative,
            usevar=args.usevar,
            icc=args.icc,
            cluster_size=args.cluster_size
        )
        result = calculator.solve()

        print(f"Result:")
        if args.solve_for == 'power':
            print(f"  Calculated Power: {result:.4f}")
        elif args.solve_for == 'sample_size':
            n1_req = result
            n2_req = np.ceil(n1_req * ratio_input).astype(int) # Use ratio_input
            n_total = n1_req + n2_req
            print(f"  Required Sample Size (n1): {n1_req}")
            print(f"  Required Sample Size (n2): {n2_req}")
            print(f"  Total Sample Size (N): {n_total}")
        elif args.solve_for == 'effect_size':
             if np.isinf(result):
                 print(f"  Minimum Detectable Effect Size (Cohen's d): Infinite (Sample size may be too small)")
             else:
                 print(f"  Minimum Detectable Effect Size (Cohen's d): {result:.4f}")

    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except NotImplementedError:
         print(f"Error: Calculation not implemented for these parameters.", file=sys.stderr)
         sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)

    print("-" * 42)


def handle_t_paired(args):
    """Handles the calculation for the paired samples t-test."""
    print("\n--- Paired Samples T-Test Calculation ---")
    print(f"Solving for: {args.solve_for}")
    print(f"Parameters:")

    effect_size_d = args.effect_size
    # Calculate effect size if alternative inputs provided
    if args.diff_mean_sd is not None:
        mean_d, sd_d = args.diff_mean_sd
        print(f"  Input Mean Difference: {mean_d}")
        print(f"  Input SD of Differences: {sd_d}")
        try:
            effect_size_d = calculate_cohens_d_paired(mean_d, sd_d)
            print(f"  (Calculated Cohen's d: {effect_size_d:.4f})")
        except ValueError as e:
             print(f"Error calculating Cohen's d: {e}", file=sys.stderr); sys.exit(1)

    if effect_size_d is not None: print(f"  Effect Size (Cohen's d of differences): {effect_size_d}")
    print(f"  Alpha: {args.alpha}")
    if args.power is not None: print(f"  Power: {args.power}")
    if args.sample_size is not None: print(f"  Sample Size (Number of Pairs): {args.sample_size}")
    print(f"  Alternative Hypothesis: {args.alternative}")
    print("-" * 41) # Adjusted width

    try:
        # Uses the same underlying calculation as one-sample t-test on differences
        calculator = PairedSamplesTTestPower(
            alpha=args.alpha,
            effect_size=effect_size_d if args.solve_for != 'effect_size' else None,
            power=args.power if args.solve_for != 'power' else None,
            sample_size=args.sample_size if args.solve_for != 'sample_size' else None,
            alternative=args.alternative
        )
        result = calculator.solve()

        print(f"Result:")
        if args.solve_for == 'power':
            print(f"  Calculated Power: {result:.4f}")
        elif args.solve_for == 'sample_size':
            print(f"  Required Sample Size (Number of Pairs): {result}")
        elif args.solve_for == 'effect_size':
             if np.isinf(result):
                 print(f"  Minimum Detectable Effect Size (Cohen's d): Infinite (Sample size may be too small)")
             else:
                 print(f"  Minimum Detectable Effect Size (Cohen's d): {result:.4f}")

    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except NotImplementedError:
         print(f"Error: Calculation not implemented for these parameters.", file=sys.stderr)
         sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)

    print("-" * 41)


def handle_anova_one_way(args):
    """Handles the calculation for the one-way ANOVA."""
    print("\n--- One-Way ANOVA Calculation ---")
    print(f"Solving for: {args.solve_for}")
    print(f"Parameters:")

    effect_size_f = args.effect_size
    # Calculate effect size if alternative inputs provided
    if args.eta_squared is not None:
        print(f"  Input Eta-squared: {args.eta_squared}")
        try:
            effect_size_f = calculate_cohens_f_from_eta_sq(args.eta_squared)
            print(f"  (Calculated Cohen's f: {effect_size_f:.4f})")
        except ValueError as e:
             print(f"Error calculating Cohen's f: {e}", file=sys.stderr); sys.exit(1)

    if effect_size_f is not None: print(f"  Effect Size (Cohen's f): {effect_size_f}")
    print(f"  Alpha: {args.alpha}")
    print(f"  Number of Groups (k): {args.k_groups}")
    if args.power is not None: print(f"  Power: {args.power}")
    if args.sample_size is not None: print(f"  Total Sample Size (N): {args.sample_size}")
    print("-" * 33) # Adjusted width

    try:
        calculator = OneWayANOVAPower(
            alpha=args.alpha,
            k_groups=args.k_groups,
            effect_size=effect_size_f if args.solve_for != 'effect_size' else None,
            power=args.power if args.solve_for != 'power' else None,
            sample_size=args.sample_size if args.solve_for != 'sample_size' else None
        )
        result = calculator.solve()

        print(f"Result:")
        if args.solve_for == 'power':
            print(f"  Calculated Power: {result:.4f}")
        elif args.solve_for == 'sample_size':
            n_total = result
            n_per_group = n_total / args.k_groups
            print(f"  Required Total Sample Size (N): {n_total}")
            if n_per_group == int(n_per_group):
                 print(f"  Sample Size Per Group (n): {int(n_per_group)}")
            else:
                 print(f"  Approx. Sample Size Per Group (n): {n_per_group:.2f} (Total N adjusted for equal groups)")
        elif args.solve_for == 'effect_size':
             if np.isinf(result):
                 print(f"  Minimum Detectable Effect Size (Cohen's f): Infinite (Sample size may be too small)")
             else:
                 print(f"  Minimum Detectable Effect Size (Cohen's f): {result:.4f}")

    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except NotImplementedError:
         print(f"Error: Calculation not implemented for these parameters.", file=sys.stderr)
         sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)

    print("-" * 33)


def handle_chisq_gof(args):
    """Handles the calculation for the Chi-Square Goodness-of-Fit test."""
    print("\n--- Chi-Square Goodness-of-Fit Calculation ---")
    print(f"Solving for: {args.solve_for}")
    print(f"Parameters:")

    effect_size_w = args.effect_size
    n_bins = args.n_bins

    # Calculate effect size if proportions provided
    if args.props is not None:
        props_list = args.props
        try:
            # Find the separator '--' (argparse might read it as a string)
            # Convert potential float representations back to string for searching
            props_str_list = [str(p) for p in props_list]
            separator = '--'
            if separator not in props_str_list:
                 raise ValueError("Separator '--' not found in --props list.")

            separator_index = props_str_list.index(separator)
            p_obs_str = props_str_list[:separator_index]
            p_exp_str = props_str_list[separator_index+1:]

            # Convert back to float
            p_obs = [float(p) for p in p_obs_str]
            p_exp = [float(p) for p in p_exp_str]

            if len(p_obs) != len(p_exp):
                 raise ValueError("Observed and expected proportion lists must have the same number of elements.")
            if args.n_bins is not None and args.n_bins != len(p_obs):
                 warnings.warn(f"--n-bins ({args.n_bins}) provided but differs from length of proportion lists ({len(p_obs)}). Using length from lists.")
            n_bins = len(p_obs) # Infer n_bins from input
            print(f"  Input Observed Proportions: {p_obs}")
            print(f"  Input Expected Proportions: {p_exp}")
            effect_size_w = calculate_cohens_w(p_obs, p_exp)
            print(f"  (Calculated Cohen's w: {effect_size_w:.4f})")
        except ValueError as e:
             print(f"Error parsing --props: {e}. Use '--' to separate observed and expected lists.", file=sys.stderr); sys.exit(1)
        except Exception as e:
             print(f"Error calculating Cohen's w: {e}", file=sys.stderr); sys.exit(1)

    if n_bins is None: # Required if not inferred from props
         parser.error("[chisq_gof] --n-bins is required if --props is not used.")

    df = n_bins - 1 - args.ddof
    if effect_size_w is not None: print(f"  Effect Size (Cohen's w): {effect_size_w}")
    print(f"  Alpha: {args.alpha}")
    print(f"  Number of Bins: {n_bins}")
    if args.ddof > 0: print(f"  Delta df (ddof): {args.ddof}")
    print(f"  (Calculated df: {df})")
    if args.power is not None: print(f"  Power: {args.power}")
    if args.sample_size is not None: print(f"  Total Sample Size (N): {args.sample_size}")
    print("-" * 44) # Adjusted width

    try:
        calculator = ChiSquareGofPower(
            alpha=args.alpha,
            n_bins=n_bins, # Use inferred or provided n_bins
            effect_size=effect_size_w if args.solve_for != 'effect_size' else None,
            power=args.power if args.solve_for != 'power' else None,
            sample_size=args.sample_size if args.solve_for != 'sample_size' else None,
            ddof=args.ddof
        )
        result = calculator.solve()

        print(f"Result:")
        if args.solve_for == 'power':
            print(f"  Calculated Power: {result:.4f}")
        elif args.solve_for == 'sample_size':
            print(f"  Required Total Sample Size (N): {result}")
        elif args.solve_for == 'effect_size':
             if np.isinf(result):
                 print(f"  Minimum Detectable Effect Size (Cohen's w): Infinite (Sample size may be too small for df)")
             else:
                 print(f"  Minimum Detectable Effect Size (Cohen's w): {result:.4f}")

    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except NotImplementedError:
         print(f"Error: Calculation not implemented for these parameters.", file=sys.stderr)
         sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)

    print("-" * 44)


def handle_chisq_indep(args):
    """Handles the calculation for the Chi-Square Test of Independence/Homogeneity."""
    print("\n--- Chi-Square Test of Independence/Homogeneity Calculation ---")
    print(f"Solving for: {args.solve_for}")
    print(f"Parameters:")
    if args.effect_size is not None: print(f"  Effect Size (Cohen's w): {args.effect_size}")
    print(f"  Alpha: {args.alpha}")
    print(f"  Degrees of Freedom (df): {args.df}")
    if args.power is not None: print(f"  Power: {args.power}")
    if args.sample_size is not None: print(f"  Total Sample Size (N): {args.sample_size}")
    print("-" * 60) # Adjusted width

    try:
        calculator = ChiSquareIndPower(
            alpha=args.alpha,
            df=args.df,
            effect_size=args.effect_size if args.solve_for != 'effect_size' else None,
            power=args.power if args.solve_for != 'power' else None,
            sample_size=args.sample_size if args.solve_for != 'sample_size' else None
        )
        result = calculator.solve()

        print(f"Result:")
        if args.solve_for == 'power':
            print(f"  Calculated Power: {result:.4f}")
        elif args.solve_for == 'sample_size':
            print(f"  Required Total Sample Size (N): {result}")
        elif args.solve_for == 'effect_size':
             if np.isinf(result):
                 print(f"  Minimum Detectable Effect Size (Cohen's w): Infinite (Sample size may be too small for df)")
             else:
                 print(f"  Minimum Detectable Effect Size (Cohen's w): {result:.4f}")

    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except NotImplementedError:
         print(f"Error: Calculation not implemented for these parameters.", file=sys.stderr)
         sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)

    print("-" * 60)


def handle_prop_one(args):
    """Handles the calculation for the one-proportion z-test."""
    print("\n--- One-Proportion Z-Test Calculation ---")
    print(f"Solving for: {args.solve_for}")
    print(f"Parameters:")

    effect_size_h = args.effect_size
    # Calculate effect size if proportions provided
    if args.props is not None:
        p1, p0 = args.props
        print(f"  Input Observed Proportion (p1): {p1}")
        print(f"  Input Null Proportion (p0): {p0}")
        try:
            effect_size_h = calculate_cohens_h(p1, p0)
            print(f"  (Calculated Cohen's h: {effect_size_h:.4f})")
        except ValueError as e:
             print(f"Error calculating Cohen's h: {e}", file=sys.stderr); sys.exit(1)

    if effect_size_h is not None: print(f"  Effect Size (Cohen's h): {effect_size_h}")
    print(f"  Alpha: {args.alpha}")
    if args.power is not None: print(f"  Power: {args.power}")
    if args.sample_size is not None: print(f"  Sample Size (N): {args.sample_size}")
    print(f"  Alternative Hypothesis: {args.alternative}")
    print("-" * 39) # Adjusted width

    try:
        calculator = OneProportionZTestPower(
            alpha=args.alpha,
            effect_size=effect_size_h if args.solve_for != 'effect_size' else None,
            power=args.power if args.solve_for != 'power' else None,
            sample_size=args.sample_size if args.solve_for != 'sample_size' else None,
            alternative=args.alternative
        )
        result = calculator.solve()

        print(f"Result:")
        if args.solve_for == 'power':
             print(f"  Calculated Power: {result:.4f}")
        elif args.solve_for == 'sample_size':
             if np.isinf(result):
                 print(f"  Required Sample Size (N): Infinite (Effect size may be 0)")
             else:
                 print(f"  Required Sample Size (N): {result}")
        elif args.solve_for == 'effect_size':
             if np.isinf(result):
                  print(f"  Minimum Detectable Effect Size (Cohen's h): Infinite (Sample size may be too small or power target too high)")
             else:
                  print(f"  Minimum Detectable Effect Size (Cohen's h): {result:.4f}")

    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except NotImplementedError:
         print(f"Error: Calculation not implemented for these parameters.", file=sys.stderr)
         sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)

    print("-" * 39)


def handle_prop_two(args):
    """Handles the calculation for the two-proportion z-test."""
    print("\n--- Two-Proportion Z-Test Calculation ---")
    print(f"Solving for: {args.solve_for}")
    print(f"Parameters:")

    effect_size_h = args.effect_size
    # Calculate effect size if proportions provided
    if args.props is not None:
        p1, p2 = args.props
        print(f"  Input Proportion 1: {p1}")
        print(f"  Input Proportion 2: {p2}")
        try:
            effect_size_h = calculate_cohens_h(p1, p2)
            print(f"  (Calculated Cohen's h: {effect_size_h:.4f})")
        except ValueError as e:
             print(f"Error calculating Cohen's h: {e}", file=sys.stderr); sys.exit(1)

    if effect_size_h is not None: print(f"  Effect Size (Cohen's h): {effect_size_h}")
    print(f"  Alpha: {args.alpha}")
    if args.power is not None: print(f"  Power: {args.power}")
    if args.sample_size is not None: print(f"  Sample Size (n1): {args.sample_size}")
    print(f"  Ratio (n2/n1): {args.ratio}")
    print(f"  Alternative Hypothesis: {args.alternative}")
    print("-" * 41) # Adjusted width

    try:
        calculator = TwoProportionZTestPower(
            alpha=args.alpha,
            effect_size=effect_size_h if args.solve_for != 'effect_size' else None,
            power=args.power if args.solve_for != 'power' else None,
            sample_size=args.sample_size if args.solve_for != 'sample_size' else None,
            ratio=args.ratio,
            alternative=args.alternative
        )
        result = calculator.solve()

        print(f"Result:")
        if args.solve_for == 'power':
             print(f"  Calculated Power: {result:.4f}")
        elif args.solve_for == 'sample_size':
             if np.isinf(result):
                 print(f"  Required Sample Size (n1): Infinite (Effect size may be 0)")
                 print(f"  Required Sample Size (n2): Infinite")
                 print(f"  Total Sample Size (N): Infinite")
             else:
                 n1 = result
                 n2 = np.ceil(n1 * args.ratio).astype(int)
                 n_total = n1 + n2
                 print(f"  Required Sample Size (n1): {n1}")
                 print(f"  Required Sample Size (n2): {n2}")
                 print(f"  Total Sample Size (N): {n_total}")
        elif args.solve_for == 'effect_size':
             if np.isinf(result):
                  print(f"  Minimum Detectable Effect Size (Cohen's h): Infinite (Sample size may be too small)")
             else:
                  print(f"  Minimum Detectable Effect Size (Cohen's h): {result:.4f}")

    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except NotImplementedError:
         print(f"Error: Calculation not implemented for these parameters.", file=sys.stderr)
         sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)

    print("-" * 41)


def handle_mcnemar(args):
    """Handles the calculation for McNemar's test."""
    print("\n--- McNemar's Test Calculation ---")
    print(f"Solving for: {args.solve_for}")
    print(f"Parameters:")

    # Validate inputs based on solve_for
    if args.solve_for in ['power', 'sample_size']:
        if args.p01 is None or args.p10 is None:
             parser.error("[mcnemar] --p01 and --p10 are required when solving for power or sample_size.")
        print(f"  Proportion 0->1 (p01): {args.p01}")
        print(f"  Proportion 1->0 (p10): {args.p10}")
    elif args.solve_for == 'effect_size':
         if args.p_disc is None:
              parser.error("[mcnemar] --p-disc (total discordant proportion) is required when solving for effect_size (MDES).")
         print(f"  Total Discordant Proportion (p_disc): {args.p_disc}")

    print(f"  Alpha: {args.alpha}")
    if args.power is not None: print(f"  Power: {args.power}")
    if args.sample_size is not None: print(f"  Sample Size (N pairs): {args.sample_size}")
    print(f"  Alternative Hypothesis: {args.alternative}")
    print(f"  Continuity Correction: {args.continuity_correction}")
    print("-" * 40) # Adjusted width

    try:
        calculator = McNemarTestPower(
            alpha=args.alpha,
            p01=args.p01 if args.solve_for != 'effect_size' else None,
            p10=args.p10 if args.solve_for != 'effect_size' else None,
            power=args.power if args.solve_for != 'power' else None,
            sample_size=args.sample_size if args.solve_for != 'sample_size' else None,
            alternative=args.alternative,
            continuity_correction=args.continuity_correction,
            # Pass p_disc if solving for MDES
            p_disc=args.p_disc if args.solve_for == 'effect_size' else None
        )
        result = calculator.solve()

        print(f"Result:")
        if args.solve_for == 'power':
             print(f"  Calculated Power: {result:.4f}")
        elif args.solve_for == 'sample_size':
             if np.isinf(result):
                 print(f"  Required Sample Size (N pairs): Infinite (Effect size may be 0 or p_disc=0)")
             else:
                 print(f"  Required Sample Size (N pairs): {result}")
        elif args.solve_for == 'effect_size':
             if np.isinf(result) or np.isnan(result):
                  print(f"  Minimum Detectable Difference |p01-p10|: Cannot be determined (N or power may be too low)")
             else:
                  print(f"  Minimum Detectable Difference |p01-p10|: {result:.4f}")

    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except NotImplementedError as e:
         print(f"Error: {e}", file=sys.stderr)
         sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)

    print("-" * 40)


def handle_linreg_simple(args):
    """Handles the calculation for simple linear regression."""
    print("\n--- Simple Linear Regression (Slope Test) Calculation ---")
    print(f"Solving for: {args.solve_for}")
    print(f"Parameters:")

    effect_size_f = None
    # Input validation specific to linreg_simple
    if args.solve_for in ['power', 'sample_size']:
        if args.effect_size is None and args.r_squared is None:
             parser.error("[linreg_simple] Either --effect-size (Cohen's f) or --r-squared must be provided when solving for power or sample_size.")
        if args.effect_size is not None and args.r_squared is not None:
             parser.error("[linreg_simple] Provide either --effect-size or --r-squared, not both.")
        if args.effect_size is not None:
             effect_size_f = args.effect_size
             print(f"  Effect Size (Cohen's f): {effect_size_f}")
        else: # args.r_squared is not None
             print(f"  R-squared: {args.r_squared}")
             try:
                 if not (0 <= args.r_squared < 1): raise ValueError("R-squared must be >= 0 and < 1.")
                 if args.r_squared == 0: effect_size_f = 0.0
                 else: effect_size_f = np.sqrt(args.r_squared / (1 - args.r_squared))
                 print(f"  (Implied Cohen's f: {effect_size_f:.4f})")
             except ValueError as e:
                 print(f"Error: {e}", file=sys.stderr); sys.exit(1)
    # For solving effect_size/r_squared, effect_size_f starts as None

    print(f"  Alpha: {args.alpha}")
    if args.power is not None: print(f"  Power: {args.power}")
    if args.sample_size is not None: print(f"  Total Sample Size (N): {args.sample_size}")
    print("-" * 55) # Adjusted width

    # Validate remaining args based on solve_for
    if args.solve_for == 'power' and args.sample_size is None:
         parser.error("[linreg_simple] --sample-size is required when solving for power.")
    if args.solve_for == 'sample_size' and args.power is None:
         parser.error("[linreg_simple] --power is required when solving for sample_size.")
    if args.solve_for in ['effect_size', 'r_squared'] and (args.power is None or args.sample_size is None):
         parser.error("[linreg_simple] --power and --sample-size are required when solving for effect_size or r_squared.")


    try:
        calculator = SimpleLinearRegressionPower(
            alpha=args.alpha,
            # Pass effect_size_f if it was provided or calculated, otherwise None
            effect_size=effect_size_f if args.solve_for not in ['effect_size', 'r_squared'] else None,
            power=args.power if args.solve_for != 'power' else None,
            sample_size=args.sample_size if args.solve_for != 'sample_size' else None
        )
        result = calculator.solve()

        print(f"Result:")
        if args.solve_for == 'power':
            print(f"  Calculated Power: {result:.4f}")
        elif args.solve_for == 'sample_size':
             if np.isinf(result):
                 print(f"  Required Total Sample Size (N): Infinite (Effect size may be 0)")
             else:
                 print(f"  Required Total Sample Size (N): {result}")
        elif args.solve_for == 'effect_size':
             if np.isinf(result):
                 print(f"  Minimum Detectable Effect Size (Cohen's f): Infinite (Sample size may be too small)")
             else:
                 print(f"  Minimum Detectable Effect Size (Cohen's f): {result:.4f}")
        elif args.solve_for == 'r_squared':
             if np.isinf(result):
                  print(f"  Minimum Detectable R-squared: Infinite (Sample size may be too small)")
             else:
                  # Convert Cohen's f back to R^2: R^2 = f^2 / (1 + f^2)
                  mdes_f = result
                  mdes_r2 = mdes_f**2 / (1 + mdes_f**2) if mdes_f != 0 else 0.0
                  print(f"  Minimum Detectable R-squared: {mdes_r2:.4f}")
                  print(f"  (Corresponding Cohen's f: {mdes_f:.4f})")

    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except NotImplementedError:
         print(f"Error: Calculation not implemented for these parameters.", file=sys.stderr)
         sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)

    print("-" * 55)


def handle_corr_pearson(args):
    """Handles the calculation for the Pearson correlation test."""
    print("\n--- Pearson Correlation Test (rho != 0) Calculation ---")
    print(f"Solving for: {args.solve_for}")
    print(f"Parameters:")
    if args.effect_size is not None: print(f"  Hypothesized Correlation (rho): {args.effect_size}")
    print(f"  Alpha: {args.alpha}")
    if args.power is not None: print(f"  Power: {args.power}")
    if args.sample_size is not None: print(f"  Sample Size (N pairs): {args.sample_size}")
    print(f"  Alternative Hypothesis: {args.alternative}")
    print("-" * 53) # Adjusted width

    try:
        calculator = PearsonCorrelationPower(
            alpha=args.alpha,
            effect_size=args.effect_size if args.solve_for != 'effect_size' else None, # effect_size is rho here
            power=args.power if args.solve_for != 'power' else None,
            sample_size=args.sample_size if args.solve_for != 'sample_size' else None,
            alternative=args.alternative
        )
        result = calculator.solve()

        print(f"Result:")
        if args.solve_for == 'power':
            print(f"  Calculated Power: {result:.4f}")
        elif args.solve_for == 'sample_size':
             if np.isinf(result):
                 print(f"  Required Sample Size (N pairs): Infinite (Effect size may be 0)")
             else:
                 print(f"  Required Sample Size (N pairs): {result}")
        elif args.solve_for == 'effect_size':
             if np.isinf(result):
                  print(f"  Minimum Detectable Correlation (rho): Infinite (Sample size may be too small)")
             else:
                  print(f"  Minimum Detectable Correlation (rho): {result:.4f}")

    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except NotImplementedError:
         print(f"Error: Calculation not implemented for these parameters.", file=sys.stderr)
         sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)

    print("-" * 53)


def handle_linreg_multiple(args):
    """Handles the calculation for multiple linear regression (overall F-test)."""
    print("\n--- Multiple Linear Regression (Overall F-Test) Calculation ---")
    print(f"Solving for: {args.solve_for}")
    print(f"Parameters:")

    effect_size_f = None
    # Input validation specific to linreg_multiple
    if args.solve_for in ['power', 'sample_size']:
        if args.effect_size is None and args.r_squared is None:
             parser.error("[linreg_multiple] Either --effect-size (Cohen's f) or --r-squared must be provided when solving for power or sample_size.")
        if args.effect_size is not None and args.r_squared is not None:
             parser.error("[linreg_multiple] Provide either --effect-size or --r-squared, not both.")
        if args.effect_size is not None:
             effect_size_f = args.effect_size
             print(f"  Effect Size (Cohen's f): {effect_size_f}")
        else: # args.r_squared is not None
             print(f"  R-squared (full model): {args.r_squared}")
             try:
                 if not (0 <= args.r_squared < 1): raise ValueError("R-squared must be >= 0 and < 1.")
                 if args.r_squared == 0: effect_size_f2 = 0.0
                 else: effect_size_f2 = args.r_squared / (1 - args.r_squared)
                 effect_size_f = np.sqrt(effect_size_f2)
                 print(f"  (Implied Cohen's f: {effect_size_f:.4f})")
             except ValueError as e:
                 print(f"Error: {e}", file=sys.stderr); sys.exit(1)
    # For solving effect_size/r_squared, effect_size_f starts as None

    print(f"  Number of Predictors (p): {args.num_predictors}")
    print(f"  Alpha: {args.alpha}")
    if args.power is not None: print(f"  Power: {args.power}")
    if args.sample_size is not None: print(f"  Total Sample Size (N): {args.sample_size}")
    print("-" * 60) # Adjusted width

    # Validate remaining args based on solve_for
    if args.solve_for == 'power' and args.sample_size is None:
         parser.error("[linreg_multiple] --sample-size is required when solving for power.")
    if args.solve_for == 'sample_size' and args.power is None:
         parser.error("[linreg_multiple] --power is required when solving for sample_size.")
    if args.solve_for in ['effect_size', 'r_squared'] and (args.power is None or args.sample_size is None):
         parser.error("[linreg_multiple] --power and --sample-size are required when solving for effect_size or r_squared.")

    try:
        calculator = MultipleLinearRegressionPower(
            alpha=args.alpha,
            num_predictors=args.num_predictors,
            # Pass effect_size_f if it was provided or calculated, otherwise None
            effect_size=effect_size_f if args.solve_for not in ['effect_size', 'r_squared'] else None,
            power=args.power if args.solve_for != 'power' else None,
            sample_size=args.sample_size if args.solve_for != 'sample_size' else None
        )
        result = calculator.solve()

        print(f"Result:")
        if args.solve_for == 'power':
            print(f"  Calculated Power: {result:.4f}")
        elif args.solve_for == 'sample_size':
             if np.isinf(result):
                 print(f"  Required Total Sample Size (N): Infinite (Effect size may be 0)")
             else:
                 print(f"  Required Total Sample Size (N): {result}")
        elif args.solve_for == 'effect_size':
             if np.isinf(result):
                 print(f"  Minimum Detectable Effect Size (Cohen's f): Infinite (Sample size may be too small)")
             else:
                 print(f"  Minimum Detectable Effect Size (Cohen's f): {result:.4f}")
        elif args.solve_for == 'r_squared':
             if np.isinf(result):
                  print(f"  Minimum Detectable R-squared: Infinite (Sample size may be too small)")
             else:
                  # Convert Cohen's f back to R^2: R^2 = f^2 / (1 + f^2)
                  mdes_f = result
                  mdes_r2 = mdes_f**2 / (1 + mdes_f**2) if mdes_f != 0 else 0.0
                  print(f"  Minimum Detectable R-squared: {mdes_r2:.4f}")
                  print(f"  (Corresponding Cohen's f: {mdes_f:.4f})")

    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except NotImplementedError:
         print(f"Error: Calculation not implemented for these parameters.", file=sys.stderr)
         sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)

    print("-" * 60)


def handle_t_one_ni(args):
    """Handles the calculation for the one-sample non-inferiority/superiority t-test."""
    test_name = "Non-Inferiority" if args.alternative == 'larger' else "Superiority"
    print(f"\n--- One-Sample {test_name} T-Test Calculation ---")
    print(f"Solving for: {args.solve_for}")
    print(f"Parameters:")
    if args.effect_size is not None: print(f"  True Effect Size (Cohen's d): {args.effect_size}")
    print(f"  Alpha: {args.alpha}")
    print(f"  Margin (standardized): {args.margin}")
    if args.power is not None: print(f"  Power: {args.power}")
    if args.sample_size is not None: print(f"  Sample Size (N): {args.sample_size}")
    print(f"  Alternative Hypothesis: {args.alternative}")
    print("-" * (38 + len(test_name))) # Adjusted width

    try:
        calculator = OneSampleNIPower(
            alpha=args.alpha,
            margin=args.margin,
            alternative=args.alternative,
            effect_size=args.effect_size if args.solve_for != 'effect_size' else None,
            power=args.power if args.solve_for != 'power' else None,
            sample_size=args.sample_size if args.solve_for != 'sample_size' else None,
        )
        result = calculator.solve()

        print(f"Result:")
        if args.solve_for == 'power':
            print(f"  Calculated Power: {result:.4f}")
        elif args.solve_for == 'sample_size':
            print(f"  Required Sample Size (N): {result}")
        elif args.solve_for == 'effect_size':
             # Result is the minimum *true* effect size needed
             if np.isinf(result):
                  print(f"  Minimum Detectable True Effect Size (Cohen's d): Infinite")
             else:
                  print(f"  Minimum Detectable True Effect Size (Cohen's d): {result:.4f}")

    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except NotImplementedError:
         print(f"Error: Calculation not implemented for these parameters.", file=sys.stderr)
         sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)

    print("-" * (38 + len(test_name)))


def handle_t_one_tost(args):
    """Handles the calculation for the one-sample equivalence t-test (TOST)."""
    print("\n--- One-Sample Equivalence T-Test (TOST) Calculation ---")
    print(f"Solving for: {args.solve_for}")
    print(f"Parameters:")
    print(f"  Assumed True Effect Size (Cohen's d): {args.effect_size}") # Note: Default is 0
    print(f"  Alpha (each test): {args.alpha}")
    print(f"  Equivalence Margin (+/- d): {args.margin}")
    if args.power is not None: print(f"  Power: {args.power}")
    if args.sample_size is not None: print(f"  Sample Size (N): {args.sample_size}")
    print("-" * 55) # Adjusted width

    try:
        calculator = OneSampleTOSTPower(
            alpha=args.alpha,
            margin=args.margin,
            effect_size=args.effect_size, # Always provided (defaults to 0)
            power=args.power if args.solve_for != 'power' else None,
            sample_size=args.sample_size if args.solve_for != 'sample_size' else None,
        )
        result = calculator.solve()

        print(f"Result:")
        if args.solve_for == 'power':
            print(f"  Calculated Power: {result:.4f}")
        elif args.solve_for == 'sample_size':
            print(f"  Required Sample Size (N): {result}")
        # MDES case excluded by argument parser validation

    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except NotImplementedError:
         print(f"Error: Calculation not implemented for these parameters.", file=sys.stderr)
         sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)

    print("-" * 55)


def handle_mann_whitney(args):
    """Handles the calculation for the Mann-Whitney U test via simulation."""
    print("\n--- Mann-Whitney U Test Calculation (Simulation) ---")
    print(f"Solving for: {args.solve_for}")
    print(f"Parameters:")
    print(f"  Effect Size (Cohen's d): {args.effect_size}") # d used to generate data
    print(f"  Alpha: {args.alpha}")
    if args.power is not None: print(f"  Power: {args.power}")
    if args.sample_size is not None: print(f"  Sample Size (n1): {args.sample_size}")
    print(f"  Ratio (n2/n1): {args.ratio}")
    print(f"Simulation Settings:")
    print(f"  Number of Simulations: {args.n_simulations}")
    if args.seed is not None: print(f"  Seed: {args.seed}")
    print(f"  Parallel Jobs (n_jobs): {args.n_jobs}")
    print("-" * 50) # Adjusted width

    try:
        calculator = MannWhitneyUTestPower(
            alpha=args.alpha,
            effect_size=args.effect_size, # Always needed for simulation model
            power=args.power if args.solve_for != 'power' else None,
            sample_size=args.sample_size if args.solve_for != 'sample_size' else None,
            ratio=args.ratio,
            n_simulations=args.n_simulations,
            seed=args.seed,
            n_jobs=args.n_jobs
        )
        result = calculator.solve() # Calls estimate_power or find_sample_size

        print(f"Result:")
        if args.solve_for == 'power':
            print(f"  Estimated Power: {result:.4f}")
        elif args.solve_for == 'sample_size':
            # Note: find_sample_size currently raises NotImplementedError
            print(f"  Estimated Required Sample Size (n1): {result}")
            n2 = np.ceil(result * args.ratio).astype(int)
            print(f"  Estimated Required Sample Size (n2): {n2}")
            print(f"  Total Estimated Sample Size (N): {result + n2}")

    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except NotImplementedError as e:
         print(f"Error: {e}", file=sys.stderr) # Show the specific error
         sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)

    print("-" * 50)


def handle_logistic_regression(args):
    """Handles the calculation for logistic regression via simulation."""
    print("\n--- Logistic Regression Calculation (Simulation) ---")
    print(f"Solving for: {args.solve_for}")
    print(f"Parameters:")
    print(f"  Odds Ratio: {args.odds_ratio}")
    print(f"  Baseline Probability: {args.baseline_prob}")
    print(f"  Predictor Mean: {args.predictor_mean}")
    print(f"  Predictor SD: {args.predictor_sd}")
    print(f"  Alpha: {args.alpha}")
    if args.power is not None: print(f"  Power: {args.power}")
    if args.sample_size is not None: print(f"  Sample Size (N): {args.sample_size}")
    print(f"Simulation Settings:")
    print(f"  Number of Simulations: {args.n_simulations}")
    if args.seed is not None: print(f"  Seed: {args.seed}")
    print(f"  Parallel Jobs (n_jobs): {args.n_jobs}")
    print("-" * 50) # Adjusted width

    try:
        # Note: effect_size in the calculator stores the odds_ratio here
        calculator = LogisticRegressionPower(
            alpha=args.alpha,
            odds_ratio=args.odds_ratio,
            baseline_prob=args.baseline_prob,
            power=args.power if args.solve_for != 'power' else None,
            sample_size=args.sample_size if args.solve_for != 'sample_size' else None,
            predictor_mean=args.predictor_mean,
            predictor_sd=args.predictor_sd,
            n_simulations=args.n_simulations,
            seed=args.seed,
            n_jobs=args.n_jobs
        )
        result = calculator.solve() # Calls estimate_power or find_sample_size

        print(f"Result:")
        if args.solve_for == 'power':
            print(f"  Estimated Power: {result:.4f}")
        elif args.solve_for == 'sample_size':
            # Note: find_sample_size currently raises NotImplementedError
            print(f"  Estimated Required Sample Size (N): {result}")

    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except NotImplementedError as e:
         print(f"Error: {e}", file=sys.stderr) # Show the specific error
         sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)

    print("-" * 50)


def handle_coxph(args):
    """Handles the calculation for the Cox Proportional Hazards model."""
    print("\n--- Cox Proportional Hazards Model (2 Groups) Calculation ---")
    print(f"Solving for: {args.solve_for}")
    print(f"Parameters:")
    if args.effect_size is not None: print(f"  Hazard Ratio (HR): {args.effect_size}")
    print(f"  Alpha: {args.alpha}")
    if args.power is not None: print(f"  Power: {args.power}")
    if args.num_events is not None: print(f"  Number of Events: {args.num_events}")
    print(f"  Proportion in Group 1: {args.prop_group1}")
    print(f"  Alternative Hypothesis: {args.alternative}")
    print("-" * 60) # Adjusted width

    try:
        # Note: Base class uses 'sample_size' slot for 'num_events' here
        calculator = CoxPHPower(
            alpha=args.alpha,
            effect_size=args.effect_size if args.solve_for != 'effect_size' else None,
            power=args.power if args.solve_for != 'power' else None,
            num_events=args.num_events if args.solve_for != 'num_events' else None,
            prop_group1=args.prop_group1,
            alternative=args.alternative
        )
        result = calculator.solve()

        print(f"Result:")
        if args.solve_for == 'power':
            print(f"  Calculated Power: {result:.4f}")
        elif args.solve_for == 'num_events':
             if np.isinf(result):
                 print(f"  Required Number of Events: Infinite (HR may be 1)")
             else:
                 print(f"  Required Number of Events: {result}")
        elif args.solve_for == 'effect_size':
             if np.isnan(result): # Can happen if num_events is too low
                  print(f"  Minimum Detectable Hazard Ratio (MDHR): Cannot be determined (num_events may be too low)")
             else:
                  # Result is the HR > 1 or < 1 depending on alternative
                  print(f"  Minimum Detectable Hazard Ratio (MDHR): {result:.4f}")

    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except NotImplementedError as e:
         print(f"Error: {e}", file=sys.stderr) # Show the specific error
         sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)

    print("-" * 60)


def handle_logrank(args):
    """Handles the calculation for the Log-rank test (uses CoxPHPower class)."""
    print("\n--- Log-rank Test (2 Groups) Calculation ---")
    print(f"Solving for: {args.solve_for}")
    print(f"Parameters:")
    if args.effect_size is not None: print(f"  Hazard Ratio (HR): {args.effect_size}")
    print(f"  Alpha: {args.alpha}")
    if args.power is not None: print(f"  Power: {args.power}")
    if args.num_events is not None: print(f"  Number of Events: {args.num_events}")
    print(f"  Proportion in Group 1: {args.prop_group1}")
    print(f"  Alternative Hypothesis: {args.alternative}")
    print("-" * 46) # Adjusted width

    try:
        # Uses the same calculation logic as CoxPHPower
        calculator = LogRankPower(
            alpha=args.alpha,
            effect_size=args.effect_size if args.solve_for != 'effect_size' else None,
            power=args.power if args.solve_for != 'power' else None,
            num_events=args.num_events if args.solve_for != 'num_events' else None,
            prop_group1=args.prop_group1,
            alternative=args.alternative
        )
        result = calculator.solve()

        print(f"Result:")
        if args.solve_for == 'power':
            print(f"  Calculated Power: {result:.4f}")
        elif args.solve_for == 'num_events':
             if np.isinf(result):
                 print(f"  Required Number of Events: Infinite (HR may be 1)")
             else:
                 print(f"  Required Number of Events: {result}")
        elif args.solve_for == 'effect_size':
             if np.isnan(result):
                  print(f"  Minimum Detectable Hazard Ratio (MDHR): Cannot be determined (num_events may be too low)")
             else:
                  print(f"  Minimum Detectable Hazard Ratio (MDHR): {result:.4f}")

    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except NotImplementedError as e:
         print(f"Error: {e}", file=sys.stderr) # Show the specific error
         sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)

    print("-" * 46)


def handle_poisson_regression(args):
    """Handles the calculation for poisson regression via simulation."""
    print("\n--- Poisson Regression Calculation (Simulation) ---")
    print(f"Solving for: {args.solve_for}")
    print(f"Parameters:")
    print(f"  Rate Ratio: {args.rate_ratio}")
    print(f"  Baseline Rate: {args.baseline_rate}")
    print(f"  Predictor Mean: {args.predictor_mean}")
    print(f"  Predictor SD: {args.predictor_sd}")
    print(f"  Alpha: {args.alpha}")
    if args.power is not None: print(f"  Power: {args.power}")
    if args.sample_size is not None: print(f"  Sample Size (N): {args.sample_size}")
    print(f"Simulation Settings:")
    print(f"  Number of Simulations: {args.n_simulations}")
    if args.seed is not None: print(f"  Seed: {args.seed}")
    print(f"  Parallel Jobs (n_jobs): {args.n_jobs}")
    print("-" * 50) # Adjusted width

    try:
        # Note: effect_size in the calculator stores the rate_ratio here
        calculator = PoissonRegressionPower(
            alpha=args.alpha,
            rate_ratio=args.rate_ratio,
            baseline_rate=args.baseline_rate,
            power=args.power if args.solve_for != 'power' else None,
            sample_size=args.sample_size if args.solve_for != 'sample_size' else None,
            predictor_mean=args.predictor_mean,
            predictor_sd=args.predictor_sd,
            n_simulations=args.n_simulations,
            seed=args.seed,
            n_jobs=args.n_jobs
        )
        result = calculator.solve() # Calls estimate_power or find_sample_size

        print(f"Result:")
        if args.solve_for == 'power':
            print(f"  Estimated Power: {result:.4f}")
        elif args.solve_for == 'sample_size':
            # Note: find_sample_size currently raises NotImplementedError
            print(f"  Estimated Required Sample Size (N): {result}")

    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except NotImplementedError as e:
         print(f"Error: {e}", file=sys.stderr) # Show the specific error
         sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)

    print("-" * 50)


if __name__ == "__main__":
    main()
