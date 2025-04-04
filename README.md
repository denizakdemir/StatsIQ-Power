# StatsIQ Power

A comprehensive statistical power calculator suite for determining sample sizes and power across all classical statistical analyses.

## Features

- **T-tests**: One-sample, independent samples, and paired t-tests
- **ANOVA**: One-way, factorial, repeated measures, and mixed-design ANOVA
- **Chi-square Tests**: Goodness of fit, independence, and homogeneity tests
- **Correlation Analyses**: Pearson, Spearman, Kendall, and partial correlations
- **Regression Analyses**: Linear, multiple, and logistic regression
- **Effect Size Calculations**: Various effect size measures and conversions
- **Visualization Tools**: Power curves, sample size curves, and analysis summaries

## Installation

```bash
pip install -r requirements.txt
```

## Usage Examples

### T-test Power Analysis

```python
from statsiq_power.modules.t_test_power import one_sample_t_test_power, sample_size_for_t_test

# Calculate power for a one-sample t-test
power = one_sample_t_test_power(effect_size=0.5, n=30, alpha=0.05)
print(f"Power: {power:.3f}")

# Calculate required sample size
n = sample_size_for_t_test(effect_size=0.5, power=0.8, alpha=0.05)
print(f"Required sample size: {n}")
```

### ANOVA Power Analysis

```python
from statsiq_power.modules.anova_power import one_way_anova_power, sample_size_for_anova

# Calculate power for a one-way ANOVA
power = one_way_anova_power(effect_size=0.3, n_per_group=20, k=3, alpha=0.05)
print(f"Power: {power:.3f}")

# Calculate required sample size
n = sample_size_for_anova(effect_size=0.3, power=0.8, k=3, alpha=0.05)
print(f"Required sample size per group: {n}")
```

### Visualization

```python
from statsiq_power.utils.visualization import power_curve, sample_size_curve, power_analysis_summary
import numpy as np

# Create power curve
effect_sizes = np.linspace(0.1, 1.0, 10)
sample_sizes = np.linspace(10, 100, 10)
fig = power_curve(effect_sizes, sample_sizes, test_type='t-test')
fig.savefig('power_curve.png')

# Create sample size curve
powers = [0.7, 0.8, 0.9]
fig = sample_size_curve(effect_sizes, powers, test_type='t-test')
fig.savefig('sample_size_curve.png')

# Create power analysis summary
fig = power_analysis_summary(effect_size=0.5, sample_size=30, test_type='t-test')
fig.savefig('power_analysis_summary.png')
```

## Documentation

For detailed documentation and examples, please visit our [documentation page](https://statsiq-power.readthedocs.io/).

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use StatsIQ Power in your research, please cite:

```bibtex
@software{statsiq_power2024,
  author = {Your Name},
  title = {StatsIQ Power: A Comprehensive Statistical Power Calculator Suite},
  year = {2024},
  url = {https://github.com/yourusername/statsiq-power}
}
``` 