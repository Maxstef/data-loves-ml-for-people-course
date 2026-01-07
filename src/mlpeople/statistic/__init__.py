# mlpeople/statistics/__init__.py
from .descriptive import (
    get_mean,
    get_median,
    get_variance,
    get_std,
    get_covariance,
    get_corrcoef,
)
from .distribution import (
    doane_bins,
    fit_distributions,
    plot_fitted_histogram,
    qqplot_sample_vs_sample,
    qqplot_from_dataframe,
)
from .clt_simulation import (
    clt_sample_proportions,
    clt_sample_means,
    clt_from_normal_population,
    clt_from_categorical_population,
    clt_from_bernoulli_population,
)

__all__ = [
    "get_mean",
    "get_median",
    "get_variance",
    "get_std",
    "get_covariance",
    "get_corrcoef",
    "doane_bins",
    "fit_distributions",
    "plot_fitted_histogram",
    "qqplot_sample_vs_sample",
    "qqplot_from_dataframe",
    "clt_sample_proportions",
    "clt_sample_means",
    "clt_from_normal_population",
    "clt_from_categorical_population",
    "clt_from_bernoulli_population",
]
