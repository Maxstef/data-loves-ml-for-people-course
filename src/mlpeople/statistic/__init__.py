# mlpeople/statistics/__init__.py
from .descriptive import get_mean, get_median, get_variance, get_std, get_covariance, get_corrcoef
from .distribution import doane_bins, fit_distributions, plot_fitted_histogram

__all__ = [
    'get_mean', 'get_median', 'get_variance', 'get_std', 'get_covariance', 'get_corrcoef',
    'doane_bins', 'fit_distributions', 'plot_fitted_histogram',
]
