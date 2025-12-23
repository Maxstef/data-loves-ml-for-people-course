# mlpeople/eda/outliers/__init__.py
from .detection import get_outlier_range_series, get_outlier_range
from .treatment import remove_outliers

__all__ = [
    'get_outlier_range_series',
    'get_outlier_range',
    'remove_outliers'
]
