# mlpeople/eda/outliers/__init__.py
from .detection import get_outlier_mask, get_outlier_range
from .treatment import remove_outliers, filter_outliers_df

__all__ = [
    'get_outlier_mask',
    'get_outlier_range',
    'remove_outliers',
    'filter_outliers_df'
]
