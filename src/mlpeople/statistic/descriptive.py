import numpy as np

def get_mean(sample, use_python=False):
    """
    Compute the mean (average) of a numerical sample.

    Parameters
    ----------
    sample : list or array-like
        Input numerical data.
    use_python : bool, default False
        If True, compute the mean using pure Python (sum / len).
        If False, use NumPy's `np.mean`.

    Returns
    -------
    float
        Mean value of the sample.
    """

    if use_python:
        return sum(sample) / len(sample)
    return np.mean(sample)

def get_median(sample, use_python=False):
    """
    Compute the median of a numerical sample.

    Parameters
    ----------
    sample : list or array-like
        Input numerical data.
    use_python : bool, default False
        If True, compute the median using pure Python.
        If False, use NumPy's `np.median`.

    Returns
    -------
    float
        Median value of the sample.
    """

    # use numpy case
    if use_python is not True:
        return np.median(sample)
    
    # calculate median with raw python case
    sorted_sample = sorted(sample)
    sample_length = len(sorted_sample)
    is_odd_count = sample_length % 2 == 1

    if is_odd_count:
        return sorted_sample[sample_length//2]
    
    return (sorted_sample[sample_length//2 - 1] + sorted_sample[sample_length//2]) / 2

def get_variance(sample, ddof=1, use_python=False):
    """
    Compute the variance of a numerical sample.

    Parameters
    ----------
    sample : list or array-like
        Input numerical data.
    ddof : int, default 1
        Delta degrees of freedom. The divisor used in calculation is N - ddof.
    use_python : bool, default False
        If True, compute variance using pure Python.
        If False, use NumPy's `np.var`.

    Returns
    -------
    float
        Variance of the sample.
    """

    # use numpy case
    if use_python is not True:
        return np.var(sample, ddof=ddof)

    # calculate variance with raw python case
    sample_mean = get_mean(sample, use_python=True)
    sample_minus_mean = [(s - sample_mean)**2 for s in sample]
    divisor = len(sample) - 1 if ddof else len(sample) # divisor = len(sample) - ddof
    return sum(sample_minus_mean)/divisor

def get_std(sample, ddof=1, use_python=False):
    """
    Compute the standard deviation of a numerical sample.

    Parameters
    ----------
    sample : list or array-like
        Input numerical data.
    ddof : int, default 1
        Delta degrees of freedom. The divisor used in calculation is N - ddof.
    use_python : bool, default False
        If True, compute standard deviation using pure Python.
        If False, use NumPy's `np.std`.

    Returns
    -------
    float
        Standard deviation of the sample.
    """

    # use numpy case
    if use_python is not True:
        return np.std(sample, ddof=ddof)
    
    # calculate standard deviation with raw python case
    # standard deviation is root square from variance
    # root square is the same as raising to the power of 1/2
    return get_variance(sample, ddof=ddof, use_python=True) ** 0.5

def get_covariance(sample_1, sample_2, ddof=1, use_python=False):
    """
    Compute the covariance between two numerical samples.

    Parameters
    ----------
    sample_1 : list or array-like
        First numerical sample.
    sample_2 : list or array-like
        Second numerical sample. Must have same length as `sample_1`.
    ddof : int, default 1
        Delta degrees of freedom. The divisor used in calculation is N - ddof.
    use_python : bool, default False
        If True, compute covariance using pure Python.
        If False, use NumPy's `np.cov`.

    Returns
    -------
    float or ndarray
        Covariance between the two samples. NumPy returns a 2x2 matrix if `use_python=False`.
    """

    # use numpy case
    if use_python is not True:
        return np.cov(sample_1, sample_2, ddof=ddof)

    # calculate covariance with raw python case
    sample_1_mean = get_mean(sample_1, use_python=True)
    sample_2_mean = get_mean(sample_2, use_python=True)
    samples_mult = [(sample_1[i] - sample_1_mean) * (sample_2[i] - sample_2_mean) for i in range(len(sample_1))]
    divisor = len(sample_1) - 1 if ddof else len(sample_1)

    return sum(samples_mult)/divisor

def get_corrcoef(sample_1, sample_2, use_python=False):
    """
    Compute the Pearson correlation coefficient between two numerical samples.

    Parameters
    ----------
    sample_1 : list or array-like
        First numerical sample.
    sample_2 : list or array-like
        Second numerical sample. Must have same length as `sample_1`.
    use_python : bool, default False
        If True, compute correlation using pure Python (manual formula).
        If False, use NumPy's `np.corrcoef`.

    Returns
    -------
    float or ndarray
        Pearson correlation coefficient. NumPy returns a 2x2 matrix if `use_python=False`.
    """

    # use numpy case
    if use_python is not True:
        return np.corrcoef(sample_1, sample_2)
    
    # calculate Pearson’s correlation coefficient with raw python case
    # use the same ddof value (0 or 1) for both covariance and standard deviations calculation
    cov_samples = get_covariance(sample_1, sample_2, ddof=1, use_python=True)
    std_1 = get_std(sample_1, ddof=1, use_python=True)
    std_2 = get_std(sample_2, ddof=1, use_python=True)

    return cov_samples / (std_1 * std_2)

