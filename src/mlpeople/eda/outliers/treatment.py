from mlpeople.statistic import get_mean, get_std

def remove_outliers(sample, sample_2=None, threshold=2):
    """
    Remove outliers from a numeric sample based on a threshold of standard deviations from the mean.

    Parameters
    ----------
    sample : list or pandas.Series or numpy.ndarray
        Primary numeric sample to check for outliers.
    sample_2 : list or pandas.Series or numpy.ndarray, optional
        Secondary sample that will be filtered in sync with `sample`. Must have the same length.
    threshold : float, default=2
        Number of standard deviations from the mean to define outliers. Values outside
        mean ± threshold * std are considered outliers.

    Returns
    -------
    tuple
        updated_sample : list
            Values from `sample` that are within the threshold range.
        updated_sample_2 : list or None
            Values from `sample_2` corresponding to non-outlier entries in `sample`.
            Returns None if `sample_2` is not provided.
        removed_indexes : list
            Indices of values in `sample` that were removed as outliers.

    Raises
    ------
    AssertionError
        If `sample_2` is provided and its length does not match `sample`.

    Notes
    -----
    - Outliers are detected using the formula:
          top_threshold = mean + threshold * std
          bottom_threshold = mean - threshold * std
    - This function preserves the order of elements.
    """

    avg = get_mean(sample)
    std = get_std(sample)

    top_threshold = avg + (std*threshold)
    bottom_threshold = avg - (std*threshold)

    updated_sample = []
    updated_sample_2 = [] if sample_2 is not None else None
    removed_indexes = []

    if sample_2 is not None:
        assert len(sample) == len(sample_2), 'sample and sample_2 must have same length'

    for i, s in enumerate(sample):
        if bottom_threshold <= s <= top_threshold:
            updated_sample.append(s)
            if sample_2 is not None:
                updated_sample_2.append(sample_2[i])
        else:
            removed_indexes.append(i)

    return updated_sample, updated_sample_2, removed_indexes