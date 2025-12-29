def age_cat(years):
    """
    Categorize an age in years into predefined age groups.

    Parameters
    ----------
    years : int or float
        Age value to categorize.

    Returns
    -------
    str Age group as a string:

    Notes
    -----
    - The function assumes non-negative age values.
    - Boundaries are inclusive at the lower end and upper end of each group
    """

    if years <= 20:
        return "0-20"
    elif years > 20 and years <= 30:
        return "20-30"
    elif years > 30 and years <= 40:
        return "30-40"
    elif years > 40 and years <= 50:
        return "40-50"
    elif years > 50 and years <= 60:
        return "50-60"
    elif years > 60 and years <= 70:
        return "60-70"
    elif years > 70:
        return "70+"
