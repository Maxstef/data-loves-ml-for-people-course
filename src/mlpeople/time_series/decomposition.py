def create_ts_additive_df(df, column=None, window=13, seasonal="month"):
    """
    Decompose a time series into additive components: trend, seasonality, residuals.

    Parameters
    ----------
    df : pandas.DataFrame with datetime index
    column : str, column to decompose
    window : int, rolling window for trend
    seasonal : str, how to define seasonality
        Options: "month", "quarter", "week", "dayofweek" or any datetime attribute of df.index
    """

    if column is None:
        column = next((col for col in df.columns if col != "Date"), None)

    df_add = df.copy(deep=True)

    # Trend: rolling mean
    df_add["trend"] = (
        df_add[column].rolling(window=window, center=True, min_periods=1).mean()
    )

    # Detrended series
    df_add["detrended"] = df_add[column] - df_add["trend"]

    # Define seasonal key
    if seasonal == "month":
        df_add["season_key"] = df_add.index.month
    elif seasonal == "quarter":
        df_add["season_key"] = df_add.index.quarter
    elif seasonal == "week":
        df_add["season_key"] = df_add.index.isocalendar().week
    elif seasonal == "dayofweek":
        df_add["season_key"] = df_add.index.dayofweek
    elif hasattr(df_add.index, seasonal):
        df_add["season_key"] = getattr(df_add.index, seasonal)
    else:
        raise ValueError(f"Unknown seasonal option: {seasonal}")

    # Seasonal component
    df_add["seasonality"] = df_add.groupby("season_key")["detrended"].transform("mean")

    # Residual
    df_add["resid"] = df_add["detrended"] - df_add["seasonality"]

    return df_add


def create_ts_multiplicative_df(df, column=None, window=13, seasonal="month"):
    """
    Decompose a time series into multiplicative components: trend, seasonality, residuals.

    Parameters
    ----------
    df : pandas.DataFrame with datetime index
    column : str, column to decompose
    window : int, rolling window for trend
    seasonal : str, how to define seasonality
        Options: "month", "quarter", "week", "dayofweek" or any datetime attribute of df.index
    """

    if column is None:
        column = next((col for col in df.columns if col != "Date"), None)

    df_mult = df.copy(deep=True)
    eps = 1e-10  # small value to avoid division by zero

    # Trend: rolling mean
    df_mult["trend"] = (
        df_mult[column].rolling(window=window, center=True, min_periods=1).mean()
    )

    # Divide series
    df_mult["detrended"] = df_mult[column] / (df_mult["trend"] + eps)

    # Define seasonal key
    if seasonal == "month":
        df_mult["season_key"] = df_mult.index.month
    elif seasonal == "quarter":
        df_mult["season_key"] = df_mult.index.quarter
    elif seasonal == "week":
        df_mult["season_key"] = df_mult.index.isocalendar().week
    elif seasonal == "dayofweek":
        df_mult["season_key"] = df_mult.index.dayofweek
    elif hasattr(df_mult.index, seasonal):
        df_mult["season_key"] = getattr(df_mult.index, seasonal)
    else:
        raise ValueError(f"Unknown seasonal option: {seasonal}")

    # Seasonal component
    df_mult["seasonality"] = df_mult.groupby("season_key")["detrended"].transform(
        "mean"
    )

    # Residual
    df_mult["resid"] = df_mult["detrended"] / (df_mult["seasonality"] + eps)

    return df_mult
