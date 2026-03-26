from statsmodels.tsa.stattools import adfuller


def display_adfuller_test_result(df, target_col=None, name=None):
    # pick column
    if target_col is None:
        if hasattr(df, "columns"):
            target_col = df.columns[0]
        else:
            target_col = None  # it's already a Series

    if name is None:
        name = target_col

    series = df[target_col] if target_col else df
    series = series.dropna()
    result = adfuller(series)

    print(f"\n{name} ADF Statistic: %f" % result[0])
    print("p-value: %f" % result[1])
    print("Critical Values:")
    for key, value in result[4].items():
        print("\t%s: %.3f" % (key, value))
