import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import root_mean_squared_error


def predict_from_decomposition(
    df,
    mode="additive",
    target_col=None,
    plot=True,
    verbose=True,
    name="",
):
    """
    Reconstruct time series from trend and seasonality components.

    NOTE: This is reconstruction, not future forecasting.
    """

    if target_col is None:
        target_col = df.columns[0]

    df_predicted = df.copy(deep=True)

    if mode == "multiplicative":
        df_predicted["predicted"] = df_predicted["trend"] * df_predicted["seasonality"]
    else:
        df_predicted["predicted"] = df_predicted["trend"] + df_predicted["seasonality"]

    valid_idx = df_predicted["predicted"].dropna().index
    target = df_predicted.loc[valid_idx, target_col]
    prediction = df_predicted.loc[valid_idx, "predicted"]

    rmse = root_mean_squared_error(target, prediction)
    smape = 100 * np.mean(
        2 * np.abs(target - prediction) / (np.abs(target) + np.abs(prediction))
    )

    if verbose:
        print(f"RMSE:  {rmse:.4f}")
        print(f"MAPE:  {smape:.2f}%")

    if plot:
        plt.figure(figsize=(15, 5))
        plt.plot(target, label="Original")
        plt.plot(prediction, label="Reconstructed")
        plt.legend()
        plt.title(f"Reconstruction from {mode} decomposition - {name}")
        plt.show()

    return df_predicted["predicted"]
