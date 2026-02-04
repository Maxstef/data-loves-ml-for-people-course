from .data import (
    generate_linear_regression_data,
    generate_regression_data_sklearn,
    data_unbiased_homoscedastic,
    data_unbiased_heteroscedastic,
    data_biased_homoscedastic,
    data_biased_heteroscedastic,
)
from .model import (
    linear_predict,
    estimate_linear,
    fit_ols,
    predict,
    fit_ols_and_get_params,
)
from .metrics import mae, rmse, rmsle, r2_score, rmse_df_for_params, t_test_coefficients
from .visualization import (
    plot_1d_predictions,
    try_parameters,
    plot_features_vs_target,
    plot_ols_predictions_with_error,
    plot_mae_rmse_rmsle,
    plot_residuals,
    plot_polynomial_fit_1d,
)
