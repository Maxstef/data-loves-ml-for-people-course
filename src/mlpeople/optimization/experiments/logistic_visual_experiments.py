from sklearn.preprocessing import StandardScaler

from mlpeople.models.logistic import (
    plot_sigmoid_fit_1d,
    fit_logistic,
    plot_decision_boundary_2d,
    plot_log_loss_curve,
    plot_predicted_probabilities,
)


def run_1d_plot_experiment(
    df, column, target_col, sample_size=500, lr=0.01, epochs=5000
):
    scaler = StandardScaler()
    df_sample = df.sample(sample_size)

    X = df_sample[[column]].to_numpy()
    X_scaled = scaler.fit_transform(X)
    y = df_sample[target_col].to_numpy()

    # Fit model
    theta, history = fit_logistic(X_scaled, y, lr=lr, epochs=epochs, verbose=False)

    # Visualize
    plot_log_loss_curve(history)
    plot_sigmoid_fit_1d(
        X_scaled,
        y,
        theta,
        title=f"Logistic Regression (1D) ({column})",
    )


def run_2d_plot_experiment(
    df, col_1, col_2, target_col, sample_size=500, lr=0.01, epochs=5000
):
    scaler = StandardScaler()
    df_sample = df.sample(sample_size)

    X = df_sample[[col_1, col_2]].to_numpy()
    X_scaled = scaler.fit_transform(X)
    y = df_sample[target_col].to_numpy()

    # Fit model
    theta, history = fit_logistic(X_scaled, y, lr=lr, epochs=epochs, verbose=False)

    # Visualize
    plot_log_loss_curve(history)
    plot_decision_boundary_2d(
        X_scaled, y, theta, title=f"Decision Boundary '{col_1}' and '{col_2}' Features"
    )
    plot_predicted_probabilities(
        X_scaled,
        y,
        theta,
        title=f"Predicted Probabilities '{col_1}' and '{col_2}' Features",
    )
