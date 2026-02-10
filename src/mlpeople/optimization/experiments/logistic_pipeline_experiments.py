from itertools import product
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from mlpeople.preprocessing.encoding import get_fitted_one_hot_encoder, keep_only_top_n
from mlpeople.preprocessing.numeric import create_binary_flag, get_fitted_scaler
from mlpeople.preprocessing.splitting import split_train_val, split_input_target
from mlpeople.models.evaluation.metrics import predict_and_score_auc


import logging

logger = logging.getLogger(__name__)


def run_experiment(
    df,
    target_col,
    test_df=None,
    test_size=0.2,
    stratify_col=None,
    drop_cols=None,
    categorical_cols=None,
    scale_mode="minmax",
    top_n_cat_values=None,
    encode_drop="if_binary",
    binary_cat_flag_cols=None,
    binary_num_flag_cols=None,
    model=None,
    random_state=42,
    skip_logging=False,
):
    """
    Run a single end-to-end experiment: split, preprocess, train, predict, score.
    todo comment
    """
    if drop_cols is None:
        drop_cols = []
    if categorical_cols is None:
        categorical_cols = []
    if model is None:
        model = LogisticRegression(solver="liblinear")

    # --- Step 1: Split ---
    train_df, val_df = split_train_val(
        df, test_size=test_size, random_state=random_state, stratify_col=stratify_col
    )

    # --- Step 2: Split inputs/targets ---
    train_inputs, train_targets, val_inputs, val_targets = split_input_target(
        train_df, val_df, target_col, drop_cols=drop_cols, verbose=False
    )

    # --- Step 3: Scale numeric ---
    numeric_cols = train_inputs.select_dtypes(include=np.number).columns.tolist()
    scaler = get_fitted_scaler(train_inputs, mode=scale_mode)
    train_inputs[numeric_cols] = scaler.transform(train_inputs[numeric_cols])
    val_inputs[numeric_cols] = scaler.transform(val_inputs[numeric_cols])
    test_inputs = None
    if test_df is not None:
        test_inputs = test_df.copy()
        test_inputs[numeric_cols] = scaler.transform(test_inputs[numeric_cols])

    # --- Step 4: Encode categorical ---
    # use only top-N alues experiment
    if top_n_cat_values is not None:
        for top_n_cat_column, top_n_cat_count in top_n_cat_values.items():
            if top_n_cat_column in categorical_cols:
                train_inputs, top_counts = keep_only_top_n(
                    train_inputs, top_n_cat_column, top_n_cat_count
                )
                top_surnames = set(top_counts.index)
                val_inputs[top_n_cat_column] = val_inputs[top_n_cat_column].where(
                    val_inputs[top_n_cat_column].isin(top_surnames)
                )
                if test_inputs is not None:
                    test_inputs[top_n_cat_column] = test_inputs[top_n_cat_column].where(
                        test_inputs[top_n_cat_column].isin(top_surnames)
                    )
            elif not skip_logging:
                logger.warning(
                    f"skipping, {top_n_cat_column} provided in top_n_cat_values but not included into categorical_cols"
                )

    encoder = get_fitted_one_hot_encoder(
        train_inputs, categorical_cols, drop=encode_drop
    )
    encoded_cols = list(encoder.get_feature_names_out(categorical_cols))
    train_inputs[encoded_cols] = encoder.transform(train_inputs[categorical_cols])
    val_inputs[encoded_cols] = encoder.transform(val_inputs[categorical_cols])
    if test_inputs is not None:
        test_inputs[encoded_cols] = encoder.transform(test_inputs[categorical_cols])

    # --- Step 5: optional custom binary flags for categorical columns ---
    if binary_cat_flag_cols is not None:
        for original_col, flag_col_setups in binary_cat_flag_cols.items():
            if original_col not in categorical_cols:
                for flag_col_setup in flag_col_setups:
                    col_name = flag_col_setup["flag_name"]
                    original_col_value = flag_col_setup["value"]
                    train_inputs[col_name] = (
                        train_inputs[original_col] == original_col_value
                    ).astype(int)
                    val_inputs[col_name] = (
                        val_inputs[original_col] == original_col_value
                    ).astype(int)
                    encoded_cols.append(col_name)
                    if test_inputs is not None:
                        test_inputs[col_name] = (
                            test_inputs[original_col] == original_col_value
                        ).astype(int)
            elif not skip_logging:
                logger.warning(
                    f"skipping, {original_col} provided in binary_cat_flag_cols but included into categorical_cols"
                )

    # --- Step 6: optional custom binary flags for numeric columns ---
    if binary_num_flag_cols is not None:
        for original_col, flag_col_setups in binary_num_flag_cols.items():
            for flag_col_setup in flag_col_setups:
                train_inputs, val_inputs, test_inputs, numeric_cols = (
                    create_binary_flag(
                        train_inputs=train_inputs,
                        val_inputs=val_inputs,
                        numeric_cols=numeric_cols,
                        column=original_col,
                        flag_name=flag_col_setup["flag_name"],
                        threshold_func=(
                            flag_col_setup["threshold_func"]
                            if flag_col_setup.get("threshold_func")
                            else lambda _: flag_col_setup["threshold"]
                        ),
                        drop_original=flag_col_setup["drop_original"],
                        test_inputs=test_inputs,
                    )
                )

    # --- Step 7: Create feature matrices ---
    X_train = train_inputs[numeric_cols + encoded_cols]
    X_val = val_inputs[numeric_cols + encoded_cols]
    X_test = None
    if test_inputs is not None:
        X_test = test_inputs[numeric_cols + encoded_cols]

    # --- Step 8: Train model ---
    model.fit(X_train, train_targets)

    # --- Step 9: Predict & score ---
    _, train_pred_proba, roc_auc_train = predict_and_score_auc(
        model, X_train, train_targets, "Training", verbose=False
    )
    _, val_pred_proba, roc_auc_val = predict_and_score_auc(
        model, X_val, val_targets, "Validation", verbose=False
    )

    # --- Step 10: Predict test if available ---
    if X_test is not None:
        y_test_proba = model.predict_proba(X_test)[:, 1]
        return (
            X_train,
            X_val,
            X_test,
            train_pred_proba,
            val_pred_proba,
            y_test_proba,
            roc_auc_train,
            roc_auc_val,
            model,
        )

    return (
        X_train,
        X_val,
        train_pred_proba,
        val_pred_proba,
        roc_auc_train,
        roc_auc_val,
        model,
    )


def run_experiments(
    df,
    target_col,
    test_size_options=[0.2],
    stratify_col_options=[None],
    drop_cols_options=[[]],
    categorical_cols_options=[[]],
    scale_mode_options=["minmax", "standard"],
    encode_drop_options=[None],
    top_n_cat_values_options=[None],
    binary_cat_flag_cols_options=[None],
    binary_num_flag_cols_options=[None],
    model_options=None,
):
    """
    Run multiple experiments over all combinations of hyperparameters.
    Returns a DataFrame with results sorted by validation AUROC.
    todo comment
    """
    if model_options is None:
        model_options = [LogisticRegression(solver="liblinear")]

    results = []

    for (
        test_size,
        stratify_col,
        drop_cols,
        categorical_cols,
        scale_mode,
        encode_drop,
        top_n_cat_values,
        binary_cat_flag_cols,
        binary_num_flag_cols,
        model,
    ) in product(
        test_size_options,
        stratify_col_options,
        drop_cols_options,
        categorical_cols_options,
        scale_mode_options,
        encode_drop_options,
        top_n_cat_values_options,
        binary_cat_flag_cols_options,
        binary_num_flag_cols_options,
        model_options,
    ):
        try:
            (
                X_train,
                X_val,
                train_pred_proba2,
                val_pred_proba2,
                roc_auc_train,
                roc_auc_val,
                model,
            ) = run_experiment(
                df=df,
                target_col=target_col,
                test_size=test_size,
                stratify_col=stratify_col,
                drop_cols=drop_cols,
                categorical_cols=categorical_cols,
                scale_mode=scale_mode,
                encode_drop=encode_drop,
                top_n_cat_values=top_n_cat_values,
                binary_cat_flag_cols=binary_cat_flag_cols,
                binary_num_flag_cols=binary_num_flag_cols,
                model=model,
                skip_logging=True,
            )

            binary_cat_flag_cols_result = ()
            if binary_cat_flag_cols is not None:
                binary_cat_flag_cols_result = tuple(
                    i["flag_name"] for j in binary_cat_flag_cols.values() for i in j
                )

            binary_num_flag_cols_result = ()
            if binary_num_flag_cols is not None:
                binary_num_flag_cols_result = tuple(
                    (i["flag_name"], i["drop_original"])
                    for j in binary_num_flag_cols.values()
                    for i in j
                )

            results.append(
                {
                    "test_size": test_size,
                    "stratify_col": stratify_col,
                    "drop_cols": tuple(drop_cols),
                    "categorical_cols": tuple(categorical_cols),
                    "scale_mode": scale_mode,
                    "encode_drop": encode_drop,
                    "top_n_cat_values": (
                        tuple((k, v) for k, v in top_n_cat_values.items())
                        if top_n_cat_values is not None
                        else ()
                    ),
                    "binary_cat_flag_cols": binary_cat_flag_cols_result,
                    "binary_num_flag_cols": binary_num_flag_cols_result,
                    "model": type(model).__name__,
                    "roc_auc_train": roc_auc_train,
                    "roc_auc_val": roc_auc_val,
                    "overfit_gap": roc_auc_train - roc_auc_val,
                }
            )

        except Exception as e:
            print("FAILED:", e)

    return pd.DataFrame(results).sort_values("roc_auc_val", ascending=False)
