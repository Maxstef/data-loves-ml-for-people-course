# src/mlpeople/optimization/experiments/polynomial_pipeline_experiments.py

from itertools import product
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import (
    MinMaxScaler,
    StandardScaler,
    OneHotEncoder,
    PolynomialFeatures,
)

# from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector
from sklearn.base import clone


from mlpeople.preprocessing.splitting import split_train_val, split_input_target
from mlpeople.models.evaluation.metrics import predict_and_score_auc
from mlpeople.preprocessing.transformers import (
    NumericBinaryFlagTransformer,
    CategoricalBinaryFlagTransformer,
    TopNCategoricalTransformer,
    NumericBinner,
)

from imblearn.pipeline import Pipeline
from imblearn.under_sampling import TomekLinks, RandomUnderSampler
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from imblearn.combine import SMOTETomek

import logging

logger = logging.getLogger(__name__)

numeric_selector = make_column_selector(dtype_include=np.number)
categorical_selector = make_column_selector(dtype_include=["object", "category"])


def binary_numeric_selector(X):

    numeric_cols = X.select_dtypes(include=[np.number, "bool"]).columns

    binary_cols = []

    for col in numeric_cols:
        values = set(X[col].dropna().unique())

        if values.issubset({0, 1, 0.0, 1.0, True, False}):
            binary_cols.append(col)

    return binary_cols


def continuous_numeric_selector(X):
    numeric_cols = X.select_dtypes(include=np.number).columns

    binary_cols = binary_numeric_selector(X)

    return [col for col in numeric_cols if col not in binary_cols]


def run_experiment_poly(
    df: pd.DataFrame,
    target_col: str,
    test_size: float = 0.2,
    stratify_col: str = None,
    drop_cols: list = None,
    scale_mode: str = "standard",
    separate_binary_numeric: bool = False,
    encode_drop: str = "if_binary",
    polynomial_degree: int = 1,
    polynomial_after_scale: bool = True,
    polynomial_interaction_only: bool = False,
    top_n_cat_values=None,
    binary_cat_flag_cols=None,
    binary_num_flag_cols=None,
    num_bin_cols=None,
    model=None,
    random_state: int = 42,
    sampler=None,
) -> dict:
    """
    Run a single experiment with optional polynomial features for numeric columns.

    Args:
        df: Training dataset.
        target_col: Name of the target column.
        test_df: Optional test dataset.
        test_size: Fraction for validation set.
        stratify_col: Column to stratify train/val split.
        drop_cols: Columns to drop from features.
        categorical_cols: Categorical columns for one-hot encoding.
        scale_mode: "minmax" or "standard".
        encode_drop: Drop strategy for one-hot encoder.
        polynomial_degree: Degree of polynomial expansion for numeric features.
        polynomial_after_scale: Apply polynomial after scaling if True.
        model: sklearn-compatible estimator.
        random_state: Random seed.
        sampler: resampling train data before training model if provided

    Returns:
        dict with:
            - pipeline: fitted sklearn Pipeline
            - train_pred_proba: np.array of training probabilities
            - val_pred_proba: np.array of validation probabilities
            - roc_auc_train: AUROC on training set
            - roc_auc_val: AUROC on validation set
    """
    if drop_cols is None:
        drop_cols = []
    if model is None:
        model = LogisticRegression(solver="liblinear")
    else:
        model = clone(model)

    # --- Step 1: Split ---
    train_df, val_df = split_train_val(
        df, test_size=test_size, random_state=random_state, stratify_col=stratify_col
    )

    # --- Step 2: Split inputs and targets ---
    train_inputs, train_targets, val_inputs, val_targets = split_input_target(
        train_df, val_df, target_col, drop_cols=drop_cols, verbose=False
    )

    # --- Step 3: Custom Feature engineering ---
    pre_steps = []

    if binary_num_flag_cols:
        pre_steps.append(
            ("num_flags", NumericBinaryFlagTransformer(binary_num_flag_cols))
        )

    if num_bin_cols:
        pre_steps.append(("num_bin", NumericBinner(num_bin_cols)))

    if binary_cat_flag_cols:
        pre_steps.append(
            ("cat_flags", CategoricalBinaryFlagTransformer(binary_cat_flag_cols))
        )

    if top_n_cat_values:
        pre_steps.append(("top_n", TopNCategoricalTransformer(top_n_cat_values)))

    feature_engineering = pre_steps

    # --- Step 4: Numeric columns & scaling ---
    if scale_mode == "minmax":
        scaler = MinMaxScaler()
    elif scale_mode == "standard":
        scaler = StandardScaler()
    else:
        raise ValueError('scale_mode must be "minmax" or "standard"')

    numeric_steps = [("scaler", scaler)]
    if polynomial_degree > 1:
        poly_features = PolynomialFeatures(
            degree=polynomial_degree,
            include_bias=False,
            interaction_only=polynomial_interaction_only,
        )

        if polynomial_after_scale:
            numeric_steps.append(("poly", poly_features))
        else:
            numeric_steps.insert(0, ("poly", poly_features))

    numeric_transformer = Pipeline(steps=numeric_steps)

    # --- Step 5: Binary columns - just "passthrough" for now ---
    if separate_binary_numeric:
        binary_transformer = "passthrough"

    # --- Step 6: Categorical columns ---
    categorical_transformer = Pipeline(
        steps=[
            (
                "onehot",
                OneHotEncoder(
                    sparse_output=False, handle_unknown="ignore", drop=encode_drop
                ),
            )
        ]
    )

    # --- Step 7: Combine transformers ---
    transformersSteps = [
        (
            "num",
            numeric_transformer,
            (
                continuous_numeric_selector
                if separate_binary_numeric
                else numeric_selector
            ),
        ),
        ("cat", categorical_transformer, categorical_selector),
    ]

    if separate_binary_numeric:
        transformersSteps.insert(
            0, ("bin", binary_transformer, binary_numeric_selector)
        )

    preprocessor = ColumnTransformer(transformers=transformersSteps)

    # --- Step 8: Build full pipeline ---
    steps = []

    if feature_engineering is not None:
        steps.extend(feature_engineering)

    # --- Step 8.1: Optional Resampling train data ---
    # --- If sampler value is provided ---
    if sampler is not None:
        # first all data preprocessing
        steps.append(("preprocessor", preprocessor))

        # second data resampling
        samplers = {
            "smote": lambda: SMOTE(random_state=random_state),
            "adasin": lambda: ADASYN(random_state=random_state),
            "smotetomek": lambda: SMOTETomek(random_state=random_state),
            "randomover": lambda: RandomOverSampler(random_state=random_state),
            "randomunder": lambda: RandomUnderSampler(random_state=random_state),
            "tomeklinks": lambda: TomekLinks(),
        }

        try:
            steps.append(("sampler", samplers[sampler]()))
        except KeyError:
            raise ValueError(f"sampler must be one of {list(samplers.keys())}")

        # last model fit with preprocessed and resampled data
        steps.append(("classifier", model))

        model_pipeline = Pipeline(
            steps,
        )

    # --- Default flow with no resampling ---
    else:
        steps.extend(
            [
                ("preprocessor", preprocessor),
                ("classifier", model),
            ]
        )

        model_pipeline = Pipeline(
            steps,
            # memory="pipeline_cache", # might give 3–10x speedups in experiment loops ?
        )

    # --- Step 9: Fit pipeline ---
    model_pipeline.fit(train_inputs, train_targets)

    # --- Step 10: Predict & score ---
    _, train_pred_proba, roc_auc_train = predict_and_score_auc(
        model_pipeline, train_inputs, train_targets, "Training", verbose=False
    )
    _, val_pred_proba, roc_auc_val = predict_and_score_auc(
        model_pipeline, val_inputs, val_targets, "Validation", verbose=False
    )

    return {
        "train_pred_proba": train_pred_proba,
        "val_pred_proba": val_pred_proba,
        "roc_auc_train": roc_auc_train,
        "roc_auc_val": roc_auc_val,
        "pipeline": model_pipeline,
    }


def run_experiments_poly(
    df,
    target_col,
    test_size_options=[0.2],
    stratify_col_options=[None],
    drop_cols_options=[[]],
    scale_mode_options=["minmax", "standard"],
    encode_drop_options=[None],
    separate_binary_numeric_options=[False],
    top_n_cat_values_options=[None],
    binary_cat_flag_cols_options=[None],
    binary_num_flag_cols_options=[None],
    num_bin_cols_options=[None],
    polynomial_interaction_only_options=[False],
    polynomial_degree_options=[1, 2, 3],
    polynomial_after_scale_options=[True, False],
    model_options=None,
    sampler_options=[None],
) -> pd.DataFrame:
    """
    Run multiple polynomial experiments over all combinations of hyperparameters.
    Returns a DataFrame sorted by validation AUROC.
    """
    if model_options is None:
        model_options = [LogisticRegression(solver="liblinear")]

    results = []

    for (
        test_size,
        stratify_col,
        drop_cols,
        scale_mode,
        encode_drop,
        separate_binary_numeric,
        top_n_cat_values,
        binary_cat_flag_cols,
        binary_num_flag_cols,
        num_bin_cols,
        polynomial_interaction_only,
        polynomial_degree,
        polynomial_after_scale,
        model,
        sampler,
    ) in product(
        test_size_options,
        stratify_col_options,
        drop_cols_options,
        scale_mode_options,
        encode_drop_options,
        separate_binary_numeric_options,
        top_n_cat_values_options,
        binary_cat_flag_cols_options,
        binary_num_flag_cols_options,
        num_bin_cols_options,
        polynomial_interaction_only_options,
        polynomial_degree_options,
        polynomial_after_scale_options,
        model_options,
        sampler_options,
    ):
        if polynomial_degree == 1 and not polynomial_after_scale:
            continue

        try:
            res = run_experiment_poly(
                df=df,
                target_col=target_col,
                test_size=test_size,
                stratify_col=stratify_col,
                drop_cols=drop_cols,
                scale_mode=scale_mode,
                encode_drop=encode_drop,
                separate_binary_numeric=separate_binary_numeric,
                top_n_cat_values=top_n_cat_values,
                binary_cat_flag_cols=binary_cat_flag_cols,
                binary_num_flag_cols=binary_num_flag_cols,
                num_bin_cols=num_bin_cols,
                polynomial_interaction_only=polynomial_interaction_only,
                polynomial_degree=polynomial_degree,
                polynomial_after_scale=polynomial_after_scale,
                model=model,
                sampler=sampler,
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
                    "scale_mode": scale_mode,
                    "encode_drop": encode_drop,
                    "model": str(model),
                    "sampler": sampler,
                    "separate_binary_numeric": separate_binary_numeric,
                    "top_n_cat_values": (
                        tuple((k, v) for k, v in top_n_cat_values.items())
                        if top_n_cat_values is not None
                        else ()
                    ),
                    "binary_cat_flag_cols": binary_cat_flag_cols_result,
                    "binary_num_flag_cols": binary_num_flag_cols_result,
                    "num_bin_cols": (
                        tuple(
                            (v["new_col"], v["drop_original"], v["bins"])
                            for k, v in num_bin_cols.items()
                        )
                        if num_bin_cols is not None
                        else ()
                    ),
                    "polynomial_degree": polynomial_degree,
                    "polynomial_interaction_only": polynomial_interaction_only,
                    "polynomial_after_scale": polynomial_after_scale,
                    "roc_auc_train": res["roc_auc_train"],
                    "roc_auc_val": res["roc_auc_val"],
                    "overfit_gap": (res["roc_auc_train"] - res["roc_auc_val"])
                    / res["roc_auc_train"],
                }
            )
        except Exception as e:
            print("FAILED:", e)

    return pd.DataFrame(results).sort_values("roc_auc_val", ascending=False)
