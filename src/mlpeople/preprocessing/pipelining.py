import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from imblearn.pipeline import Pipeline as ImblPipeline
from sklearn.pipeline import Pipeline as SkPipeline
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE, SMOTENC

# Selectors
numeric_selector = make_column_selector(dtype_include=np.number)
categorical_selector = make_column_selector(dtype_exclude=np.number)


def get_model_pipeline(
    model=None,
    num_impute_strategy="most_frequent",
    num_impute_value=0,  # applicable only if num_impute_strategy="constant"
    cat_impute_strategy="most_frequent",
    cat_impute_value="missing",  # applicable only if cat_impute_strategy="constant"
    cat_encoder_drop=None,
):
    """
    Create a basic preprocessing + model pipeline.

    Steps:
    1. Impute numeric features (strategy or constant)
    2. Scale numeric features
    3. Impute categorical features (strategy or constant)
    4. One-hot encode categorical features
    5. Fit model

    Args:
        model: sklearn estimator. Defaults to LogisticRegression with 'liblinear'.
        num_impute_strategy: Imputation strategy for numeric columns.
        num_impute_value: Value to use if num_impute_strategy="constant".
        cat_impute_strategy: Imputation strategy for categorical columns.
        cat_impute_value: Value to use if cat_impute_strategy="constant".
        cat_encoder_drop: Column to drop in OneHotEncoder (e.g., "first").

    Returns:
        sklearn.pipeline.Pipeline object
    """
    if model is None:
        model = LogisticRegression(solver="liblinear")

    # Numeric pipeline
    num_imputer_params = {"strategy": num_impute_strategy}
    if num_impute_strategy == "constant":
        num_imputer_params["fill_value"] = num_impute_value

    numeric_pipeline = Pipeline(
        [("imputer", SimpleImputer(**num_imputer_params)), ("scaler", StandardScaler())]
    )

    # Categorical pipeline
    cat_imputer_params = {"strategy": cat_impute_strategy}
    if cat_impute_strategy == "constant":
        cat_imputer_params["fill_value"] = cat_impute_value

    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(**cat_imputer_params)),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore", sparse_output=False, drop=cat_encoder_drop
                ),
            ),
        ]
    )

    # Combine
    preprocessor = ColumnTransformer(
        [
            ("num", numeric_pipeline, numeric_selector),
            ("cat", categorical_pipeline, categorical_selector),
        ]
    )

    pipeline = Pipeline([("preprocessing", preprocessor), ("model", model)])

    return pipeline


def get_model_pipeline_resample_ext(
    model=None,
    sampler=None,  # None | "smote" | "smotetomek"
    num_impute_strategy="most_frequent",
    cat_impute_strategy="most_frequent",
    cat_encoder_drop=None,
    random_state=42,
):
    """
    Create a preprocessing + resampling + model pipeline.

    Notes:
    - SMOTE / SMOTETomek require numeric input.
    - ColumnTransformer will reorder columns: numeric first, categorical second.

    Steps:
    1. Impute numeric and categorical features
    2. Scale numeric and one-hot encode categorical features
    3. Optional: apply SMOTE or SMOTETomek to balance classes
    4. Fit model

    Args:
        model: sklearn estimator. Defaults to LogisticRegression.
        sampler: "smote" or "smotetomek" or None.
        num_impute_strategy: strategy for numeric imputation.
        cat_impute_strategy: strategy for categorical imputation.
        cat_encoder_drop: drop option for OneHotEncoder.
        random_state: for reproducibility of resampling.

    Returns:
        imblearn.pipeline.Pipeline object
    """

    # Default model option
    if model is None:
        model = LogisticRegression(solver="liblinear")

    # Numeric pipeline:
    # Imputation before scaling.
    numeric_pipeline = ImblPipeline(
        [
            ("imputer", SimpleImputer(strategy=num_impute_strategy)),
            ("scaler", StandardScaler()),
        ]
    )

    # Categorical pipeline:
    # Impute first -> then encode.
    # handle_unknown="ignore" prevents inference crashes when unseen categories appear in test data.
    categorical_pipeline = ImblPipeline(
        [
            ("imputer", SimpleImputer(strategy=cat_impute_strategy)),
            ("encoder", OneHotEncoder(handle_unknown="ignore", drop=cat_encoder_drop)),
        ]
    )

    # Apply preprocessing to all columns BEFORE resampling.
    # Output is a fully numeric matrix safe for SMOTE.
    preprocessor = ColumnTransformer(
        [
            ("num", numeric_pipeline, numeric_selector),
            ("cat", categorical_pipeline, categorical_selector),
        ]
    )

    steps = [("preprocessing", preprocessor)]

    # Resampling AFTER full preprocessing.
    # This is correct because SMOTE operates in feature space.
    if sampler == "smote":
        steps.append(("sampler", SMOTE(random_state=random_state)))

    elif sampler == "smotetomek":
        steps.append(("sampler", SMOTETomek(random_state=random_state)))

    steps.append(("model", model))

    return ImblPipeline(steps)


def get_smotenc_pipeline(
    model=None,
    X=None,
    num_impute_strategy="most_frequent",
    cat_impute_strategy="most_frequent",
    cat_encoder_drop=None,
    random_state=42,
):
    """
    Create a preprocessing pipeline with SMOTENC resampling for categorical + numeric data.

    Required sequence:
    1. Impute numeric and categorical columns (raw values only)
    2. Apply SMOTENC to balance dataset
    3. Scale numeric and one-hot encode categorical features
    4. Fit model

    Notes:
    - SMOTENC requires exact categorical column indices after ColumnTransformer.
    - Must pass dataframe X to detect categorical columns.

    Args:
        model: sklearn estimator. Defaults to LogisticRegression.
        X: pandas DataFrame used to detect numeric/categorical columns.
        num_impute_strategy: strategy for numeric imputation.
        cat_impute_strategy: strategy for categorical imputation.
        cat_encoder_drop: drop option for OneHotEncoder.
        random_state: for reproducibility.

    Returns:
        imblearn.pipeline.Pipeline object
    """
    # Fail fast with a clear error
    if X is None:
        raise ValueError(
            "X must be provided so SMOTENC can determine categorical column indices."
        )

    if model is None:
        model = LogisticRegression(solver="liblinear")

    # Detect column types from the provided dataframe.
    num_cols = numeric_selector(X)
    cat_cols = categorical_selector(X)

    # ColumnTransformer ALWAYS outputs columns in the order transformers are listed.
    # Since we put numeric first and categorical second, we can safely construct positional indices.
    num_idx = list(range(len(num_cols)))
    cat_idx = list(range(len(num_cols), len(num_cols) + len(cat_cols)))

    # PRE-SMOTE:
    # Only impute missing values.
    # Encoding here would BREAK SMOTENC because it expects categorical features in their original form.
    numeric_imputer = SkPipeline(
        [("imputer", SimpleImputer(strategy=num_impute_strategy))]
    )

    categorical_imputer = SkPipeline(
        [("imputer", SimpleImputer(strategy=cat_impute_strategy))]
    )

    pre_smote = ColumnTransformer(
        [
            ("num", numeric_imputer, numeric_selector),
            ("cat", categorical_imputer, categorical_selector),
        ]
    )

    # SMOTENC must know EXACT categorical column positions AFTER ColumnTransformer reorders them.
    smote = SMOTENC(categorical_features=cat_idx, random_state=random_state)

    # POST-SMOTE:
    # Now it is safe to scale numeric features and one-hot encode categoricals.
    numeric_pipeline = SkPipeline([("scaler", StandardScaler())])

    categorical_pipeline = SkPipeline(
        [("encoder", OneHotEncoder(handle_unknown="ignore", drop=cat_encoder_drop))]
    )

    post_smote = ColumnTransformer(
        [
            ("num", numeric_pipeline, num_idx),
            ("cat", categorical_pipeline, cat_idx),
        ]
    )

    pipeline = ImblPipeline(
        [
            ("pre_smote", pre_smote),
            ("smote", smote),
            ("post_smote", post_smote),
            ("model", model),
        ]
    )

    return pipeline
