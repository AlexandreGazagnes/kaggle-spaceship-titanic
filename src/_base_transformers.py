from src.helpers import *
from src.imports import *

# passthrough
passthrough = "passthrough"


# Just imputer


def imp_pipe(imputer="k"):
    """build a simple pipeline with imputers"""

    _imputer = (
        KNNImputer() if "k" in imputer.lower() else SimpleImputer(strategy="median")
    )

    return Pipeline(
        steps=[
            ("imputer", _imputer),
        ]
    )


def sca_pipe(imputer="k", scaler=StandardScaler):
    """ """

    _imputer = (
        KNNImputer() if "k" in imputer.lower() else SimpleImputer(strategy="median")
    )

    _scaler = scaler()

    return Pipeline(
        steps=[
            ("imputer", _imputer),
            ("scaler", _scaler),
        ]
    )


def pca_pipe(imputer, percentage_var, scaler=StandardScaler):
    """ """

    _imputer = (
        KNNImputer() if "k" in imputer.lower() else SimpleImputer(strategy="median")
    )

    _scaler = scaler()

    percentage_var = percentage_var / 100 if percentage_var > 1 else percentage_var

    return Pipeline(
        steps=[
            ("imputer", _imputer),
            ("scaler", _scaler),
            ("pca", PCA(n_components=percentage_var)),
        ]
    )


def cat_pipe():
    """ """

    return Pipeline(
        steps=[
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )


def cat_khi_pipe(percentile: int = 50):
    """ """

    if not 0 < percentile < 100:
        percentile *= 100

    return Pipeline(
        steps=[
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ("selector", SelectPercentile(chi2, percentile=percentile)),
        ]
    )


# numeric_features = ["age", "fare"]
# numeric_transformer = Pipeline(
#     steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
# )

# categorical_features = ["embarked", "sex", "pclass"]
# categorical_transformer = Pipeline(
#     steps=[
#         ("encoder", OneHotEncoder(handle_unknown="ignore")),
#         ("selector", SelectPercentile(chi2, percentile=50)),
#     ]
# )
# preprocessor = ColumnTransformer(
#     transformers=[
#         ("num", numeric_transformer, numeric_features),
#         ("cat", categorical_transformer, categorical_features),
#     ]
# )
