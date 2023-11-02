from src.imports import *
from src.helpers import *

# passthrough
_passthrough = "passthrough"


# Just imputer
_knIm = Pipeline(steps=[("imputer", KNNImputer())])
_simIm = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])


# Imputer Scaler
_knIm_sca = Pipeline(steps=[("imputer", KNNImputer()), ("scaler", StandardScaler())])
_simIm_sca = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
)


# knIm_sca_pca_60 = Pipeline(
#     steps=[
#         ("imputer", KNNImputer()),
#         ("scaler", StandardScaler()),
#         ("pca", PCA(n_components=0.60)),
#     ]
# )
_knIm_sca_pca_80 = Pipeline(
    steps=[
        ("imputer", KNNImputer()),
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=0.80)),
    ]
)
_knIm_sca_pca_90 = Pipeline(
    steps=[
        ("imputer", KNNImputer()),
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=0.90)),
    ]
)
_knIm_sca_pca_95 = Pipeline(
    steps=[
        ("imputer", KNNImputer()),
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=0.95)),
    ]
)
knIm_sca_pca_99 = Pipeline(
    steps=[
        ("imputer", KNNImputer()),
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=0.99)),
    ]
)

# _simIm_sca_pca_60 = Pipeline(
#     steps=[
#         ("imputer", SimpleImputer(strategy="median")),
#         ("scaler", StandardScaler()),
#         ("pca", PCA(n_components=0.60)),
#     ]
# )
_simIm_sca_pca_80 = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=0.80)),
    ]
)
_simIm_sca_pca_90 = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=0.90)),
    ]
)
_simIm_sca_pca_95 = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=0.95)),
    ]
)
_simIm_sca_pca_99 = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=0.99)),
    ]
)

_cat = Pipeline(steps=[("encoder", OneHotEncoder(handle_unknown="ignore"))])

_cat_khi2_50 = Pipeline(
    steps=[
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ("selector", SelectPercentile(chi2, percentile=50)),
    ]
)
_cat_khi2_25 = Pipeline(
    steps=[
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ("selector", SelectPercentile(chi2, percentile=25)),
    ]
)
_cat_khi2_75 = Pipeline(
    steps=[
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ("selector", SelectPercentile(chi2, percentile=75)),
    ]
)
_cat_khi2_90 = Pipeline(
    steps=[
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ("selector", SelectPercentile(chi2, percentile=90)),
    ]
)
_cat_khi2_95 = Pipeline(
    steps=[
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ("selector", SelectPercentile(chi2, percentile=95)),
    ]
)


# class NumTrans:
#     """steps for column transformer"""

#     _num_passthrough = [
#         ("num", "passthrough", make_column_selector(dtype_include=np.number)),
#     ]

#     _num_knn = [
#         ("num", KNNImputer(), make_column_selector(dtype_include=np.number)),
#     ]
#     _num_knn_scaler = [
#         ("num", KNNImputer(), make_column_selector(dtype_include=np.number)),
#         ("scaler", StandardScaler(), make_column_selector(dtype_include=np.number)),
#     ]


# _onehot = [
#     (
#         "cat",
#         OneHotEncoder(handle_unknown="ignore"),
#         make_column_selector(dtype_include=object),
#     ),
# ]


# class Transformer:
#     """Num and OneHot transformers"""

#     only_num = ColumnTransformer(
#         [_num_passthrough], remainder="drop", verbose_feature_names_out=True
#     )

#     num_onehot = ColumnTransformer(
#         [_num_passthrough, _onehot], remainder="drop", verbose_feature_names_out=True
#     )


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
