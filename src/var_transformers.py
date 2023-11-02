from src.imports import *
from src.helpers import *
from src._base_transformers import *

from sklearn.compose import make_column_selector as mcs


# Just Num
class JustNum:
    """ """

    # passthrough
    passthrough = ColumnTransformer(
        transformers=[("_passthrough", _passthrough, mcs(dtype_include=np.number))],
        remainder="drop",
        verbose_feature_names_out=True,
    )

    # Just Num + imputer
    knIm = ColumnTransformer(
        transformers=[("_knIm", _knIm, mcs(dtype_include=np.number))],
        remainder="drop",
        verbose_feature_names_out=True,
    )

    simIm = ColumnTransformer(
        transformers=[("_simIm", _simIm, mcs(dtype_include=np.number))],
        remainder="drop",
        verbose_feature_names_out=True,
    )

    # Just Num + Imputer + Scaler
    knIm_sca = ColumnTransformer(
        transformers=[("_knIm_sca", _knIm_sca, mcs(dtype_include=np.number))],
        remainder="drop",
        verbose_feature_names_out=True,
    )

    simIm_sca = ColumnTransformer(
        transformers=[("_simIm_sca", _simIm_sca, mcs(dtype_include=np.number))],
        remainder="drop",
        verbose_feature_names_out=True,
    )

    # Just num + imputer + scaler + PCA
    knIm_sca_pca_80 = ColumnTransformer(
        transformers=[
            ("_knIm_sca_pca_80", _knIm_sca_pca_80, mcs(dtype_include=np.number))
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )
    knIm_sca_pca_90 = ColumnTransformer(
        transformers=[
            ("_knIm_sca_pca_90", _knIm_sca_pca_90, mcs(dtype_include=np.number))
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )
    knIm_sca_pca_90 = ColumnTransformer(
        transformers=[
            ("_knIm_sca_pca_90", _knIm_sca_pca_90, mcs(dtype_include=np.number))
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )
    knIm_sca_pca_95 = ColumnTransformer(
        transformers=[
            ("_knIm_sca_pca_95", _knIm_sca_pca_95, mcs(dtype_include=np.number))
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )
    knIm_sca_pca_99 = ColumnTransformer(
        transformers=[
            ("_knIm_sca_pca_99", _knIm_sca_pca_99, mcs(dtype_include=np.number))
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


class NumOneHot:
    """ """

    base = ColumnTransformer(
        transformers=[
            ("_passthrough", passthrough, mcs(dtype_include=np.number)),
            ("cat", _cat, mcs(dtype_include=object)),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )

    knIm_oneHot = ColumnTransformer(
        transformers=[
            ("_knIm", _knIm, mcs(dtype_include=np.number)),
            ("cat", _cat, mcs(dtype_include=object)),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )

    simIm_oneHot = ColumnTransformer(
        transformers=[
            ("_simIm", _simIm, mcs(dtype_include=np.number)),
            ("cat", _cat, mcs(dtype_include=object)),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )

    # Just Num + Imputer + Scaler + onehot
    knIm_sca_oneHot = ColumnTransformer(
        transformers=[
            ("_knIm_sca", _knIm_sca, mcs(dtype_include=np.number)),
            ("cat", _cat, mcs(dtype_include=object)),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )

    simIm_sca_oneHot = ColumnTransformer(
        transformers=[
            ("_simIm_sca", _simIm_sca, mcs(dtype_include=np.number)),
            ("cat", _cat, mcs(dtype_include=object)),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )

    # Just num + imputer + scaler + PCA + OneHot
    knIm_sca_pca_80_oneHot = ColumnTransformer(
        transformers=[
            (
                "_knIm_sca_pca_80",
                _knIm_sca_pca_80,
                mcs(dtype_include=np.number),
                ("cat", _cat, mcs(dtype_include=object)),
            ),
            ("cat", _cat, mcs(dtype_include=object)),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )
    knIm_sca_pca_90_oneHot = ColumnTransformer(
        transformers=[
            (
                "_knIm_sca_pca_90",
                _knIm_sca_pca_90,
                mcs(dtype_include=np.number),
            ),
            ("cat", _cat, mcs(dtype_include=object)),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )
    knIm_sca_pca_90_oneHot = ColumnTransformer(
        transformers=[
            (
                "_knIm_sca_pca_90",
                _knIm_sca_pca_90,
                mcs(dtype_include=np.number),
            ),
            ("cat", _cat, mcs(dtype_include=object)),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )
    knIm_sca_pca_95_oneHot = ColumnTransformer(
        transformers=[
            (
                "_knIm_sca_pca_95",
                _knIm_sca_pca_95,
                mcs(dtype_include=np.number),
            ),
            ("cat", _cat, mcs(dtype_include=object)),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )
    knIm_sca_pca_99_oneHot = ColumnTransformer(
        transformers=[
            (
                "_knIm_sca_pca_99",
                _knIm_sca_pca_99,
                mcs(dtype_include=np.number),
            ),
            ("cat", _cat, mcs(dtype_include=object)),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


# t0ter = select only num + knn + scaler

# t0quad_1 = select only num + knn + scaler + pca 0.6
# t0quad_2 = select only num + knn + scaler + pca 0.8
# t0quad_3 = select only num + knn + scaler + pca 0.9
# t0quad_4 = select only num + knn + scaler + pca 0.95
# t0quad_5 = select only num + knn + scaler + pca 0.99


# t0quint_1 = select only num + simpleImputer + scaler + pca 0.6
# t0quint_2 = select only num + simpleImputer + scaler + pca 0.8
# t0quint_3 = select only num + simpleImputer + scaler + pca 0.9
# t0quint_4 = select only num + simpleImputer + scaler + pca 0.95
# t0quint_5 = select only num + simpleImputer + scaler + pca 0.99


# t1 = select num + onehot
#
# t1bis = (num + knn) + onehot
# t1ter = (num + knn + scaler) + onehot

# t1quad_1 = ( num + knn + scaler + pca 0.60) + onehot
# t1quad_2 = ( num + knn + scaler + pca 0.80) + onehot
# t1quad_3 = ( num + knn + scaler + pca 0.90) + onehot
# t1quad_4 = ( num + knn + scaler + pca 0.95) + onehot
# t1quad_5 = ( num + knn + scaler + pca 0.99) + onehot
