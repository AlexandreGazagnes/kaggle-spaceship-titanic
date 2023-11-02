from src.imports import *
from src.helpers import *
from src._base_transformers import *

from sklearn.compose import make_column_selector as mcs


# Just Num
class JustNum:
    """ """

    # passthrough
    base = ColumnTransformer(
        transformers=[
            ("passthrough", passthrough, mcs(dtype_include=np.number)),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )

    # Just Num + imputer
    knIm = ColumnTransformer(
        transformers=[("knIm", imp_pipe("k"), mcs(dtype_include=np.number))],
        remainder="drop",
        verbose_feature_names_out=True,
    )

    simIm = ColumnTransformer(
        transformers=[("simIm", imp_pipe("s"), mcs(dtype_include=np.number))],
        remainder="drop",
        verbose_feature_names_out=True,
    )

    # Just Num + Imputer + Scaler
    knIm_sca = ColumnTransformer(
        transformers=[("knIm_sca", sca_pipe("k"), mcs(dtype_include=np.number))],
        remainder="drop",
        verbose_feature_names_out=True,
    )

    simIm_sca = ColumnTransformer(
        transformers=[("simIm_sca", sca_pipe("s"), mcs(dtype_include=np.number))],
        remainder="drop",
        verbose_feature_names_out=True,
    )

    # Just num + imputer + scaler + PCA
    knIm_sca_pca_60 = ColumnTransformer(
        transformers=[
            ("knIm_sca_pca_60", pca_pipe("k", 60), mcs(dtype_include=np.number))
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )
    knIm_sca_pca_80 = ColumnTransformer(
        transformers=[
            ("knIm_sca_pca_80", pca_pipe("k", 80), mcs(dtype_include=np.number))
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )
    knIm_sca_pca_90 = ColumnTransformer(
        transformers=[
            ("knIm_sca_pca_90", pca_pipe("k", 90), mcs(dtype_include=np.number))
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )

    knIm_sca_pca_95 = ColumnTransformer(
        transformers=[
            ("knIm_sca_pca_95", pca_pipe("k", 95), mcs(dtype_include=np.number))
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )
    knIm_sca_pca_99 = ColumnTransformer(
        transformers=[
            ("knIm_sca_pca_99", pca_pipe("k", 99), mcs(dtype_include=np.number))
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


class NumOneHot:
    """ """

    base = ColumnTransformer(
        transformers=[
            ("passthrough", passthrough, mcs(dtype_include=np.number)),
            ("cat", cat_pipe(), mcs(dtype_include=object)),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )

    knIm = ColumnTransformer(
        transformers=[
            ("knIm", imp_pipe("k"), mcs(dtype_include=np.number)),
            ("cat", cat_pipe(), mcs(dtype_include=object)),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )

    simIm = ColumnTransformer(
        transformers=[
            ("simIm", imp_pipe("s"), mcs(dtype_include=np.number)),
            ("cat", cat_pipe(), mcs(dtype_include=object)),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )

    # Just Num + Imputer + Scaler + onehot
    knIm_sca = ColumnTransformer(
        transformers=[
            ("knIm_sca", sca_pipe("k"), mcs(dtype_include=np.number)),
            ("cat", cat_pipe(), mcs(dtype_include=object)),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )

    simIm_sca = ColumnTransformer(
        transformers=[
            ("simIm_sca", sca_pipe("s"), mcs(dtype_include=np.number)),
            ("cat", cat_pipe(), mcs(dtype_include=object)),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )

    # Just num + imputer + scaler + PCA + OneHot
    knIm_sca_pca_80 = ColumnTransformer(
        transformers=[
            (
                "_knIm_sca_pca_80",
                pca_pipe("k", 80),
                mcs(dtype_include=np.number),
                ("cat", cat_pipe(), mcs(dtype_include=object)),
            ),
            ("cat", cat_pipe(), mcs(dtype_include=object)),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )
    knIm_sca_pca_90 = ColumnTransformer(
        transformers=[
            (
                "_knIm_sca_pca_90",
                pca_pipe("k", 90),
                mcs(dtype_include=np.number),
            ),
            ("cat", cat_pipe(), mcs(dtype_include=object)),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )

    knIm_sca_pca_95 = ColumnTransformer(
        transformers=[
            (
                "_knIm_sca_pca_95",
                pca_pipe("k", 95),
                mcs(dtype_include=np.number),
            ),
            ("cat", cat_pipe(), mcs(dtype_include=object)),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )
    knIm_sca_pca_99 = ColumnTransformer(
        transformers=[
            (
                "_knIm_sca_pca_99",
                pca_pipe("k", 99),
                mcs(dtype_include=np.number),
            ),
            ("cat", cat_pipe(), mcs(dtype_include=object)),
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
