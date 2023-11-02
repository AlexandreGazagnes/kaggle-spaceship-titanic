# from src.imports import *
from src.class_transformers import *
from src.tools import *
from src.var_transformers import *


scaler_list = [
    "passthrough",
    StandardScaler(),
    RobustScaler(),
    Normalizer(),
    QuantileTransformer(n_quantiles=100),
    # QuantileTransformer(n_quantiles=100, output_distribution="uniform"),
    # MinMaxScaler(),
]

transformers_list = [
    # {JustNum},
    JustNum.base,
    # JustNum.knIm,
    # JustNum.simIm,
    # JustNum.knIm_sca,
    # JustNum.simIm_sca,
    # JustNum.knIm_sca_pca_60,
    # JustNum.knIm_sca_pca_80,
    # JustNum.knIm_sca_pca_90,
    # JustNum.knIm_sca_pca_95,
    # JustNum.knIm_sca_pca_99,
    # # {OneHot},
    # NumOneHot.base,
    # NumOneHot.knIm,
    # NumOneHot.simIm,
    # NumOneHot.knIm_sca,
    # NumOneHot.simIm_sca,
    # # NumOneHot.knIm_sca_pca_60,
    # NumOneHot.knIm_sca_pca_80,
    # NumOneHot.knIm_sca_pca_90,
    # NumOneHot.knIm_sca_pca_95,
    # NumOneHot.knIm_sca_pca_99,
]


pipeline = Pipeline(
    [
        ("feat_enhancer", FeatEnhancer()),
        ("column_cleaner", ColumnCleaner()),
        ("column_selector", ColumnSelector()),
        ("log_transformer", LogTransformer(threshold=3)),
        ("preprocessor", NumOneHot.base),
        ("sampler_1", RandomUnderSampler()),
        ("imputer", KNNImputer()),
        ("scaler", StandardScaler()),
        ("sampler_2", RandomUnderSampler()),
        ("estimator", LogisticRegression()),
    ]
)


classifer_list = [
    LogisticRegression(),
    # RandomForestClassifier(),
    # XGBClassifier(),
    # LGBMClassifier(),
    # CatBoostClassifier(),
]

# param_grid = {
#     "log_transformer__threshold": [0.3, 0.5, 1, 1.5, 3],
#     "preprocessor": [t1],
#     "sampler_1": ["passthrough"],  #  RandomUnderSampler()
#     "sampler_2": ["passthrough", RandomUnderSampler()],
#     "scaler": scaler_list,
#     "estimator": classifer_list,
# }


param_grid = {
    "log_transformer__threshold": [0.3, 0.5, 0.75, 1, 1.5],
    "preprocessor": transformers_list,
    "sampler_1": ["passthrough"],  #  RandomUnderSampler()
    "sampler_2": [RandomUnderSampler()],  # "passthrough"
    "scaler": scaler_list,
    "estimator": [
        LogisticRegression()
        #         RandomForestClassifier(max_depth=128, min_samples_split=4, n_estimators=1280)
    ],
}


# grid = GridSearchCV(
#     pipeline,
#     param_grid,
#     cv=cv(),
#     scoring="accuracy",
#     n_jobs=-1,
#     return_train_score=True,
#     verbose=2,
# )
