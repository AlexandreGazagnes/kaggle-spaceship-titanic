# from src.imports import *
from src.class_transformers import *
from src.tools import *


scaler_list = [
    "passthrough",
    StandardScaler(),
    RobustScaler(),
    Normalizer(),
    QuantileTransformer(n_quantiles=100),
    # QuantileTransformer(n_quantiles=100, output_distribution="uniform"),
    # MinMaxScaler(),
]


pipeline = Pipeline(
    [
        ("feat_enhancer", FeatEnhancer()),
        ("column_cleaner", ColumnCleaner()),
        ("column_selector", ColumnSelector()),
        ("log_transformer", LogTransformer(threshold=3)),
        ("preprocessor", t1),
        ("sampler_1", RandomUnderSampler()),
        ("imputer", KNNImputer()),
        ("scaler", StandardScaler()),
        ("sampler_2", RandomUnderSampler()),
        ("estimator", LogisticRegression()),
    ]
)


classifer_list = [
    RandomForestClassifier(),
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
    "preprocessor": [t1],
    "sampler_1": ["passthrough"],  #  RandomUnderSampler()
    "sampler_2": ["passthrough", RandomUnderSampler()],
    "scaler": scaler_list,
    "estimator": [
        RandomForestClassifier(max_depth=128, min_samples_split=4, n_estimators=1280)
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
