# from src.imports import *
from src.transformers import *
from src.tools import *
from src.transformers import FeatEnhancer, ColumnCleaner, ColumnSelector, LogTransformer
from src.transformers import JustNumTransformer as JNT
from src.transformers import NumOneHotTransformer as NOT

scaler_list = [
    # "passthrough",
    StandardScaler(),
    # RobustScaler(),
    # Normalizer(),
    # QuantileTransformer(n_quantiles=100),
    # QuantileTransformer(n_quantiles=100, output_distribution="uniform"),
    # MinMaxScaler(),
]

transformers_list = [
    # JustNumTransformer
    # JNT.base(),
    # JNT.imp(imputer="k"),
    # JNT.imp(imputer="s"),
    JNT.sca(imputer="k", scaler=StandardScaler),
    # JNT.sca(imputer="s", scaler=StandardScaler),
    # JNT.pca(imputer="k", percentage_var=60),
    # JNT.pca(imputer="k", percentage_var=80),
    # JNT.pca(imputer="k", percentage_var=90),
    # JNT.pca(imputer="k", percentage_var=95),
    # JNT.pca(imputer="k", percentage_var=99),
    # # {OneHot},
    # NOT.base(),
    # NOT.imp(imputer="k"),
    # NOT.imp(imputer="s"),
    # NOT.sca(imputer="k", scaler=StandardScaler),
    # NOT.sca(imputer="s", scaler=StandardScaler),
    # NOT.pca(imputer="k", percentage_var=60),
    # NOT.pca(imputer="k", percentage_var=80),
    # NOT.pca(imputer="k", percentage_var=90),
    # NOT.pca(imputer="k", percentage_var=95),
    # NOT.pca(imputer="k", percentage_var=99),
]


pipeline = Pipeline(
    [
        ("feat_enhancer", FeatEnhancer()),
        ("column_cleaner", ColumnCleaner()),
        ("column_selector", ColumnSelector()),
        ("log_transformer", LogTransformer(threshold=3)),
        ("preprocessor", NOT.base()),
        ("sampler_1", RandomUnderSampler()),
        ("imputer", KNNImputer()),
        ("scaler", StandardScaler()),
        ("sampler_2", RandomUnderSampler()),
        ("estimator", LogisticRegression()),
    ]
)


classifer_list = [
    # LogisticRegression(),
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
    "log_transformer__threshold": [0.5, 0.75, 1],  # 0.3,  1.5
    "preprocessor": transformers_list,
    "sampler_1": ["passthrough"],  #  RandomUnderSampler()
    "scaler": scaler_list,
    "sampler_2": [RandomUnderSampler()],  # "passthrough"
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
