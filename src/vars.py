from src.imports import *
from src.transformers import ColumnSelector, LogTransformer

pipeline = Pipeline(
    [
        ("column_selector", ColumnSelector()),
        ("log_transformer", LogTransformer()),
        ("preprocessor", ColumnTransformer([])),
        ("sampler_1", RandomUnderSampler()),
        ("imputer", KNNImputer()),
        ("scaler", StandardScaler()),
        ("sampler_2", RandomUnderSampler()),
        ("estimator", LogisticRegression()),
    ]
)

t0 = [
    ("num", "passthrough", make_column_selector(dtype_include=np.number)),
]

t1 = [
    ("num", "passthrough", make_column_selector(dtype_include=np.number)),
    ("cat", OneHotEncoder(), make_column_selector(dtype_include=object)),
]


scaler_list = [
    StandardScaler(),
    "passthrough",
    MinMaxScaler(),
    RobustScaler(),
    Normalizer(),
    # QuantileTransformer(n_quantiles=100, output_distribution="uniform"),
    QuantileTransformer(n_quantiles=100),
]


# grid = GridSearchCV(
#     pipeline,
#     param_grid,
#     cv=10,
#     scoring="accuracy",
#     n_jobs=-1,
#     return_train_score=True,
#     verbose=2,
# )


def resultize(grid: GridSearchCV, head: int = 10):
    """Build a dataframe from the results of a grid search."""

    res = grid.cv_results_
    res = pd.DataFrame(res)

    res = res.sort_values(by="rank_test_score")
    res.drop(columns="rank_test_score", inplace=True)

    cols = [i for i in res.columns if "split" not in i]

    res = res[cols]

    res = res.round(2)

    return res.head(head)


# StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42)

# RandomSh
