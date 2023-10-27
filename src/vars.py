from src.imports import *


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


def resultize(grid):
    res = grid.cv_results_
    res = pd.DataFrame(res)

    res = res.sort_values(by="rank_test_score")
    res.drop(columns="rank_test_score", inplace=True)

    cols = [i for i in res.columns if "split" not in i]

    res = res[cols]

    res = res.round(2)

    return res


StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42)

RandomSh
