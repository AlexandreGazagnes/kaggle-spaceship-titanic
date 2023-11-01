from src.imports import *
from src.helpers import *


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


def cv(n=5, test_size=0.33):
    return StratifiedShuffleSplit(n_splits=n, test_size=test_size)


def save_res(
    grid: GridSearchCV,
    fn: str = None,
    verbose: int = 0,
):
    """save grid search results to csv"""

    if not fn:
        fn = f"./results/{now()}"

    res = resultize(grid)
    if verbose:
        display(res)

    fn += ".csv"
    res.astype(str).to_csv(fn, index=False)

    return fn


def save_preds(
    grid: GridSearchCV,
    test: pd.DataFrame = None,
    fn: str = None,
    verbose: int = 0,
) -> pd.DataFrame:
    """make predictions on test set and save them to csv"""

    if not test:
        test = pd.read_csv("./data/source/test.csv")

    if not fn:
        fn = f"./data/preds/{now()}"

    preds = pd.DataFrame(
        zip(test.PassengerId, grid.predict(test)),
        columns=["PassengerId", "Transported"],
    )
    preds.Transported = preds.Transported.astype(bool)

    if verbose:
        display(preds.head())

    fn += ".csv"
    preds.to_csv(fn, index=False)

    return fn


def save_params(
    grid: GridSearchCV,
    fn: str = None,
    verbose: int = 0,
):
    """save grid search best params to json"""

    if not fn:
        fn = f"./params/{now()}"

    params = grid.best_params_
    if verbose:
        display(params)

    fn += ".json"
    with open(fn, "w") as f:
        txt = str(params)
        f.write(txt)

    return fn


def save_model(
    grid: GridSearchCV,
    fn: str = None,
    verbose: int = 0,
):
    """save grid search best model to pickle"""

    if not fn:
        fn = f"./models/{now()}"

    model = grid.best_estimator_
    if verbose:
        display(model)

    fn += ".pk"
    with open(fn, "wb") as f:
        pickle.dump(model, f)

    return fn


def save_predict_all(
    grid: GridSearchCV,
    verbose=1,
) -> None:
    """save models, res, etc and all predictions to csv"""

    _now = now()

    save_model(grid=grid, fn=f"./models/{_now}")
    save_params(grid=grid, fn=f"./params/{_now}")
    save_res(grid=grid, fn=f"./results/{_now}")
    save_preds(grid=grid, fn=f"./data/preds/{_now}")
