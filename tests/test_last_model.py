import pytest

from src.imports import *
from src.transformers import *
from src.tools import *
from src.helpers import *
from src.vars import *


# def test_model(current_score=0.79541, best_score=0.96, threshold=0.1):
#     """Test the last model score"""

#     dist = best_score - current_score
#     quantity = threshold * dist
#     valid_score = current_score + quantity

#     train = pd.read_csv("data/source/train.csv")
#     models = os.listdir("models")


def test_model(current_score=0.79635, best_score=0.82183, threshold=0.15):
    """Test the last model score"""

    train = pd.read_csv("data/source/train.csv")
    models = sorted(os.listdir("models"))

    model_fn = models[-2]

    with open(f"models/{model_fn}", "rb") as f:
        model = pickle.load(f)

    X, y = train.drop("Transported", axis=1), train["Transported"]

    grid = GridSearchCV(model, param_grid={}, cv=10, verbose=1, n_jobs=1)
    grid.fit(X, y)

    new_score = grid.cv_results_["mean_test_score"][0]

    if new_score < current_score:
        raise ValueError(
            f"Progress negative -> new score : {new_score} <  current_score {current_score}"
        )

    dist = best_score - current_score
    progress = new_score - current_score
    progress_rate = round(progress / dist, 4)
    if progress_rate < threshold:
        raise ValueError(
            f"Progress too low -> new_score {round(new_score, 4)} > {round(current_score,4)} but progress_rate {progress_rate} < threshold {threshold}"
        )
    # quantity = threshold * dist
    # valid_score = current_score + quantity4
