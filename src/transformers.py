from src.imports import *


class ColumnSelector(TransformerMixin, BaseEstimator):
    """Selects columns from a dataframe."""

    def __init__(
        self, drop_cols=[], clean_cols=["PassengerId", "Cabin", "Name", "Cabin"]
    ) -> None:
        """Initialize the ColumnSelector class."""

        self.drop_cols = drop_cols
        self.clean_cols = clean_cols

    def fit(self, X, y=None):
        """Fit the ColumnSelector class."""
        return self

    def transform(self, X, y=None):
        """Transform the ColumnSelector class."""
        X = X.drop(self.clean_cols, axis=1)
        X = X.drop(self.drop_cols, axis=1)

        return X


class LogTransformer(BaseEstimator, TransformerMixin):
    """Log transforms columns with skewness above a threshold."""

    def __init__(self, threshold=3) -> None:
        """Initialize the LogTransformer class."""

        self.threshold = threshold

    def fit(self, X, y=None):
        """Fit the LogTransformer class."""

        return self

    def transform(self, X, y=None):
        """Transform the LogTransformer class."""

        # sep num cat
        num = X.select_dtypes(include=np.number)
        cat = X.select_dtypes(exclude=np.number)

        # skew threshold
        skew = num.skew()
        do_log_columns = skew[skew >= self.threshold].index.tolist()
        dont_log_columns = skew[skew < self.threshold].index.tolist()

        # log dont log
        do_log = num.loc[:, do_log_columns]
        do_log = np.log1p(do_log)
        dont_log = num.loc[:, dont_log_columns]

        # concat
        X = pd.concat([cat, dont_log, do_log], axis=1, ignore_index=True)
        X.columns = cat.columns.tolist() + dont_log_columns + do_log_columns

        return X
