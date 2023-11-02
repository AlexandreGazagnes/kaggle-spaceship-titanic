from src.imports import *
from src.helpers import *


class FeatEnhancer(BaseEstimator, TransformerMixin):
    """Boleanize, manage name and split/cast cabin"""

    def __init__(
        self,
        bool_cols: list = ["CryoSleep", "VIP"],
        split_cols: list = ["Cabin"],
    ) -> None:
        """init method"""

        self.bool_cols = bool_cols
        self.split_cols = split_cols

    def fit(self, X, y=None):
        """fit"""

        return self

    def transform(self, X, y=None) -> pd.DataFrame:
        """transform"""

        _X = X.copy()

        # bool cols => 1 /0
        for col in self.bool_cols:
            _X[col] = _X[col].replace({True: 1, False: 0})

        # cabins
        for col in self.split_cols:
            for i in range(3):
                _X[f"_{col}_{i}"] = _X[col].apply(
                    lambda val: val.split("/")[i].upper()
                    if isinstance(val, str)
                    else val
                )
                try:
                    _X[f"_{col}_{i}"] = _X[f"_{col}_{i}"].astype(float)
                except Exception as e:
                    _X[f"_{col}_{i}_int"] = _X[f"_{col}_{i}"].replace(cab_dict)

        # split Name
        # _X.Name.apply(lambda i : len(i.split(" ")) if isinstance(i, str) else i) # no because only 2 on train
        _X["_len_Name"] = _X.Name.apply(lambda i: len(i) if isinstance(i, str) else i)
        _X["_FirstName"] = _X.Name.apply(
            lambda i: i.split(" ")[0] if isinstance(i, str) else i
        )
        _X["_LastName"] = _X.Name.apply(
            lambda i: i.split(" ")[1] if isinstance(i, str) else i
        )
        family_dict = _X._LastName.value_counts().to_dict()
        _X["_FamilySize"] = _X["_LastName"].replace(family_dict)

        return _X


class ColumnCleaner(TransformerMixin, BaseEstimator):
    """Selects columns from a dataframe."""

    def __init__(
        self,
        clean_cols=["PassengerId", "Cabin", "Name", "_FirstName", "_LastName"],
    ) -> None:
        """Initialize the ColumnSelector class."""

        self.clean_cols = clean_cols

    def fit(self, X, y=None):
        """Fit the ColumnSelector class."""

        return self

    def transform(self, X, y=None) -> pd.DataFrame:
        """Transform the ColumnSelector class."""

        _X = X.drop(columns=self.clean_cols, errors="ignore").copy()

        return _X


class ColumnSelector(TransformerMixin, BaseEstimator):
    """Selects columns from a dataframe."""

    def __init__(self, drop_cols: list = None, keep_cols: list = None) -> None:
        """Initialize the ColumnSelector class."""

        self.drop_cols = drop_cols
        self.keep_cols = keep_cols

    def fit(self, X, y=None):
        """Fit the ColumnSelector class."""

        return self

    def transform(self, X, y=None) -> pd.DataFrame:
        """Transform the ColumnSelector class."""

        _X = X.copy()

        if self.keep_cols:
            _X = _X[self.keep_cols]

        if self.drop_cols:
            _X = _X.drop(columns=self.drop_cols, errors="ignore")

        return _X


class LogTransformer(BaseEstimator, TransformerMixin):
    """Log transforms columns with skewness above a threshold."""

    def __init__(self, threshold=3) -> None:
        """Initialize the LogTransformer class."""

        self.threshold = threshold

    def fit(self, X, y=None):
        """Fit the LogTransformer class."""

        return self

    def transform(self, X, y=None) -> pd.DataFrame:
        """Transform the LogTransformer class."""

        _X = X.copy()

        # sep num cat
        num = _X.select_dtypes(include=np.number)
        cat = _X.select_dtypes(exclude=np.number)

        # skew threshold
        skew = num.skew()
        do_log_columns = skew[skew >= self.threshold].index.tolist()
        dont_log_columns = skew[skew < self.threshold].index.tolist()

        # log dont log
        do_log = num.loc[:, do_log_columns]
        do_log = np.log1p(do_log)
        dont_log = num.loc[:, dont_log_columns]

        # concat
        _X = pd.concat([cat, dont_log, do_log], axis=1, ignore_index=True)
        _X.columns = (
            cat.columns.tolist()
            + dont_log_columns
            + ["_log_" + i for i in do_log_columns]
        )

        return _X
