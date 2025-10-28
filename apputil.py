# apputil.py (or a file you import from)
from __future__ import annotations
from typing import Iterable, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


class GroupEstimate:
    """
    Group-based estimator that predicts an aggregate (mean or median)
    of a continuous target y for combinations of categorical inputs X.

    Parameters
    ----------
    estimate : str
        'mean' or 'median' (controls how group estimates are computed).
    """

    def __init__(self, estimate: str = "mean"):
        if estimate not in {"mean", "median"}:
            raise ValueError("estimate must be 'mean' or 'median'")
        self.estimate = estimate
        self._group_map = {}  # mapping: tuple(key_values) -> float
        self._columns: List[str] = []
        self._default_category: Optional[str] = None
        self._default_map = {}  # mapping for default_category -> float

    # -------------------------
    # Fit
    # -------------------------
    def fit(
        self,
        X: Union[pd.DataFrame, Iterable[Iterable]],
        y: Union[pd.Series, Iterable],
        default_category: Optional[str] = None,
    ) -> "GroupEstimate":
        """
        Fit the GroupEstimate.

        Parameters
        ----------
        X : DataFrame or 2D iterable
            Categorical columns used to form groups. If iterable, will be coerced to DataFrame.
        y : 1D iterable
            Continuous target values. Must be same length as X and contain no missing values.
        default_category : str or None
            If supplied, must be a column name in X. When a combination is missing at predict time,
            and default_category was provided, the prediction will fall back to the estimate
            computed for that single category (if available).
        """
        # convert X,y to DataFrame / Series
        if isinstance(X, pd.DataFrame):
            Xdf = X.copy()
        else:
            Xdf = pd.DataFrame(X)

        yser = pd.Series(y).reset_index(drop=True)

        if len(Xdf) != len(yser):
            raise ValueError("X and y must have the same number of rows")

        if yser.isna().any():
            raise ValueError("y must not contain missing values")

        # normalize column names (string)
        Xdf.columns = [str(c) for c in Xdf.columns.tolist()]
        self._columns = list(Xdf.columns)

        if default_category is not None:
            if default_category not in self._columns:
                raise ValueError("default_category must be one of the columns in X")
            self._default_category = default_category

        # Combine for grouping
        df = Xdf.copy()
        df["_target"] = yser.values

        # Group by all columns -> compute mean or median
        if self.estimate == "mean":
            grouped = df.groupby(self._columns, observed=True)["_target"].mean()
        else:
            grouped = df.groupby(self._columns, observed=True)["_target"].median()

        # Save mapping: tuple(key_values) -> float
        # For single-column group, we still store keys as tuples to keep uniformity.
        self._group_map = {}
        for idx, val in grouped.items():
            # pandas returns single value index as that value (not a tuple)
            if len(self._columns) == 1:
                key = (idx,)  # make it a tuple
            else:
                key = tuple(idx)
            self._group_map[key] = float(val)

        # If default_category provided, compute per-category aggregates
        if self._default_category:
            if self.estimate == "mean":
                default_grouped = df.groupby(self._default_category, observed=True)["_target"].mean()
            else:
                default_grouped = df.groupby(self._default_category, observed=True)["_target"].median()

            self._default_map = {k: float(v) for k, v in default_grouped.items()}

        return self

    # -------------------------
    # Predict
    # -------------------------
    def predict(self, X_: Union[pd.DataFrame, Iterable[Iterable]]) -> np.ndarray:
        """
        Predict the group estimate for each row in X_.

        Parameters
        ----------
        X_ : DataFrame or 2D iterable
            Observations to predict. Must contain the same number of columns and same ordering
            as the X provided during fit.

        Returns
        -------
        numpy.ndarray of floats (with np.nan for missing groups)
        """
        if not self._columns:
            raise RuntimeError("Model is not fitted. Call fit(X, y) before predict().")

        # Convert to DataFrame
        if isinstance(X_, pd.DataFrame):
            Xpred = X_.copy()
        else:
            Xpred = pd.DataFrame(X_)

        # Normalize columns: if Xpred has column names and they match our columns -> keep order.
        # If it has no columns or different names, coerce by positional order.
        if list(Xpred.columns) == self._columns:
            Xpred = Xpred[self._columns].copy()
        else:
            # If number of columns mismatches, raise
            if Xpred.shape[1] != len(self._columns):
                raise ValueError(
                    f"Predict input must have {len(self._columns)} columns (got {Xpred.shape[1]})"
                )
            Xpred.columns = self._columns

        # Build keys and map
        results: List[float] = []
        missing_count = 0
        missing_and_no_default = 0
        missing_but_filled_by_default = 0

        for _, row in Xpred.iterrows():
            if len(self._columns) == 1:
                key = (row[self._columns[0]],)
            else:
                key = tuple(row[col] for col in self._columns)

            if key in self._group_map:
                results.append(self._group_map[key])
            else:
                # try default category fallback if available
                if self._default_category:
                    default_key = row[self._default_category]
                    if pd.isna(default_key):
                        # can't fallback to default if that value is missing
                        results.append(np.nan)
                        missing_count += 1
                        missing_and_no_default += 1
                    elif default_key in self._default_map:
                        results.append(self._default_map[default_key])
                        missing_count += 1
                        missing_but_filled_by_default += 1
                    else:
                        # default category value not found -> leave NaN
                        results.append(np.nan)
                        missing_count += 1
                        missing_and_no_default += 1
                else:
                    results.append(np.nan)
                    missing_count += 1
                    missing_and_no_default += 1

        # Print summary about missing groups
        if missing_count > 0:
            if self._default_category:
                print(
                    f"{missing_count} rows had missing group-combinations; "
                    f"{missing_but_filled_by_default} were filled using '{self._default_category}' "
                    f"estimates and {missing_and_no_default} remain NaN."
                )
            else:
                print(f"{missing_count} rows had missing group-combinations and were returned as NaN.")

        return np.array(results, dtype=float)


# -------------------------
# Example usage (for local testing)
# -------------------------
if __name__ == "__main__":  # quick manual check
    # sample data
    df = pd.DataFrame(
        {
            "country": ["G", "G", "M", "M", "B", "B"],
            "roast": ["Light", "Light", "Medium", "Light", "Dark", "Light"],
            "rating": [88.0, 89.0, 91.0, 90.0, 92.0, 87.0],
        }
    )

    X = df[["country", "roast"]]
    y = df["rating"]

    gm = GroupEstimate(estimate="mean")
    gm.fit(X, y)
    queries = [["G", "Light"], ["M", "Medium"], ["C", "Dark"]]
    print("No default:", gm.predict(queries))  # last should be nan

    # with default category fallback
    gm2 = GroupEstimate(estimate="mean")
    gm2.fit(X, y, default_category="country")
    print("With default:", gm2.predict(queries))  # last should be filled using country-level mean