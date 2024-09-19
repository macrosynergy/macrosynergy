from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
import pandas as pd
import numpy as np
from linearmodels.panel import RandomEffects


class RandomEffects(BaseEstimator):
    """
    A custom sklearn estimator that fits a random effects model using linearmodels.
    """

    def __init__(self, group_col):
        self.group_col = group_col

    def fit(self, X, y):
        """
        Fit the random effects model.

        Parameters
        ----------
        X : pandas DataFrame, shape (n_samples, n_features)
            Training data, including the group column.
        y : array-like, shape (n_samples,)
            Target values.
        """
        # Check that X and y have correct shape
        # X, y = check_X_y(X, y, accept_pd_dataframe=True)

        # Ensure X is a DataFrame
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame.")

        # Check if group_col exists in X
        if self.group_col not in X.columns:
            raise ValueError(f"Group column '{self.group_col}' not found in X.")

        # Store the group column and reset index
        self.groups_ = X[self.group_col]
        X_features = X.drop(columns=[self.group_col])
        X_features = X_features.reset_index(drop=True)
        y = pd.Series(y).reset_index(drop=True)
        groups = self.groups_.reset_index(drop=True)

        # Prepare the data for linearmodels
        # Set the index to (group, observation)
        data = pd.concat([X_features, y], axis=1)
        data[self.group_col] = groups
        data.set_index([self.group_col, data.index], inplace=True)

        # Fit the random effects model
        self.model_ = RandomEffects(data[y.name], data.drop(columns=[y.name]))
        self.result_ = self.model_.fit()

        # Store the fixed effects parameters
        self.params_ = self.result_.params

        return self

    def predict(self, X):
        """
        Predict using the random effects model.

        Parameters
        ----------
        X : pandas DataFrame, shape (n_samples, n_features)
            Test data, including the group column.

        Returns
        -------
        y_pred : array-like, shape (n_samples,)
            Predicted values.
        """
        pass

    def get_params(self):
        """Get parameters for this estimator."""
        return {"group_col": self.group_col}

    def set_params(self, **params):
        """Set parameters for this estimator."""
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def demean(self, df):
        mu = df.groupby(level=1).transform("mean")
        return (df - mu + df.mean(0)).to_numpy()

    def _s2(self, eps, _nobs, _scale=1.0):
        return _scale * float(np.squeeze(eps.T @ eps)) / _nobs

    def _cov(self, x, eps, _nobs):
        s2 = self._s2(eps, _nobs)
        cov = s2 * np.linalg.inv(x.T @ x)
        return cov

    def _fit(self, df_y, df_x):
        if isinstance(df_y, pd.Series):
            df_y = df_y.to_frame()
        if isinstance(df_x, pd.Series):
            df_x = df_x.to_frame()

        y_demeaned = self.demean(df_y)
        x_demeaned = self.demean(df_x)

        params, ssr, _, _ = np.linalg.lstsq(x_demeaned, y_demeaned, rcond=None)
        eps = y_demeaned - x_demeaned @ params

        # Between Estimation
        xbar = df_x.groupby(level=1).mean()
        ybar = df_y.groupby(level=1).mean()
        params, ssr, _, _ = np.linalg.lstsq(xbar, ybar, rcond=None)
        u = np.asarray(ybar.values) - np.asarray(xbar.values) @ params

        # Estimate variances
        nobs = df_y.shape[0]
        neffects = u.shape[0]
        nvars = df_x.shape[1]

        # Idiosyncratic variance
        sigma2_e = float(np.squeeze(eps.T @ eps)) / (nobs - nvars - neffects + 1)

        cid_count = np.asarray(df_y.groupby(level=1).count())
        cid_bar = neffects / ((1.0 / cid_count)).sum()

        sigma2_a = max(0.0, (ssr / (neffects - nvars)) - sigma2_e / cid_bar)

        # Theta
        theta = 1.0 - np.sqrt(sigma2_e / (cid_count * sigma2_a + sigma2_e))

        index = df_y.index
        reindex = index.levels[1][index.codes[1]]
        ybar = (theta * ybar).loc[reindex]
        xbar = (theta * xbar).loc[reindex]

        y = np.asarray(df_y)
        x = np.asarray(df_x)

        y = y - ybar.values
        x = x - xbar.values

        params, ssr, _, _ = np.linalg.lstsq(x, y, rcond=None)
        eps = y - x @ params

        cov_matrix = self._cov(x, eps, nobs)

        index = df_y.index
        fitted = pd.DataFrame(df_x.values @ params, index, ["fitted_values"])
        effects = pd.DataFrame(
            np.asarray(df_y) - np.asarray(fitted) - eps,
            index,
            ["estimated_effects"],
        )
        idiosyncratic = pd.DataFrame(eps, index, ["idiosyncratic"])

        residual_ss = float(np.squeeze(eps.T @ eps))

        y_demeaned = y - y.mean(0)
        total_ss = float(np.squeeze(y_demeaned.T @ y_demeaned))
        r2 = 1 - residual_ss / total_ss
        return


if __name__ == "__main__":
    import pandas as pd
    from sklearn.model_selection import train_test_split

    # Sample data
    data = pd.DataFrame(
        {
            "Group": np.random.choice(["A", "B", "C"], size=100),
            "Feature1": np.random.randn(100),
            "Feature2": np.random.randn(100),
            "Target": np.random.randn(100),
        }
    )

    X = data[["Feature1", "Feature2", "Group"]]
    y = data["Target"]

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Initialize and fit the model
    model = RandomEffectsRegressor(group_col="Group")
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)
