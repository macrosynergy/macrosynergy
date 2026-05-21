import numbers
from copy import deepcopy
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler

from macrosynergy.learning.forecasting.torch.models.attention import MacroAttentionNet


class AttentionRegressor(BaseEstimator, RegressorMixin):
    """
    Sklearn-compatible macro attention regressor.

    This wrapper converts panel observations into rolling lookback sequences and
    trains a small Transformer encoder.
    """

    def __init__(
        self,
        lookback=36,
        d_model=32,
        n_heads=2,
        n_layers=1,
        dim_feedforward=None,
        dropout_p=0.1,
        head_activation="identity",
        loss_func=None,
        optimizer="AdamW",
        batch_size=32,
        learning_rate=3e-4,
        weight_decay=1e-4,
        epochs=500,
        patience=50,
        train_pct=0.8,
        x_scaler=None,
        y_scaler=None,
        inverse_transform_preds=False,
        device=None,
        random_state=42,
        verbose=False,
    ):
        self.lookback = lookback
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dim_feedforward = dim_feedforward
        self.dropout_p = dropout_p
        self.head_activation = head_activation

        self.loss_func = loss_func
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.patience = patience
        self.train_pct = train_pct

        self.x_scaler = x_scaler
        self.y_scaler = y_scaler
        self.inverse_transform_preds = inverse_transform_preds

        self.device = device
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y):
        self._check_fit_params(X, y)

        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        X_df = self._normalize_X(X)
        y_ser = self._normalize_y(y, X_df.index)

        self.feature_names_in_ = list(X_df.columns)
        self.n_features_in_ = X_df.shape[1]

        self.x_scaler_ = deepcopy(
            self.x_scaler if self.x_scaler is not None else StandardScaler(with_mean=False)
        )
        self.y_scaler_ = deepcopy(
            self.y_scaler if self.y_scaler is not None else StandardScaler(with_mean=False)
        )

        X_scaled = pd.DataFrame(
            self.x_scaler_.fit_transform(X_df),
            index=X_df.index,
            columns=X_df.columns,
        )

        y_arr = y_ser.to_numpy().reshape(-1, 1)
        y_scaled = pd.Series(
            self.y_scaler_.fit_transform(y_arr).ravel(),
            index=y_ser.index,
            name=y_ser.name,
        )

        X_seq, y_seq, self.train_index_ = self._make_sequences(X_scaled, y_scaled)

        split = int(len(X_seq) * self.train_pct)
        if split <= 0 or split >= len(X_seq):
            raise ValueError("train_pct leaves empty train or validation set.")

        X_train, y_train = X_seq[:split], y_seq[:split]
        X_val, y_val = X_seq[split:], y_seq[split:]

        self.device_ = torch.device(
            self.device
            if self.device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.model_ = MacroAttentionNet(
            n_inputs=self.n_features_in_,
            n_outputs=1,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            dim_feedforward=self.dim_feedforward,
            dropout_p=self.dropout_p,
            max_seq_len=self.lookback,
            head_activation=self.head_activation,
        ).to(self.device_)

        loss_func = self.loss_func if self.loss_func is not None else nn.MSELoss()
        loss_func = loss_func.to(self.device_)

        optimizer = self._make_optimizer(self.model_)

        train_dataset = torch.utils.data.TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1),
        )

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
        )

        X_val_t = torch.tensor(X_val, dtype=torch.float32).to(self.device_)
        y_val_t = torch.tensor(y_val, dtype=torch.float32).reshape(-1, 1).to(self.device_)

        best_state = None
        best_val_loss = np.inf
        stale_epochs = 0

        for epoch in range(self.epochs):
            self.model_.train()

            for xb, yb in train_loader:
                xb = xb.to(self.device_)
                yb = yb.to(self.device_)

                pred = self.model_(xb)
                loss = loss_func(pred, yb)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model_.parameters(), 1.0)
                optimizer.step()

            self.model_.eval()
            with torch.no_grad():
                val_pred = self.model_(X_val_t)
                val_loss = loss_func(val_pred, y_val_t).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = deepcopy(self.model_.state_dict())
                stale_epochs = 0
            else:
                stale_epochs += 1

            if self.verbose and epoch % 25 == 0:
                print(f"epoch={epoch}, val_loss={val_loss:.6f}")

            if stale_epochs >= self.patience:
                break

        if best_state is not None:
            self.model_.load_state_dict(best_state)

        self.best_val_loss_ = best_val_loss
        self.is_fitted_ = True

        return self

    def predict(self, X):
        self._check_is_fitted()

        X_df = self._normalize_X(X)

        missing_cols = set(self.feature_names_in_) - set(X_df.columns)
        if missing_cols:
            raise ValueError(f"Missing columns: {sorted(missing_cols)}")

        X_df = X_df[self.feature_names_in_]

        X_scaled = pd.DataFrame(
            self.x_scaler_.transform(X_df),
            index=X_df.index,
            columns=X_df.columns,
        )

        X_seq, pred_index = self._make_prediction_sequences(X_scaled)

        self.model_.eval()
        with torch.no_grad():
            X_t = torch.tensor(X_seq, dtype=torch.float32).to(self.device_)
            pred = self.model_(X_t).cpu().numpy().reshape(-1, 1)

        if self.inverse_transform_preds:
            pred = self.y_scaler_.inverse_transform(pred)

        return pd.Series(pred.ravel(), index=pred_index, name="prediction")

    def predict_with_attention(self, X):
        self._check_is_fitted()

        X_df = self._normalize_X(X)
        X_df = X_df[self.feature_names_in_]

        X_scaled = pd.DataFrame(
            self.x_scaler_.transform(X_df),
            index=X_df.index,
            columns=X_df.columns,
        )

        X_seq, pred_index = self._make_prediction_sequences(X_scaled)

        self.model_.eval()
        with torch.no_grad():
            X_t = torch.tensor(X_seq, dtype=torch.float32).to(self.device_)
            pred, attn = self.model_(X_t, return_attention=True)

        pred = pred.cpu().numpy().reshape(-1, 1)
        attn = attn.cpu().numpy()

        if self.inverse_transform_preds:
            pred = self.y_scaler_.inverse_transform(pred)

        pred_ser = pd.Series(pred.ravel(), index=pred_index, name="prediction")
        attn_df = pd.DataFrame(
            attn,
            index=pred_index,
            columns=[f"lag_{i}" for i in range(self.lookback, 0, -1)],
        )

        return pred_ser, attn_df

    def _make_optimizer(self, model):
        if self.optimizer == "AdamW":
            return torch.optim.AdamW(
                model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )

        if self.optimizer == "SGD":
            return torch.optim.SGD(
                model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )

        if self.optimizer == "SGD+mom":
            return torch.optim.SGD(
                model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                momentum=0.9,
            )

        raise ValueError("optimizer must be one of 'AdamW', 'SGD', or 'SGD+mom'.")

    def _make_sequences(self, X, y):
        X_seq = []
        y_seq = []
        indices = []

        joined = X.join(y.rename("__target__"), how="inner")

        for cid, panel in joined.groupby(level="cid", sort=False):
            panel = panel.sort_index(level="real_date")

            values = panel[self.feature_names_in_].to_numpy(dtype=float)
            targets = panel["__target__"].to_numpy(dtype=float)
            idx = panel.index

            for end in range(self.lookback - 1, len(panel)):
                window = values[end - self.lookback + 1 : end + 1]
                target = targets[end]

                if np.isnan(window).any() or np.isnan(target):
                    continue

                X_seq.append(window)
                y_seq.append(target)
                indices.append(idx[end])

        if not X_seq:
            raise ValueError("No valid sequences produced. Reduce lookback or check missing data.")

        return np.asarray(X_seq), np.asarray(y_seq), pd.MultiIndex.from_tuples(indices)

    def _make_prediction_sequences(self, X):
        X_seq = []
        indices = []

        for cid, panel in X.groupby(level="cid", sort=False):
            panel = panel.sort_index(level="real_date")

            values = panel[self.feature_names_in_].to_numpy(dtype=float)
            idx = panel.index

            for end in range(self.lookback - 1, len(panel)):
                window = values[end - self.lookback + 1 : end + 1]

                if np.isnan(window).any():
                    continue

                X_seq.append(window)
                indices.append(idx[end])

        if not X_seq:
            raise ValueError("No valid prediction sequences produced.")

        return np.asarray(X_seq), pd.MultiIndex.from_tuples(indices)

    def _normalize_X(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame.")

        X = X.copy()

        if {"cid", "real_date"}.issubset(X.columns):
            X["real_date"] = pd.to_datetime(X["real_date"])
            X = X.set_index(["cid", "real_date"])

        if not isinstance(X.index, pd.MultiIndex):
            raise ValueError("X must be indexed by ['cid', 'real_date'].")

        if list(X.index.names) != ["cid", "real_date"]:
            X.index = X.index.set_names(["cid", "real_date"])

        return X.sort_index()

    def _normalize_y(self, y, index):
        if isinstance(y, pd.DataFrame):
            if y.shape[1] != 1:
                raise ValueError("y DataFrame must have exactly one column.")
            y = y.iloc[:, 0]

        if not isinstance(y, pd.Series):
            y = pd.Series(y, index=index)

        y = y.copy()

        if not isinstance(y.index, pd.MultiIndex):
            y.index = index

        if list(y.index.names) != ["cid", "real_date"]:
            y.index = y.index.set_names(["cid", "real_date"])

        return y.sort_index()

    def _check_fit_params(self, X, y):
        if not isinstance(self.lookback, numbers.Integral) or self.lookback < 2:
            raise ValueError("lookback must be an integer greater than 1.")

        if not isinstance(self.batch_size, numbers.Integral) or self.batch_size < 1:
            raise ValueError("batch_size must be a positive integer.")

        if not isinstance(self.epochs, numbers.Integral) or self.epochs < 1:
            raise ValueError("epochs must be a positive integer.")

        if not isinstance(self.patience, numbers.Integral) or self.patience < 1:
            raise ValueError("patience must be a positive integer.")

        if not 0 < self.train_pct < 1:
            raise ValueError("train_pct must be between 0 and 1.")

    def _check_is_fitted(self):
        if not getattr(self, "is_fitted_", False):
            raise RuntimeError("AttentionRegressor is not fitted.")