from dataclasses import dataclass
from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from macrosynergy.learning.forecasting.torch.samplers import TimeSeriesSampler


class MLPTrainer:
    """
    Trainer utility for fitting a PyTorch regression model with time-based train/validation
    splitting, optional scaling, and early stopping.

    Parameters
    ----------
    train_pct : float, optional
        Fraction of unique dates used for the training set (remainder used for validation).
        Default is 0.8.
    batch_size : int, optional
        Batch size used by the training and evaluation dataloaders. Default is 256.
    use_ts_sampler : bool, optional
        Whether to use a time-series batch sampler (contiguous batches by time order)
        instead of random shuffling. Default is False.
    learning_rate : float, optional
        Learning rate used by the AdamW optimizer. Default is 1e-3.
    weight_decay : float, optional
        Weight decay (L2 penalty) used by the AdamW optimizer. Default is 0.0.
    epochs : int, optional
        Maximum number of training epochs. Default is 50.
    loss_fn : torch.nn.Module, optional
        Loss function used for optimization. Default is ``nn.MSELoss()``.
    x_scaler : object or None, optional
        Feature scaler implementing ``fit`` and ``transform`` (e.g. ``StandardScaler``).
        If None, no scaling is applied to inputs. Default is ``StandardScaler(with_mean=False)``.
    y_scaler : object or None, optional
        Target scaler implementing ``fit`` and ``transform`` (e.g. ``StandardScaler``).
        If None, no scaling is applied to targets. Default is ``StandardScaler(with_mean=False)``.
    patience : int, optional
        Number of epochs without validation improvement tolerated before early stopping.
        Default is 5.
    reg_turnover : float, optional
        Strength of an L1 penalty on successive prediction differences, intended to
        discourage excessive turnover. If 0, no turnover penalty is applied. Default is 0.0.
    random_state : int, optional
        Random seed used for PyTorch initialization and training. Default is 0.
    verbose : bool, optional
        Whether to print periodic training diagnostics. Default is False.
    """

    def __init__(
        self,
        train_pct: float = 0.8,
        batch_size: int = 256,
        use_ts_sampler: bool = False,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        epochs: int = 50,
        loss_fn: nn.Module = nn.MSELoss(),
        x_scaler=StandardScaler(with_mean=False),
        y_scaler=StandardScaler(with_mean=False),
        patience: int = 5,
        reg_turnover: float = 0.0,
        random_state: int = 0,
        verbose: bool = False,
    ):
        self.train_pct = train_pct
        self.batch_size = batch_size
        self.use_ts_sampler = use_ts_sampler
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.loss_fn = loss_fn
        self.patience = patience
        self.reg_turnover = reg_turnover
        self.random_state = random_state
        self.verbose = verbose

        self.x_scaler = x_scaler
        self.y_scaler = y_scaler

    def fit(self, model, X, y):
        torch.manual_seed(self.random_state)

        X_train, y_train, X_val, y_val = self._time_split(X, y)

        X_train_s, X_val_s = self._fit_transform_X(X_train, X_val)
        y_train_s, y_val_s = self._fit_transform_y(y_train, y_val)

        train_loader, train_loader_eval, valid_loader = self._make_loaders(
            X_train_s, y_train_s, X_val_s, y_val_s
        )

        optimizer = torch.optim.AdamW(
            params=model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        best_score = np.inf
        best_state = None
        counter = 0

        for epoch in range(self.epochs):
            model.train()
            for X_i, y_i in train_loader:
                optimizer.zero_grad()
                preds = model(X_i)
                loss = self.loss_fn(preds, y_i)

                if self.reg_turnover > 0:
                    pweight_levels = preds[1:] - preds[:-1]
                    pweight_l1 = torch.mean(torch.abs(pweight_levels))
                    loss = loss + self.reg_turnover * pweight_l1

                loss.backward()
                optimizer.step()

            train_loss = self._eval_loss(model, train_loader_eval, self.loss_fn)
            valid_loss = self._eval_loss(model, valid_loader, self.loss_fn)

            if valid_loss < best_score:
                best_score = valid_loss
                best_state = deepcopy(model.state_dict())
                counter = 0
            else:
                counter += 1
                if counter >= self.patience:
                    break

            if self.verbose and (epoch % 5 == 0):
                print(
                    f"Epoch {epoch+1}: train_loss={train_loss:.6g}, valid_loss={valid_loss:.6g}"
                )

        if best_state is not None:
            model.load_state_dict(best_state)

        return model, self.x_scaler, self.y_scaler

    def _time_split(self, X, y):
        dates = sorted(X.index.get_level_values(1).unique())
        cut = int(self.train_pct * len(dates))
        train_dates, val_dates = dates[:cut], dates[cut:]

        X_train = X[X.index.get_level_values(1).isin(train_dates)]
        y_train = y[y.index.get_level_values(1).isin(train_dates)]
        X_val = X[X.index.get_level_values(1).isin(val_dates)]
        y_val = y[y.index.get_level_values(1).isin(val_dates)]
        return X_train, y_train, X_val, y_val

    def _make_loaders(self, X_train_s, y_train_s, X_val_s, y_val_s):
        train_dataset = torch.utils.data.TensorDataset(
            torch.Tensor(X_train_s), torch.Tensor(y_train_s)
        )
        valid_dataset = torch.utils.data.TensorDataset(
            torch.Tensor(X_val_s), torch.Tensor(y_val_s)
        )

        if not self.use_ts_sampler:
            train_loader = torch.utils.data.DataLoader(
                dataset=train_dataset, batch_size=self.batch_size, shuffle=True
            )
            train_loader_eval = torch.utils.data.DataLoader(
                dataset=train_dataset, batch_size=self.batch_size, shuffle=False
            )
        else:
            train_loader = torch.utils.data.DataLoader(
                dataset=train_dataset,
                batch_sampler=TimeSeriesSampler(
                    dataset=train_dataset, batch_size=self.batch_size, shuffle=True
                ),
            )
            train_loader_eval = torch.utils.data.DataLoader(
                dataset=train_dataset,
                batch_sampler=TimeSeriesSampler(
                    dataset=train_dataset, batch_size=self.batch_size, shuffle=False
                ),
            )

        valid_loader = torch.utils.data.DataLoader(
            dataset=valid_dataset,
            batch_sampler=TimeSeriesSampler(
                dataset=valid_dataset, batch_size=self.batch_size, shuffle=False
            ),
        )

        return train_loader, train_loader_eval, valid_loader

    def _fit_transform_X(self, X_train, X_val):
        if self.x_scaler is None:
            return X_train, X_val
        self.x_scaler.fit(X_train)
        return self.x_scaler.transform(X_train), self.x_scaler.transform(X_val)

    def _fit_transform_y(self, y_train, y_val):
        if self.y_scaler is None:
            return y_train, y_val
        self.y_scaler.fit(y_train)
        return self.y_scaler.transform(y_train), self.y_scaler.transform(y_val)

    @staticmethod
    def _eval_loss(model, loader, loss_fn):
        model.eval()
        total = 0.0
        with torch.no_grad():
            for X_i, y_i in loader:
                preds = model(X_i)
                total += loss_fn(preds, y_i).item()
        return total / len(loader)
