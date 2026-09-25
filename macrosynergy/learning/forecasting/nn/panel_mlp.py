import logging
import numbers
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.model_selection import BaseCrossValidator
from sklearn.preprocessing import StandardScaler

from macrosynergy.learning.forecasting.torch.losses import NegCrossSectionalIC
from macrosynergy.learning.forecasting.torch.models.panel_mlp import SharedHeadPanelMLP
from macrosynergy.learning.forecasting.torch.samplers import PanelBatchSampler

logger = logging.getLogger(__name__)

# Epoch interval at which the training diagnostics are logged
_DIAGNOSTIC_LOG_EVERY = 5


class PanelMLPRegressor(BaseEstimator, RegressorMixin):
    """
    Feed-forward neural network over (security, period) observations with a shared head.

    Parameters
    ----------
    n_latent : Union[int, List[int]], optional
        Number of units in the hidden layer(s). Default is 32.
    encoder_activation : str, optional
        Activation for the encoder. Default is "relu".
    head_activation : str, optional
        Activation for the output. Default is "identity".
    fit_encoder_intercept : bool, optional
        Whether to fit intercepts in the encoder. Default is True.
    fit_head_intercept : bool, optional
        Whether to fit an intercept in the head. Default is True.
    dropout_p : float or list, optional
        Dropout probability for the encoder. Default is 0.
    normalization : str, optional
        "layer", "batch" or None. Default is None.
    embedding_dim : int, optional
        Width of a learned per-security embedding. Default is None, for complete pooling.
    reg_embedding : float, optional
        Weight on the L2 penalty shrinking the embedding toward zero, i.e. toward the
        pooled model. Default is 0.
    signal_modifier : nn.Module, optional
        Module applied to the (period, security) grid of outputs. Default is None.
    loss_func : nn.Module, optional
        Loss, evaluated on the (period, security) grid. Default is `NegCrossSectionalIC()`.
    optimizer : str, optional
        "AdamW", "Adam", "SGD" or "SGD+mom". Default is "AdamW".
    batch_mode : str, optional
        `PanelBatchSampler` mode: "period", "asset", "block" or "mc". Default is "period".
    batch_periods : int, optional
        Periods per batch. Default is 32.
    batch_assets : int, optional
        Securities per batch, required by the "asset" and "block" modes and available to
        "mc" as an alternative to `asset_fraction`. Default is None.
    asset_fraction : float, optional
        Fraction of securities drawn per batch under "mc". Default is None.
    draws_per_block : int, optional
        Independent security draws per period block under "mc". Default is None, which
        takes `ceil(1 / asset_fraction)`.
    learning_rate : float, optional
        Learning rate. Default is 3e-4.
    weight_decay : float, optional
        Weight decay. Default is 1e-4.
    epochs : int, optional
        Maximum number of epochs. Default is 100.
    patience : int, optional
        Epochs without validation improvement before stopping, or None to disable early
        stopping. Default is 10.
    train_splitter : float or BaseCrossValidator, optional
        Fraction of dates used for training, or a splitter. Default is 0.7.
    x_scaler : TransformerMixin, optional
        Scaler for the features, fitted on the training split. Default is
        `StandardScaler()`.
    y_scaler : TransformerMixin, optional
        Scaler for the target. Default is None, since the ranking objectives are
        scale-free and portfolio objectives need raw returns.
    verbose : bool, optional
        Whether to print progress. Default is False.
    random_state : int or list of int, optional
        Seed, or seeds forming an ensemble. Default is 42.

    Notes
    -----
    A sibling of `MLPRegressor` that differs in one structural respect: what an
    observation is.

    `MLPRegressor` treats a *period* as an observation and gives every security its own
    output unit, so each security's behaviour is carried by a private parameter vector
    fitted on that security's history alone. `PanelMLPRegressor` treats a *(security,
    period) pair* as an observation and maps its features to a forecast through one
    shared network, so every parameter is estimated on the whole panel and securities
    differ through their characteristics rather than through free coefficients. See
    `SharedHeadPanelMLP` for what that buys and what it requires.

    **Features must vary across securities.** With macro-only inputs every security
    receives an identical forecast at each date and the cross-sectional dispersion of the
    predictions is zero by construction. `prediction_dispersion_` records that dispersion
    after fitting, so the degenerate case is visible immediately rather than being
    mistaken for an absence of skill.

    **How the losses still apply.** The portfolio, ranking and benchmark objectives all
    expect a (period, security) matrix. `PanelBatchSampler` yields rectangular batches — a
    block of periods, optionally restricted to a subset of securities — so each batch is
    scattered back into such a matrix before the loss sees it, with absent cells left as
    NaN for the loss's own masking to handle. Every existing objective therefore works
    unchanged.

    Two consequences of that arrangement are worth stating. First, the batching mode and
    the loss must agree: a loss that is a ratio over the batch, such as `NegSharpeRatio`,
    requires "period" mode, whereas a loss that is a mean over per-period statistics, such
    as `NegCrossSectionalIC`, admits any mode. The "mc" mode, which holds a block of periods
    fixed and resamples the securities within it, is the one intended for regularising a
    panel with many more securities than periods, and it is only meaningful with a
    per-period objective: a realised portfolio volatility computed on a fraction `f` of
    the names is biased upward by roughly `1 / sqrt(f)`. Second, a `signal_modifier` is applied to the
    full grid and the absent cells are zeroed afterwards, so a modifier that pins gross
    exposure will do so over the grid rather than over the observed securities. On an
    unbalanced panel the ranking objectives, which are invariant to both the scale and the
    level of the outputs, are the cleaner pairing.

    This estimator does not accept `refit`, a custom `torch_model`, a scheduler or
    `reg_turnover`. It is a research prototype for the shared-head question, not a
    drop-in replacement.
    """

    def __init__(
        self,
        n_latent=32,
        encoder_activation="relu",
        head_activation="identity",
        fit_encoder_intercept=True,
        fit_head_intercept=True,
        dropout_p=0,
        normalization=None,
        embedding_dim=None,
        reg_embedding=0,
        signal_modifier=None,
        loss_func=None,
        optimizer="AdamW",
        batch_mode="period",
        batch_periods=32,
        batch_assets=None,
        asset_fraction=None,
        draws_per_block=None,
        learning_rate=3e-4,
        weight_decay=1e-4,
        epochs=100,
        patience=10,
        train_splitter=0.7,
        x_scaler=None,
        y_scaler=None,
        verbose=False,
        random_state=42,
    ):
        self.n_latent = n_latent
        self.encoder_activation = encoder_activation
        self.head_activation = head_activation
        self.fit_encoder_intercept = fit_encoder_intercept
        self.fit_head_intercept = fit_head_intercept
        self.dropout_p = dropout_p
        self.normalization = normalization
        self.embedding_dim = embedding_dim
        self.reg_embedding = reg_embedding
        self.signal_modifier = signal_modifier
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.batch_mode = batch_mode
        self.batch_periods = batch_periods
        self.batch_assets = batch_assets
        self.asset_fraction = asset_fraction
        self.draws_per_block = draws_per_block
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.patience = patience
        self.train_splitter = train_splitter
        self.x_scaler = x_scaler
        self.y_scaler = y_scaler
        self.verbose = verbose
        self.random_state = random_state

    # ------------------------------------------------------------------ fitting

    def fit(self, X, y):
        """
        Fit the network.

        Parameters
        ----------
        X : pd.DataFrame
            Features, indexed by (security, date). Each row is one observation.
        y : pd.Series or single-column pd.DataFrame
            Forward return of that security over that period.
        """
        self._check_fit_params(X, y)

        loss_func = self.loss_func if self.loss_func is not None else NegCrossSectionalIC()
        x_scaler = self.x_scaler if self.x_scaler is not None else StandardScaler()
        seeds = (
            self.random_state
            if isinstance(self.random_state, list)
            else [self.random_state]
        )

        y = y.iloc[:, 0] if isinstance(y, pd.DataFrame) else y

        # Integer codes for the two panel dimensions, ordered by their sorted unique
        # values so that "consecutive periods" is meaningful
        self.assets_ = np.sort(X.index.get_level_values(0).unique().to_numpy())
        self.periods_ = np.sort(X.index.get_level_values(1).unique().to_numpy())
        self.n_assets_ = len(self.assets_)
        self.n_features_ = X.shape[1]

        asset_codes = pd.Index(self.assets_).get_indexer(X.index.get_level_values(0))
        period_codes = pd.Index(self.periods_).get_indexer(X.index.get_level_values(1))

        splits = self._create_splits(X, y, self.train_splitter)

        self.models_ = []
        self.scalers_ = []
        self.y_scalers_ = []
        self.training_history_ = []
        self.selected_epochs_ = []

        for seed in seeds:
            for fold, (train_rows, valid_rows) in enumerate(splits):
                torch.manual_seed(seed)

                scaler = clone(x_scaler)
                scaler.fit(X.iloc[train_rows])
                y_scaler = clone(self.y_scaler) if self.y_scaler is not None else None
                if y_scaler is not None:
                    y_scaler.fit(y.iloc[train_rows].to_frame())

                train_loader = self._make_loader(
                    X, y, scaler, y_scaler, asset_codes, period_codes, train_rows,
                    shuffle=True, seed=seed,
                )
                train_eval_loader = self._make_loader(
                    X, y, scaler, y_scaler, asset_codes, period_codes, train_rows,
                    shuffle=False, seed=seed,
                )
                valid_loader = (
                    self._make_loader(
                        X, y, scaler, y_scaler, asset_codes, period_codes, valid_rows,
                        shuffle=False, seed=seed,
                    )
                    if valid_rows is not None and len(valid_rows) > 0
                    else None
                )

                model = SharedHeadPanelMLP(
                    n_inputs=self.n_features_,
                    n_latent=self.n_latent,
                    n_assets=self.n_assets_,
                    embedding_dim=self.embedding_dim,
                    encoder_activation=self.encoder_activation,
                    head_activation=self.head_activation,
                    fit_encoder_intercept=self.fit_encoder_intercept,
                    fit_head_intercept=self.fit_head_intercept,
                    dropout_p=self.dropout_p,
                    normalization=self.normalization,
                )
                optimizer = self._make_optimizer(model)

                model, history, selected = self._train(
                    model=model,
                    optimizer=optimizer,
                    loss_func=loss_func,
                    train_loader=train_loader,
                    train_eval_loader=train_eval_loader,
                    valid_loader=valid_loader,
                    context={"seed": seed, "fold": fold},
                )

                self.models_.append(model)
                self.scalers_.append(scaler)
                self.y_scalers_.append(y_scaler)
                self.training_history_.append(history)
                self.selected_epochs_.append(selected)

        # Cross-sectional dispersion of the fitted predictions: zero means the features
        # carry no security-specific information and no ranking is possible, whatever the
        # loss reports
        preds = self.predict(X)
        self.prediction_dispersion_ = float(
            preds.groupby(level=1).std().mean()
        )
        if self.prediction_dispersion_ == 0 or np.isnan(self.prediction_dispersion_):
            logger.warning(
                "Predictions carry no cross-sectional dispersion: with a shared head "
                "this means the features do not vary across securities, so no ranking "
                "is representable."
            )

        return self

    def _train(self, model, optimizer, loss_func, train_loader, train_eval_loader,
               valid_loader, context):
        """Train one network, with early stopping when a validation loader is present."""
        best_state, best_score, counter, best_epoch = None, np.inf, 0, 0
        history = []

        for epoch in range(self.epochs):
            model.train()
            for batch in train_loader:
                optimizer.zero_grad()
                grid_pred, grid_true = self._forward_grid(model, batch)
                loss = loss_func(grid_pred, grid_true)
                if self.reg_embedding > 0:
                    loss = loss + self.reg_embedding * model.embedding_penalty()
                loss.backward()
                optimizer.step()

            if valid_loader is None or self.patience is None:
                continue

            report = (epoch % _DIAGNOSTIC_LOG_EVERY == 0) or (epoch == self.epochs - 1)
            train_stats = self._evaluate(model, train_eval_loader, loss_func, report)
            valid_stats = self._evaluate(model, valid_loader, loss_func, report)

            record = dict(context)
            record.update(epoch=epoch + 1, **{f"train_{k}": v for k, v in train_stats.items()})
            record.update({f"valid_{k}": v for k, v in valid_stats.items()})
            history.append(record)
            logger.debug("panel mlp epoch diagnostics", extra={"mlp_diagnostics": record})

            if valid_stats["loss"] < best_score:
                best_score = valid_stats["loss"]
                best_state = deepcopy(model.state_dict())
                best_epoch = epoch + 1
                counter = 0
            else:
                counter += 1

            if self.verbose and report:
                print(
                    f"Epoch {epoch + 1}: train loss = {train_stats['loss']:.4f}, "
                    f"valid loss = {valid_stats['loss']:.4f}, best = {best_score:.4f}"
                )

            if counter >= self.patience:
                break

        if best_state is not None:
            model.load_state_dict(best_state)

        return model, history, best_epoch

    def _evaluate(self, model, loader, loss_func, diagnostics=False):
        """Loss, and optionally the cross-sectional IC, over a loader."""
        model.eval()
        total, n_batches, ics = 0.0, 0, []

        with torch.no_grad():
            for batch in loader:
                grid_pred, grid_true = self._forward_grid(model, batch)
                total += float(loss_func(grid_pred, grid_true))
                n_batches += 1
                if diagnostics:
                    ic, usable = NegCrossSectionalIC(min_names=10).period_ic(grid_pred, grid_true)
                    ics.extend(ic[usable].tolist())

        return {
            "loss": total / max(n_batches, 1),
            "ic": float(np.mean(ics)) if ics else np.nan,
            "periods": len(ics),
        }

    # ------------------------------------------------------------ grid handling

    def _forward_grid(self, model, batch):
        """
        Run one batch through the network and scatter it into a (period, security) grid.

        Parameters
        ----------
        model : SharedHeadPanelMLP
            Network to evaluate.
        batch : tuple of torch.Tensor
            Features, targets, period codes and security codes of the batch's rows.

        Returns
        -------
        tuple of torch.Tensor
            The predictions and targets as (n_periods, n_securities) matrices. Cells with
            no observation carry a zero prediction and a NaN target, which every loss in
            this package masks out.
        """
        X_i, y_i, period_i, asset_i = batch
        preds = model(X_i, asset_i if model.embedding is not None else None)

        _, period_pos = torch.unique(period_i, return_inverse=True)
        _, asset_pos = torch.unique(asset_i, return_inverse=True)
        n_periods = int(period_pos.max()) + 1
        n_assets = int(asset_pos.max()) + 1
        flat = period_pos * n_assets + asset_pos

        grid_pred = torch.zeros(
            n_periods * n_assets, dtype=preds.dtype, device=preds.device
        ).scatter(0, flat, preds.reshape(-1)).view(n_periods, n_assets)

        grid_true = torch.full(
            (n_periods * n_assets,), float("nan"), dtype=y_i.dtype, device=y_i.device
        )
        grid_true[flat] = y_i.reshape(-1)
        grid_true = grid_true.view(n_periods, n_assets)

        if self.signal_modifier is not None:
            observed = torch.isfinite(grid_true)
            grid_pred = self.signal_modifier(grid_pred)
            grid_pred = torch.where(observed, grid_pred, torch.zeros_like(grid_pred))

        return grid_pred, grid_true

    def _make_loader(self, X, y, scaler, y_scaler, asset_codes, period_codes, rows,
                     shuffle, seed):
        """Build a DataLoader over a subset of rows, driven by `PanelBatchSampler`."""
        X_s = scaler.transform(X.iloc[rows])
        y_s = y.iloc[rows].to_numpy().reshape(-1, 1)
        if y_scaler is not None:
            y_s = y_scaler.transform(y_s)

        dataset = torch.utils.data.TensorDataset(
            torch.Tensor(X_s),
            torch.Tensor(y_s).reshape(-1),
            torch.as_tensor(period_codes[rows], dtype=torch.long),
            torch.as_tensor(asset_codes[rows], dtype=torch.long),
        )
        sampler = PanelBatchSampler(
            period_index=period_codes[rows],
            asset_index=asset_codes[rows],
            mode=self.batch_mode,
            batch_periods=self.batch_periods,
            batch_assets=self.batch_assets,
            asset_fraction=self.asset_fraction,
            draws_per_block=self.draws_per_block,
            shuffle=shuffle,
            seed=seed,
        )
        return torch.utils.data.DataLoader(dataset=dataset, batch_sampler=sampler)

    # --------------------------------------------------------------- prediction

    def predict(self, X):
        """
        Forecast every observation.

        Parameters
        ----------
        X : pd.DataFrame
            Features, indexed by (security, date).

        Returns
        -------
        pd.Series
            One forecast per row, averaged across the fitted networks.
        """
        self._check_predict_params(X)

        asset_codes = torch.as_tensor(
            pd.Index(self.assets_).get_indexer(X.index.get_level_values(0)),
            dtype=torch.long,
        )
        # A security unseen in training has no embedding; fall back to the pooled model
        asset_codes = asset_codes.clamp(min=0)

        outputs = []
        with torch.no_grad():
            for model, scaler in zip(self.models_, self.scalers_):
                model.eval()
                X_s = torch.Tensor(scaler.transform(X))
                preds = model(X_s, asset_codes if model.embedding is not None else None)
                outputs.append(preds.reshape(-1).numpy())

        return pd.Series(np.mean(np.stack(outputs, axis=0), axis=0), index=X.index)

    # ------------------------------------------------------------------- checks

    def _create_splits(self, X, y, splitter):
        """Positional train/validation row indices, one pair per fold."""
        if isinstance(splitter, BaseCrossValidator):
            return list(splitter.split(X, y))

        dates = np.sort(X.index.get_level_values(1).unique().to_numpy())
        cut = int(splitter * len(dates))
        if cut == 0 or cut == len(dates):
            raise ValueError(
                "train_splitter produces an empty training or validation split over the "
                f"{len(dates)} dates present."
            )
        is_train = np.isin(X.index.get_level_values(1).to_numpy(), dates[:cut])
        return [(np.flatnonzero(is_train), np.flatnonzero(~is_train))]

    def _make_optimizer(self, model):
        if self.optimizer == "AdamW":
            return torch.optim.AdamW(
                model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
            )
        if self.optimizer == "Adam":
            return torch.optim.Adam(
                model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
            )
        if self.optimizer == "SGD":
            return torch.optim.SGD(
                model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
            )
        if self.optimizer == "SGD+mom":
            return torch.optim.SGD(
                model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                momentum=0.9,
            )
        raise ValueError(f"Unsupported optimizer: {self.optimizer}")

    def _check_fit_params(self, X, y):
        self._check_predict_params(X)

        if not isinstance(y, (pd.Series, pd.DataFrame)):
            raise TypeError("y must be a pandas Series or DataFrame.")
        if isinstance(y, pd.DataFrame) and y.shape[1] != 1:
            raise ValueError(
                "y must hold a single target: an observation is one security in one "
                "period, so it has one forward return."
            )
        if not X.index.equals(y.index):
            raise ValueError("X and y must have the same multi-index.")
        if not isinstance(self.batch_mode, str):
            raise TypeError("batch_mode must be a string.")
        if self.batch_mode not in {"period", "asset", "block", "mc"}:
            raise ValueError(
                "batch_mode must be one of 'period', 'asset', 'block' or 'mc'."
            )
        if self.batch_mode in ("asset", "block") and self.batch_assets is None:
            raise ValueError(
                "batch_assets must be set when batch_mode is 'asset' or 'block'."
            )
        if self.batch_mode == "mc" and (self.batch_assets is None) == (
            self.asset_fraction is None
        ):
            raise ValueError(
                "batch_mode 'mc' needs exactly one of batch_assets or asset_fraction."
            )
        if not isinstance(self.reg_embedding, numbers.Real) or self.reg_embedding < 0:
            raise ValueError("reg_embedding must be a non-negative real number.")
        if self.reg_embedding > 0 and self.embedding_dim is None:
            raise ValueError("reg_embedding has no effect unless embedding_dim is set.")

    def _check_predict_params(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame.")
        if not isinstance(X.index, pd.MultiIndex):
            raise ValueError("X must be multi-indexed.")
        if X.index.get_level_values(0).dtype != "object":
            raise TypeError("The outer index of X must be strings.")
        if X.index.get_level_values(1).dtype != "datetime64[ns]":
            raise TypeError("The inner index of X must be datetime.")
        if not X.apply(lambda col: pd.api.types.is_numeric_dtype(col)).all():
            raise TypeError("All columns in X must be numeric.")
        if X.isnull().values.any():
            raise ValueError("X must not contain missing values.")
