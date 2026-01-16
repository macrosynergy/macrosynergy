from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, RegressorMixin
from copy import deepcopy

from macrosynergy.learning.forecasting.torch import MultiLayerPerceptron, MLPTrainer
import torch


class MLPRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        # Neural network structure hyperparameters
        n_latent,
        # Neural network training hyperparameters
        weight_decay=1e-4,
        reg_turnover=0,
        batch_size=16,
        learning_rate=3e-4,
        use_ts_sampler=True,
        fit_intercept=True,
        activation="tanh",
        epochs=10000,
        patience=1000,
        train_pct=0.7,
        # Other stuff
        verbose=False,
        random_state=42,
    ):
        # assert economic_objective in ["sharpe", "sortino", "meanvariance", "meanvarianceskew", "crra"]

        self.n_latent = n_latent
        self.weight_decay = weight_decay
        self.reg_turnover = reg_turnover
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.use_ts_sampler = use_ts_sampler
        self.fit_intercept = fit_intercept
        self.activation = activation
        self.epochs = epochs
        self.patience = patience
        self.train_pct = train_pct
        # self.long_only = long_only
        self.verbose = verbose
        self.random_state = random_state

        self.model = None

    def fit(self, X, y):
        torch.manual_seed(self.random_state)

        # Initialize model
        self.model = MultiLayerPerceptron(
            n_inputs=X.shape[1],
            n_latent=self.n_latent,
            n_outputs=y.shape[1],
            fit_intercept=self.fit_intercept,
            activation=self.activation,
        )

        trainer = MLPTrainer(
            train_pct=self.train_pct,
            batch_size=self.batch_size,
            use_ts_sampler=self.use_ts_sampler,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            epochs=self.epochs,
            reg_turnover=self.reg_turnover,
            patience=self.patience,
            random_state=self.random_state,
            verbose=self.verbose,
            x_scaler=StandardScaler(with_mean=False),
            y_scaler=StandardScaler(with_mean=False),
        )

        self.model, self.x_scaler, self.y_scaler = trainer.fit(self.model, X, y)

        return self

    def predict(self, X):
        # Scale data
        X_scaled = self.X_scaler.transform(X)

        # Switch to evaluation mode
        self.model.eval()
        with torch.no_grad():
            # Convert to torch and pass through network
            X_scaled_torch = torch.Tensor(X_scaled)
            preds = self.model(X_scaled_torch).numpy()

            # Convert back to original
            preds = self.y_scaler.inverse_transform(preds)

        return preds
