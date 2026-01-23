from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, RegressorMixin
from copy import deepcopy

from macrosynergy.learning.forecasting.torch import MultiLayerPerceptron, MLPTrainer
import torch


class MLPRegressor(BaseEstimator, RegressorMixin):
    """
    Scikit-learn compatible multi-layer perceptron (MLP) regressor implemented in PyTorch.

    This estimator wraps :class:`macrosynergy.learning.forecasting.torch.MultiLayerPerceptron`
    and trains it via :class:`macrosynergy.learning.forecasting.torch.MLPTrainer`, including
    optional scaling of inputs and targets using :class:`sklearn.preprocessing.StandardScaler`.

    Parameters
    ----------
    n_latent : int
        Number of hidden units in the latent layer of the MLP.
    loss_func : torch.nn.Module, optional
        Loss function used during training. Default is ``nn.MSELoss()``.
    weight_decay : float, optional
        L2 regularization strength applied via the optimizer. Default is 1e-4.
    reg_turnover : float, optional
        Additional turnover regularization penalty applied by the trainer. Default is 0.
    batch_size : int, optional
        Batch size used during training. Default is 16.
    learning_rate : float, optional
        Learning rate used by the optimizer. Default is 3e-4.
    use_ts_sampler : bool, optional
        Whether to use time-series batch sampling during training. Default is True.
    encoder_activation : str, optional
        Activation function for the encoder (hidden) component of the network.
        Default is "tanh".
    head_activation : str, optional
        Activation function for the head (output) component of the network.
        Default is "identity".
    fit_encoder_intercept : bool, optional
        Whether to include an intercept (bias term) in the encoder layers.
        Default is False.
    fit_head_intercept : bool, optional
        Whether to include an intercept (bias term) in the output layer.
        Default is True.
    epochs : int, optional
        Maximum number of training epochs. Default is 10000.
    patience : int, optional
        Number of epochs to wait for improvement before early stopping. Default is 1000.
    train_pct : float, optional
        Fraction of samples used for training (remainder used for validation). Default is 0.7.
    verbose : bool, optional
        Whether to print training diagnostics. Default is False.
    random_state : int, optional
        Random seed used for PyTorch initialization and training. Default is 42.
    inverse_transform_preds : bool, optional
        Whether to inverse-transform predictions back to the original target scale using
        the fitted target scaler. Default is False.
    """

    def __init__(
        self,
        # Neural network structure hyperparameters
        n_latent,
        # Neural network training hyperparameters
        loss_func=torch.nn.MSELoss(),
        weight_decay=1e-4,
        reg_turnover=0,
        batch_size=16,
        learning_rate=3e-4,
        use_ts_sampler=True,
        encoder_activation = "tanh",
        head_activation = "identity",
        fit_encoder_intercept = False,
        fit_head_intercept = True,
        epochs=10000,
        patience=1000,
        train_pct=0.7,
        # Other stuff
        verbose=False,
        random_state=42,
        inverse_transform_preds=False
    ):
        self.n_latent = n_latent
        self.loss_func = loss_func
        self.weight_decay = weight_decay
        self.reg_turnover = reg_turnover
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.use_ts_sampler = use_ts_sampler
        self.encoder_activation = encoder_activation
        self.head_activation = head_activation
        self.fit_encoder_intercept = fit_encoder_intercept
        self.fit_head_intercept = fit_head_intercept
        self.epochs = epochs
        self.patience = patience
        self.train_pct = train_pct
        # self.long_only = long_only
        self.verbose = verbose
        self.random_state = random_state
        self.inverse_transform_preds = inverse_transform_preds

        self.model = None

    def fit(self, X, y):
        torch.manual_seed(self.random_state)

        # Initialize model
        self.model = MultiLayerPerceptron(
            n_inputs=X.shape[1],
            n_latent=self.n_latent,
            n_outputs=y.shape[1],
            encoder_activation=self.encoder_activation,
            head_activation=self.head_activation,
            fit_encoder_intercept=self.fit_encoder_intercept,
            fit_head_intercept=self.fit_head_intercept,
        )

        trainer = MLPTrainer(
            train_pct=self.train_pct,
            batch_size=self.batch_size,
            use_ts_sampler=self.use_ts_sampler,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            epochs=self.epochs,
            loss_fn=self.loss_func,
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
        X_scaled = self.x_scaler.transform(X)

        # Switch to evaluation mode
        self.model.eval()
        with torch.no_grad():
            # Convert to torch and pass through network
            X_scaled_torch = torch.Tensor(X_scaled)
            preds = self.model(X_scaled_torch).numpy()

            # Convert back to original
            if self.inverse_transform_preds:
                preds = self.y_scaler.inverse_transform(preds)

        return preds
