import numbers

import torch
import torch.nn as nn

from sklearn.base import BaseEstimator

from macrosynergy.learning.forecasting.torch.models.mlps import MultiLayerPerceptron


class HeteroskedasticMLP(nn.Module, BaseEstimator):
    r"""
    Multi-layer perceptron predicting both a conditional mean and a conditional variance.

    Parameters
    ----------
    n_inputs : int
        Number of input features.
    n_latent : Union[int, list[int]]
        Number of units in the hidden layer(s).
    n_signals : int
        Number of assets. The network emits two numbers per asset, so its output width is
        `2 * n_signals`.
    encoder_activation : str, optional
        Activation for the encoder. Default is "tanh".
    fit_encoder_intercept : bool, optional
        Whether to fit intercepts in the encoder. Default is True.
    fit_head_intercept : bool, optional
        Whether to fit intercepts in the two heads. Default is True.
    dropout_p : float or list, optional
        Dropout probability for the encoder. Default is 0.
    normalization : str, optional
        "layer", "batch", or None. Default is None.
    min_log_var : float, optional
        Lower clamp on the predicted log variance. Default is -10.
    max_log_var : float, optional
        Upper clamp on the predicted log variance. Default is 10.

    Notes
    -----
    An ordinary regression assumes the residual variance is the same everywhere and
    estimates one number for it. This network instead makes the variance a function of
    the same features that drive the mean:

    .. code-block:: none

        z          = encoder(x)
        mu(x)      = W_mu z + b_mu
        log s2(x)  = W_v  z + b_v

    so each asset gets a conditional mean *and* a conditional variance at every date. In
    familiar terms this is a heteroskedastic regression whose variance equation shares
    its design matrix with the mean equation, fitted jointly by maximum likelihood — see
    `GaussianNLL`.

    The output is one tensor of width `2 * n_signals`, the means first and the log
    variances second, so that it satisfies the single-tensor contract that the loss
    functions and the training loop expect. `split` and `precision_weighted` recover the
    pieces.

    **What the variance head is and is not.** It captures *aleatoric* uncertainty: the
    part of the target that is unpredictable given the features, which is nearly all of a
    single stock's return. It says nothing about *epistemic* uncertainty — how unsure the
    model is about its own parameters — for which the spread across ensemble members and
    dropout samples is the available estimate. The two answer different questions and a
    high aleatoric variance does not mean the mean estimate is unreliable, only that the
    target is noisy there.

    The log variance is clamped because an unclamped variance head can drive `s2` toward
    zero on points it happens to fit early in training, which sends the likelihood to
    infinity and the gradients with it.
    """

    def __init__(
        self,
        n_inputs,
        n_latent,
        n_signals,
        encoder_activation="tanh",
        fit_encoder_intercept=True,
        fit_head_intercept=True,
        dropout_p=0,
        normalization=None,
        min_log_var=-10.0,
        max_log_var=10.0,
    ):
        super().__init__()

        if not isinstance(n_signals, numbers.Integral):
            raise TypeError("n_signals must be an integer.")
        if n_signals < 1:
            raise ValueError("n_signals must be at least 1.")
        if not isinstance(min_log_var, numbers.Real) or not isinstance(max_log_var, numbers.Real):
            raise TypeError("min_log_var and max_log_var must be real numbers.")
        if min_log_var >= max_log_var:
            raise ValueError("min_log_var must be less than max_log_var.")

        self.n_inputs = n_inputs
        self.n_latent = n_latent
        self.n_signals = n_signals
        self.encoder_activation = encoder_activation
        self.fit_encoder_intercept = fit_encoder_intercept
        self.fit_head_intercept = fit_head_intercept
        self.dropout_p = dropout_p
        self.normalization = normalization
        self.min_log_var = min_log_var
        self.max_log_var = max_log_var

        # The encoder is reused verbatim, so the two networks differ only in their heads
        backbone = MultiLayerPerceptron(
            n_inputs=n_inputs,
            n_latent=n_latent,
            n_outputs=n_signals,
            encoder_activation=encoder_activation,
            fit_encoder_intercept=fit_encoder_intercept,
            fit_head_intercept=fit_head_intercept,
            dropout_p=dropout_p,
            normalization=normalization,
        )
        self.encoder = backbone.encoder

        latent_width = backbone.n_latent[-1]
        self.head = nn.Linear(latent_width, n_signals, bias=fit_head_intercept)
        self.log_var_head = nn.Linear(latent_width, n_signals, bias=fit_head_intercept)

    def forward(self, x):
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor. Dimension: (batch_size, n_inputs).

        Returns
        -------
        torch.Tensor
            Dimension: (batch_size, 2 * n_signals), the conditional means followed by the
            conditional log variances.
        """
        latent = self.encoder(x)
        mu = self.head(latent)
        log_var = self.log_var_head(latent).clamp(self.min_log_var, self.max_log_var)
        return torch.cat([mu, log_var], dim=1)

    @staticmethod
    def split(output):
        """
        Separate a stacked output into its mean and log-variance halves.

        Parameters
        ----------
        output : torch.Tensor
            Dimension: (batch_size, 2 * n_signals).

        Returns
        -------
        tuple of torch.Tensor
            The conditional means and the conditional log variances.
        """
        if output.shape[1] % 2 != 0:
            raise ValueError(
                "A heteroskedastic output must have an even width, holding a mean and a "
                "log variance per asset."
            )
        n_signals = output.shape[1] // 2
        return output[:, :n_signals], output[:, n_signals:]

    @classmethod
    def precision_weighted(cls, output, power=1.0, normalize=None, eps=1e-8):
        r"""
        Convert a stacked output into a signal shrunk by its conditional uncertainty.

        Parameters
        ----------
        output : torch.Tensor
            Dimension: (batch_size, 2 * n_signals).
        power : float, optional
            Exponent on the precision. 0 leaves the means untouched, 0.5 divides by the
            conditional standard deviation, 1 by the conditional variance. Default is 1.
        normalize : str, optional
            "demean" to subtract the cross-sectional mean, "gross" to also scale to unit
            gross exposure, or None to leave the scale alone. Default is None.
        eps : float, optional
            Small value guarding the normalisation. Default is 1e-8.

        Notes
        -----
        The signal is

        .. code-block:: none

            signal_i = mu_i / (s2_i ^ power)

        which is the shrinkage the variance head exists to provide: where the target is
        predictable given the features, positions are taken at full size; where it is
        noisy, they are cut in favour of assets whose outcome the features do explain.
        At `power = 1` this is precision weighting, which is the weighting a
        mean-variance optimiser would apply under a diagonal covariance, and it is what
        generalised least squares does to observations.

        **Shrinkage must be relative, not absolute, and this is the trap to avoid.** Under
        a normalisation that pins gross exposure — `nn.Softmax`, or `LongShortModule`, or
        `ActiveWeightModule` — scaling every position down and then renormalising changes
        nothing at all. Uncertainty can only reallocate *between* assets, unless the book
        is allowed to de-gross toward a fallback. Passing `normalize=None` and letting the
        book shrink in absolute size is one option; the other, and the more useful one
        where a benchmark exists, is to apply the shrinkage to *active* weights, so that
        an uncertain model sits at benchmark weight rather than at zero.
        """
        mu, log_var = cls.split(output)
        signal = mu * torch.exp(-power * log_var)

        if normalize == "demean":
            signal = signal - signal.mean(dim=-1, keepdim=True)
        elif normalize == "gross":
            signal = signal - signal.mean(dim=-1, keepdim=True)
            signal = signal / (signal.abs().sum(dim=-1, keepdim=True) + eps)
        elif normalize is not None:
            raise ValueError("normalize must be one of 'demean', 'gross' or None.")

        return signal
