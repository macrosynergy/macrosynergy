import numbers

import torch
import torch.nn as nn

from sklearn.base import BaseEstimator

from macrosynergy.learning.forecasting.torch.models.mlps import MultiLayerPerceptron


class SharedHeadPanelMLP(nn.Module, BaseEstimator):
    r"""
    Multi-layer perceptron over (security, period) observations with a single shared head.

    Parameters
    ----------
    n_inputs : int
        Number of input features per observation.
    n_latent : Union[int, list[int]]
        Number of units in the hidden layer(s).
    n_assets : int, optional
        Number of securities, required only when `embedding_dim` is set. Default is None.
    embedding_dim : int, optional
        Width of a learned per-security embedding concatenated to the features. Default
        is None, for no embedding.
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
    embedding_init_scale : float, optional
        Standard deviation of the embedding's initialisation. Default is 0.01, which
        starts every security near the pooled model.

    Notes
    -----
    The contrast with `MultiLayerPerceptron` as used for allocation is where a security's
    identity comes from.

    There, an observation is a *period*, the securities are the output units, and
    security `i` is represented by its own row of the head matrix — a free parameter
    vector estimated from that security's history alone. With hundreds of securities and
    tens or low hundreds of periods, each of those vectors sees very few observations,
    and no security's data informs any other's. In the language of panel econometrics
    that is a fixed-effects specification with no pooling.

    Here, an observation is a *(security, period) pair*, and the network maps its feature
    vector to that security's forecast through one shared set of parameters:

    .. code-block:: none

        x_{i,t} = [ macro state at t , characteristics of security i at t ]
        yhat_{i,t} = head( encoder( x_{i,t} ) )

    Two securities differ in their forecasts, and in their sensitivity to the macro state,
    exactly to the extent that their characteristics differ. A hidden layer applied to the
    concatenated vector forms products of macro and characteristic inputs, so the
    derivative of the forecast with respect to the macro state is itself a function of the
    security's characteristics — a conditional beta, rather than a free one. Every
    parameter is then estimated from all `n_securities x n_periods` observations.

    **The corollary is the constraint to design around.** If the features are macro only,
    every security shares the same input at date `t` and therefore receives the same
    forecast, and the cross-sectional dispersion of the predictions is identically zero.
    A shared head is inert without per-security features; supplying them is not an
    enhancement but the precondition.

    `embedding_dim` restores a controlled amount of per-security freedom: a learned vector
    of `embedding_dim` numbers per security, concatenated to the features. Initialised
    near zero and shrunk by weight decay, this is partial pooling — the random-effects
    rung between complete pooling and the free per-security vectors of the multi-output
    design. At `embedding_dim = 4` it costs four parameters per security rather than
    seventeen, and unlike those seventeen it is explicitly regularised toward the pooled
    model.
    """

    def __init__(
        self,
        n_inputs,
        n_latent,
        n_assets=None,
        embedding_dim=None,
        encoder_activation="relu",
        head_activation="identity",
        fit_encoder_intercept=True,
        fit_head_intercept=True,
        dropout_p=0,
        normalization=None,
        embedding_init_scale=0.01,
    ):
        super().__init__()

        if embedding_dim is not None:
            if not isinstance(embedding_dim, numbers.Integral):
                raise TypeError("embedding_dim must be an integer.")
            if embedding_dim < 1:
                raise ValueError("embedding_dim must be at least 1.")
            if n_assets is None:
                raise ValueError("n_assets must be provided when embedding_dim is set.")
            if not isinstance(n_assets, numbers.Integral):
                raise TypeError("n_assets must be an integer.")
            if n_assets < 1:
                raise ValueError("n_assets must be at least 1.")
        if not isinstance(embedding_init_scale, numbers.Real):
            raise TypeError("embedding_init_scale must be a real number.")
        if embedding_init_scale < 0:
            raise ValueError("embedding_init_scale must be non-negative.")

        self.n_inputs = n_inputs
        self.n_latent = n_latent
        self.n_assets = n_assets
        self.embedding_dim = embedding_dim
        self.encoder_activation = encoder_activation
        self.head_activation = head_activation
        self.fit_encoder_intercept = fit_encoder_intercept
        self.fit_head_intercept = fit_head_intercept
        self.dropout_p = dropout_p
        self.normalization = normalization
        self.embedding_init_scale = embedding_init_scale

        if embedding_dim is not None:
            self.embedding = nn.Embedding(n_assets, embedding_dim)
            # Start every security at the pooled model, so that the fit has to earn any
            # per-security deviation rather than beginning with an arbitrary one
            nn.init.normal_(self.embedding.weight, mean=0.0, std=embedding_init_scale)
            width = n_inputs + embedding_dim
        else:
            self.embedding = None
            width = n_inputs

        backbone = MultiLayerPerceptron(
            n_inputs=width,
            n_latent=n_latent,
            n_outputs=1,
            encoder_activation=encoder_activation,
            head_activation=head_activation,
            fit_encoder_intercept=fit_encoder_intercept,
            fit_head_intercept=fit_head_intercept,
            dropout_p=dropout_p,
            normalization=normalization,
        )
        self.encoder = backbone.encoder
        self.head = backbone.head

    def forward(self, x, asset_codes=None):
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Features of each observation. Dimension: (n_observations, n_inputs).
        asset_codes : torch.Tensor, optional
            Integer security code of each observation, required when an embedding is in
            use. Dimension: (n_observations,).

        Returns
        -------
        torch.Tensor
            One forecast per observation. Dimension: (n_observations, 1).
        """
        if self.embedding is not None:
            if asset_codes is None:
                raise ValueError(
                    "asset_codes must be provided when the network has an embedding."
                )
            x = torch.cat([x, self.embedding(asset_codes)], dim=1)

        return self.head(self.encoder(x))

    def embedding_penalty(self):
        """
        Sum of squared embedding entries, for shrinking securities toward the pooled model.

        Returns
        -------
        torch.Tensor
            A scalar, zero when no embedding is in use.

        Notes
        -----
        Applied as an explicit penalty rather than left to the optimiser's `weight_decay`,
        so that the strength of the pooling can be set independently of the regularisation
        on the shared parameters. Those are two different decisions: how much to smooth
        the common mapping, and how much to let individual securities depart from it.
        """
        if self.embedding is None:
            return torch.zeros(())
        return (self.embedding.weight**2).sum()
