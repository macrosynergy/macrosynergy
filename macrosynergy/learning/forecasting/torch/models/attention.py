import numbers

import torch
import torch.nn as nn


class LagAttentionNet(nn.Module):
    def __init__(self, n_inputs, n_outputs=1, d_model=16, dropout_p=0.2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_inputs, d_model),
            nn.Tanh(),
            nn.Dropout(dropout_p),
        )
        self.score = nn.Linear(d_model, 1)
        self.head = nn.Linear(d_model, n_outputs)

    def forward(self, x, return_attention=False):
        # x: [batch, lookback, n_inputs]
        h = self.encoder(x)
        w = torch.softmax(self.score(h).squeeze(-1), dim=1)
        z = torch.sum(h * w.unsqueeze(-1), dim=1)
        y = self.head(z)

        if return_attention:
            return y, w
        return y

    
# class MacroAttentionNet(nn.Module):
#     """
#     Small Transformer encoder for macro panel sequences.

#     Parameters
#     ----------
#     n_inputs : int
#         Number of input features per time step.
#     n_outputs : int
#         Number of outputs. Usually 1.
#     d_model : int
#         Transformer embedding dimension.
#     n_heads : int
#         Number of attention heads.
#     n_layers : int
#         Number of Transformer encoder layers.
#     dim_feedforward : int | None
#         Feedforward dimension. Defaults to 4 * d_model.
#     dropout_p : float
#         Dropout probability.
#     max_seq_len : int
#         Maximum lookback length.
#     head_activation : str
#         Output activation: "identity", "tanh", "relu", or "sigmoid".
#     """

#     def __init__(
#         self,
#         n_inputs,
#         n_outputs=1,
#         d_model=32,
#         n_heads=2,
#         n_layers=1,
#         dim_feedforward=None,
#         dropout_p=0.1,
#         max_seq_len=120,
#         head_activation="identity",
#     ):
#         super().__init__()

#         self._check_init_params(
#             n_inputs=n_inputs,
#             n_outputs=n_outputs,
#             d_model=d_model,
#             n_heads=n_heads,
#             n_layers=n_layers,
#             dim_feedforward=dim_feedforward,
#             dropout_p=dropout_p,
#             max_seq_len=max_seq_len,
#             head_activation=head_activation,
#         )

#         self.n_inputs = n_inputs
#         self.n_outputs = n_outputs
#         self.d_model = d_model
#         self.n_heads = n_heads
#         self.n_layers = n_layers
#         self.dim_feedforward = dim_feedforward or 4 * d_model
#         self.dropout_p = dropout_p
#         self.max_seq_len = max_seq_len
#         self.head_activation = head_activation

#         self.activation_map = {
#             "identity": lambda: nn.Identity(),
#             "tanh": lambda: nn.Tanh(),
#             "relu": lambda: nn.ReLU(inplace=True),
#             "sigmoid": lambda: nn.Sigmoid(),
#         }

#         self.input_projection = nn.Linear(n_inputs, d_model)

#         self.position_embedding = nn.Parameter(
#             torch.randn(1, max_seq_len, d_model) * 0.01
#         )

#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=d_model,
#             nhead=n_heads,
#             dim_feedforward=self.dim_feedforward,
#             dropout=dropout_p,
#             activation="gelu",
#             batch_first=True,
#             norm_first=True,
#         )

#         self.encoder = nn.TransformerEncoder(
#             encoder_layer=encoder_layer,
#             num_layers=n_layers,
#         )

#         self.attention_pool = nn.Sequential(
#             nn.Linear(d_model, d_model),
#             nn.Tanh(),
#             nn.Linear(d_model, 1),
#         )

#         self.head = nn.Sequential(
#             nn.LayerNorm(d_model),
#             nn.Linear(d_model, n_outputs),
#             self.activation_map[head_activation](),
#         )

#     def forward(self, x, padding_mask=None, return_attention=False):
#         """
#         Parameters
#         ----------
#         x : torch.Tensor
#             Shape: (batch_size, seq_len, n_inputs).
#         padding_mask : torch.Tensor | None
#             Boolean tensor of shape (batch_size, seq_len). True means padded/missing.
#         return_attention : bool
#             Whether to return temporal attention weights.

#         Returns
#         -------
#         torch.Tensor | tuple[torch.Tensor, torch.Tensor]
#             Predictions of shape (batch_size, n_outputs), optionally with attention
#             weights of shape (batch_size, seq_len).
#         """
#         if x.ndim != 3:
#             raise ValueError("x must have shape (batch_size, seq_len, n_inputs).")

#         batch_size, seq_len, n_inputs = x.shape

#         if n_inputs != self.n_inputs:
#             raise ValueError(
#                 f"Expected {self.n_inputs} input features, received {n_inputs}."
#             )

#         if seq_len > self.max_seq_len:
#             raise ValueError(
#                 f"Sequence length {seq_len} exceeds max_seq_len={self.max_seq_len}."
#             )

#         h = self.input_projection(x)
#         h = h + self.position_embedding[:, :seq_len, :]

#         h = self.encoder(h, src_key_padding_mask=padding_mask)

#         scores = self.attention_pool(h).squeeze(-1)

#         if padding_mask is not None:
#             scores = scores.masked_fill(padding_mask, float("-inf"))

#         weights = torch.softmax(scores, dim=1)
#         pooled = torch.sum(h * weights.unsqueeze(-1), dim=1)

#         output = self.head(pooled)

#         if return_attention:
#             return output, weights

#         return output

#     def _check_init_params(
#         self,
#         n_inputs,
#         n_outputs,
#         d_model,
#         n_heads,
#         n_layers,
#         dim_feedforward,
#         dropout_p,
#         max_seq_len,
#         head_activation,
#     ):
#         for name, value in {
#             "n_inputs": n_inputs,
#             "n_outputs": n_outputs,
#             "d_model": d_model,
#             "n_heads": n_heads,
#             "n_layers": n_layers,
#             "max_seq_len": max_seq_len,
#         }.items():
#             if not isinstance(value, numbers.Integral):
#                 raise TypeError(f"{name} must be an integer.")
#             if value < 1:
#                 raise ValueError(f"{name} must be at least 1.")

#         if d_model % n_heads != 0:
#             raise ValueError("d_model must be divisible by n_heads.")

#         if dim_feedforward is not None:
#             if not isinstance(dim_feedforward, numbers.Integral):
#                 raise TypeError("dim_feedforward must be an integer or None.")
#             if dim_feedforward < 1:
#                 raise ValueError("dim_feedforward must be at least 1.")

#         if not isinstance(dropout_p, numbers.Real):
#             raise TypeError("dropout_p must be numeric.")
#         if dropout_p < 0 or dropout_p > 0.5:
#             raise ValueError("dropout_p must be between 0 and 0.5.")

#         if head_activation not in {"identity", "tanh", "relu", "sigmoid"}:
#             raise ValueError(
#                 "head_activation must be one of 'identity', 'tanh', 'relu', or 'sigmoid'."
#             )