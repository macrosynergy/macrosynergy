import torch
import torch.nn as nn

import numbers

class ConstrainedLongOnlyModule(nn.Module):
    """
    Converts neural network scores for a collection of assets to long-only portfolio
    weights that respects a concentration bound.

    Parameters
    ----------
    concentration_bound : float
        The maximum allowed weight for any single asset in the portfolio. Acceptable 
        values are between 0 and 1. Default is 0.2.

    Notes
    -----
    Whilst this can be used as a standalone layer, this has been designed to be used as
    the final layer of a neural network to ensure that the outputs can be interpreted as 
    fractions of capital allocated to a collection of assets.
    """
    def __init__(self, concentration_bound = 0.2):
        super().__init__()

        # Checks 
        if not isinstance(concentration_bound, numbers.Real):
            raise TypeError("concentration_bound must be a real number.")

        if not (0 < concentration_bound < 1):
            raise ValueError("concentration_bound must be between 0 and 1.")

        # Attributes
        self.concentration_bound = concentration_bound

    def forward(self, x):
        n_assets = x.size(-1)

        # Check that the concentration bound works with the number of assets
        if self.concentration_bound * n_assets <= 1:
            raise ValueError(
                "concentration_bound is too small for the number of assets: "
                "n_assets * concentration_bound must be greater than 1."
            )
        
        alpha = (1 - self.concentration_bound) / (self.concentration_bound * n_assets - 1)

        phi = alpha + torch.sigmoid(x)

        return phi / phi.sum(dim=-1, keepdim=True)