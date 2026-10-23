import numbers

import torch
import torch.nn as nn

class LongShortModule(nn.Module):
    """
    Normalizes a neural network layer so that the sum of absolute values equals one, when 
    the raw sum of absolute values is at least eps. Otherwise, small a small value eps is
    used to prevent division by zero.

    Parameters
    ----------
    dollar_neutral : bool, default=False
        If True, the layer is first demeaned to ensure that the sum of the outputs equals
        zero.
    eps : float, default=0
        A small value to prevent division by zero when normalizing the layer output.
    
    Notes
    -----
    Whilst this can be used as a standalone layer, this has been designed to be used as
    the final layer of a neural network to ensure that the outputs can be interpreted as 
    fractions of capital allocated to a collection of assets, with allowance for both
    long and short positions.
    The eps parameter ensures numerical stability by preventing division by zero when the
    sum of absolute values is very small.
    If dollar_neutral is set to True, equal capital is allocated to long and short
    positions. 
    """
    def __init__(self, dollar_neutral = False, eps = 0):
        super().__init__()

        # Checks
        if not isinstance(dollar_neutral, bool):
            raise TypeError("dollar_neutral must be a boolean value.")
        if not isinstance(eps, numbers.Real):
            raise TypeError("eps must be a real number.")
        if eps < 0:
            raise ValueError("eps must be non-negative.")

        # Attributes
        self.dollar_neutral = dollar_neutral
        self.eps = eps
        
    def forward(self, x):
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor for the layer. Should have dimension (batch_size, n_assets)
            where n_assets is the number of assets in the portfolio.
        """
        if self.dollar_neutral:
            x = x - x.mean(axis = -1, keepdim=True)
            
        x = x / x.abs().sum(dim=-1, keepdim=True).clamp_min(self.eps)
        
        return x