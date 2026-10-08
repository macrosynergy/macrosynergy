import torch
import torch.nn as nn

class LongShortModule(nn.Module):
    """
    Normalizes a neural network layer so that the sum of absolute values equals one.

    Parameters
    ----------
    dollar_neutral : bool, default=False
        If True, the layer is first demeaned to ensure that the sum of the outputs equals
        zero. 
    
    Notes
    -----
    Whilst this can be used as a standalone layer, this has been designed to be used as
    the final layer of a neural network to ensure that the outputs can be interpreted as 
    fractions of capital allocated to a collection of assets, with allowance for both
    long and short positions.
    
    If dollar_neutral is set to True, equal capital is allocated to long and short
    positions. 
    """
    def __init__(self, dollar_neutral = False):
        super().__init__()

        if not isinstance(dollar_neutral, bool):
            raise TypeError("dollar_neutral must be a boolean value.")
            
        self.dollar_neutral = dollar_neutral
        
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
            
        x = x / (x.abs().sum(dim=-1, keepdim=True))
        
        return x