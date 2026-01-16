import torch 
import torch.nn as nn

class MultiOutputSharpe(nn.Module):
    """
    Negative Sharpe ratio loss for multi-output regression problems.

    Notes
    -----
    When a neural network is designed so that the output can be interpreted as signals
    or portfolio weights for each output, a stylized Sharpe ratio can be calculated 
    by multiplying the true returns by the respective signals or weights, before
    downsampling to portfolio returns. The Sharpe ratio, excluding trading frictions such
    as transaction costs, can be calculated over the batch. 

    Neural networks are most naturally formulated as minimization problems, so the 
    negative Sharpe ratio is used as a loss function. 
    """
    def __init__(self, skip_validation = True):
        super().__init__()
        self.skip_validation = skip_validation

    def forward(self, y_true, y_pred):
        """
        Return :math:`- \hat{E_t}[sharpe] = 1/T sum[sharpe_t]`.
        """
        if not self.skip_validation:
            self._check_forward_params(y_true, y_pred)

        returns = y_true * y_pred
        portfolio_returns = torch.sum(returns, axis = 1)
        
        return - torch.mean(portfolio_returns)/torch.std(portfolio_returns)
    
    def _check_forward_params(self, y_true, y_pred):
        """ Check parameters for forward method """
        if not isinstance(y_true, torch.Tensor):
            raise TypeError("y_true must be a torch.Tensor")
        if y_true.shape[1] < 2:
            raise ValueError("y_true must have at least 2 outputs (shape[1] >= 2)")
        if not isinstance(y_pred, torch.Tensor):
            raise TypeError("y_pred must be a torch.Tensor")
        if y_pred.shape[1] < 2:
            raise ValueError("y_pred must have at least 2 outputs (shape[1] >= 2)")
        if y_true.shape != y_pred.shape:
            raise ValueError("y_true and y_pred must have the same shape")