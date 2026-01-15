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
    def forward(self, y_true, y_pred):
        """
        Return :math:`- \hat{E_t}[sharpe] = 1/T sum[sharpe_t]`.
        """
        returns = y_true * y_pred
        portfolio_returns = torch.sum(returns, axis = 1)
        
        return - torch.mean(portfolio_returns)/torch.std(portfolio_returns)