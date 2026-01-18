import torch 
import torch.nn as nn 

class MultiOutputMCR(nn.Module):
    """
    Negative mean-concentration risk ratio loss for multi-output regression problems.

    Notes
    -----
    By mean-concentration risk ratio, we refer to the ratio of the mean return within a time
    period, to the standard deviation of returns within that time period. This differs 
    from a Sharpe ratio in that the Sharpe is a temporal quantity, whereas this 
    statistic is cross-sectional. Maximisation of such a statistic would encourage
    positive returns at each time period whilst penalising diversity in the cross-sectional
    return distribution. The goal is to encourage prevent the model from concentrating
    returns in a small subset of the outputs.

    This statistic can be calculated for each sample in a batch, and then averaged over
    the batch. Neural networks are most naturally formulated as minimization problems, so
    the negative mean-concentration risk ratio is used as a loss function. 
    """
    def __init__(self, skip_validation = True, unbiased = True):
        super().__init__()
        
        # Checks
        if not isinstance(skip_validation, bool):
            raise TypeError("skip_validation must be a boolean")
        if not isinstance(unbiased, bool):
            raise TypeError("unbiased must be a boolean")
        
        # Attributes
        self.skip_validation = skip_validation
        self.unbiased = unbiased

    def forward(self, y_pred, y_true):
        """
        Evaluate batch negative mean-concentration risk ratio loss.

        Parameters
        ----------
        y_pred : torch.Tensor
            Predicted outputs (signals or portfolio weights).
        y_true : torch.Tensor
            True outputs (returns).
        """
        if not self.skip_validation:
            self._check_forward_params(y_pred, y_true)

        returns = y_true * y_pred
        mean_returns = torch.mean(returns, axis = 1)
        std_returns = torch.std(returns, axis = 1)

        return - torch.mean(mean_returns / std_returns)
    
    def _check_forward_params(self, y_pred, y_true):
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