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
    def forward(self, y_true, y_pred):
        returns = y_true * y_pred
        mean_returns = torch.mean(returns, axis = 1)
        std_returns = torch.std(returns, axis = 1)

        return - torch.mean(mean_returns / std_returns)