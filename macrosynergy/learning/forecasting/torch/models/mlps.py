import torch
import torch.nn as nn

import numbers

class MultiLayerPerceptron(nn.Module):
    """
    Multi-layer perceptron models in PyTorch.

    Parameters
    ----------
    n_inputs : int
        Number of input features. Must be at least 1.
    n_latent : Union[int, list[int]]
        Number of latent features in a single hidden layer or list specifying the size of
        each hidden layer.
    n_outputs : int
        Number of output variables. Must be at least 1.
    encoder_activation : Union[str, nn.Module], optional
        Activation function for the encoder layers.
        Default is "tanh". String options include "relu" and "sigmoid".
        Alternatively, a custom PyTorch activation module can be provided.
    head_activation : str, optional
        Activation function for the head layers.
        Default is "identity" for no activation. String options include "tanh", "relu"
        and "sigmoid". Alternatively, a custom PyTorch activation module can be provided.
    fit_encoder_intercept : bool, optional
        Whether to fit intercepts in the encoder layers. Default is False.
    fit_head_intercept : bool, optional
        Whether to fit intercepts in the output head. Default is True.
    """
    def __init__(
        self,
        n_inputs,
        n_latent,
        n_outputs,
        encoder_activation = "tanh",
        head_activation = "identity",
        fit_encoder_intercept = False,
        fit_head_intercept = True,
    ):
        super().__init__()

        # Checks
        self._check_init_params(
            n_inputs,
            n_latent,
            n_outputs,
            encoder_activation,
            head_activation,
            fit_encoder_intercept,
            fit_head_intercept,
        )

        # Attributes
        self.n_inputs = n_inputs
        self.n_latent = list(n_latent)
        self.n_outputs = n_outputs
        self.encoder_activation = encoder_activation
        self.head_activation = head_activation
        self.fit_encoder_intercept = fit_encoder_intercept
        self.fit_head_intercept = fit_head_intercept

        self.activation_map = {
            "tanh": lambda: nn.Tanh(),
            "relu": lambda: nn.ReLU(inplace=True),
            "sigmoid": lambda: nn.Sigmoid(),
            "identity": lambda: nn.Identity(),
        }

        # Encoder
        self.encoder = self._build_encoder(self.n_inputs, self.n_latent, self.encoder_activation, self.fit_encoder_intercept)

        # Projection head
        self.head = self._build_head(self.n_latent[-1], self.n_outputs, self.head_activation, self.fit_head_intercept)

    def forward(self, x):
        """
        Forward pass through the network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, n_inputs).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, n_outputs).
        """
        latent = self.encoder(x)
        output = self.head(latent)
        return output

    def _build_encoder(self, n_inputs, n_latent, encoder_activation, fit_encoder_intercept):
        raise NotImplementedError("Subclasses must implement _build_encoder method.")
    
    def _build_head(self, n_latent, n_outputs, head_activation, fit_head_intercept):
        raise NotImplementedError("Subclasses must implement _build_head method.")

    def _check_init_params(
        self,
        n_inputs,
        n_latent,
        n_outputs,
        encoder_activation,
        head_activation,
        fit_encoder_intercept,
        fit_head_intercept,
    ):
        # n_inputs
        if not isinstance(n_inputs, numbers.Integral):
            raise TypeError("n_inputs must be an integer.")
        if n_inputs < 1:
            raise ValueError("n_inputs must be at least 1.")
        # n_latent
        if not isinstance(n_latent, numbers.Integral):
            if not isinstance(n_latent, list):
                raise TypeError("n_latent must be either an integer or a list of integers.")
            if not all(isinstance(x, numbers.Integral) for x in n_latent):
                raise TypeError("When n_latent is a list, all elements must be integers.")
            if len(n_latent) <= 1:
                raise ValueError("When n_latent is a list, it must contain more than one element.")
        else:
            if n_latent < 1:
                raise ValueError("When n_latent is an integer, it must be at least 1.")
        # n_outputs
        if not isinstance(n_outputs, numbers.Integral):
            raise TypeError("n_outputs must be an integer.")
        if n_outputs < 1:
            raise ValueError("n_outputs must be at least 1.")
        # encoder_activation
        if not (isinstance(encoder_activation, str) or isinstance(encoder_activation, nn.Module)) :
            raise TypeError("encoder_activation must be a string or PyTorch-compatible activation function.")
        if isinstance(encoder_activation, str):
            if encoder_activation not in {"tanh", "relu", "sigmoid"}:
                raise ValueError(
                    "encoder_activation must be one of 'tanh', 'relu', or 'sigmoid'."
                )
        else:
            if not hasattr(encoder_activation, "forward"):
                raise ValueError("encoder_activation must be a valid PyTorch activation module.")
        # head_activation
        if not (isinstance(head_activation, str) or isinstance(head_activation, nn.Module)):
            raise TypeError("head_activation must be a string or PyTorch-compatible activation function.")
        if isinstance(head_activation, str):
            if head_activation not in {"tanh", "relu", "sigmoid", "identity"}:
                raise ValueError(
                    "head_activation must be one of 'tanh', 'relu', 'sigmoid', or 'identity'."
                )
        else:
            if not hasattr(head_activation, "forward"):
                raise ValueError("head_activation must be a valid PyTorch activation module.")
        # fit_encoder_intercept
        if not isinstance(fit_encoder_intercept, bool):
            raise TypeError("fit_encoder_intercept must be a boolean.")
        # fit_head_intercept
        if not isinstance(fit_head_intercept, bool):
            raise TypeError("fit_head_intercept must be a boolean.")

class MultiHeadRetNet(BaseMultiLayerPerceptron):
    """
    Multi-layer perceptron model with multiple output heads and a shared encoder. The 
    head is a linear layer without activation, suitable for return forecasting tasks, 
    ergo "RetNet".

    Parameters
    ----------
    n_inputs : int
        Number of input features. Must be at least 1.
    n_latent : Union[int, list[int]]
        Number of latent features in a single hidden layer or list specifying the size of
        each hidden layer.
    n_outputs : int
        Number of output variables. Must be at least 1.
    encoder_activation : str, optional
        Activation function for the encoder layers.
        Default is "tanh". Options include "relu" and "sigmoid".
    head_activation : str, optional
        Activation function for the head layers.
        Default is "identity" for no activation. Options include "tanh", "relu"
        and "sigmoid".
    fit_encoder_intercept : bool, optional
        Whether to fit intercepts in the encoder layers. Default is False.
    fit_head_intercept : bool, optional
        Whether to fit intercepts in the output head. Default is True.
    """
    def _build_encoder(self, n_inputs, n_latent, encoder_activation, fit_encoder_intercept):
        encoder_modules = [nn.Linear(n_inputs, n_latent[0], bias = fit_encoder_intercept), self.activation_map[encoder_activation]()]
        if len(n_latent) > 1:
            for layer_idx in range(1, len(n_latent)):
                encoder_modules.append(
                    nn.Linear(n_latent[layer_idx - 1], n_latent[layer_idx], bias = fit_encoder_intercept)
                )
                encoder_modules.append(self.activation_map[encoder_activation]())
        
        return nn.Sequential(*encoder_modules)
    
    def _build_head(self, n_latent, n_outputs, head_activation, fit_head_intercept):

        return nn.Linear(n_latent, n_outputs, bias = fit_head_intercept)
    
class MultiHeadPANet(BaseMultiLayerPerceptron):
    """
    Multi-layer perceptron model with multiple output heads and a shared encoder. The 
    head is a linear layer with either tanh or sigmoid activation to constrain outputs to
    bounded ranges that represent trading signal strengths rather than expected returns, 
    ergo "PANet".

    Parameters
    ----------
    n_inputs : int
        Number of input features. Must be at least 1.
    n_latent : Union[int, list[int]]
        Number of latent features in a single hidden layer or list specifying the size of
        each hidden layer.
    n_outputs : int
        Number of output variables. Must be at least 1.
    encoder_activation : str, optional
        Activation function for the encoder layers.
        Default is "tanh". Options include "relu" and "sigmoid".
    head_activation : str, optional
        Activation function for the head layers.
        Default is "identity" for no activation. Options include "tanh", "relu"
        and "sigmoid".
    fit_encoder_intercept : bool, optional
        Whether to fit intercepts in the encoder layers. Default is False.
    fit_head_intercept : bool, optional
        Whether to fit intercepts in the output head. Default is True.
    """
    def _build_encoder(self, n_inputs, n_latent, encoder_activation, fit_encoder_intercept):
        encoder_modules = [nn.Linear(n_inputs, n_latent[0], bias = fit_encoder_intercept), self.activation_map[encoder_activation]()]
        if len(n_latent) > 1:
            for layer_idx in range(1, len(n_latent)):
                encoder_modules.append(
                    nn.Linear(n_latent[layer_idx - 1], n_latent[layer_idx], bias = fit_encoder_intercept)
                )
                encoder_modules.append(self.activation_map[encoder_activation]())
        
        return nn.Sequential(*encoder_modules)
    
    def _build_head(self, n_latent, n_outputs, head_activation, fit_head_intercept):
        head = nn.Sequential(
            nn.Linear(n_latent, n_outputs, bias = fit_head_intercept),
            self.activation_map[head_activation]()
        )
        return head