import torch
import torch.nn as nn

import numbers

class MultiLayerPerceptron(nn.Module):
    r"""
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
    encoder_activation : str, optional
        Activation function for the encoder layers.
        Default is "tanh". Other options include "relu" and "sigmoid".
    head_activation : str, optional
        Activation function for the head layers.
        Default is "identity" for no activation. Other options include "tanh", "relu"
        and "sigmoid".
    fit_encoder_intercept : bool, optional
        Whether to fit intercepts in the encoder layers. Default is False.
    fit_head_intercept : bool, optional
        Whether to fit intercepts in the output head. Default is True.

    Notes
    -----
    A multi-layer perceptron is a feed-forward neural network that learns a (hopefully)
    optimal representation of the feature set for a prediction task, or for a collection
    of tasks. The intitial set is transformed into a new, "learnt", collection of features.
    This is the "first hidden layer" of the network. Each learnt feature is the composition
    of the linear combination of initial features and a non-linear activation function. 
    The choice of activation is currently "relu" (:math:`f(x) = \max(0, x)`), "tanh" 
    (:math:`f(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}`), or "sigmoid"
    (:math:`f(x) = \frac{1}{1 + e^{-x}}`). This new feature set can be further transformed
    in the same manner by creating a second hidden layer, and so on. 

    The part of the network that describes how the initial features are transformed into 
    the final features (before mapping to the outputs) is called the "encoder". The
    component that maps the final learnt features to the outputs is called the "projection head".
    When multiple outputs are being modelled, this is usually referred to as having a 
    "multi-head" architecture.

    What's the advantage of a feedforward neural network over other models on tabular 
    datasets? Structure and customizability. 32 neurons in a hidden layer means that 32 features
    are being learnt. I can shrink these features towards priors, if I have any beliefs.
    I can regularize network outputs to encourage smoothness (temporal regularization)
    and consistency with known relationships (spatial regularization). I can customize
    loss functions to optimize economically informed losses rather than generic 
    distance metrics. I can penalize correlation against existing strategies, if so 
    desired. People often refer to neural network flexibility in the context of learning
    an arbitrarily complex function. While this is true, I would use the word "flexibility"
    to refer to the ability to customize architectures and loss functions to suit
    a particular problem.

    Future work
    -----------
    - Add dropout layers for regularization.
    - Support for skip connections.
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
        if isinstance(n_latent, numbers.Integral):
            self.n_latent = [n_latent]
        else:
                self.n_latent = n_latent

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
        # Identify encoder activation
        activation_func = self.activation_map[encoder_activation]
        # Build encoder
        encoder_modules = [nn.Linear(n_inputs, n_latent[0], bias = fit_encoder_intercept), activation_func()]
        if len(n_latent) > 1:
            for layer_idx in range(1, len(n_latent)):
                encoder_modules.append(
                    nn.Linear(n_latent[layer_idx - 1], n_latent[layer_idx], bias = fit_encoder_intercept)
                )
                encoder_modules.append(activation_func())
        
        return nn.Sequential(*encoder_modules)
    
    def _build_head(self, n_latent, n_outputs, head_activation, fit_head_intercept):
        head = nn.Sequential(
            nn.Linear(n_latent, n_outputs, bias = fit_head_intercept),
            self.activation_map[head_activation]()
        )
        return head

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
            if not all(x >= 1 for x in n_latent):
                raise ValueError("When n_latent is a list, all elements must be at least 1.")
        else:
            if n_latent < 1:
                raise ValueError("When n_latent is an integer, it must be at least 1.")
        # n_outputs
        if not isinstance(n_outputs, numbers.Integral):
            raise TypeError("n_outputs must be an integer.")
        if n_outputs < 1:
            raise ValueError("n_outputs must be at least 1.")
        # encoder_activation
        if not isinstance(encoder_activation, str):
            raise TypeError("encoder_activation must be a string.")
        if encoder_activation not in {"tanh", "relu", "sigmoid"}:
            raise ValueError(
                "encoder_activation must be one of 'tanh', 'relu', or 'sigmoid'."
            )
        # head_activation
        if not isinstance(head_activation, str):
            raise TypeError("head_activation must be a string.")
        if head_activation not in {"tanh", "relu", "sigmoid", "identity"}:
            raise ValueError(
                "head_activation must be one of 'tanh', 'relu', 'sigmoid', or 'identity'."
            )
        # fit_encoder_intercept
        if not isinstance(fit_encoder_intercept, bool):
            raise TypeError("fit_encoder_intercept must be a boolean.")
        # fit_head_intercept
        if not isinstance(fit_head_intercept, bool):
            raise TypeError("fit_head_intercept must be a boolean.")
        
if __name__=="__main__":
    print("========================================")
    print("MLP: 5-32-1 structure, tanh activation")
    model = MultiLayerPerceptron(
        n_inputs=5,
        n_latent = 32,
        n_outputs=1,
    )
    print(model)
    print("========================================")
    print("MLP: 10-[64,32,16]-3 structure, relu activation, sigmoid head, encoder intercept, no head intercept")
    model = MultiLayerPerceptron(
        n_inputs=10,
        n_latent = [64,32,16],
        n_outputs=3,
        encoder_activation="relu",
        head_activation="sigmoid",
        fit_encoder_intercept=True,
        fit_head_intercept=False,
    )
    print(model)
    print("========================================")
