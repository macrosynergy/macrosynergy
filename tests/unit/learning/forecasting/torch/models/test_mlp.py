import unittest 
from unittest import TestCase 

import torch
import torch.nn as nn 

from macrosynergy.learning.forecasting.torch.models.mlps import MultiLayerPerceptron

from parameterized import parameterized
import itertools

class TestMultiLayerPerceptron(TestCase):
    @classmethod
    def setUpClass(self):
        # Instantiate some models upfront
        self.single_output_single_layer_mlp = MultiLayerPerceptron(
            n_inputs=10,
            n_latent=32,
            n_outputs=1,
        )
        self.single_output_multi_layer_mlp = MultiLayerPerceptron(
            n_inputs=10,
            n_latent=[32, 16],
            n_outputs=1,
        )
        self.multi_output_single_layer_mlp = MultiLayerPerceptron(
            n_inputs=10,
            n_latent=32,
            n_outputs=3,
        )
        self.multi_output_multi_layer_mlp = MultiLayerPerceptron(
            n_inputs=10,
            n_latent=[32, 16],
            n_outputs=3,
        ) 


    def test_valid_init(self):
        """ Test correct model instantiation with valid inputs """
        # Single output, single hidden layer
        self.assertIsInstance(self.single_output_single_layer_mlp, MultiLayerPerceptron)
        self.assertEqual(self.single_output_single_layer_mlp.n_inputs, 10)
        self.assertEqual(self.single_output_single_layer_mlp.n_latent, [32])
        self.assertEqual(self.single_output_single_layer_mlp.n_outputs, 1)
        self.assertEqual(self.single_output_single_layer_mlp.encoder_activation, "tanh")
        self.assertEqual(self.single_output_single_layer_mlp.head_activation, "identity")
        self.assertEqual(self.single_output_single_layer_mlp.fit_encoder_intercept, False)
        self.assertEqual(self.single_output_single_layer_mlp.fit_head_intercept, True)
        # Single output, multiple hidden layers
        self.assertIsInstance(self.single_output_multi_layer_mlp, MultiLayerPerceptron)
        self.assertEqual(self.single_output_multi_layer_mlp.n_inputs, 10)
        self.assertEqual(self.single_output_multi_layer_mlp.n_latent, [32, 16])
        self.assertEqual(self.single_output_multi_layer_mlp.n_outputs, 1)
        self.assertEqual(self.single_output_multi_layer_mlp.encoder_activation, "tanh")
        self.assertEqual(self.single_output_multi_layer_mlp.head_activation, "identity")
        self.assertEqual(self.single_output_multi_layer_mlp.fit_encoder_intercept, False)
        self.assertEqual(self.single_output_multi_layer_mlp.fit_head_intercept, True)
        # Multiple output, single hidden layer
        self.assertIsInstance(self.multi_output_single_layer_mlp, MultiLayerPerceptron)
        self.assertEqual(self.multi_output_single_layer_mlp.n_inputs, 10)
        self.assertEqual(self.multi_output_single_layer_mlp.n_latent, [32])
        self.assertEqual(self.multi_output_single_layer_mlp.n_outputs, 3)
        self.assertEqual(self.multi_output_single_layer_mlp.encoder_activation, "tanh")
        self.assertEqual(self.multi_output_single_layer_mlp.head_activation, "identity")
        self.assertEqual(self.multi_output_single_layer_mlp.fit_encoder_intercept, False)
        self.assertEqual(self.multi_output_single_layer_mlp.fit_head_intercept, True)
        # Multiple output, multiple hidden layers
        self.assertIsInstance(self.multi_output_multi_layer_mlp, MultiLayerPerceptron)
        self.assertEqual(self.multi_output_multi_layer_mlp.n_inputs, 10)
        self.assertEqual(self.multi_output_multi_layer_mlp.n_latent, [32, 16])
        self.assertEqual(self.multi_output_multi_layer_mlp.n_outputs, 3)
        self.assertEqual(self.multi_output_multi_layer_mlp.encoder_activation, "tanh")
        self.assertEqual(self.multi_output_multi_layer_mlp.head_activation, "identity")
        self.assertEqual(self.multi_output_multi_layer_mlp.fit_encoder_intercept, False)
        self.assertEqual(self.multi_output_multi_layer_mlp.fit_head_intercept, True)

    @parameterized.expand(itertools.product([True, False], [True, False], ["tanh", "relu", "sigmoid"], ["identity", "tanh", "relu", "sigmoid"]))
    def test_valid_init_custom(self, fit_encoder_intercept, fit_head_intercept, encoder_activation, head_activation):
        """ Test correct model instantiation with varying network structures """
        model = MultiLayerPerceptron(
            n_inputs=10,
            n_latent=[64, 32, 16],
            n_outputs=2,
            encoder_activation=encoder_activation,
            head_activation=head_activation,
            fit_encoder_intercept=fit_encoder_intercept,
            fit_head_intercept=fit_head_intercept,
        )
        self.assertIsInstance(model, MultiLayerPerceptron)
        self.assertEqual(model.n_inputs, 10)
        self.assertEqual(model.n_latent, [64, 32, 16])
        self.assertEqual(model.n_outputs, 2)
        self.assertEqual(model.encoder_activation, encoder_activation)
        self.assertEqual(model.head_activation, head_activation)
        self.assertEqual(model.fit_encoder_intercept, fit_encoder_intercept)
        self.assertEqual(model.fit_head_intercept, fit_head_intercept)

    def test_types_init(self):
        """ Test correct input validation checks """
        """ n_inputs """
        with self.assertRaises(TypeError):
            MultiLayerPerceptron(
                n_inputs="10",
                n_latent=4,
                n_outputs=1,
            )
        with self.assertRaises(TypeError):
            MultiLayerPerceptron(
                n_inputs=10.5,
                n_latent=4,
                n_outputs=1,
            )
        with self.assertRaises(TypeError):
            MultiLayerPerceptron(
                n_inputs=[345],
                n_latent=4,
                n_outputs=1,
            )
        with self.assertRaises(TypeError):
            MultiLayerPerceptron(
                n_inputs=3.4,
                n_latent=4,
                n_outputs=1,
            )
        with self.assertRaises(ValueError):
            MultiLayerPerceptron(
                n_inputs=0,
                n_latent=4,
                n_outputs=1,
            )
        with self.assertRaises(ValueError):
            MultiLayerPerceptron(
                n_inputs=-2,
                n_latent=4,
                n_outputs=1,
            )
        """ n_latent """
        # Start with single hidden layer
        with self.assertRaises(TypeError):
            MultiLayerPerceptron(
                n_inputs=10,
                n_latent="4",
                n_outputs=1,
            )
        with self.assertRaises(TypeError):
            MultiLayerPerceptron(
                n_inputs=10,
                n_latent=4.5,
                n_outputs=1,
            )
        with self.assertRaises(ValueError):
            MultiLayerPerceptron(
                n_inputs=10,
                n_latent=0,
                n_outputs=1,
            )
        with self.assertRaises(ValueError):
            MultiLayerPerceptron(
                n_inputs=10,
                n_latent=-2,
                n_outputs=1,
            )
        # Single element of a list
        with self.assertRaises(TypeError):
            MultiLayerPerceptron(
                n_inputs=10,
                n_latent=["4"],
                n_outputs=1,
            )
        with self.assertRaises(TypeError):
            MultiLayerPerceptron(
                n_inputs=10,
                n_latent=[4.5],
                n_outputs=1,
            )
        with self.assertRaises(ValueError):
            MultiLayerPerceptron(
                n_inputs=10,
                n_latent=[0],
                n_outputs=1,
            )
        with self.assertRaises(ValueError):
            MultiLayerPerceptron(
                n_inputs=10,
                n_latent=[-2],
                n_outputs=1,
            )
        # Multiple elements of a list
        with self.assertRaises(TypeError):
            MultiLayerPerceptron(
                n_inputs=10,
                n_latent=[16, "32", 64],
                n_outputs=1,
            )
        with self.assertRaises(TypeError):
            MultiLayerPerceptron(
                n_inputs=10,
                n_latent=[16, 32.5, 64],
                n_outputs=1,
            )
        with self.assertRaises(ValueError):
            MultiLayerPerceptron(
                n_inputs=10,
                n_latent=[16, 0, 64],
                n_outputs=1,
            )
        with self.assertRaises(ValueError):
            MultiLayerPerceptron(
                n_inputs=10,
                n_latent=[16, -32, 64],
                n_outputs=1,
            )

        """ n_outputs """
        with self.assertRaises(TypeError):
            MultiLayerPerceptron(
                n_inputs=10,
                n_latent=4,
                n_outputs="1",
            )
        with self.assertRaises(TypeError):
            MultiLayerPerceptron(
                n_inputs=10,
                n_latent=4,
                n_outputs=1.5,
            )
        with self.assertRaises(ValueError):
            MultiLayerPerceptron(
                n_inputs=10,
                n_latent=4,
                n_outputs=0,
            )
        with self.assertRaises(ValueError):
            MultiLayerPerceptron(
                n_inputs=10,
                n_latent=4,
                n_outputs=-3,
            )
        
        """ encoder_activation """
        with self.assertRaises(TypeError):
            MultiLayerPerceptron(
                n_inputs=10,
                n_latent=4,
                n_outputs=1,
                encoder_activation=123,
            )
        with self.assertRaises(ValueError):
            MultiLayerPerceptron(
                n_inputs=10,
                n_latent=4,
                n_outputs=1,
                encoder_activation="sdfsdfs",
            )
        with self.assertRaises(ValueError):
            MultiLayerPerceptron(
                n_inputs=10,
                n_latent=4,
                n_outputs=1,
                encoder_activation="identity",
            )

        """ head_activation """
        with self.assertRaises(TypeError):
            MultiLayerPerceptron(
                n_inputs=10,
                n_latent=4,
                n_outputs=1,
                head_activation=456,
            )
        with self.assertRaises(ValueError):
            MultiLayerPerceptron(
                n_inputs=10,
                n_latent=4,
                n_outputs=1,
                head_activation="sdfsdfs",
            )

        """ fit_encoder_intercept """
        with self.assertRaises(TypeError):
            MultiLayerPerceptron(
                n_inputs=10,
                n_latent=4,
                n_outputs=1,
                fit_encoder_intercept="True",
            )

        """ fit_head_intercept """
        with self.assertRaises(TypeError):
            MultiLayerPerceptron(
                n_inputs=10,
                n_latent=4,
                n_outputs=1,
                fit_head_intercept="False",
            )
    

    def test_build_encoder(self):
        """ Test that the encoder is built correctly """
        """ Single hidden layer, single output """
        # Tanh
        model_encoder = nn.Sequential(
            nn.Linear(10, 32, bias=False),
            nn.Tanh(),
        )
        created_encoder = self.single_output_single_layer_mlp._build_encoder(10, [32], "tanh", False)
        self.assertIsInstance(created_encoder, nn.Sequential)
        self.assertEqual(len(model_encoder), len(created_encoder))
        self.assertIsInstance(created_encoder[1], nn.Tanh)
        # ReLU
        model_encoder = nn.Sequential(
            nn.Linear(10, 32, bias=False),
            nn.ReLU(inplace=True),
        )
        created_encoder = self.single_output_single_layer_mlp._build_encoder(10, [32], "relu", False)
        self.assertIsInstance(created_encoder, nn.Sequential)
        self.assertEqual(len(model_encoder), len(created_encoder))
        self.assertIsInstance(created_encoder[1], nn.ReLU)
        # Sigmoid
        model_encoder = nn.Sequential(
            nn.Linear(10, 32, bias=False),
            nn.Sigmoid(),
        )
        created_encoder = self.single_output_single_layer_mlp._build_encoder(10, [32], "sigmoid", False)
        self.assertIsInstance(created_encoder, nn.Sequential)
        self.assertEqual(len(model_encoder), len(created_encoder))
        self.assertIsInstance(created_encoder[1], nn.Sigmoid)

        """ Multiple hidden layers, single output """
        # Tanh
        model_encoder = nn.Sequential(
            nn.Linear(10, 32, bias=False),
            nn.Tanh(),
            nn.Linear(32, 16, bias=False),
            nn.Tanh(),
        )
        created_encoder = self.single_output_multi_layer_mlp._build_encoder(10, [32, 16], "tanh", False)
        self.assertIsInstance(created_encoder, nn.Sequential)
        self.assertEqual(len(model_encoder), len(created_encoder))
        self.assertIsInstance(created_encoder[1], nn.Tanh)
        self.assertIsInstance(created_encoder[3], nn.Tanh)
        # Sigmoid
        model_encoder = nn.Sequential(
            nn.Linear(10, 32, bias=False),
            nn.Sigmoid(),
            nn.Linear(32, 16, bias=False),
            nn.Sigmoid(),
        )
        created_encoder = self.single_output_multi_layer_mlp._build_encoder(10, [32, 16], "sigmoid", False)
        self.assertIsInstance(created_encoder, nn.Sequential)
        self.assertEqual(len(model_encoder), len(created_encoder))
        self.assertIsInstance(created_encoder[1], nn.Sigmoid)
        self.assertIsInstance(created_encoder[3], nn.Sigmoid)
        # ReLU
        model_encoder = nn.Sequential(
            nn.Linear(10, 32, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(32, 16, bias=False),
            nn.ReLU(inplace=True),
        )
        created_encoder = self.single_output_multi_layer_mlp._build_encoder(10, [32, 16], "relu", False)
        self.assertIsInstance(created_encoder, nn.Sequential)
        self.assertEqual(len(model_encoder), len(created_encoder))
        self.assertIsInstance(created_encoder[1], nn.ReLU)
        self.assertIsInstance(created_encoder[3], nn.ReLU)

        """ Single hidden layer, multiple output """
        # Tanh
        model_encoder = nn.Sequential(
            nn.Linear(10, 32, bias=False),
            nn.Tanh(),
        )
        created_encoder = self.multi_output_single_layer_mlp._build_encoder(10, [32], "tanh", False)
        self.assertIsInstance(created_encoder, nn.Sequential)
        self.assertEqual(len(model_encoder), len(created_encoder))
        self.assertIsInstance(created_encoder[1], nn.Tanh)
        # Sigmoid
        model_encoder = nn.Sequential(
            nn.Linear(10, 32, bias=False),
            nn.Sigmoid(),
        )
        created_encoder = self.multi_output_single_layer_mlp._build_encoder(10, [32], "sigmoid", False)
        self.assertIsInstance(created_encoder, nn.Sequential)
        self.assertEqual(len(model_encoder), len(created_encoder))
        self.assertIsInstance(created_encoder[1], nn.Sigmoid)
        # ReLU
        model_encoder = nn.Sequential(
            nn.Linear(10, 32, bias=False),
            nn.ReLU(inplace=True),
        )
        created_encoder = self.multi_output_single_layer_mlp._build_encoder(10, [32], "relu", False)
        self.assertIsInstance(created_encoder, nn.Sequential)
        self.assertEqual(len(model_encoder), len(created_encoder))
        self.assertIsInstance(created_encoder[1], nn.ReLU)

        """ Multiple hidden layers, multiple output """
        # Tanh
        model_encoder = nn.Sequential(
            nn.Linear(10, 32, bias=False),
            nn.Tanh(),
            nn.Linear(32, 16, bias=False),
            nn.Tanh(),
        )
        created_encoder = self.multi_output_multi_layer_mlp._build_encoder(10, [32, 16], "tanh", False)
        self.assertIsInstance(created_encoder, nn.Sequential)
        self.assertEqual(len(model_encoder), len(created_encoder))
        self.assertIsInstance(created_encoder[1], nn.Tanh)
        self.assertIsInstance(created_encoder[3], nn.Tanh)
        # Sigmoid
        model_encoder = nn.Sequential(
            nn.Linear(10, 32, bias=False),
            nn.Sigmoid(),
            nn.Linear(32, 16, bias=False),
            nn.Sigmoid(),
        )
        created_encoder = self.multi_output_multi_layer_mlp._build_encoder(10, [32, 16], "sigmoid", False)
        self.assertIsInstance(created_encoder, nn.Sequential)
        self.assertEqual(len(model_encoder), len(created_encoder))
        self.assertIsInstance(created_encoder[1], nn.Sigmoid)
        self.assertIsInstance(created_encoder[3], nn.Sigmoid)
        # ReLU
        model_encoder = nn.Sequential(
            nn.Linear(10, 32, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(32, 16, bias=False),
            nn.ReLU(inplace=True),
        )
        created_encoder = self.multi_output_multi_layer_mlp._build_encoder(10, [32, 16], "relu", False)
        self.assertIsInstance(created_encoder, nn.Sequential)
        self.assertEqual(len(model_encoder), len(created_encoder))
        self.assertIsInstance(created_encoder[1], nn.ReLU)
        self.assertIsInstance(created_encoder[3], nn.ReLU)

    def test_build_head(self):
        """ Test that the projection head is built correctly """
        # Single output
        model_head = nn.Sequential(
            nn.Linear(32, 1, bias=True),
            nn.Identity(),
        )
        created_head = self.single_output_single_layer_mlp._build_head(32, 1, "identity", True)
        self.assertIsInstance(created_head, nn.Sequential)
        self.assertEqual(len(model_head), len(created_head))
        self.assertIsInstance(created_head[1], nn.Identity)
        model_head = nn.Sequential(
            nn.Linear(32, 1, bias=True),
            nn.Tanh(),
        )
        created_head = self.single_output_single_layer_mlp._build_head(32, 1, "tanh", True)
        self.assertIsInstance(created_head, nn.Sequential)
        self.assertEqual(len(model_head), len(created_head))
        self.assertIsInstance(created_head[1], nn.Tanh)
        # Multiple output
        model_head = nn.Sequential(
            nn.Linear(32, 3, bias=True),
            nn.Identity(),
        )
        created_head = self.multi_output_single_layer_mlp._build_head(32, 3, "identity", True)
        self.assertIsInstance(created_head, nn.Sequential)
        self.assertEqual(len(model_head), len(created_head))
        self.assertIsInstance(created_head[1], nn.Identity)
        model_head = nn.Sequential(
            nn.Linear(32, 3, bias=True),
            nn.Tanh(),
        )
        created_head = self.multi_output_single_layer_mlp._build_head(32, 3, "tanh", True)
        self.assertIsInstance(created_head, nn.Sequential)
        self.assertEqual(len(model_head), len(created_head))
        self.assertIsInstance(created_head[1], nn.Tanh)

    def test_valid_forward(self):
        pass