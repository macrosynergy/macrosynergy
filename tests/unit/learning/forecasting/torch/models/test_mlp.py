import unittest 
from unittest import TestCase 

import torch
import torch.nn as nn 

from macrosynergy.learning.forecasting.torch.models.mlps import MultiLayerPerceptron

class TestMultiLayerPerceptron(TestCase):
    @classmethod
    def setUpClass(self):
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
        pass

    def test_build_head(self):
        pass

    def test_valid_forward(self):
        pass

    def test_types_forward(self):
        pass