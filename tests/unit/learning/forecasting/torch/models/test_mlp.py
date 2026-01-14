import unittest 
from unittest import TestCase 

import torch
import torch.nn as nn 

from macrosynergy.learning.forecasting.torch.models.mlps import MultiLayerPerceptron

class TestMultiLayerPerceptron(TestCase):
    @classmethod
    def setUpClass(self):
        pass 

    def test_valid_init(self):
        pass 

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
        

    def test_build_encoder(self):
        pass

    def test_build_head(self):
        pass

    def test_valid_forward(self):
        pass

    def test_types_forward(self):
        pass