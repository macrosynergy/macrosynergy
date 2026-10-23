import unittest

import numpy as np
import torch
import torch.nn as nn

from macrosynergy.learning.forecasting.torch.modules.portfolio_layers import LongShortModule

class TestLongShortModule(unittest.TestCase):
    def test_types_init(self):
        """
        Test that constructor arguments raise appropriate
        Type and Value Errors. 
        """
        # dollar_neutral should be a boolean
        with self.assertRaises(TypeError):
            LongShortModule(dollar_neutral="not_a_boolean")

        # eps should be a non-negative real number
        with self.assertRaises(TypeError):
            LongShortModule(eps="not_a_number")
        with self.assertRaises(ValueError):
            LongShortModule(eps=-0.1)

    def test_valid_init(self):
        """
        Test that valid initialization works correctly.
        """
        # Test default initialization works
        try:
            default_module = LongShortModule()
        except Exception as e:
            self.fail(f"Default initialization failed with exception: {e}")

        self.assertIsInstance(default_module, LongShortModule)
        self.assertEqual(default_module.dollar_neutral, False)
        self.assertEqual(default_module.eps, 0)

        # Test custom initialization works as expected
        try:
            module = LongShortModule(dollar_neutral=True, eps=1e-5)
        except Exception as e:
            self.fail(f"Initialization failed with dexception: {e}")

        self.assertIsInstance(module, LongShortModule)
        self.assertEqual(module.dollar_neutral, True)
        self.assertEqual(module.eps, 1e-5)

    def test_valid_forward(self):
        """
        Test that the forward pass produces valid portfolio weights.
        """
        # Default initialization
        module = LongShortModule()
        x = torch.randn(10, 6)  # batch of 10 samples, 6 assets
        weights = module(x)
        self.assertTrue(torch.all(weights.abs() <= 1))
        self.assertTrue(torch.allclose(weights.abs().sum(dim=-1), torch.ones(10)))


        # Dollar neutrality
        module = LongShortModule(dollar_neutral=True, eps = 0)
        x = torch.randn(10, 5)  # batch of 10 samples, 5 assets
        weights = module(x)
        self.assertTrue(torch.all(weights.abs() <= 1))
        self.assertTrue(torch.allclose(weights.abs().sum(dim=-1), torch.ones(10)))
        self.assertTrue(torch.allclose(weights.sum(dim=-1), torch.zeros(10)))

        # Dollar neutrality with epsilon greater than 0
        module = LongShortModule(dollar_neutral=True, eps=1e-4)
        x = torch.randn(10, 5)  # batch of 10 samples, 5 assets
        weights = module(x)
        self.assertTrue(torch.all(weights.abs() <= 1))
        self.assertTrue(torch.allclose(weights.abs().sum(dim=-1), torch.ones(10)))
        self.assertTrue(torch.allclose(weights.sum(dim=-1), torch.zeros(10)))