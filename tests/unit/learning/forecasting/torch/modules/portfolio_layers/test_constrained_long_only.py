import unittest

import numpy as np
import torch
import torch.nn as nn

from macrosynergy.learning.forecasting.torch.modules.portfolio_layers import ConstrainedLongOnlyModule

class TestConstrainedLongOnly(unittest.TestCase):
    def test_types_init(self):
        """
        Test that constructor arguments raise appropriate
        Type and Value Errors. 
        """
        # concentration_bound should be a float between 0 and 1
        with self.assertRaises(TypeError):
            ConstrainedLongOnlyModule(concentration_bound="not_a_float")
        with self.assertRaises(ValueError):
            ConstrainedLongOnlyModule(concentration_bound=-0.1)
        with self.assertRaises(ValueError):
            ConstrainedLongOnlyModule(concentration_bound=1.1)

    def test_valid_init(self):
        """
        Test that valid initialization works correctly.
        """
        # Test default initialization works
        try:
            default_module = ConstrainedLongOnlyModule()
        except Exception as e:
            self.fail(f"Default initialization failed with exception: {e}")

        self.assertIsInstance(default_module, ConstrainedLongOnlyModule)
        self.assertEqual(default_module.concentration_bound, 0.2) 

        # Test custom initialization works as expected
        try:
            module = ConstrainedLongOnlyModule(concentration_bound=0.5)
        except Exception as e:
            self.fail(f"Initialization failed with exception: {e}")

        self.assertIsInstance(module, ConstrainedLongOnlyModule)
        self.assertEqual(module.concentration_bound, 0.5)

    def test_valid_forward(self):
        pass