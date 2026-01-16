import torch 
import torch.nn as nn 

from macrosynergy.learning.forecasting.torch.losses.sharpe_loss import MultiOutputSharpe

import unittest 

from parameterized import parameterized

class TestMultiOutputSharpe(unittest.TestCase):
    @classmethod 
    def setUpClass(self):
        pass 

    def test_types_init(self):
        """ Test types of init parameters """
        # skip_validation
        self.assertRaises(TypeError, MultiOutputSharpe, skip_validation = "invalid_string")

    @parameterized.expand([True, False])
    def test_valid_init(self, skip_validation):
        try: 
            sharpe = MultiOutputSharpe(skip_validation = skip_validation)
        except Exception as e:
            self.fail(f"MultiOutputSharpe raised {type(e)} unexpectedly!")
        self.assertEqual(sharpe.skip_validation, skip_validation)

    def test_types_forward(self):
        sharpe = MultiOutputSharpe(skip_validation = False)

        # y_true
        self.assertRaises(
            TypeError,
            sharpe.forward,
            y_true = "invalid_string",
            y_pred = torch.randn(10, 5),
        )
        self.assertRaises(
            ValueError,
            sharpe.forward,
            y_true = torch.randn(10, 1),
            y_pred = torch.randn(10, 3),
        )

        # y_pred
        self.assertRaises(
            TypeError,
            sharpe.forward,
            y_true = torch.randn(10, 5),
            y_pred = "invalid_string",
        )
        self.assertRaises(
            ValueError,
            sharpe.forward,
            y_true = torch.randn(10, 3),
            y_pred = torch.randn(10, 1),
        )

        # ValueError raised when shapes do not match
        self.assertRaises(
            ValueError,
            sharpe.forward,
            y_true = torch.randn(10, 4),
            y_pred = torch.randn(10, 3),
        )

    def test_valid_forward(self):
        pass