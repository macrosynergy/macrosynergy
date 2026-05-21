import itertools
import unittest
from unittest import TestCase

import torch
from parameterized import parameterized

from macrosynergy.learning.forecasting.torch.models.attention import MacroAttentionNet


class TestMacroAttentionNet(TestCase):
    @classmethod
    def setUpClass(self):
        self.model = MacroAttentionNet(
            n_inputs=6,
            n_outputs=1,
            d_model=12,
            n_heads=3,
            n_layers=2,
            dropout_p=0.1,
            max_seq_len=24,
            head_activation="identity",
        )

    def test_valid_init(self):
        self.assertIsInstance(self.model, MacroAttentionNet)
        self.assertEqual(self.model.n_inputs, 6)
        self.assertEqual(self.model.n_outputs, 1)
        self.assertEqual(self.model.d_model, 12)
        self.assertEqual(self.model.n_heads, 3)
        self.assertEqual(self.model.n_layers, 2)
        self.assertEqual(self.model.max_seq_len, 24)

    @parameterized.expand(
        itertools.product(
            [1, 3],
            [1, 2],
            ["identity", "tanh", "relu", "sigmoid"],
        )
    )
    def test_valid_init_custom(self, n_heads, n_layers, head_activation):
        model = MacroAttentionNet(
            n_inputs=4,
            n_outputs=2,
            d_model=12,
            n_heads=n_heads,
            n_layers=n_layers,
            head_activation=head_activation,
        )
        self.assertIsInstance(model, MacroAttentionNet)
        self.assertEqual(model.n_heads, n_heads)
        self.assertEqual(model.n_layers, n_layers)
        self.assertEqual(model.head_activation, head_activation)

    def test_forward_shape(self):
        x = torch.randn(8, 12, 6)
        out = self.model(x)
        self.assertEqual(tuple(out.shape), (8, 1))

    def test_forward_with_attention(self):
        x = torch.randn(5, 12, 6)
        out, attn = self.model(x, return_attention=True)
        self.assertEqual(tuple(out.shape), (5, 1))
        self.assertEqual(tuple(attn.shape), (5, 12))
        self.assertTrue(torch.allclose(attn.sum(dim=1), torch.ones(5), atol=1e-5))

    def test_forward_with_padding_mask(self):
        x = torch.randn(4, 10, 6)
        padding_mask = torch.zeros(4, 10, dtype=torch.bool)
        padding_mask[:, -2:] = True

        out, attn = self.model(x, padding_mask=padding_mask, return_attention=True)
        self.assertEqual(tuple(out.shape), (4, 1))
        self.assertTrue(torch.all(attn[:, -2:] < 1e-6))

    def test_invalid_init(self):
        with self.assertRaises(ValueError):
            MacroAttentionNet(n_inputs=4, d_model=10, n_heads=3)
        with self.assertRaises(ValueError):
            MacroAttentionNet(n_inputs=4, dropout_p=0.8)
        with self.assertRaises(ValueError):
            MacroAttentionNet(n_inputs=4, head_activation="bad")

    def test_invalid_forward(self):
        with self.assertRaises(ValueError):
            self.model(torch.randn(5, 6))
        with self.assertRaises(ValueError):
            self.model(torch.randn(5, 12, 5))
        with self.assertRaises(ValueError):
            self.model(torch.randn(5, 30, 6))


if __name__ == "__main__":
    unittest.main()
