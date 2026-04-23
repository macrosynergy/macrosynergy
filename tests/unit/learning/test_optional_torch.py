"""Tests that torch-dependent classes are lazily imported and produce
a clear error when torch is not installed."""

import importlib
import sys
import unittest
from unittest.mock import patch


# Names that should always be importable (no torch needed)
NON_TORCH_NAMES = [
    "SignalOptimizer",
    "LADRegressor",
    "NaiveRegressor",
    "PanelStandardScaler",
]

# Names that require torch
TORCH_NAMES = [
    "MultiLayerPerceptron",
    "TimeSeriesSampler",
    "MultiOutputSharpe",
    "MultiOutputMCR",
    "MLPRegressor",
]


def _reload_learning_modules():
    """Remove cached learning submodules so imports are re-evaluated."""
    to_remove = [
        k
        for k in sys.modules
        if k.startswith("macrosynergy.learning.forecasting.torch")
        or k == "macrosynergy.learning.forecasting.nn.regressors"
    ]
    for k in to_remove:
        del sys.modules[k]


class TestTorchOptionalDependency(unittest.TestCase):
    def test_non_torch_imports_without_torch(self):
        """Non-torch classes should import successfully even if torch is missing."""
        import macrosynergy.learning as learning

        for name in NON_TORCH_NAMES:
            self.assertTrue(
                hasattr(learning, name),
                f"{name} should be importable without torch",
            )

    def test_torch_imports_fail_without_torch(self):
        """Torch-dependent classes should raise ImportError with install hint."""
        _reload_learning_modules()
        try:
            with patch.dict(sys.modules, {"torch": None}):
                for name in TORCH_NAMES:
                    with self.assertRaises(ImportError, msg=f"{name} should raise ImportError") as ctx:
                        importlib.import_module("macrosynergy.learning.forecasting.torch")
                    self.assertIn(
                        "macrosynergy[torch]",
                        str(ctx.exception),
                        "Error message should include install instructions",
                    )
                    _reload_learning_modules()
        finally:
            _reload_learning_modules()

    def test_torch_imports_succeed_with_torch(self):
        """Torch-dependent classes should import when torch is available."""
        torch = None
        try:
            import torch  # noqa: F811
        except ImportError:
            pass

        if torch is None:
            self.skipTest("torch not installed")

        import macrosynergy.learning as learning

        for name in TORCH_NAMES:
            obj = getattr(learning, name, None)
            self.assertIsNotNone(
                obj,
                f"{name} should be importable when torch is installed",
            )


if __name__ == "__main__":
    unittest.main()
