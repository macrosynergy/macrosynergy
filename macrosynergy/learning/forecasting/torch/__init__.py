try:
    import torch as _torch  # noqa: F401
except ImportError as e:
    raise ImportError(
        "PyTorch is required for this module but is not installed. "
        "Install it with: pip install macrosynergy[torch]"
    ) from e

from .models import MultiLayerPerceptron
from .samplers import TimeSeriesSampler
from .losses import (
    MultiOutputSharpe,
    MultiOutputMCR,
)
from .trainer import MLPTrainer

__all__ = [
    # models
    "MultiLayerPerceptron",
    # samplers
    "TimeSeriesSampler",
    # losses
    "MultiOutputSharpe",
    "MultiOutputMCR",
    # trainer
    "MLPTrainer",
]
