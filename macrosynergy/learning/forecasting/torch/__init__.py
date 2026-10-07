try:
    import torch as _torch  # noqa: F401
except ImportError as e:
    raise ImportError(
        "PyTorch is required for this module but is not installed. "
        "Install it with: pip install macrosynergy[torch]"
    ) from e

from .models import MultiLayerPerceptron, HeteroskedasticMLP
from .samplers import TimeSeriesSampler, PanelBatchSampler
from .losses import (
    MultiOutputSharpe,
    MultiOutputMCR,
    NegMeanPortfolioReturn,
    PortfolioVariance,
    NegMeanVarianceUtility,
    NegMeanVarianceExAnteVol,
    NegMeanVarianceSkewnessUtility,
    NegSharpeRatio,
    NegSharpeRatioExAnteVol,
    AssetBaggingLoss,
    NegCrossSectionalIC,
    NegRankIC,
    NegThreeComponentLoss,
    RankingRiskLoss,
    GaussianNLL,
    ActiveWeightModule,
    ActiveReturnLoss,
    BenchmarkWeightedIC,
    BenchmarkFeasibilityPenalty,
)
from .modules import (
    LongShortModule,
    SwiGLU
)

__all__ = [
    # models
    "MultiLayerPerceptron",
    "HeteroskedasticMLP",
    # samplers
    "TimeSeriesSampler",
    "PanelBatchSampler",
    # losses
    "MultiOutputSharpe",
    "MultiOutputMCR",
    "NegMeanPortfolioReturn",
    "PortfolioVariance",
    "NegMeanVarianceUtility",
    "NegMeanVarianceExAnteVol",
    "NegMeanVarianceSkewnessUtility",
    "NegSharpeRatio",
    "NegSharpeRatioExAnteVol",
    "AssetBaggingLoss",
    "NegCrossSectionalIC",
    "NegRankIC",
    "NegThreeComponentLoss",
    "RankingRiskLoss",
    "GaussianNLL",
    "ActiveWeightModule",
    "ActiveReturnLoss",
    "BenchmarkWeightedIC",
    "BenchmarkFeasibilityPenalty",
    # modules
    "LongShortModule",
    "SwiGLU",
]
