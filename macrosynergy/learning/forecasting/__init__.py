from .linear_model import (
    LADRegressor,
    SignWeightedLADRegressor,
    TimeWeightedLADRegressor,
    SignWeightedLinearRegression,
    TimeWeightedLinearRegression,
    ModifiedLinearRegression,
    ModifiedSignWeightedLinearRegression,
    ModifiedTimeWeightedLinearRegression,
    GlobalLocalRegression,
    LinearMultiTargetRegression,
)

from .model_systems import (
    CorrelationVolatilitySystem,
    LADRegressionSystem,
    LinearRegressionSystem,
    RidgeRegressionSystem,
)

from .naive_predictors import (
    NaiveRegressor,
)

from .neighbors.nearest_neighbors import KNNClassifier

from .meta_estimators import ProbabilityEstimator, FIExtractor, DataFrameTransformer, CountryByCountryRegression, TimeWeightedWrapper

from .ensemble import (
    VotingClassifier,
    VotingRegressor,
)

from .factor_models import (
    PLSTransformer,
)

from .model_inference import (
    LarsSurrogateModel,
)

def __getattr__(name):
    _torch_names = {
        "MultiLayerPerceptron",
        "TimeSeriesSampler",
        "MultiOutputSharpe",
        "MultiOutputMCR",
        "PortfolioVariance",
        "NegMeanVarianceUtility",
        "NegMeanVarianceExAnteVol",
        "NegMeanPortfolioReturn",
        "NegSharpeRatio",
        "NegSharpeRatioExAnteVol",
        "LongShortModule",
        "PanelBatchSampler",
        "NegMeanVarianceSkewnessUtility",
        "AssetBaggingLoss",
        "AssetBaggingLoss",
    "NegCrossSectionalIC",
        "NegRankIC",
        "NegThreeComponentLoss",
        "RankingRiskLoss",
        "HeteroskedasticMLP",
        "GaussianNLL",
        "ActiveWeightModule",
        "ActiveReturnLoss",
        "BenchmarkWeightedIC",
        "BenchmarkFeasibilityPenalty",
        "SwiGLU",
    }
    _nn_names = {
        "MLPRegressor",
        "PanelMLPRegressor",
    }
    if name in _torch_names:
        from .torch import (
            MultiLayerPerceptron,
            TimeSeriesSampler,
            MultiOutputSharpe,
            MultiOutputMCR,
            PortfolioVariance,
            NegMeanVarianceUtility,
            NegMeanVarianceExAnteVol,
            NegMeanPortfolioReturn,
            NegSharpeRatio,
            NegSharpeRatioExAnteVol,
            LongShortModule,
            SwiGLU,
            PanelBatchSampler,
            NegMeanVarianceSkewnessUtility,
            AssetBaggingLoss,
            NegCrossSectionalIC,
            NegRankIC,
            NegThreeComponentLoss,
            RankingRiskLoss,
            HeteroskedasticMLP,
            GaussianNLL,
            ActiveWeightModule,
            ActiveReturnLoss,
            BenchmarkWeightedIC,
            BenchmarkFeasibilityPenalty,
        )
        return locals()[name]
    if name in _nn_names:
        from .nn import MLPRegressor, PanelMLPRegressor
        return locals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "LADRegressor",
    "KNNClassifier",
    "SignWeightedLADRegressor",
    "TimeWeightedLADRegressor",
    "SignWeightedLinearRegression",
    "TimeWeightedLinearRegression",
    "NaiveRegressor",
    "CorrelationVolatilitySystem",
    "LADRegressionSystem",
    "LinearRegressionSystem",
    "RidgeRegressionSystem",
    "ModifiedLinearRegression",
    "ModifiedSignWeightedLinearRegression",
    "ModifiedTimeWeightedLinearRegression",
    "ProbabilityEstimator",
    "VotingClassifier",
    "VotingRegressor",
    "FIExtractor",
    "DataFrameTransformer",
    "GlobalLocalRegression",
    "CountryByCountryRegression",
    "TimeWeightedWrapper",
    "PLSTransformer",
    "LinearMultiTargetRegression",
    "MultiLayerPerceptron",
    "TimeSeriesSampler",
    "MultiOutputSharpe",
    "MultiOutputMCR",
    "NegMeanPortfolioReturn",
    "PortfolioVariance",
    "NegMeanVarianceUtility",
    "NegMeanVarianceExAnteVol",
    "NegSharpeRatio",
    "NegSharpeRatioExAnteVol",
    "PanelMLPRegressor",
    "MLPRegressor",
    "LongShortModule",
    "PanelBatchSampler",
    "NegMeanVarianceSkewnessUtility",
    "NegCrossSectionalIC",
    "NegRankIC",
    "NegThreeComponentLoss",
    "RankingRiskLoss",
    "HeteroskedasticMLP",
    "GaussianNLL",
    "ActiveWeightModule",
    "ActiveReturnLoss",
    "BenchmarkWeightedIC",
    "BenchmarkFeasibilityPenalty",
    "SwiGLU",
    "LarsSurrogateModel",
]