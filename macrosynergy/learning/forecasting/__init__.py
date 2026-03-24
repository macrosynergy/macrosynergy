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

def __getattr__(name):
    _torch_names = {
        "MultiLayerPerceptron",
        "TimeSeriesSampler",
        "MultiOutputSharpe",
        "MultiOutputMCR",
    }
    _nn_names = {
        "MLPRegressor",
    }
    if name in _torch_names:
        from .torch import (
            MultiLayerPerceptron,
            TimeSeriesSampler,
            MultiOutputSharpe,
            MultiOutputMCR,
        )
        return locals()[name]
    if name in _nn_names:
        from .nn import MLPRegressor
        return MLPRegressor
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
    "MLPRegressor",
]