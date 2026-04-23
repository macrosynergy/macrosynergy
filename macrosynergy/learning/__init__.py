from .splitters import (
    ExpandingKFoldPanelSplit,
    RollingKFoldPanelSplit,
    RecencyKFoldPanelSplit,
    ExpandingIncrementPanelSplit,
    ExpandingFrequencyPanelSplit,
)
from .cv_tools import panel_cv_scores

from .preprocessing import (
    LassoSelector,
    LarsSelector,
    MapSelector,
    ZnScoreAverager,
    PanelMinMaxScaler,
    PanelStandardScaler,
    BasePanelScaler,
    BasePanelSelector,
    PanelPCA,
    KendallSignificanceSelector,
    BaseImputer,
    ConstantImputer,
    CrossSectionalImputer,
    EstimatorImputer,
    GaussianConditionalImputer,
    FactorAvailabilitySelector,
)
from .model_evaluation import (
    neg_mean_abs_corr,
    panel_significance_probability,
    sharpe_ratio,
    sortino_ratio,
    multi_output_sharpe,
    multi_output_sortino,
    regression_accuracy,
    regression_balanced_accuracy,
    correlation_coefficient,
    create_panel_metric,
    regression_mcc,
)
from .sequential import SignalOptimizer, BetaEstimator, ReturnForecaster

from .forecasting import (
    LADRegressor,
    NaiveRegressor,
    SignWeightedLinearRegression,
    TimeWeightedLinearRegression,
    SignWeightedLADRegressor,
    TimeWeightedLADRegressor,
    ModifiedLinearRegression,
    ModifiedSignWeightedLinearRegression,
    ModifiedTimeWeightedLinearRegression,
    LinearRegressionSystem,
    LADRegressionSystem,
    RidgeRegressionSystem,
    CorrelationVolatilitySystem,
    KNNClassifier,
    ProbabilityEstimator,
    VotingRegressor,
    VotingClassifier,
    FIExtractor,
    DataFrameTransformer,
    GlobalLocalRegression,
    CountryByCountryRegression,
    PLSTransformer,
    LinearMultiTargetRegression,
    TimeWeightedWrapper,
)

from .random_effects import RandomEffects


def __getattr__(name):
    _torch_names = {
        "MultiLayerPerceptron",
        "TimeSeriesSampler",
        "MultiOutputSharpe",
        "MultiOutputMCR",
        "MLPRegressor",
    }
    if name in _torch_names:
        from .forecasting import __getattr__ as _fget
        return _fget(name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # splitters
    "ExpandingKFoldPanelSplit",
    "RollingKFoldPanelSplit",
    "RecencyKFoldPanelSplit",
    "ExpandingIncrementPanelSplit",
    "ExpandingFrequencyPanelSplit",
    "BasePanelSplit",
    # cv_tools
    "panel_cv_scores",
    # preprocessing
    "BasePanelSelector",
    "LassoSelector",
    "LarsSelector",
    "MapSelector",
    "KendallSignificanceSelector",
    "FactorAvailabilitySelector",
    # transformers
    "BasePanelScaler",
    "PanelMinMaxScaler",
    "PanelStandardScaler",
    "ZnScoreAverager",
    "PanelPCA",
    "PLSTransformer",
    # imputers
    "BaseImputer",
    "ConstantImputer",
    "CrossSectionalImputer",
    "EstimatorImputer",
    "GaussianConditionalImputer",
    # metrics
    "neg_mean_abs_corr",
    "panel_significance_probability",
    "sharpe_ratio",
    "sortino_ratio",
    "multi_output_sharpe",
    "multi_output_sortino",
    "regression_accuracy",
    "regression_balanced_accuracy",
    "create_panel_metric",
    "correlation_coefficient",
    "regression_mcc",
    # Sequential forecasting
    "SignalOptimizer",
    "ReturnForecaster",
    # forecasting
    "NaiveRegressor",
    "LADRegressor",
    "KNNClassifier",
    "SignWeightedLADRegressor",
    "TimeWeightedLADRegressor",
    "SignWeightedLinearRegression",
    "TimeWeightedLinearRegression",
    "ModifiedLinearRegression",
    "ModifiedSignWeightedLinearRegression",
    "ModifiedTimeWeightedLinearRegression",
    "VotingRegressor",
    "VotingClassifier",
    "GlobalLocalRegression",
    "TimeWeightedWrapper",
    "LinearMultiTargetRegression",
    "MLPRegressor",
    # market beta estimation
    "BetaEstimator",
    # regression system
    "LADRegressionSystem",
    "RidgeRegressionSystem",
    "LinearRegressionSystem",
    "CorrelationVolatilitySystem",
    # random effects
    "RandomEffects",
    # Meta estimators
    "ProbabilityEstimator",
    "FIExtractor",
    "DataFrameTransformer",
    "CountryByCountryRegression",
    # torch
    "MultiLayerPerceptron",
    "TimeSeriesSampler",
    "MultiOutputSharpe",
    "MultiOutputMCR",
]
