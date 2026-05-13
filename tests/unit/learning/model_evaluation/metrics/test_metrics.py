import numpy as np
import pandas as pd
import unittest
import scipy.stats as stats

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    r2_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
)
from macrosynergy.learning import (
    panel_significance_probability,
    regression_accuracy,
    regression_balanced_accuracy,
    sharpe_ratio,
    sharpe_stability_ratio_scorer,
    sortino_ratio,
    correlation_coefficient,
    create_panel_metric,
    regression_mcc,
)

from parameterized import parameterized


class TestMetrics(unittest.TestCase):
    def setUp(self):
        cids = ["AUD", "CAD", "GBP", "USD"]

        df_cids = pd.DataFrame(index=cids, columns=["earliest", "latest"])
        df_cids.loc["AUD"] = ["2020-06-01", "2020-12-31"]
        df_cids.loc["CAD"] = ["2020-06-01", "2020-12-31"]
        df_cids.loc["GBP"] = ["2020-09-01", "2020-12-31"]
        df_cids.loc["USD"] = ["2020-10-01", "2020-12-31"]

        tuples = []

        for cid in cids:
            # get list of all elidgible dates
            sdate = df_cids.loc[cid]["earliest"]
            edate = df_cids.loc[cid]["latest"]
            all_days = pd.date_range(sdate, edate)
            work_days = all_days[all_days.weekday < 5]
            for work_day in work_days:
                tuples.append((cid, work_day))

        n_predictions = len(tuples)

        # generate random predictions
        classifier_predictions = np.random.choice([0, 1], size=n_predictions)
        regressor_predictions = np.random.normal(size=n_predictions)

        # generate random true values
        classifier_true = np.random.choice([0, 1], size=n_predictions)
        regressor_true = np.random.normal(size=n_predictions)

        self.classifier_predictions = pd.Series(
            data=classifier_predictions,
            index=pd.MultiIndex.from_tuples(tuples, names=["cid", "real_date"]),
        )
        self.regressor_predictions = pd.Series(
            data=regressor_predictions,
            index=pd.MultiIndex.from_tuples(tuples, names=["cid", "real_date"]),
        )
        self.classifier_true = pd.Series(
            data=classifier_true,
            index=pd.MultiIndex.from_tuples(tuples, names=["cid", "real_date"]),
        )
        self.regressor_true = pd.Series(
            data=regressor_true,
            index=pd.MultiIndex.from_tuples(tuples, names=["cid", "real_date"]),
        )

    def test_types_accuracy(self):
        # y_true
        with self.assertRaises(TypeError):
            regression_accuracy("hello", self.regressor_predictions)
        with self.assertRaises(TypeError):
            regression_accuracy(
                np.zeros(len(self.regressor_predictions)), self.regressor_predictions
            )
        with self.assertRaises(ValueError):
            regression_accuracy(
                self.regressor_predictions.reset_index(drop=True),
                self.regressor_predictions,
            )
        # y_pred
        with self.assertRaises(TypeError):
            regression_accuracy(self.regressor_true, "hello")
        with self.assertRaises(ValueError):
            regression_accuracy(self.regressor_true, self.regressor_predictions.head(5))
        # type
        with self.assertRaises(ValueError):
            regression_accuracy(
                self.regressor_true, self.regressor_predictions, type="hello"
            )
        with self.assertRaises(TypeError):
            regression_accuracy(self.regressor_true, self.regressor_predictions, type=1)

    def test_valid_accuracy(self):
        # Test accuracy over the panel
        accuracy = regression_accuracy(
            self.regressor_true, self.regressor_predictions, type="panel"
        )
        true_signs = np.sign(self.regressor_true)
        pred_signs = np.sign(self.regressor_predictions)
        expected_result = accuracy_score(true_signs, pred_signs)
        self.assertEqual(
            accuracy,
            expected_result,
            "regression_accuracy should return the same value as sklearn.metrics.accuracy_score",
        )
        # Test accuracy over cross-sections
        unique_cross_sections = self.regressor_true.index.get_level_values(0).unique()
        accuracies = []
        for cross_section in unique_cross_sections:
            true_signs = np.sign(self.regressor_true.loc[cross_section])
            pred_signs = np.sign(self.regressor_predictions.loc[cross_section])
            accuracy = accuracy_score(true_signs, pred_signs)
            accuracies.append(accuracy)
        expected_result = np.mean(accuracies)
        accuracy = regression_accuracy(
            self.regressor_true, self.regressor_predictions, type="cross_section"
        )
        self.assertEqual(
            accuracy,
            expected_result,
            "regression_accuracy should return the same value as sklearn.metrics.accuracy_score, when applied over cross-sections",
        )
        # Test accuracy over time
        unique_dates = self.regressor_true.index.get_level_values(1).unique()
        accuracies = []
        for date in unique_dates:
            true_signs = np.sign(
                self.regressor_true[
                    self.regressor_true.index.get_level_values(1) == date
                ]
            )
            pred_signs = np.sign(
                self.regressor_predictions[
                    self.regressor_predictions.index.get_level_values(1) == date
                ]
            )
            accuracy = accuracy_score(true_signs, pred_signs)
            accuracies.append(accuracy)
        expected_result = np.mean(accuracies)
        accuracy = regression_accuracy(
            self.regressor_true, self.regressor_predictions, type="time_periods"
        )
        self.assertEqual(
            accuracy,
            expected_result,
            "regression_accuracy should return the same value as sklearn.metrics.accuracy_score, when applied over time periods",
        )

    def test_types_balanced_accuracy(self):
        # y_true
        with self.assertRaises(TypeError):
            regression_balanced_accuracy("hello", self.regressor_predictions)
        with self.assertRaises(TypeError):
            regression_balanced_accuracy(
                np.zeros(len(self.regressor_predictions)), self.regressor_predictions
            )
        with self.assertRaises(ValueError):
            regression_balanced_accuracy(
                self.regressor_predictions.reset_index(drop=True),
                self.regressor_predictions,
            )
        # y_pred
        with self.assertRaises(TypeError):
            regression_balanced_accuracy(self.regressor_true, "hello")
        with self.assertRaises(ValueError):
            regression_balanced_accuracy(
                self.regressor_true, self.regressor_predictions.head(5)
            )
        # type
        with self.assertRaises(ValueError):
            regression_balanced_accuracy(
                self.regressor_true, self.regressor_predictions, type="hello"
            )
        with self.assertRaises(TypeError):
            regression_balanced_accuracy(
                self.regressor_true, self.regressor_predictions, type=1
            )

    def test_valid_balanced_accuracy(self):
        # Test balanced accuracy over the panel
        balanced_accuracy = regression_balanced_accuracy(
            self.regressor_true, self.regressor_predictions, type="panel"
        )
        true_signs = np.sign(self.regressor_true)
        pred_signs = np.sign(self.regressor_predictions)
        expected_result = balanced_accuracy_score(true_signs, pred_signs)
        self.assertEqual(
            balanced_accuracy,
            expected_result,
            "regression_balanced_accuracy should return the same value as sklearn.metrics.balanced_accuracy_score",
        )
        # Test balanced accuracy over cross-sections
        unique_cross_sections = self.regressor_true.index.get_level_values(0).unique()
        balanced_accuracies = []
        for cross_section in unique_cross_sections:
            true_signs = np.sign(self.regressor_true.loc[cross_section])
            pred_signs = np.sign(self.regressor_predictions.loc[cross_section])
            balanced_accuracy = balanced_accuracy_score(true_signs, pred_signs)
            balanced_accuracies.append(balanced_accuracy)
        expected_result = np.mean(balanced_accuracies)
        balanced_accuracy = regression_balanced_accuracy(
            self.regressor_true, self.regressor_predictions, type="cross_section"
        )
        self.assertEqual(
            balanced_accuracy,
            expected_result,
            "regression_balanced_accuracy should return the same value as sklearn.metrics.balanced_accuracy_score, when applied over cross-sections",
        )
        # Test balanced accuracy over time
        unique_dates = self.regressor_true.index.get_level_values(1).unique()
        balanced_accuracies = []
        for date in unique_dates:
            true_signs = np.sign(
                self.regressor_true[
                    self.regressor_true.index.get_level_values(1) == date
                ]
            )
            pred_signs = np.sign(
                self.regressor_predictions[
                    self.regressor_predictions.index.get_level_values(1) == date
                ]
            )
            balanced_accuracy = balanced_accuracy_score(true_signs, pred_signs)
            balanced_accuracies.append(balanced_accuracy)
        expected_result = np.mean(balanced_accuracies)
        balanced_accuracy = regression_balanced_accuracy(
            self.regressor_true, self.regressor_predictions, type="time_periods"
        )
        self.assertEqual(
            balanced_accuracy,
            expected_result,
            "regression_balanced_accuracy should return the same value as sklearn.metrics.balanced_accuracy_score, when applied over time periods",
        )

    def test_types_panel_significance_probability(self):
        with self.assertRaises(TypeError):
            panel_significance_probability(
                "self.regressor_true", self.regressor_predictions
            )
        with self.assertRaises(TypeError):
            panel_significance_probability(
                self.regressor_true.reset_index(), self.regressor_predictions
            )
        with self.assertRaises(ValueError):
            panel_significance_probability(
                self.regressor_true, self.regressor_predictions[:-1]
            )

    def test_valid_panel_significance_probability(self):
        map_result = panel_significance_probability(
            self.regressor_true, self.regressor_predictions
        )
        self.assertIsInstance(
            map_result, float, "panel_significance_probability should return a float"
        )
        self.assertGreaterEqual(
            map_result,
            0,
            "panel_significance_probability should return a value greater or equal to zero",
        )
        self.assertLessEqual(
            map_result,
            1,
            "panel_significance_probability should return a value less or equal to one",
        )

    def test_types_binary_sharpe_ratio(self):
        # y_true
        with self.assertRaises(TypeError):
            sharpe_ratio("hello", self.regressor_predictions)
        with self.assertRaises(TypeError):
            sharpe_ratio(
                np.zeros(len(self.regressor_predictions)), self.regressor_predictions
            )
        with self.assertRaises(ValueError):
            sharpe_ratio(
                self.regressor_predictions.reset_index(drop=True),
                self.regressor_predictions,
            )
        # y_pred
        with self.assertRaises(TypeError):
            sharpe_ratio(self.regressor_true, "hello")
        with self.assertRaises(ValueError):
            sharpe_ratio(self.regressor_true, self.regressor_predictions.head(5))
        # type
        with self.assertRaises(ValueError):
            sharpe_ratio(self.regressor_true, self.regressor_predictions, type="hello")
        with self.assertRaises(TypeError):
            sharpe_ratio(self.regressor_true, self.regressor_predictions, type=1)

    def test_types_general_sharpe_ratio(self):
        # y_true
        with self.assertRaises(TypeError):
            sharpe_ratio("hello", self.regressor_predictions, binary=False)
        with self.assertRaises(TypeError):
            sharpe_ratio(
                np.zeros(len(self.regressor_predictions)),
                self.regressor_predictions,
                binary=False,
            )
        with self.assertRaises(ValueError):
            sharpe_ratio(
                self.regressor_predictions.reset_index(drop=True),
                self.regressor_predictions,
                binary=False,
            )
        # y_pred
        with self.assertRaises(TypeError):
            sharpe_ratio(self.regressor_true, "hello", binary=False)
        with self.assertRaises(ValueError):
            sharpe_ratio(
                self.regressor_true, self.regressor_predictions.head(5), binary=False
            )
        # type
        with self.assertRaises(ValueError):
            sharpe_ratio(
                self.regressor_true,
                self.regressor_predictions,
                type="hello",
                binary=False,
            )
        with self.assertRaises(TypeError):
            sharpe_ratio(
                self.regressor_true, self.regressor_predictions, type=1, binary=False
            )
        # binary
        with self.assertRaises(TypeError):
            sharpe_ratio(
                self.regressor_true, self.regressor_predictions, binary="hello"
            )
        # thresh
        with self.assertRaises(TypeError):
            sharpe_ratio(
                self.regressor_true,
                self.regressor_predictions,
                thresh="hello",
                binary=True,
            )
        with self.assertRaises(ValueError):
            sharpe_ratio(
                self.regressor_true, self.regressor_predictions, thresh=-1, binary=True
            )
        with self.assertRaises(ValueError):
            sharpe_ratio(
                self.regressor_true, self.regressor_predictions, thresh=0, binary=True
            )

    def test_valid_binary_sharpe_ratio(self):
        # Test sharpe ratio over the panel
        sharpe_result = sharpe_ratio(self.regressor_true, self.regressor_predictions)
        self.assertIsInstance(
            sharpe_result, float, "sharpe_ratio should return a float"
        )
        portfolio_returns = np.where(
            self.regressor_predictions > 0, self.regressor_true, -self.regressor_true
        )
        self.assertEqual(
            sharpe_result,
            np.mean(portfolio_returns) / np.std(portfolio_returns),
            "sharpe_ratio should return the same value as the ratio of the mean of the portfolio returns to the standard deviation of the portfolio returns",
        )
        # Test sharpe ratio over cross-sections
        unique_cross_sections = self.regressor_true.index.get_level_values(0).unique()
        sharpes = []
        for cross_section in unique_cross_sections:
            portfolio_returns = np.where(
                self.regressor_predictions.loc[cross_section] > 0,
                self.regressor_true.loc[cross_section],
                -self.regressor_true.loc[cross_section],
            )
            sharpe = np.mean(portfolio_returns) / np.std(portfolio_returns)
            sharpes.append(sharpe)
        expected_result = np.mean(sharpes)
        sharpe = sharpe_ratio(
            self.regressor_true, self.regressor_predictions, type="cross_section"
        )
        self.assertEqual(
            sharpe,
            expected_result,
            "sharpe_ratio should return the same value as the ratio of the mean of the portfolio returns to the standard deviation of the portfolio returns, when applied over cross-sections",
        )
        # Test sharpe ratio over time
        unique_dates = self.regressor_true.index.get_level_values(1).unique()
        sharpes = []
        for date in unique_dates:
            portfolio_returns = np.where(
                self.regressor_predictions[
                    self.regressor_predictions.index.get_level_values(1) == date
                ]
                > 0,
                self.regressor_true[
                    self.regressor_true.index.get_level_values(1) == date
                ],
                -self.regressor_true[
                    self.regressor_true.index.get_level_values(1) == date
                ],
            )
            sharpe = np.mean(portfolio_returns) / np.std(portfolio_returns)
            sharpes.append(sharpe)
        expected_result = np.mean(sharpes)
        sharpe = sharpe_ratio(
            self.regressor_true, self.regressor_predictions, type="time_periods"
        )
        self.assertEqual(
            sharpe,
            expected_result,
            "sharpe_ratio should return the same value as the ratio of the mean of the portfolio returns to the standard deviation of the portfolio returns, when applied over time periods",
        )

    @parameterized.expand([None, 1, 3, 5.1])
    def test_valid_general_sharpe_ratio(self, thresh):
        # Calculate portfolio weights
        if thresh is not None:
            portfolio_returns = (
                self.regressor_predictions
                / self.regressor_predictions.groupby(level=1).transform("std")
            ).clip(-thresh, thresh) * self.regressor_true
        else:
            portfolio_returns = (
                self.regressor_predictions
                / self.regressor_predictions.groupby(level=1).transform("std")
            ) * self.regressor_true
        # Test sharpe ratio over the panel
        sharpe_result = sharpe_ratio(
            self.regressor_true, self.regressor_predictions, binary=False, thresh=thresh
        )
        self.assertIsInstance(
            sharpe_result, float, "sharpe_ratio should return a float"
        )
        numerator = np.mean(portfolio_returns)
        denominator = np.std(portfolio_returns)
        if denominator == 0 or np.isnan(denominator):
            denominator = 1
        self.assertEqual(
            sharpe_result,
            numerator / denominator,
            "sharpe_ratio should return the same value as the ratio of the mean of the portfolio returns to the standard deviation of the portfolio returns",
        )
        # Test sharpe ratio over cross-sections
        unique_cross_sections = self.regressor_true.index.get_level_values(0).unique()
        sharpes = []
        for cross_section in unique_cross_sections:
            sharpe = np.mean(portfolio_returns.loc[cross_section]) / np.std(
                portfolio_returns.loc[cross_section]
            )
            numerator = np.mean(portfolio_returns.loc[cross_section])
            denominator = np.std(portfolio_returns.loc[cross_section])
            if denominator == 0 or np.isnan(denominator):
                denominator = 1
            sharpe = numerator / denominator
            sharpes.append(sharpe)
        expected_result = np.mean(sharpes)
        sharpe = sharpe_ratio(
            self.regressor_true,
            self.regressor_predictions,
            type="cross_section",
            binary=False,
            thresh=thresh,
        )
        self.assertEqual(
            sharpe,
            expected_result,
            "sharpe_ratio should return the same value as the ratio of the mean of the portfolio returns to the standard deviation of the portfolio returns, when applied over cross-sections",
        )
        # Test sharpe ratio over time
        unique_dates = self.regressor_true.index.get_level_values(1).unique()
        sharpes = []
        for date in unique_dates:
            numerator = np.mean(
                portfolio_returns[portfolio_returns.index.get_level_values(1) == date]
            )
            denominator = np.std(
                portfolio_returns[portfolio_returns.index.get_level_values(1) == date]
            )
            if denominator == 0 or np.isnan(denominator):
                denominator = 1
            sharpe = numerator / denominator
            sharpes.append(sharpe)
        expected_result = np.mean(sharpes)
        sharpe = sharpe_ratio(
            self.regressor_true,
            self.regressor_predictions,
            type="time_periods",
            binary=False,
            thresh=thresh,
        )
        self.assertEqual(
            sharpe,
            expected_result,
            "sharpe_ratio should return the same value as the ratio of the mean of the portfolio returns to the standard deviation of the portfolio returns, when applied over time periods",
        )

    def test_types_binary_sortino_ratio(self):
        # y_true
        with self.assertRaises(TypeError):
            sortino_ratio("hello", self.regressor_predictions)
        with self.assertRaises(TypeError):
            sortino_ratio(
                np.zeros(len(self.regressor_predictions)), self.regressor_predictions
            )
        with self.assertRaises(ValueError):
            sortino_ratio(
                self.regressor_predictions.reset_index(drop=True),
                self.regressor_predictions,
            )
        # y_pred
        with self.assertRaises(TypeError):
            sortino_ratio(self.regressor_true, "hello")
        with self.assertRaises(ValueError):
            sortino_ratio(self.regressor_true, self.regressor_predictions.head(5))
        # type
        with self.assertRaises(ValueError):
            sortino_ratio(self.regressor_true, self.regressor_predictions, type="hello")
        with self.assertRaises(TypeError):
            sortino_ratio(self.regressor_true, self.regressor_predictions, type=1)

    def test_valid_binary_sortino_ratio(self):
        # Test sortino ratio over the panel
        sortino_result = sortino_ratio(self.regressor_true, self.regressor_predictions)
        self.assertIsInstance(
            sortino_result, float, "sortino_ratio should return a float"
        )
        portfolio_returns = np.where(
            self.regressor_predictions > 0, self.regressor_true, -self.regressor_true
        )
        numerator = np.mean(portfolio_returns)
        denominator = np.std(portfolio_returns[portfolio_returns < 0])
        if denominator == 0 or np.isnan(denominator):
            denominator = 1
        self.assertEqual(
            sortino_result,
            numerator / denominator,
            "sortino_ratio should return the same value as the ratio of the mean of the portfolio returns to the standard deviation of the negative portfolio returns",
        )
        # Test sortino ratio over cross-sections
        unique_cross_sections = self.regressor_true.index.get_level_values(0).unique()
        sortinos = []
        for cross_section in unique_cross_sections:
            portfolio_returns = np.where(
                self.regressor_predictions.loc[cross_section] > 0,
                self.regressor_true.loc[cross_section],
                -self.regressor_true.loc[cross_section],
            )
            numerator = np.mean(portfolio_returns)
            denominator = np.std(portfolio_returns[portfolio_returns < 0])
            if denominator == 0 or np.isnan(denominator):
                denominator = 1
            sortino = numerator / denominator
            sortinos.append(sortino)
        expected_result = np.mean(sortinos)
        sortino = sortino_ratio(
            self.regressor_true, self.regressor_predictions, type="cross_section"
        )
        self.assertEqual(
            sortino,
            expected_result,
            "sortino_ratio should return the same value as the ratio of the mean of the portfolio returns to the standard deviation of the negative portfolio returns, when applied over cross-sections",
        )
        # Test sortino ratio over time
        unique_dates = self.regressor_true.index.get_level_values(1).unique()
        sortinos = []
        for date in unique_dates:
            portfolio_returns = np.where(
                self.regressor_predictions[
                    self.regressor_predictions.index.get_level_values(1) == date
                ]
                > 0,
                self.regressor_true[
                    self.regressor_true.index.get_level_values(1) == date
                ],
                -self.regressor_true[
                    self.regressor_true.index.get_level_values(1) == date
                ],
            )
            numerator = np.mean(portfolio_returns)
            denominator = np.std(portfolio_returns[portfolio_returns < 0])
            if denominator == 0 or np.isnan(denominator):
                denominator = 1
            sortino = numerator / denominator
            sortinos.append(sortino)
        expected_result = np.mean(sortinos)
        sortino = sortino_ratio(
            self.regressor_true, self.regressor_predictions, type="time_periods"
        )
        self.assertEqual(
            sortino,
            expected_result,
            "sortino_ratio should return the same value as the ratio of the mean of the portfolio returns to the standard deviation of the negative portfolio returns, when applied over time periods",
        )

    @parameterized.expand([None, 1, 3, 5.1])
    def test_valid_general_sortino_ratio(self, thresh):
        # Calculate portfolio weights
        if thresh is not None:
            portfolio_returns = (
                self.regressor_predictions
                / self.regressor_predictions.groupby(level=1).transform("std")
            ).clip(-thresh, thresh) * self.regressor_true
        else:
            portfolio_returns = (
                self.regressor_predictions
                / self.regressor_predictions.groupby(level=1).transform("std")
            ) * self.regressor_true
        # Test sortino ratio over the panel
        sortino_result = sortino_ratio(
            self.regressor_true, self.regressor_predictions, binary=False, thresh=thresh
        )
        self.assertIsInstance(
            sortino_result, float, "sortino_ratio should return a float"
        )
        numerator = np.mean(portfolio_returns)
        denominator = np.std(portfolio_returns[portfolio_returns < 0])
        if denominator == 0 or np.isnan(denominator):
            denominator = 1
        self.assertEqual(
            sortino_result,
            numerator / denominator,
            "sortino_ratio should return the same value as the ratio of the mean of the portfolio returns to the downside deviation of the portfolio returns",
        )
        # Test sortino ratio over cross-sections
        unique_cross_sections = self.regressor_true.index.get_level_values(0).unique()
        sortinos = []
        for cross_section in unique_cross_sections:
            numerator = np.mean(portfolio_returns.loc[cross_section])
            denominator = np.std(
                portfolio_returns.loc[cross_section][
                    portfolio_returns.loc[cross_section] < 0
                ]
            )
            if denominator == 0 or np.isnan(denominator):
                denominator = 1
            sortino = numerator / denominator
            sortinos.append(sortino)
        expected_result = np.mean(sortinos)
        sortino = sortino_ratio(
            self.regressor_true,
            self.regressor_predictions,
            type="cross_section",
            binary=False,
            thresh=thresh,
        )
        self.assertEqual(
            sortino,
            expected_result,
            "sortino_ratio should return the same value as the ratio of the mean of the portfolio returns to the standard deviation of the portfolio returns, when applied over cross-sections",
        )
        # Test sortino ratio over time
        unique_dates = self.regressor_true.index.get_level_values(1).unique()
        sortinos = []
        for date in unique_dates:
            numerator = np.mean(
                portfolio_returns[portfolio_returns.index.get_level_values(1) == date]
            )
            denominator = np.std(
                portfolio_returns[portfolio_returns.index.get_level_values(1) == date][
                    portfolio_returns[
                        portfolio_returns.index.get_level_values(1) == date
                    ]
                    < 0
                ]
            )
            if denominator == 0 or np.isnan(denominator):
                denominator = 1
            sortino = numerator / denominator
            sortinos.append(sortino)
        expected_result = np.mean(sortinos)
        sortino = sortino_ratio(
            self.regressor_true,
            self.regressor_predictions,
            type="time_periods",
            binary=False,
            thresh=thresh,
        )
        self.assertEqual(
            sortino,
            expected_result,
            "sortino_ratio should return the same value as the ratio of the mean of the portfolio returns to the standard deviation of the portfolio returns, when applied over time periods",
        )

    def test_types_correlation_coefficient(self):
        # y_true
        with self.assertRaises(TypeError):
            correlation_coefficient("hello", self.regressor_predictions)
        with self.assertRaises(TypeError):
            correlation_coefficient(
                "hello", self.regressor_predictions, correlation_type="spearman"
            )
        with self.assertRaises(TypeError):
            correlation_coefficient(
                "hello", self.regressor_predictions, correlation_type="kendall"
            )
        with self.assertRaises(TypeError):
            correlation_coefficient(
                np.zeros(len(self.regressor_predictions)), self.regressor_predictions
            )
        with self.assertRaises(TypeError):
            correlation_coefficient(
                np.zeros(len(self.regressor_predictions)),
                self.regressor_predictions,
                correlation_type="spearman",
            )
        with self.assertRaises(TypeError):
            correlation_coefficient(
                np.zeros(len(self.regressor_predictions)),
                self.regressor_predictions,
                correlation_type="kendall",
            )
        with self.assertRaises(ValueError):
            correlation_coefficient(
                self.regressor_predictions.reset_index(drop=True),
                self.regressor_predictions,
            )
        with self.assertRaises(ValueError):
            correlation_coefficient(
                self.regressor_predictions.reset_index(drop=True),
                self.regressor_predictions,
                correlation_type="spearman",
            )
        with self.assertRaises(ValueError):
            correlation_coefficient(
                self.regressor_predictions.reset_index(drop=True),
                self.regressor_predictions,
                correlation_type="kendall",
            )
        # y_pred
        with self.assertRaises(TypeError):
            correlation_coefficient(self.regressor_true, "hello")
        with self.assertRaises(TypeError):
            correlation_coefficient(
                self.regressor_true, "hello", correlation_type="spearman"
            )
        with self.assertRaises(TypeError):
            correlation_coefficient(
                self.regressor_true, "hello", correlation_type="kendall"
            )
        with self.assertRaises(ValueError):
            correlation_coefficient(
                self.regressor_true, self.regressor_predictions.head(5)
            )
        with self.assertRaises(ValueError):
            correlation_coefficient(
                self.regressor_true,
                self.regressor_predictions.head(5),
                correlation_type="spearman",
            )
        with self.assertRaises(ValueError):
            correlation_coefficient(
                self.regressor_true,
                self.regressor_predictions.head(5),
                correlation_type="kendall",
            )
        # type
        with self.assertRaises(ValueError):
            correlation_coefficient(
                self.regressor_true, self.regressor_predictions, type="hello"
            )
        with self.assertRaises(ValueError):
            correlation_coefficient(
                self.regressor_true,
                self.regressor_predictions,
                type="hello",
                correlation_type="spearman",
            )
        with self.assertRaises(ValueError):
            correlation_coefficient(
                self.regressor_true,
                self.regressor_predictions,
                type="hello",
                correlation_type="kendall",
            )
        with self.assertRaises(TypeError):
            correlation_coefficient(
                self.regressor_true, self.regressor_predictions, type=1
            )
        with self.assertRaises(TypeError):
            correlation_coefficient(
                self.regressor_true,
                self.regressor_predictions,
                type=1,
                correlation_type="spearman",
            )
        with self.assertRaises(TypeError):
            correlation_coefficient(
                self.regressor_true,
                self.regressor_predictions,
                type=1,
                correlation_type="kendall",
            )

    def test_valid_correlation_coefficient(self):
        # Test correlation coefficients over the panel
        pearson = correlation_coefficient(
            self.regressor_true,
            self.regressor_predictions,
            correlation_type="pearson",
            type="panel",
        )
        expected_pearson = stats.pearsonr(
            self.regressor_true, self.regressor_predictions
        )[
            0
        ]  # statisic
        self.assertEqual(
            pearson,
            expected_pearson,
            "correlation_coefficient should return the same value as scipy.stats.pearsonr",
        )
        spearman = correlation_coefficient(
            self.regressor_true,
            self.regressor_predictions,
            correlation_type="spearman",
            type="panel",
        )
        expected_spearman = stats.spearmanr(
            self.regressor_true, self.regressor_predictions
        ).correlation
        self.assertEqual(
            spearman,
            expected_spearman,
            "correlation_coefficient should return the same value as scipy.stats.spearmanr",
        )
        kendall = correlation_coefficient(
            self.regressor_true,
            self.regressor_predictions,
            correlation_type="kendall",
            type="panel",
        )
        expected_kendall = stats.kendalltau(
            self.regressor_true, self.regressor_predictions
        ).correlation
        self.assertEqual(
            kendall,
            expected_kendall,
            "correlation_coefficient should return the same value as scipy.stats.kendalltau",
        )
        # Test correlation coefficients over cross-sections
        unique_cross_sections = self.regressor_true.index.get_level_values(0).unique()
        pearsons = []
        spearmans = []
        kendalls = []
        for cross_section in unique_cross_sections:
            pearson = correlation_coefficient(
                self.regressor_true[
                    self.regressor_true.index.get_level_values(0) == cross_section
                ],
                self.regressor_predictions.loc[cross_section],
                correlation_type="pearson",
            )
            expected_pearson = stats.pearsonr(
                self.regressor_true.loc[cross_section],
                self.regressor_predictions.loc[cross_section],
            )[0]
            self.assertEqual(
                pearson,
                expected_pearson,
                "correlation_coefficient should return the same value as scipy.stats.pearsonr, when applied over cross-sections",
            )
            pearsons.append(pearson)
            spearman = correlation_coefficient(
                self.regressor_true[
                    self.regressor_true.index.get_level_values(0) == cross_section
                ],
                self.regressor_predictions.loc[cross_section],
                correlation_type="spearman",
            )
            expected_spearman = stats.spearmanr(
                self.regressor_true.loc[cross_section],
                self.regressor_predictions.loc[cross_section],
            ).correlation
            self.assertEqual(
                spearman,
                expected_spearman,
                "correlation_coefficient should return the same value as scipy.stats.spearmanr, when applied over cross-sections",
            )
            spearmans.append(spearman)
            kendall = correlation_coefficient(
                self.regressor_true[
                    self.regressor_true.index.get_level_values(0) == cross_section
                ],
                self.regressor_predictions.loc[cross_section],
                correlation_type="kendall",
            )
            expected_kendall = stats.kendalltau(
                self.regressor_true.loc[cross_section],
                self.regressor_predictions.loc[cross_section],
            ).correlation
            self.assertEqual(
                kendall,
                expected_kendall,
                "correlation_coefficient should return the same value as scipy.stats.kendalltau, when applied over cross-sections",
            )
            kendalls.append(kendall)
        expected_pearson = np.mean(pearsons)
        expected_spearman = np.mean(spearmans)
        expected_kendall = np.mean(kendalls)
        pearson = correlation_coefficient(
            self.regressor_true,
            self.regressor_predictions,
            correlation_type="pearson",
            type="cross_section",
        )
        spearman = correlation_coefficient(
            self.regressor_true,
            self.regressor_predictions,
            correlation_type="spearman",
            type="cross_section",
        )
        kendall = correlation_coefficient(
            self.regressor_true,
            self.regressor_predictions,
            correlation_type="kendall",
            type="cross_section",
        )
        self.assertEqual(
            pearson,
            expected_pearson,
            "correlation_coefficient should return the same value as scipy.stats.pearsonr, when applied over cross-sections",
        )
        self.assertEqual(
            spearman,
            expected_spearman,
            "correlation_coefficient should return the same value as scipy.stats.spearmanr, when applied over cross-sections",
        )
        self.assertEqual(
            kendall,
            expected_kendall,
            "correlation_coefficient should return the same value as scipy.stats.kendalltau, when applied over cross-sections",
        )
        # Test correlation coefficients over time
        unique_dates = self.regressor_true.index.get_level_values(1).unique()
        pearsons = []
        spearmans = []
        kendalls = []
        for date in unique_dates:
            pearson = correlation_coefficient(
                self.regressor_true[
                    self.regressor_true.index.get_level_values(1) == date
                ],
                self.regressor_predictions[
                    self.regressor_predictions.index.get_level_values(1) == date
                ],
                correlation_type="pearson",
            )
            expected_pearson = stats.pearsonr(
                self.regressor_true[
                    self.regressor_true.index.get_level_values(1) == date
                ],
                self.regressor_predictions[
                    self.regressor_predictions.index.get_level_values(1) == date
                ],
            )[
                0
            ]  # statistic
            self.assertEqual(
                pearson,
                expected_pearson,
                "correlation_coefficient should return the same value as scipy.stats.pearsonr, when applied over time periods",
            )
            pearsons.append(pearson)
            spearman = correlation_coefficient(
                self.regressor_true[
                    self.regressor_true.index.get_level_values(1) == date
                ],
                self.regressor_predictions[
                    self.regressor_predictions.index.get_level_values(1) == date
                ],
                correlation_type="spearman",
            )
            expected_spearman = stats.spearmanr(
                self.regressor_true[
                    self.regressor_true.index.get_level_values(1) == date
                ],
                self.regressor_predictions[
                    self.regressor_predictions.index.get_level_values(1) == date
                ],
            ).correlation
            self.assertEqual(
                spearman,
                expected_spearman,
                "correlation_coefficient should return the same value as scipy.stats.spearmanr, when applied over time periods",
            )
            spearmans.append(spearman)
            kendall = correlation_coefficient(
                self.regressor_true[
                    self.regressor_true.index.get_level_values(1) == date
                ],
                self.regressor_predictions[
                    self.regressor_predictions.index.get_level_values(1) == date
                ],
                correlation_type="kendall",
            )
            expected_kendall = stats.kendalltau(
                self.regressor_true[
                    self.regressor_true.index.get_level_values(1) == date
                ],
                self.regressor_predictions[
                    self.regressor_predictions.index.get_level_values(1) == date
                ],
            ).correlation
            self.assertEqual(
                kendall,
                expected_kendall,
                "correlation_coefficient should return the same value as scipy.stats.kendalltau, when applied over time periods",
            )
            kendalls.append(kendall)
        expected_pearson = np.mean(pearsons)
        expected_spearman = np.mean(spearmans)
        expected_kendall = np.mean(kendalls)
        pearson = correlation_coefficient(
            self.regressor_true,
            self.regressor_predictions,
            correlation_type="pearson",
            type="time_periods",
        )
        spearman = correlation_coefficient(
            self.regressor_true,
            self.regressor_predictions,
            correlation_type="spearman",
            type="time_periods",
        )
        kendall = correlation_coefficient(
            self.regressor_true,
            self.regressor_predictions,
            correlation_type="kendall",
            type="time_periods",
        )
        self.assertEqual(
            pearson,
            expected_pearson,
            "correlation_coefficient should return the same value as scipy.stats.pearsonr, when applied over time periods",
        )
        self.assertEqual(
            spearman,
            expected_spearman,
            "correlation_coefficient should return the same value as scipy.stats.spearmanr, when applied over time periods",
        )
        self.assertEqual(
            kendall,
            expected_kendall,
            "correlation_coefficient should return the same value as scipy.stats.kendalltau, when applied over time periods",
        )

    def test_types_create_panel_metric(self):
        # y_true
        with self.assertRaises(TypeError):
            create_panel_metric("hello", self.regressor_predictions, r2_score)
        with self.assertRaises(TypeError):
            create_panel_metric(
                np.zeros(len(self.regressor_predictions)),
                self.regressor_predictions,
                r2_score,
            )
        with self.assertRaises(ValueError):
            create_panel_metric(
                self.regressor_predictions.reset_index(drop=True),
                self.regressor_predictions,
                r2_score,
            )
        # y_pred
        with self.assertRaises(TypeError):
            create_panel_metric(self.regressor_true, "hello", r2_score)
        with self.assertRaises(ValueError):
            create_panel_metric(
                self.regressor_true, self.regressor_predictions.head(5), r2_score
            )
        # metric
        with self.assertRaises(TypeError):
            create_panel_metric(
                self.regressor_true, self.regressor_predictions, "hello"
            )
        with self.assertRaises(ValueError):
            create_panel_metric(
                self.regressor_true, self.regressor_predictions, np.mean
            )
        # type
        with self.assertRaises(TypeError):
            create_panel_metric(
                self.regressor_true, self.regressor_predictions, r2_score, type=1
            )
        with self.assertRaises(ValueError):
            create_panel_metric(
                self.regressor_true, self.regressor_predictions, r2_score, type="siu"
            )

    @parameterized.expand(
        [r2_score, mean_absolute_error, mean_absolute_percentage_error]
    )
    def test_valid_create_panel_metric_regr(self, metric):
        # Test metrics over the panel
        metric_result = create_panel_metric(
            self.regressor_true, self.regressor_predictions, metric
        )
        expected_result = metric(self.regressor_true, self.regressor_predictions)
        self.assertEqual(
            metric_result,
            expected_result,
            "create_panel_metric should return the same value as the metric function passed as argument",
        )
        # Test metrics over cross-sections
        unique_cross_sections = self.regressor_true.index.get_level_values(0).unique()
        metrics = []
        for cross_section in unique_cross_sections:
            metric_value = metric(
                self.regressor_true.loc[cross_section],
                self.regressor_predictions.loc[cross_section],
            )
            metrics.append(metric_value)
        expected_result = np.mean(metrics)
        metric_result = create_panel_metric(
            self.regressor_true,
            self.regressor_predictions,
            metric,
            type="cross_section",
        )
        self.assertEqual(
            metric_result,
            expected_result,
            "create_panel_metric should return the same value as the metric function passed as argument, when applied over cross-sections",
        )
        # Test metrics over time
        unique_dates = self.regressor_true.index.get_level_values(1).unique()
        metrics = []
        for date in unique_dates:
            metric_value = metric(
                self.regressor_true[
                    self.regressor_true.index.get_level_values(1) == date
                ],
                self.regressor_predictions[
                    self.regressor_predictions.index.get_level_values(1) == date
                ],
            )
            metrics.append(metric_value)
        expected_result = np.mean(metrics)
        metric_result = create_panel_metric(
            self.regressor_true, self.regressor_predictions, metric, type="time_periods"
        )
        self.assertEqual(
            metric_result,
            expected_result,
            "create_panel_metric should return the same value as the metric function passed as argument, when applied over time periods",
        )

    @parameterized.expand([accuracy_score, balanced_accuracy_score])
    def test_valid_create_panel_metric_clf(self, metric):
        # Test metrics over the panel
        metric_result = create_panel_metric(
            self.classifier_true, self.classifier_predictions, metric
        )
        expected_result = metric(self.classifier_true, self.classifier_predictions)
        self.assertEqual(
            metric_result,
            expected_result,
            "create_panel_metric should return the same value as the metric function passed as argument",
        )
        # Test metrics over cross-sections
        unique_cross_sections = self.classifier_true.index.get_level_values(0).unique()
        metrics = []
        for cross_section in unique_cross_sections:
            metric_value = metric(
                self.classifier_true.loc[cross_section],
                self.classifier_predictions.loc[cross_section],
            )
            metrics.append(metric_value)
        expected_result = np.mean(metrics)
        metric_result = create_panel_metric(
            self.classifier_true,
            self.classifier_predictions,
            metric,
            type="cross_section",
        )
        self.assertEqual(
            metric_result,
            expected_result,
            "create_panel_metric should return the same value as the metric function passed as argument, when applied over cross-sections",
        )
        # Test metrics over time
        unique_dates = self.classifier_true.index.get_level_values(1).unique()
        metrics = []
        for date in unique_dates:
            metric_value = metric(
                self.classifier_true[
                    self.classifier_true.index.get_level_values(1) == date
                ],
                self.classifier_predictions[
                    self.classifier_predictions.index.get_level_values(1) == date
                ],
            )
            metrics.append(metric_value)
        expected_result = np.mean(metrics)
        metric_result = create_panel_metric(
            self.classifier_true,
            self.classifier_predictions,
            metric,
            type="time_periods",
        )
        self.assertEqual(
            metric_result,
            expected_result,
            "create_panel_metric should return the same value as the metric function passed as argument, when applied over time periods",
        )


class TestSharpeStabilityRatioScorer(unittest.TestCase):
    def setUp(self):
        cids = ["AUD", "CAD", "GBP", "USD"]
        # 60 months of monthly observations per cross-section so that the
        # default window=12 SSR can be computed (needs >= 2*window + 2).
        dates = pd.date_range("2010-01-31", periods=60, freq="ME")
        index = pd.MultiIndex.from_product(
            [cids, dates], names=["cid", "real_date"]
        )

        rng = np.random.default_rng(42)
        self.y_true = pd.Series(
            rng.normal(0.01, 0.05, size=len(index)), index=index
        ).sort_index()
        self.y_pred = pd.Series(
            rng.normal(0.0, 1.0, size=len(index)), index=index
        ).sort_index()

        # A short panel that should trigger the insufficient-data fallback.
        short_dates = pd.date_range("2010-01-31", periods=10, freq="ME")
        short_index = pd.MultiIndex.from_product(
            [cids, short_dates], names=["cid", "real_date"]
        )
        self.y_true_short = pd.Series(
            rng.normal(0.01, 0.05, size=len(short_index)), index=short_index
        ).sort_index()
        self.y_pred_short = pd.Series(
            rng.normal(0.0, 1.0, size=len(short_index)), index=short_index
        ).sort_index()

    def test_types(self):
        # y_true must be a multi-indexed pd.Series
        with self.assertRaises(TypeError):
            sharpe_stability_ratio_scorer("hello", self.y_pred)
        with self.assertRaises(TypeError):
            sharpe_stability_ratio_scorer(
                np.zeros(len(self.y_pred)), self.y_pred
            )
        with self.assertRaises(ValueError):
            sharpe_stability_ratio_scorer(
                self.y_true.reset_index(drop=True), self.y_pred
            )

        # y_pred must be array-like and matching length
        with self.assertRaises(TypeError):
            sharpe_stability_ratio_scorer(self.y_true, "hello")
        with self.assertRaises(ValueError):
            sharpe_stability_ratio_scorer(self.y_true, self.y_pred.head(5))

    def test_returns_float(self):
        result = sharpe_stability_ratio_scorer(self.y_true, self.y_pred)
        self.assertIsInstance(result, float)
        self.assertTrue(np.isfinite(result))

    def test_accepts_ndarray_y_pred(self):
        result_series = sharpe_stability_ratio_scorer(self.y_true, self.y_pred)
        result_array = sharpe_stability_ratio_scorer(
            self.y_true, self.y_pred.values
        )
        self.assertAlmostEqual(result_series, result_array)

    def test_insufficient_data_returns_zero(self):
        # Underlying sharpe_stability_ratio returns NaN when the cross-sectional
        # mean time series is too short; the scorer should map that to 0.0.
        result = sharpe_stability_ratio_scorer(
            self.y_true_short, self.y_pred_short
        )
        self.assertEqual(result, 0.0)

    def test_matches_underlying_function(self):
        from macrosynergy.pnl import sharpe_stability_ratio

        pr = pd.Series(
            np.where(
                self.y_pred.values > 0, self.y_true.values, -self.y_true.values
            ),
            index=self.y_true.index,
        )
        ts = pr.groupby(level=1).mean().sort_index().dropna()
        expected = sharpe_stability_ratio(
            ts, window=12, annualization_factor=12, min_periods=12
        )

        result = sharpe_stability_ratio_scorer(
            self.y_true, self.y_pred, window=12, annualization_factor=12
        )
        self.assertAlmostEqual(result, float(expected))

    def test_sign_flips_with_inverted_predictions(self):
        ssr_pos = sharpe_stability_ratio_scorer(self.y_true, self.y_pred)
        ssr_neg = sharpe_stability_ratio_scorer(self.y_true, -self.y_pred)
        # Flipping the sign of every prediction flips the per-period return
        # sign, which flips the mean rolling Sharpe and therefore the SSR.
        self.assertAlmostEqual(ssr_pos, -ssr_neg)

    def test_usable_as_sklearn_scorer(self):
        from sklearn.metrics import make_scorer

        scorer = make_scorer(sharpe_stability_ratio_scorer)
        self.assertTrue(callable(scorer))


if __name__ == "__main__":
    Test = TestMetrics()
    Test.setUp()
    Test.test_valid_panel_significance_probability()
