import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline

from macrosynergy.learning import GlobalLocalRegression, ReturnClusterer

CIDS = ["AUD", "CAD", "GBP", "USD"]
CID_TO_GROUP = {"AUD": 0, "CAD": 0, "GBP": 1, "USD": 1}
CLUSTER_MAP = {"dollar": ["AUD", "CAD", "USD"], "europe": ["GBP", "EUR"]}


@pytest.fixture
def panel():
    """AUD and CAD returns follow one common factor, GBP and USD another."""
    rng = np.random.default_rng(0)

    dates = pd.bdate_range("2010-01-01", periods=120)
    index = pd.MultiIndex.from_product([CIDS, dates], names=["cid", "real_date"])
    factors = rng.normal(size=(2, len(dates)))

    y = pd.Series(
        np.concatenate([factors[CID_TO_GROUP[cid]] for cid in CIDS])
        + 0.3 * rng.normal(size=len(index)),
        index=index,
    )
    X = pd.DataFrame(
        rng.normal(size=(len(index), 2)), index=index, columns=["CPI", "GROWTH"]
    )

    return X, y


class TestFit:
    @pytest.mark.parametrize("target", ["series", "frame", "array"])
    def test_clusters_correlated_cross_sections(self, panel, target):
        X, y = panel
        y = {"series": y, "frame": y.to_frame(), "array": y.to_numpy()}[target]

        rc = ReturnClusterer(col_name="GROUP", n_clusters=2).fit(X, y)

        assert rc.clusters_ == {0: ["AUD", "CAD"], 1: ["GBP", "USD"]}
        assert rc.cid_to_group_ == CID_TO_GROUP

    def test_relative_returns_cluster_deviations_from_the_mean(self, panel):
        X, y = panel
        # Scaling AUD leaves raw correlations unchanged, but makes AUD dominate the
        # cross-sectional mean, so the other deviations from it all move together
        y = y.where(y.index.get_level_values(0) != "AUD", 20 * y)

        raw = ReturnClusterer(col_name="GROUP", n_clusters=2).fit(X, y)
        relative = ReturnClusterer(
            col_name="GROUP", n_clusters=2, relative_returns=True
        ).fit(X, y)

        assert raw.clusters_ == {0: ["AUD", "CAD"], 1: ["GBP", "USD"]}
        assert relative.clusters_ == {0: ["AUD"], 1: ["CAD", "GBP", "USD"]}

    def test_matches_global_local_regression(self, panel):
        X, y = panel

        rc = ReturnClusterer(col_name="GROUP", n_clusters=3).fit(X, y)
        glr = GlobalLocalRegression(groups="cluster", n_clusters=3).fit(X, y)

        assert rc.clusters_ == glr.clusters_

    def test_too_many_clusters_falls_back_to_one(self, panel):
        X, y = panel

        with pytest.warns(UserWarning, match="Falling back"):
            rc = ReturnClusterer(col_name="GROUP", n_clusters=5).fit(X, y)

        assert rc.clusters_ == {0: CIDS}

    def test_single_cross_section_forms_one_cluster(self, panel):
        X, y = panel

        rc = ReturnClusterer(col_name="GROUP", n_clusters=1).fit(
            X.loc[["AUD"]], y.loc[["AUD"]]
        )

        assert rc.clusters_ == {0: ["AUD"]}

    def test_cluster_map_needs_no_target(self, panel):
        X, _ = panel

        rc = ReturnClusterer(col_name="GROUP", cluster_map=CLUSTER_MAP).fit(X)

        assert rc.clusters_ == CLUSTER_MAP
        assert rc.cid_to_group_ == {
            "AUD": "dollar",
            "CAD": "dollar",
            "USD": "dollar",
            "GBP": "europe",
            "EUR": "europe",
        }

    @pytest.mark.parametrize(
        "params, error, match",
        [
            ({"n_clusters": 2.0}, TypeError, "n_clusters"),
            ({"cluster_map": {1: CIDS}}, TypeError, "group names"),
            ({}, ValueError, "n_clusters must be provided"),
            ({"n_clusters": 0}, ValueError, "at least 1"),
            ({"n_clusters": 2, "min_overlap": 1}, ValueError, "at least 2"),
            ({"n_clusters": 2, "linkage": "ward"}, ValueError, "linkage"),
            ({"n_clusters": 2, "col_name": "CPI"}, ValueError, "already has a column"),
            ({"cluster_map": {"g": ["AUD", "CAD", "GBP"]}}, ValueError, "not assigned"),
            (
                {"cluster_map": {"g1": ["AUD", "GBP"], "g2": ["GBP", "CAD", "USD"]}},
                ValueError,
                "more than one group",
            ),
        ],
    )
    def test_invalid_parameters_raise(self, panel, params, error, match):
        X, y = panel

        with pytest.raises(error, match=match):
            ReturnClusterer(**{"col_name": "GROUP", **params}).fit(X, y)

    def test_invalid_inputs_raise(self, panel):
        X, y = panel
        rc = ReturnClusterer(col_name="GROUP", n_clusters=2)

        with pytest.raises(TypeError, match="pandas dataframe"):
            rc.fit(X.to_numpy(), y)
        with pytest.raises(ValueError, match="multi-indexed"):
            rc.fit(X.reset_index(drop=True), y)
        with pytest.raises(ValueError, match="y must be provided"):
            rc.fit(X)
        with pytest.raises(ValueError, match="one value per row"):
            rc.fit(X, y.iloc[:-1])
        with pytest.raises(ValueError, match="one value per row"):
            rc.fit(X, pd.concat([y, y], axis=1))
        with pytest.raises(ValueError, match="numeric"):
            rc.fit(X, y.astype(str))


class TestTransform:
    def test_appends_categorical_group_column(self, panel):
        X, y = panel
        X_orig = X.copy()
        rc = ReturnClusterer(col_name="GROUP", n_clusters=2).fit(X, y)

        X_new = rc.transform(X)

        pd.testing.assert_frame_equal(X, X_orig)
        pd.testing.assert_frame_equal(X_new[X.columns], X)
        assert list(X_new["GROUP"].cat.categories) == [0, 1]
        np.testing.assert_array_equal(
            X_new["GROUP"], X.index.get_level_values(0).map(CID_TO_GROUP)
        )

    def test_unseen_cross_sections_have_missing_group(self, panel):
        X, y = panel
        X_test = X.rename(index={"GBP": "EUR", "USD": "JPY"}, level=0)

        rc = ReturnClusterer(col_name="GROUP", n_clusters=2).fit(X, y)
        groups = rc.transform(X_test)["GROUP"]

        assert groups.loc[["EUR", "JPY"]].isna().all()

    def test_cluster_map_covers_cross_sections_outside_training(self, panel):
        X, _ = panel
        X_test = X.rename(index={"GBP": "EUR", "USD": "JPY"}, level=0)

        rc = ReturnClusterer(col_name="GROUP", cluster_map=CLUSTER_MAP).fit(X)
        groups = rc.transform(X_test)["GROUP"]

        assert (groups.loc["EUR"] == "europe").all()
        assert groups.loc["JPY"].isna().all()

    def test_invalid_inputs_raise(self, panel):
        X, y = panel
        rc = ReturnClusterer(col_name="GROUP", n_clusters=2)

        with pytest.raises(NotFittedError):
            rc.transform(X)
        with pytest.raises(NotFittedError):
            rc.get_feature_names_out()

        rc.fit(X, y)
        with pytest.raises(TypeError, match="pandas dataframe"):
            rc.transform(X.to_numpy())
        with pytest.raises(ValueError, match="multi-indexed"):
            rc.transform(X.reset_index(drop=True))
        with pytest.raises(ValueError, match="must match"):
            rc.transform(X[["CPI"]])
        with pytest.raises(ValueError, match="must match"):
            rc.transform(X.rename(columns={"CPI": "RIR"}))

    def test_pipeline(self, panel):
        X, y = panel
        pipe = Pipeline(
            [
                ("clusters", ReturnClusterer(col_name="GROUP", n_clusters=2)),
                (
                    "regions",
                    ReturnClusterer(col_name="REGION", cluster_map=CLUSTER_MAP),
                ),
            ]
        )

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            X_new = pipe.fit_transform(X, y)

        assert list(X_new.columns) == ["CPI", "GROWTH", "GROUP", "REGION"]
        assert list(pipe.get_feature_names_out()) == list(X_new.columns)
