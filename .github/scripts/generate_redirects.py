all_old_files_list = [
    "api_reference.html",
    "common_definitions.html",
    "contribution_guide.html",
    "faqs.html",
    "genindex.html",
    "getting_started.html",
    "index.html",
    "macrosynergy/download/dataquery.html",
    "macrosynergy/download/exceptions.html",
    "macrosynergy/download.html",
    "macrosynergy/download/index.html",
    "macrosynergy/download/jpmaqs.html",
    "macrosynergy/learning/cv_tools.html",
    "macrosynergy/learning.html",
    "macrosynergy/learning/metrics.html",
    "macrosynergy/learning/panel_time_series_split.html",
    "macrosynergy/learning/predictors.html",
    "macrosynergy/learning/index.html",
    "macrosynergy/learning/signal_optimizer.html",
    "macrosynergy/learning/transformers.html",
    "macrosynergy/management/constants.html",
    "macrosynergy/management/decorators.html",
    "macrosynergy/management/index.html",
    "macrosynergy/management.html",
    "macrosynergy/management/simulate.html",
    "macrosynergy/management/simulate/simulate_quantamental_data.html",
    "macrosynergy/management/simulate/simulate_vintage_data.html",
    "macrosynergy/management/types.html",
    "macrosynergy/management/utils/check_availability.html",
    "macrosynergy/management/utils/core.html",
    "macrosynergy/management/utils/df_utils.html",
    "macrosynergy/management/simulate/index.html",
    "macrosynergy/management/utils.html",
    "macrosynergy/management/utils/math.html",
    "macrosynergy/management/validation.html",
    "macrosynergy/panel/basket.html",
    "macrosynergy/panel/category_relations.html",
    "macrosynergy/panel/converge_row.html",
    "macrosynergy/panel/granger_causality_test.html",
    "macrosynergy/panel/historic_vol.html",
    "macrosynergy/panel.html",
    "macrosynergy/panel/linear_composite.html",
    "macrosynergy/panel/make_blacklist.html",
    "macrosynergy/panel/make_relative_value.html",
    "macrosynergy/panel/make_zn_scores.html",
    "macrosynergy/panel/panel_calculator.html",
    "macrosynergy/panel/return_beta.html",
    "macrosynergy/panel/view_correlations.html",
    "macrosynergy/panel/view_grades.html",
    "macrosynergy/panel/view_metrics.html",
    "macrosynergy/panel/index.html",
    "macrosynergy/panel/view_ranges.html",
    "macrosynergy/panel/view_timelines.html",
    "macrosynergy/pnl.html",
    "macrosynergy/pnl/naive_pnl.html",
    "macrosynergy/signal.html",
    "macrosynergy/signal/signal_return_relations.html",
    "macrosynergy/signal/target_positions.html",
    "macrosynergy/version.html",
    "macrosynergy/visuals/correlation.html",
    "macrosynergy/visuals/facetplot.html",
    "macrosynergy/visuals/grades.html",
    "macrosynergy/visuals/heatmap.html",
    "macrosynergy/visuals.html",
    "macrosynergy/visuals/lineplot.html",
    "macrosynergy/visuals/metrics.html",
    "macrosynergy/visuals/multiple_reg_scatter.html",
    "macrosynergy/visuals/plotter.html",
    "macrosynergy/visuals/ranges.html",
    "macrosynergy/visuals/table.html",
    "macrosynergy/visuals/index.html",
    "macrosynergy/visuals/timelines.html",
    "macrosynergy/visuals/view_panel_dates.html",
    "macrosynergy_academy.html",
    "release_notes.html",
    "usage_examples.html",
    "05_dev_guide.html",
    "06_definitions.html",
    "02_installation.html",
    "01_context.html",
    "03_dataquery.html",
]


"""
[
    {
        "Condition": {
            "KeyPrefixEquals": "/"
        },
        "Redirect": {
            "HostName": "macrosynergy-docs-test.s3.eu-west-2.amazonaws.com",
            "ReplaceKeyPrefixWith": "main/index.html"
        }
    }
]
"""

OLD_BASE_URL = "https://www.macrosynergy.com/"
NEW_BASE_URL = (
    "http://macrosynergy-docs-test.s3-website.eu-west-2.amazonaws.com/stable/"
)
import json


def new_url(
    old_path: str,
    old_base_url: str = OLD_BASE_URL,
    new_base_url: str = NEW_BASE_URL,
) -> str:
    assert old_path.endswith(".html")
    fpath = old_path.replace("/", ".")
    fpath = fpath.replace("index.html", ".html")
    return new_base_url + fpath


def make_redirects_json(
    old_files_list: list,
    old_base_url: str = OLD_BASE_URL,
    new_base_url: str = NEW_BASE_URL,
) -> list:
    return [
        {
            "Condition": {"KeyPrefixEquals": old_file},
            "Redirect": {
                "HostName": new_base_url,
                "ReplaceKeyWith": new_url(old_file, old_base_url, new_base_url),
            },
        }
        for old_file in old_files_list
    ]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", type=str, default="redirects.json")
    args = parser.parse_args()
    with open(args.output, "w") as f:
        json.dump(make_redirects_json(all_old_files_list), f, indent=4)
