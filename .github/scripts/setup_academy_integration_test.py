"""
Script to clone the academy repository and install the requirements for the integration tests.
"""
from typing import List, Optional, Tuple, Dict, Any
import git
import os
import shutil
import requests
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

NOTEBOOK_URL: str = "https://academy.macrosynergy.com/academy/Introductions/Introduction%20to%20Macrosynergy%20package/_build/html/_sources/Introduction%20to%20Macrosynergy%20package.ipynb"
TEST_DIR: str = "tests/integration/notebooks"

GH_USER: str = os.environ.get("GH_USERNAME", None)
GH_TOKEN: str = os.environ.get("GH_TOKEN", None)
UTILS_REPO: str = "github.com/macrosynergy/msy-utils"


def download_notebook(url: str, filename: str, max_retries: int = 3) -> None:
    """
    Download a notebook from the academy website.
    """
    while max_retries > 0:
        try:
            response = requests.get(url)
            if response.status_code == 200:
                with open(filename, "wb") as f:
                    f.write(response.content)
                return
        except Exception as e:
            if 400 <= response.status_code < 500:
                raise e
            logger.warn(
                f"Failed to download notebook. Retrying in 5 seconds. Error: {e}"
            )
            max_retries -= 1
            time.sleep(5)

    raise Exception("Failed to download notebook.")


def clone_notebook_runner(
    repo: str,
    branch: str,
    path: str,
    user: Optional[str] = GH_USER,
    token: Optional[str] = GH_TOKEN,
) -> None:
    """
    Clone the notebook runner repository.
    """
    try:
        print("user is", user)
        print("token is", token)
        git.Repo.clone_from(
            #f"https://{user}:{token}@{repo}",
            f"https://{repo}",
            path,
            branch=branch,
            depth=1,
        )

        logger.info("Cloned notebook runner repository.")
        return
    except Exception as excA:
        try:
            git.Repo.clone_from(f"https://{repo}", path, branch=branch, depth=1)
            logger.info("Cloned notebook runner repository.")
            return
        except Exception as excB:
            logger.error(f"Failed to clone notebook runner repository. Error: {excB}")
            raise excB from excA


def setup_test(test_dir: str = TEST_DIR) -> None:
    """
    Setup the integration test.
    """
    logger.info("Setting up integration test.")
    os.makedirs(test_dir, exist_ok=True)
    notebook_filename = os.path.join(test_dir, "notebook.ipynb")
    download_notebook(NOTEBOOK_URL, notebook_filename)
    print("Downloaded notebook.")

    clone_notebook_runner(UTILS_REPO, "feature/notebook_script", "notebook-runner")

    logger.info("Integration test setup complete.")


def run_test(test_dir: str = TEST_DIR) -> None:
    """
    Run the integration test.
    """
    logger.info("Running integration test.")
    os.system(f"python notebook-runner/notebook_scripts/run_notebooks.py {test_dir}")
    logger.info("Integration test complete.")


def main(test_dir: str = TEST_DIR) -> None:
    """
    Main function.
    """
    setup_test(test_dir)
    run_test(test_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--test-dir", default=TEST_DIR)
    args = parser.parse_args()

    main(args.test_dir)
