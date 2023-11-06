import os
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from joblib import Parallel, delayed
import unittest

class TestAll(unittest.TestCase):

    def test_d_are_error_output(self):
        """
        Traverses repository and fails if any notebook has any errors in code output
        """
        print("Test 4: Do any notebooks contain error outputs?")

        def run_test(nb):
            ep = ExecutePreprocessor(timeout=600)
            with open(nb, "rb") as f:
                notebook = nbformat.read(f, as_version=nbformat.NO_CONVERT)
                ep.preprocess(notebook, {"metadata": {"path": "./"}})
                print("Executed notebook:", nb)
                for cell in notebook["cells"]:
                    if cell["cell_type"] == "code":
                        outputs = cell["outputs"]
                        for output in outputs:
                            if (
                                "output_type" in output
                                and output["output_type"] == "error"
                            ):
                                return f"{nb} contains code cell output errors"
            return None

        filenames = [
            os.path.join(root, ff)
            for root, dirs, files in os.walk(os.getcwd())
            for ff in files
            if ff.endswith(".ipynb")
        ]

        errors = Parallel(n_jobs=-1, backend="threading")(
            delayed(run_test)(nb) for nb in filenames
        )

        errors = [e for e in errors if e is not None]

        if errors:
            self.fail("\n".join(errors))

        print("Test complete")