import unittest
import requests
from packaging import version
from macrosynergy import __version__


def get_version(test: bool = False):
    if test:
        url = "https://test.pypi.org/pypi/macrosynergy/json"
    else:
        url = "https://pypi.org/pypi/macrosynergy/json"

    with requests.get(url) as r:
        assert r.ok, f"Incorrect: {r.status_code}, {r.text}"
        js = r.json()

    return js['info']['version']


class TestPyPIVersionNumber(unittest.TestCase):
    def test_version_number_test_pypi(self):
        # test for TestPyPI version number
        self.assertGreaterEqual(
            version.parse(__version__),
            version.parse(get_version(test=True))
        )

    def test_version_number_pypi(self):
        # test for PyPI version number
        self.assertGreater(
            version.parse(__version__),
            version.parse(get_version(test=False))
        )


if __name__ == "__main__":
    unittest.main()
