import unittest
import requests
from packaging import version
from macrosynergy import __version__


class TestPyPIVersionNumber(unittest.TestCase):

    def test_version_number(self):

        # test for TestPyPI version number

        with requests.get(f'https://test.pypi.org/pypi/macrosynergy/json') as r:
            self.assertEqual(200, r.status_code, f"Incorrect: {r.status_code}, {r.text}")
            js = r.json()

        latest_version = js['info']['version']
        self.assertGreater(version.parse(__version__), version.parse(latest_version))

        # prevents publishing a version that is older or equal to the latest version on TestPyPI

if __name__ == "__main__":
    unittest.main()
