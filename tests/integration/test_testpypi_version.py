import unittest
import requests
from packaging import version
# already part of python

try:
    from .version import git_revision as __git_revision__
    from .version import version as __version__
except ModuleNotFoundError:
    from setup import get_version_info
    FULLVERSION, GIT_REVISION = get_version_info()
    __version__ = FULLVERSION
    __git_revision__ = GIT_REVISION


class TestPyPIVersionNumber(unittest.TestCase):

    def test_version_number(self):

        # test for TestPyPI version number

        package = 'macrosynergy'
        with requests.get(f'https://test.pypi.org/pypi/{package}/json') as r:
            self.assertEqual(200, r.status_code, f"Incorrect: {r.status_code}, {r.text}")
            js = r.json()

        latest_version = js['info']['version']
        self.assertGreater(version.parse(__version__), version.parse(latest_version),
                           f"The version being published to TestPyPI is not newer than the latest version already on TestPyPI. -- {__version__} <= {latest_version}")

        # prevents publishing a version that is older or equal to the latest version on TestPyPI

if __name__ == "__main__":
    unittest.main()
