import requests
import unittest


class TestPyPIVersion(unittest.TestCase):
    def test_version(self):
        url = "https://pypi.org/pypi/macrosynergy/json"
        with requests.get(url) as r:
            print(r.json())


if __name__ == '__main__':
    unittest.main()
