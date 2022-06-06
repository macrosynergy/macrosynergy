import unittest
from unittest import mock
from macrosynergy.dataquery import auth


class TestCertAuth(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, True)  # add assertion here


class TestOAuth(unittest.TestCase):
    def test_init(self):
        oauth = auth.OAuth(client_id="test-id", client_secret="SECRET")

        self.assertEqual(auth.OAUTH_BASE_URL, oauth.base_url)
        self.assertEqual("test-id", oauth.client_id)
        self.assertEqual("SECRET", oauth.client_secret)

        self.assertIsNone(oauth.status_code)
        self.assertIsNone(oauth.last_response)
        self.assertIsNone(oauth.last_url)

    def test_invalid_dtype_client_id(self):
        with self.assertRaises(AssertionError):
            auth.OAuth(client_id=b"test-id", client_secret="SECRET")

    def test_invalid_dtype_client_secret(self):
        with self.assertRaises(AssertionError):
            auth.OAuth(client_id="test-id", client_secret=b"SECRET")

    def test_valid_token(self):
        oauth = auth.OAuth(client_id="test-id", client_secret="SECRET")
        self.assertFalse(oauth._valid_token())


if __name__ == '__main__':
    unittest.main()
