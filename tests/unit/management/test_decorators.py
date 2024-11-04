import unittest
from unittest.mock import Mock
import warnings
from inspect import signature

from typing import List, Tuple, Dict, Union, Optional

from macrosynergy.management.decorators import deprecate, is_matching_subscripted_type


class TestDeprecateDecorator(unittest.TestCase):
    def setUp(self):
        self.mock_new_func = Mock(name="new_func")
        self.mock_old_func = Mock(name="old_func")
        self.mock_new_func.__name__ = "new_func"
        self.mock_old_func.__name__ = "old_func"
        self.mock_new_func.__doc__ = "This is a new function."
        self.mock_old_func.__doc__ = "This is an old function."

    def test_deprecate_warning(self):
        @deprecate(
            new_func=self.mock_new_func,
            deprecate_version="0.8.0",
            remove_after="1.5.0",
            message="The {old_method} method is deprecated in version {deprecate_version}. Use {new_method} instead.",
            macrosynergy_package_version="1.0.0",
        )
        def old_func():
            return "result"

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = old_func()

            self.assertEqual(len(w), 1)
            self.assertTrue(issubclass(w[-1].category, FutureWarning))
            self.assertIn("old_func", str(w[-1].message))
            self.assertIn("deprecated", str(w[-1].message))
            self.assertIn("new_func", str(w[-1].message))

        self.assertEqual(result, "result")

    def test_default_message(self):
        def test_new_func():
            return "result"

        @deprecate(new_func=test_new_func, deprecate_version="0.0.4")
        def old_func():
            return "result"

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = old_func()

            self.assertEqual(len(w), 1)
            self.assertTrue(issubclass(w[-1].category, FutureWarning))
            self.assertIn("old_func was deprecated in version", str(w[-1].message))
            self.assertIn("Use test_new_func instead.", str(w[-1].message))

        self.assertEqual(result, "result")

    def test_invalid_message_format(self):
        with self.assertRaises(ValueError) as context:

            @deprecate(
                new_func=self.mock_new_func,
                deprecate_version="1.0.0",
                message="Invalid message without format strings",
            )
            def old_func():
                pass

        self.assertIn(
            "The message must contain the following format strings",
            str(context.exception),
        )

    def test_invalid_remove_after_version(self):
        with self.assertRaises(ValueError) as context:

            @deprecate(
                new_func=self.mock_new_func,
                deprecate_version="1.0.0",
                remove_after="invalid_version",
            )
            def old_func():
                pass

        self.assertIn("must be a valid version string", str(context.exception))

    def test_remove_after_less_than_deprecate_version(self):
        with self.assertRaises(ValueError) as context:

            @deprecate(
                new_func=self.mock_new_func,
                deprecate_version="1.0.0",
                remove_after="0.9.0",
            )
            def old_func():
                pass

        self.assertIn(
            "must be greater than the version in which it is deprecated",
            str(context.exception),
        )

    def test_signature_and_docstring(self):
        @deprecate(new_func=self.mock_new_func, deprecate_version="0.8.0")
        def old_func():
            "Old function docstring"
            return "result"

        self.assertEqual(old_func.__doc__, self.mock_new_func.__doc__)
        self.assertEqual(old_func.__signature__, signature(self.mock_new_func))

if __name__ == "__main__":
    unittest.main()
