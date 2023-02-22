# from macrosynergy.dataquery import api
from macrosynergy.download import JPMaQSDownload
import warnings
import unittest

class TestDataQueryInterface(unittest.TestCase):
    
    def test_deprecation_warning(self):
        with warnings.catch_warnings(record=True) as w:
            from macrosynergy.dataquery import api
            for warning in w:
                self.assertEqual(warning.category, DeprecationWarning)
                self.assertIn("has been moved to macrosynergy.download.jpmaqs", str(warning.message))
                
            
    def test_successful_deprecation(self):
        from macrosynergy.dataquery import api

        self.assertEqual(api.Interface, JPMaQSDownload)



if __name__ == "__main__":
    unittest.main()
