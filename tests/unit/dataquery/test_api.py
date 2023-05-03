# from macrosynergy.dataquery import api
import warnings
import unittest

class TestDataQueryInterface(unittest.TestCase):
    
    def test_deprecation_warning(self):
        warnings.simplefilter("default")
        with self.assertWarns(DeprecationWarning):
            from macrosynergy.dataquery.api import Interface
            with Interface(client_id='client_id', client_secret='client_secret',
                            check_connection=False) as interface:
                pass
            
            # assertWarns uses warnings.catch_warnings() internally,
            # which means that using warnings.catch_warnings() alongside it will
            # cause the test to fail. This is because only one of the two can 
            # actually catch the warning.

    def test_successful_deprecation(self):
        from macrosynergy.dataquery import api
        from macrosynergy.download import JPMaQSDownload
        self.assertTrue(issubclass(api.Interface, JPMaQSDownload))



if __name__ == "__main__":
    unittest.main()
