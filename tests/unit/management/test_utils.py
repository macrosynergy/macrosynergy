import unittest
import random
import io
import os
import numpy as np
import pandas as pd
import warnings
import datetime
import yaml
import json
from unittest.mock import patch, MagicMock, Mock, mock_open
from typing import List, Tuple, Dict, Union, Any, Set
from macrosynergy.management.simulate_quantamental_data import make_test_df
from macrosynergy.management.utils import (
    get_dict_max_depth,
    rec_search_dict,
    is_valid_iso_date,
    convert_dq_to_iso,
    convert_iso_to_dq,
    convert_to_iso_format,
    form_full_url,
    generate_random_date,
    common_cids,
    drop_nan_series
)

from macrosynergy.management.utils import (
    Config,)


class TestFunctions(unittest.TestCase):
    def test_get_dict_max_depth(self):
        d: dict = {"a": 1, "b": {"c": 2, "d": {"e": 3}}}
        self.assertEqual(get_dict_max_depth(d), 3)

        d: int = 10
        self.assertEqual(get_dict_max_depth(d), 0)

        dx: dict = {0: "a"}
        for i in range(1, 100):
            dx = {i: dx}
        self.assertEqual(get_dict_max_depth(dx), 100)

    def test_rec_search_dict(self):
        d: dict = {"a": 1, "b": {"c": 2, "d": {"e": 3}}}
        self.assertEqual(rec_search_dict(d, "e"), 3)

        self.assertEqual(rec_search_dict('Some string', "KEY"), None)

        dx: dict = {0: "a"}
        for i in range(1, 100):
            dx = {i: dx}
        self.assertEqual(rec_search_dict(dx, 0), "a")

        d = {"12": 1, "123": 2, "234": 3, "1256": 4, "246": 5}
        self.assertEqual(rec_search_dict(d=d, key="25", match_substring=True), 4)
        self.assertEqual(rec_search_dict(d=d, key="99", match_substring=True), None)

        d = {"12": 1, "123": [2], "234": '3', "1256": 4.0, "246": {"a": 1}}
        for k in d.keys():
            self.assertEqual(rec_search_dict(d=d, key=k, match_substring=True, match_type=type(d[k])), d[k])

    def test_is_valid_iso_date(self):
        d1: str = "2020-01-01"
        d2: str = "2020-01-01T00:00:00"
        d3: str = "2020-01-01T00:00:00Z"
        d5: str = "12-900-56"
        d6: str = "foo"
        d7: str = "bar"
        d8: str = "Ze-ld-a"

        self.assertTrue(is_valid_iso_date(d1))
        for d in [d2, d3, d5, d6, d7, d8]:
            self.assertFalse(is_valid_iso_date(d))

    def test_convert_dq_to_iso(self):
        d: List[Tuple[str, str]] = [
            ("20200101", "2020-01-01"),
            ("20110101", "2011-01-01"),
            ("20120101", "2012-01-01"),
            ("20130101", "2013-01-01"),
            ("20140101", "2014-01-01"),
            ("20150101", "2015-01-01"),
        ]

        for i in range(len(d)):
            self.assertEqual(convert_dq_to_iso(d[i][0]), d[i][1])

        # generate 20 random dates
        dts = [generate_random_date() for i in range(20)]
        for dt in dts:
            self.assertEqual(convert_iso_to_dq(dt), dt.replace("-", ""))
            self.assertEqual(convert_dq_to_iso(dt.replace("-", "")), dt)

    def test_convert_iso_to_dq(self):
        d: List[Tuple[str, str]] = [
            ("2020-01-01", "20200101"),
            ("2011-01-01", "20110101"),
            ("2012-01-01", "20120101"),
            ("2013-01-01", "20130101"),
            ("2014-01-01", "20140101"),
            ("2015-01-01", "20150101"),
        ]

        for i in range(len(d)):
            self.assertEqual(convert_iso_to_dq(d[i][0]), d[i][1])

        # generate 20 random dates
        dts: List[str] = [generate_random_date() for i in range(20)]
        for dt in dts:
            self.assertEqual(convert_iso_to_dq(dt), dt.replace("-", ""))

    def test_convert_to_iso_format(self):
        """
        dd-mm-yyyy
        dd-mmm-yyyy
        dd-mm-yy
        ddmmyyyy (if len==8)

        iterate separators from ["-", "/", ".", " "]

        """
        return

        tests: List[Tuple[str, str]] = [
            ("01*01*2020", "2020-01-01"),
            ("01*MAY*2020", "2020-05-01"),
            ("01012020", "2020-01-01"),
        ]
        seps: List[str] = ["-", "/", ".", " "]
        for t, r in tests:
            for sep in seps:
                print(t, r, sep)
                self.assertEqual(convert_to_iso_format(t.replace("*", sep)), r)

    def test_form_full_url(self):
        url: str = "https://www.google.com"
        params: Dict[str, Union[str, int]] = {"a": 1, "b": 2}
        # html safe url
        self.assertEqual(form_full_url(url, params), "https://www.google.com?a=1&b=2")
        # url = http://foo.bar
        # params = {'banana': '!@#$%^&*()_+{}|:"<>?'}', 'apple': '><?>?<?><?'}
        url = "http://foo.bar"
        params = {"banana": """!@#$%^&*()_+{}|:"<>?}""", "apple": "><?>?<?><?"}

        exp_out: str = (
            "http://foo.bar?banana=%21%40%23%24%25%5E%26%"
            "2A%28%29_%2B%7B%7D%7C%3A%22%3C%3E%3F%7D&apple"
            "=%3E%3C%3F%3E%3F%3C%3F%3E%3C%3F"
        )
        self.assertEqual(form_full_url(url, params), exp_out)

    def test_generate_random_date(self):
        # get 20 random dates
        strts: List[str] = [generate_random_date() for i in range(10)]
        ends: List[str] = [generate_random_date() for i in range(10)]
        
        for st, ed in zip(strts, ends):
            stD = datetime.datetime.strptime(st, "%Y-%m-%d")
            edD = datetime.datetime.strptime(ed, "%Y-%m-%d")
            if stD > edD:
                stD, edD = edD, stD
                # generate random date between st and ed
                rd = generate_random_date(stD, edD)
                rdD = datetime.datetime.strptime(rd, "%Y-%m-%d")
                self.assertTrue(stD <= rdD <= edD)

        strts = ['2020-01-01', '2023-05-02', '2021-12-31']
        ends = ['2020-01-03', '2023-05-03', '2021-12-31']
        endst = ['2020-01-02', '2023-05-03', '2021-12-31']
        for st, ed, edt in zip(strts, ends, endst):
            stD = datetime.datetime.strptime(st, "%Y-%m-%d")
            edD = datetime.datetime.strptime(ed, "%Y-%m-%d")
            edtD = datetime.datetime.strptime(edt, "%Y-%m-%d")
            if stD > edD:
                stD, edD = edD, stD
            # generate random date between st and ed
            rd = generate_random_date(stD, edD)
            rdD = datetime.datetime.strptime(rd, "%Y-%m-%d")
            self.assertTrue(stD <= rdD <= edD)
            self.assertTrue(rdD <= edtD)
            
            # when generate st=ed, rd=ed
            rdD = generate_random_date(edD, edD)
            self.assertEqual(rdD, edD.strftime("%Y-%m-%d"))

    def test_common_cids(self):
        cids : List[str] = ["AUD", "USD", "GBP", "EUR", "CAD"]
        xcats : List[str] = ["FXXR", "IR", "EQXR", "CRY", "FXFW"]
        df : pd.DataFrame = make_test_df(cids=cids, xcats=xcats)
        
        # check normal case
        com_cids : List[str] = common_cids(df=df, xcats=xcats)
        self.assertEqual(set(com_cids), set(cids))
        
        self.assertRaises(TypeError, common_cids, df=1, xcats=xcats)
        self.assertRaises(TypeError, common_cids, df=df, xcats=1)
        self.assertRaises(ValueError, common_cids, df=df, xcats=['xcat'])
        self.assertRaises(ValueError, common_cids, df=df, xcats=["apple", "banana"])
        self.assertRaises(TypeError, common_cids, df=df, xcats=[1,2,3])
        
        # test A
        dfA : pd.DataFrame = df.copy()
        dfA = dfA[~((dfA["cid"] == "USD") & (dfA["xcat"].isin(["FXXR", "IR"])))]
        dfA = dfA[~((dfA["cid"] == "CAD") & (dfA["xcat"].isin(["FXXR", "IR"])))]
        
        com_cids : List[str] = common_cids(df=dfA, xcats=xcats)
        self.assertEqual(set(com_cids), set(["AUD", "GBP", "EUR"]))
        
        comm_cids : List[str] = common_cids(df=dfA, xcats=["FXXR", "IR"])
        self.assertEqual(set(comm_cids), set(["AUD", "GBP", "EUR",]))
        
        # test B
        dfB : pd.DataFrame = df.copy()
        # remove "FXXR", "IR", "EQXR" from "AUD", "USD"
        dfB = dfB[~((dfB["cid"] == "AUD") & (dfB["xcat"].isin(["FXXR", "IR", "EQXR"])))]
        dfB = dfB[~((dfB["cid"] == "USD") & (dfB["xcat"].isin(["FXXR", "IR", "EQXR"])))]

        com_cids : List[str] = common_cids(df=dfB, xcats=xcats)
        self.assertEqual(set(com_cids), set(["GBP", "EUR", "CAD"]))

        comm_cids : List[str] = common_cids(df=dfB, xcats=["FXFW", "CRY"])
        self.assertEqual(set(comm_cids), set(cids))


    def test_drop_nan_series(self):
        cids : List[str] = ["AUD", "USD", "GBP", "EUR", "CAD"]
        xcats : List[str] = ["FXXR", "IR", "EQXR", "CRY", "FXFW"]
        df_orig : pd.DataFrame = make_test_df(cids=cids, xcats=xcats)

        # set warnings to error. test if a warning is raised in the obvious "clean" case
        warnings.simplefilter("error")
        for boolx in [True, False]:
            try:
                dfx : pd.DataFrame = drop_nan_series(df=df_orig, raise_warning=boolx)
                self.assertTrue(dfx.equals(df_orig))
            except Warning as w:
                self.fail("Warning raised unexpectedly")

        df_test : pd.DataFrame = df_orig.copy()
        df_test.loc[(df_test["cid"] == "AUD") & (df_test["xcat"].isin(["FXXR", "IR"])), "value"] = pd.NA

        warnings.simplefilter("always")
        with warnings.catch_warnings(record=True) as w:
            dfx : pd.DataFrame = drop_nan_series(df=df_test, raise_warning=True)
            self.assertEqual(len(w), 2)
            for ww in w:
                self.assertTrue(issubclass(ww.category, UserWarning))
                
            found_tickers : Set = set(dfx['cid'] + '_' + dfx['xcat'])
            if any([x in found_tickers for x in ["AUD_FXXR", "AUD_IR"]]):
                self.fail("NaN series not dropped")

        with warnings.catch_warnings(record=True) as w:
            dfx : pd.DataFrame = drop_nan_series(df=df_test, raise_warning=False)
            self.assertEqual(len(w), 0)
            found_tickers : Set = set(dfx['cid'] + '_' + dfx['xcat'])
            if any([x in found_tickers for x in ["AUD_FXXR", "AUD_IR"]]):
                self.fail("NaN series not dropped")      
          
        self.assertRaises(TypeError, drop_nan_series, df=1, raise_warning=True)
        self.assertRaises(TypeError, drop_nan_series, df=df_test, raise_warning=1)
        
        df_test_q = df_test.dropna(how="any")
        with warnings.catch_warnings(record=True) as w:
            dfx : pd.DataFrame = drop_nan_series(df=df_test_q, raise_warning=True)
            dfu : pd.DataFrame = drop_nan_series(df=df_test_q, raise_warning=False)
            self.assertEqual(len(w), 0)
            self.assertTrue(dfx.equals(df_test_q))
            self.assertTrue(dfu.equals(df_test_q))
            
        df_test : pd.DataFrame = df_orig.copy()
        bcids : List[str] = ["AUD", "USD", "GBP",]
        bxcats : List[str] = ["FXXR", "IR", "EQXR",]
        df_test.loc[(df_test["cid"].isin(bcids)) & (df_test["xcat"].isin(bxcats)), "value"] = pd.NA
        with warnings.catch_warnings(record=True) as w:
            dfx : pd.DataFrame = drop_nan_series(df=df_test, raise_warning=True)
            self.assertEqual(len(w), 9)
            for ww in w:
                self.assertTrue(issubclass(ww.category, UserWarning))
                
            found_tickers : Set = set(dfx['cid'] + '_' + dfx['xcat'])
            if any([x in found_tickers for x in [f"{cid}_{xcat}" for cid in bcids for xcat in bxcats]]):
                self.fail("NaN series not dropped")

            dfu : pd.DataFrame = drop_nan_series(df=df_test, raise_warning=False)
            self.assertEqual(len(w), 9)



####################################################################################################
####################################################################################################

class TestJPMaQSAPIConfigObject(unittest.TestCase):

    def mock_yaml_content(self):
        yaml_content_yml =\
        """
        # JPMaQS API Configuration File
        JPMAQS_CREDENTIALS:
            OAUTH:
                client_id: 'CLIENT_ID'
                client_secret: 'CLIENT_SECRET'
            CERT:
                crt: 'path/to/CERTIFICATE'
                key: 'path/to/KEY'
                username: 'USERNAME'
                password: 'PASSWORD'
            PROXYSETTINGS:
                proxy:
                    http: 'HTTP_PROXY:PORT'
                    ssl: 'SSL_PROXY:PORT'
            # other settings
            proxies:
                https: 'HTTPS_PROXY:PORTX'
                ftp: 'FTP_PROXY:PORTZ'
                smtp: 'SMTP_PROXY:PORTY'

        """
        return yaml_content_yml
    
    def mock_json_content(self):
        return json.dumps(yaml.load(io.StringIO(self.mock_yaml_content()), Loader=yaml.FullLoader))
    
    def _mock_isfile(self, path):
            return path in ([f"config.{ext}" for ext in ["yml", "yaml", "json"]] + ['path/to/KEY', 'path/to/CERTIFICATE']
                            + ['path/to/KEY_ALT', 'path/to/CERTIFICATE_ALT'])
           

    def test_init(self):
        
        yaml_content_yml = self.mock_yaml_content()
        json_content_json = json.dumps(yaml.load(io.StringIO(yaml_content_yml), Loader=yaml.FullLoader))

        yaml_dict = yaml.load(io.StringIO(yaml_content_yml), Loader=yaml.FullLoader)
        json_dict = json.loads(json_content_json)
        # check that yaml and json are the same
        self.assertEqual(yaml_dict, json_dict)

        self.yaml_content_yml = yaml_content_yml
        self.json_content_json = json_content_json
        self.yaml_dict = yaml_dict
        self.json_dict = json_dict
    

    def test_init_yaml(self):
        """
        Mock the open function to return the yaml content.
        Also mock path/to/CERTIFICATE and path/to/KEY as files
        """

        

        # Patch the built-in 'open' function with the mock file object
        oauth : Dict[str, str] = {}
        cert : Dict[str, str] = {}
        proxy : Dict[str, str] = {}

        m = mock_open(read_data=self.mock_yaml_content())
        def _mock_is_file(path):
            return self._mock_isfile(path)
        
        # it's going to read yml. so it's going to read the file, then it's goinj gto do isfile on the path/to/CERTIFICATE and path/to/KEY
        with patch('os.path.isfile', side_effect= lambda path: _mock_is_file(path)):
            with patch('builtins.open', m):

                config = Config("config.yml")
                oauth = config.oauth(mask=False)
                cert = config.cert(mask=False)
                proxy = config.proxy(mask=False)
        


        # Assert that the 'open' function was called with the correct file name
        m.assert_called_once_with('config.yml', 'r')
        # Assert that the necessary attributes or properties of the object were correctly set
        
        # check [client_id, client_secret] in oauth
        self.assertEqual(oauth['client_id'], 'CLIENT_ID')
        self.assertEqual(oauth['client_secret'], 'CLIENT_SECRET')

        # check [crt, key, username, password] in cert
        self.assertEqual(cert['crt'], 'path/to/CERTIFICATE')
        self.assertEqual(cert['key'], 'path/to/KEY')
        self.assertEqual(cert['username'], 'USERNAME')
        self.assertEqual(cert['password'], 'PASSWORD')

        # check [http, ssl] in proxy
        self.assertEqual(proxy['http'], 'HTTP_PROXY:PORT')
        self.assertEqual(proxy['ssl'], 'SSL_PROXY:PORT')
        self.assertEqual(proxy['https'], 'HTTPS_PROXY:PORTX')
        self.assertEqual(proxy['ftp'], 'FTP_PROXY:PORTZ')
        self.assertEqual(proxy['smtp'], 'SMTP_PROXY:PORTY')



    def test_init_json(self):
        """
        Mock the open function to return the yaml content.
        Also mock path/to/CERTIFICATE and path/to/KEY as files
        """

        

        # Patch the built-in 'open' function with the mock file object
        oauth : Dict[str, str] = {}
        cert : Dict[str, str] = {}
        proxy : Dict[str, str] = {}

        m = mock_open(read_data=self.mock_json_content())
        def _mock_is_file(path):
            return self._mock_isfile(path)
        
        # it's going to read yml. so it's going to read the file, then it's goinj gto do isfile on the path/to/CERTIFICATE and path/to/KEY
        with patch('os.path.isfile', side_effect= lambda path: _mock_is_file(path)):
            with patch('builtins.open', m):

                config = Config("config.json")
                oauth = config.oauth(mask=False)
                cert = config.cert(mask=False)
                proxy = config.proxy(mask=False)
        


        # Assert that the 'open' function was called with the correct file name
        m.assert_called_once_with('config.json', 'r')
        # Assert that the necessary attributes or properties of the object were correctly set
        
        # check [client_id, client_secret] in oauth
        self.assertEqual(oauth['client_id'], 'CLIENT_ID')
        self.assertEqual(oauth['client_secret'], 'CLIENT_SECRET')

        # check [crt, key, username, password] in cert
        self.assertEqual(cert['crt'], 'path/to/CERTIFICATE')
        self.assertEqual(cert['key'], 'path/to/KEY')
        self.assertEqual(cert['username'], 'USERNAME')
        self.assertEqual(cert['password'], 'PASSWORD')

        # check [http, ssl] in proxy
        self.assertEqual(proxy['http'], 'HTTP_PROXY:PORT')
        self.assertEqual(proxy['ssl'], 'SSL_PROXY:PORT')
        self.assertEqual(proxy['https'], 'HTTPS_PROXY:PORTX')
        self.assertEqual(proxy['ftp'], 'FTP_PROXY:PORTZ')
        self.assertEqual(proxy['smtp'], 'SMTP_PROXY:PORTY')



    def test_overwrite_args(self):
        """
        Mock the open function to return the yaml content.
        Also mock path/to/CERTIFICATE and path/to/KEY as files
        """

        

        # Patch the built-in 'open' function with the mock file object
        client_id = 'CLIENT_ID--'
        client_secret = 'CLIENT_SECRET--'


        oauth : Dict[str, str] = {}
        cert : Dict[str, str] = {}
        proxy : Dict[str, str] = {}

        m = mock_open(read_data=self.mock_json_content())
        def _mock_is_file(path):
            return self._mock_isfile(path)
        
        # it's going to read yml. so it's going to read the file, then it's going to do isfile on the path/to/CERTIFICATE and path/to/KEY
        with patch('os.path.isfile', side_effect= lambda path: _mock_is_file(path)):
            with patch('builtins.open', m):
                config = Config("config.json", client_id=client_id, client_secret=client_secret)
                oauth = config.oauth(mask=False)
                cert = config.cert(mask=False)
                proxy = config.proxy(mask=False)
        


        m.assert_called_once_with('config.json', 'r')        
        # check [client_id, client_secret] in oauth
        self.assertEqual(oauth['client_id'], client_id)
        self.assertEqual(oauth['client_secret'], client_secret)

        # check [crt, key, username, password] in cert
        self.assertEqual(cert['crt'], 'path/to/CERTIFICATE')
        self.assertEqual(cert['key'], 'path/to/KEY')
        self.assertEqual(cert['username'], 'USERNAME')
        self.assertEqual(cert['password'], 'PASSWORD')

        # check [http, ssl] in proxy
        self.assertEqual(proxy['http'], 'HTTP_PROXY:PORT')
        self.assertEqual(proxy['ssl'], 'SSL_PROXY:PORT')
        self.assertEqual(proxy['https'], 'HTTPS_PROXY:PORTX')
        self.assertEqual(proxy['ftp'], 'FTP_PROXY:PORTZ')
        self.assertEqual(proxy['smtp'], 'SMTP_PROXY:PORTY')

        ####### test overwrite cert and proxy

        oauth : Dict[str, str] = {}
        cert : Dict[str, str] = {}
        proxy : Dict[str, str] = {}

        m = mock_open(read_data=self.mock_yaml_content())
        def _mock_is_file(path):
            return self._mock_isfile(path)
        
        with patch('os.path.isfile', side_effect= lambda path: _mock_is_file(path)):
            with patch('builtins.open', m):
                username = 'uname--'
                password = 'pass--'
                crtx = 'path/to/CERTIFICATE_ALT'
                keyx = 'path/to/KEY_ALT'
                proxyL = {'http': 'vpn.com:8090'}

                config = Config("config.yml", 
                                               username=username, 
                                               password=password,
                                                crt=crtx,
                                                key=keyx,
                                               proxy=proxyL)
                oauth = config.oauth(mask=False)
                cert = config.cert(mask=False)
                proxy = config.proxy(mask=False)
                
        m.assert_called_once_with('config.yml', 'r')

        config_dict = yaml.safe_load(io.StringIO(self.mock_json_content()))
        self.assertEqual(oauth['client_id'], config_dict['JPMAQS_CREDENTIALS']['OAUTH']['client_id'])
        self.assertEqual(oauth['client_id'], rec_search_dict(config_dict, 'client_id'))
        self.assertEqual(oauth['client_secret'], config_dict['JPMAQS_CREDENTIALS']['OAUTH']['client_secret'])
        self.assertEqual(oauth['client_secret'], rec_search_dict(config_dict, 'client_secret'))
        self.assertEqual(cert['crt'], crtx)
        self.assertEqual(cert['key'], keyx)
        self.assertEqual(cert['username'], username)
        self.assertEqual(cert['password'], password)
        self.assertEqual(proxy['http'], proxyL['http'])
        
    def test_partial_args(self):
        """
        Mock the open function to return the yaml content.
        Also mock path/to/CERTIFICATE and path/to/KEY as files
        """

        # Patch the built-in 'open' function with the mock file object
        client_id = 'CLIENT_ID--'
        client_secret = 'CLIENT_SECRET--'


        oauth : Dict[str, str] = {}
        cert : Dict[str, str] = {}
        proxy : Dict[str, str] = {}

        
        config = Config(client_id=client_id, client_secret=client_secret)
        oauth = config.oauth(mask=False)
        cert = config.cert(mask=False)
        proxy = config.proxy(mask=False)

        self.assertEqual(oauth['client_id'], client_id)
        self.assertEqual(oauth['client_secret'], client_secret)

        self.assertEqual(cert, None)
        self.assertEqual(proxy, None)
        

        def _mock_is_file(path):
            return self._mock_isfile(path)
        
        with patch('os.path.isfile', side_effect= lambda path: _mock_is_file(path)):
                username = 'uname--'
                password = 'pass--'
                crtx = 'path/to/CERTIFICATE_ALT'
                keyx = 'path/to/KEY_ALT'
                proxyL = {'http': 'vpn.com:8090'}

                config = Config(username=username, 
                                                password=password,
                                                crt=crtx,
                                                key=keyx,
                                                proxy=proxyL)
                
                oauth = config.oauth(mask=False)
                cert = config.cert(mask=False)
                proxy = config.proxy(mask=False)
                
        self.assertEqual(oauth, None)
        self.assertEqual(cert['crt'], crtx)
        self.assertEqual(cert['key'], keyx)
        self.assertEqual(cert['username'], username)
        self.assertEqual(cert['password'], password)
        self.assertEqual(proxy['http'], proxyL['http'])
        
        
    


    


        

if __name__ == "__main__":
    unittest.main()
