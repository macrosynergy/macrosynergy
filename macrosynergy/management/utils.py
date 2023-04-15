"""
Generic dataframe and type conversion functions specific to the
Macrosynergy package and JPMaQS dataframes/data.
"""

import pandas as pd
import numpy as np
import datetime
import os
import yaml
import json
from typing import Any, List, Dict, Optional, Callable, Union
import requests, requests.compat
import itertools


##############################
#   Helpful Functions
##############################


def generate_random_date(
    start: Optional[Union[str, datetime.datetime, pd.Timestamp]] = "1990-01-01",
    end: Optional[Union[str, datetime.datetime, pd.Timestamp]] = "2020-01-01",
) -> str:
    """
    Generates a random date between two dates.

    Parameters
    :param <str> start: The start date, in the ISO format (YYYY-MM-DD).
    :param <str> end: The end date, in the ISO format (YYYY-MM-DD).

    Returns
    :return <str>: The random date.
    """

    if not isinstance(start, (str, datetime.datetime, pd.Timestamp)):
        raise TypeError(
            "Argument `start` must be a string, datetime.datetime, or pd.Timestamp."
        )
    if not isinstance(end, (str, datetime.datetime, pd.Timestamp)):
        raise TypeError(
            "Argument `end` must be a string, datetime.datetime, or pd.Timestamp."
        )

    start: pd.Timestamp = pd.Timestamp(start)
    end: pd.Timestamp = pd.Timestamp(end)
    if start == end:
        return start.strftime("%Y-%m-%d")
    else:
        return pd.Timestamp(
            np.random.randint(start.value, end.value, dtype=np.int64)
        ).strftime("%Y-%m-%d")


def get_dict_max_depth(d: dict) -> int:
    """
    Returns the maximum depth of a dictionary.

    Parameters
    :param <dict> d: The dictionary to be searched.

    Returns
    :return <int>: The maximum depth of the dictionary.
    """
    return (
        1 + max(map(get_dict_max_depth, d.values()), default=0)
        if isinstance(d, dict)
        else 0
    )

def rec_search_dict(d : dict,
                    key : str,
                    match_substring : bool = False,
                    match_type=None):
    """
    Recursively searches a dictionary for a key and returns the value
    associated with it.

    :param <dict> d: The dictionary to be searched.
    :param <str> key: The key to be searched for.
    :param <bool> match_substring: If True, the function will return
        the value of the first key that contains the substring
        specified by the key parameter. If False, the function will
        return the value of the first key that matches the key
        parameter exactly. Default is False.
    :param <Any> match_type: If not None, the function will look for
        a key that matches the search parameters and has
        the specified type. Default is None.
    :return Any: The value associated with the key, or None if the key
        is not found.
    """
    if not isinstance(d, dict):
        return None

    for k, v in d.items():
        if match_substring:
            if key in k:
                if match_type is None or isinstance(v, match_type):
                    return v
        else:
            if k == key:
                if match_type is None or isinstance(v, match_type):
                    return v

        if isinstance(v, dict):
            result = rec_search_dict(v, key, match_substring, match_type)
            if result is not None:
                return result

    return None


def is_valid_iso_date(date: str) -> bool:
    try:
        datetime.datetime.strptime(date, "%Y-%m-%d")
        return True
    except ValueError:
        return False


def convert_to_iso_format(date: Any = None) -> str:
    raise NotImplementedError("This function is not yet implemented.")
    """
    Converts a datetime like object or string to an ISO date string.

    Parameters
    :param <Any> date: The date to be converted. This can be a
        datetime object, a string, pd.Timestamp, or np.datetime64.

    Returns
    :return <str>: The ISO date string (YYYY-MM-DD).
    """
    if date is None:
        ValueError("Argument `date` cannot be None.")

    r: Optional[str] = None
    if isinstance(date, str):
        r: Optional[str] = None
        if is_valid_iso_date(date):
            r = date
        else:
            if len(date) == 8:
                try:
                    r = convert_dq_to_iso(date)
                except Exception as e:
                    if isinstance(e, (ValueError, AssertionError)):
                        pass
            else:
                for sep in ["-", "/", ".", " "]:
                    if sep in date:
                        try:
                            sd = date.split(sep)
                            dx = date
                            if len(sd) == 3:
                                if len(sd[1]) == 3:
                                    sd[1] = {
                                        "JAN": "01",
                                        "FEB": "02",
                                        "MAR": "03",
                                        "APR": "04",
                                        "MAY": "05",
                                        "JUN": "06",
                                        "JUL": "07",
                                        "AUG": "08",
                                        "SEP": "09",
                                        "OCT": "10",
                                        "NOV": "11",
                                        "DEC": "12",
                                    }[sd[1].upper()]
                                    dx = sep.join(sd)
                                r = datetime.datetime.strptime(
                                    dx, "%d" + sep + "%m" + sep + "%Y"
                                ).strftime("%Y-%m-%d")
                                break
                        except Exception as e:
                            if isinstance(e, ValueError):
                                pass
                            else:
                                raise e

        if r is None:
            raise RuntimeError("Could not convert date to ISO format.")
    elif isinstance(date, (datetime.datetime, pd.Timestamp, np.datetime64)):
        r = date.strftime("%Y-%m-%d")
    else:
        raise TypeError(
            "Argument `date` must be a string, datetime.datetime, pd.Timestamp or np.datetime64."
        )

    assert is_valid_iso_date(r), "Failed to convert date to ISO format."
    return r


def convert_iso_to_dq(date: str) -> str:
    if is_valid_iso_date(date):
        r = date.replace("-", "")
        assert len(r) == 8, "Date formatting failed"
        return r
    else:
        raise ValueError("Incorrect date format, should be YYYY-MM-DD")


def convert_dq_to_iso(date: str) -> str:
    if len(date) == 8:
        r = datetime.datetime.strptime(date, "%Y%m%d").strftime("%Y-%m-%d")
        assert is_valid_iso_date(r), "Failed to format date"
        return r
    else:
        raise ValueError("Incorrect date format, should be YYYYMMDD")


def form_full_url(url: str, params: Dict = {}) -> str:
    """
    Forms a full URL from a base URL and a dictionary of parameters.
    Useful for logging and debugging.

    :param <str> url: base URL.
    :param <dict> params: dictionary of parameters.

    :return <str>: full URL
    """
    return requests.compat.quote(
        (f"{url}?{requests.compat.urlencode(params)}" if params else url),
        safe="%/:=&?~#+!$,;'@()*[]",
    )


##############################
#   Dataframe Functions
##############################


def standardise_dataframe(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    idx_cols: list = ["cid", "xcat", "real_date"]
    if not set(df.columns).issuperset(set(idx_cols)):
        fail_str: str = (
            f"Error : Tried to standardize DataFrame but failed."
            f"DataFrame not in the correct format. Please ensure "
            f"that the DataFrame has the following columns: "
            f"'cid', 'xcat', 'real_date', along with any other "
            "variables you wish to include (e.g. 'value', 'mop_lag', "
            "'eop_lag', 'grading')."
        )

        try:
            dft: pd.DataFrame = df.reset_index()
            found_cols: list = dft.columns.tolist()
            fail_str += f"\nFound columns: {found_cols}"
            assert set(dft.columns).issuperset(set(idx_cols)), fail_str
            df = dft.copy()
        except:
            raise ValueError(fail_str)

        # check if there is atleast one more column
        if len(df.columns) < 4:
            raise ValueError(fail_str)

        df["real_date"] = pd.to_datetime(df["real_date"], format="%Y-%m-%d")
        df["cid"] = df["cid"].astype(str)
        df["xcat"] = df["xcat"].astype(str)
        df = df.sort_values(by=["cid", "xcat", "real_date"])
        df = df.reset_index(drop=True)

        non_idx_cols: list = sorted(list(set(df.columns) - set(idx_cols)))
        return df[idx_cols + non_idx_cols]


def wide_to_long(
    df: pd.DataFrame,
    wide_var: str = "cid",
    val_col: str = "value",
) -> pd.DataFrame:
    """
    Converts a wide dataframe to a long dataframe.

    Parameters
    :param <pd.DataFrame> df: The dataframe to be converted.
    :param <str> wide_var: The variable name of the wide variable.
        In case the columns are ... cid_1, cid_2, cid_3, ... then
        wide_var should be "cid", else "xcat" or "real_date" must be
        passed.

    Returns
    :return <pd.DataFrame>: The converted dataframe.
    """
    idx_cols: list = ["cid", "xcat", "real_date"]

    if not isinstance(df, pd.DataFrame):
        raise ValueError("Error: The input must be a pandas DataFrame.")

    if wide_var not in ["cid", "xcat", "real_date"]:
        raise ValueError(
            "Error: The wide_var must be one of 'cid', 'xcat', 'real_date'."
        )

    """ 
    if wide_var == "cid":
     then the columns are real_date, xcat, cidX, cidY, cidZ, ...
     convert to real_date, xcat, cid, value
    """
    # use stack and unstack to convert to long format
    df = df.set_index(idx_cols).stack().reset_index()
    df.columns = idx_cols + [wide_var, val_col]

    return standardise_dataframe(df)


##############################
#   Class Definitions
##############################


class Config(object):
    def __init__(
        self,
        config_path: Optional[str] = None,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        crt: Optional[str] = None,
        key: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        proxy: Optional[dict] = None,
        proxies: Optional[dict] = None,
    ):
        """
        Class for handling the configuration of the JPMaQS API.

        Initialising Parameters

        :param <str> config_path: The path to the config file. If set to
            None or "env", the class will attempt to load the config
            file from the following environment variables:
            For OAuth authentication:
                - JPMAQS_API_CLIENT_ID : your_client_id
                - JPMAQS_API_CLIENT_SECRET : your_client_secret

            For certificate authentication:
                - JPMAQS_API_CRT : path_to_crt_file
                - JPMAQS_API_KEY : path_to_key_file
                - JPMAQS_API_USERNAME : your_username
                - JPMAQS_API_PASSWORD : your_password

            For proxy settings:
                - JPMAQS_API_PROXY : proxy_json_string

                (See https://requests.readthedocs.io/en/master/user/advanced/#proxies)
        """

        if not isinstance(config_path, (str, type(None))):
            raise ValueError(
                "Config file must be a string containing the path"
                " to the config file or None. Use `env` to use"
                " environment variables."
            )

        oauth_var_names: List[str] = ["client_id", "client_secret"]
        cert_var_names: List[str] = ["crt", "key", "username", "password"]
        oauth_vars: List[Optional[str]] = [client_id, client_secret]
        cert_vars: List[Optional[str]] = [crt, key, username, password]
        proxy_vars: Optional[dict] = [proxy, proxies]
        proxy_var_names: List[str] = ["proxy", "proxies"]

        loaded_vars: Dict[str, Optional[str]] = {
            var: None for var in oauth_var_names + cert_var_names + proxy_var_names
        }

        if isinstance(config_path, str):
            if config_path == "env":
                for var in loaded_vars.keys():
                    loaded_vars[var] = os.environ.get(f"JPMAQS_API_{var.upper()}", None)

            else:
                config_dict: Optional[dict] = None
                if config_path.endswith(".json"):
                    with open(config_path, "r") as f:
                        config_dict = json.load(f)
                elif config_path.endswith(".yaml") or config_path.endswith(".yml"):
                    with open(config_path, "r") as f:
                        config_dict = yaml.safe_load(f)

                if not config_dict:
                    raise ValueError("Config file could not be loaded.")
                else:
                    # make all keys lowercase
                    config_dict = {k.lower(): v for k, v in config_dict.items()}
                    for var in loaded_vars.keys():
                        loaded_vars[var] = rec_search_dict(d=config_dict, key=var, match_type=str)

                    for var in proxy_var_names:
                        loaded_vars[var] = rec_search_dict(d=config_dict, key=var, match_type=dict)

                    if loaded_vars["crt"] is None:
                        loaded_vars["crt"] = rec_search_dict(d=config_dict, key="cert",
                                                             match_substring=True, match_type=str)

        all_args_present: Callable = lambda x: all([v is not None for v in x])

        # overwrite any loaded variables with any passed variables
        for pr_args, pr_arg_names in zip(
            [oauth_vars, cert_vars, proxy_vars],
            [oauth_var_names, cert_var_names, proxy_var_names],
        ):
            for arg, arg_name in zip(pr_args, pr_arg_names):
                if arg is not None:
                    loaded_vars[arg_name] = arg

        r_auth: Dict[str, Optional[str]] = {}
        # any complete set of credentials will now be in r_auth
        for vx, varsx in zip(
            [oauth_var_names, cert_var_names, proxy_var_names],
            ["oauth", "cert", "proxy", "proxy"],
        ):
            if all_args_present([loaded_vars[v] for v in vx]):
                r_auth[varsx] = {v: loaded_vars[v] for v in vx}
            
        for px, pxn in zip(proxy_vars, proxy_var_names):
            if px is not None:
                r_auth[pxn] = px

        if not r_auth:
            raise ValueError(
                f"Failed to load authentication details from config file or environment variables."
                f" Please ensure the environment variables or config file path the specification"
                " in the package documentation."
            )

        if "cert" in r_auth.keys():
            for kx in ["crt", "key"]:
                if not os.path.isfile(r_auth["cert"][kx]):
                    raise FileNotFoundError(
                        f"Please ensure the file path - {r_auth['cert'][kx]} - is correct, "
                        "and the program has read access to the file."
                    )

        for pkx in proxy_var_names:
            if loaded_vars[pkx] is not None:
                if isinstance(loaded_vars[pkx], str):
                    try:
                        loaded_vars[pkx] = json.loads(loaded_vars[pkx])
                    except json.decoder.JSONDecodeError:
                        raise ValueError(
                            "Proxy settings could not be loaded. Please ensure the proxy settings are"
                            " in the correct format.\n Problematic string: \n"
                            f"{loaded_vars[pkx]}"
                        )

                assert isinstance(
                    loaded_vars[pkx], dict
                ), "Proxy settings must be a dictionary or JSON-like string."
        if "proxy" in r_auth.keys():
            if r_auth["proxy"] is not None:
                if "proxies" in r_auth["proxy"].keys():
                    if r_auth["proxy"]["proxies"] is not None:
                        r_auth["proxy"].update(r_auth["proxy"]["proxies"])
                        del r_auth["proxy"]["proxies"]
                if "proxy" in r_auth["proxy"].keys():
                    if r_auth["proxy"]["proxy"] is not None:
                        r_auth["proxy"].update(r_auth["proxy"]["proxy"])
                        del r_auth["proxy"]["proxy"]


        self._credentials: Dict[str, dict] = r_auth
        self.config_path: Optional[str] = config_path

        self._config_type: Optional[str] = "yaml"  # default
        if (not isinstance(config_path, type(None))) and config_path.endswith(".json"):
            self._config_type = "json"

    def oauth(self, mask: bool = True):
        if "oauth" not in self._credentials.keys():
            return None
        else:
            rdict: Dict[str, str] = {
                "client_id": self._credentials["oauth"]["client_id"],
                "client_secret": self._credentials["oauth"]["client_secret"],
            }
            if mask:
                ix: int = (
                    (len(self._credentials["oauth"]["client_id"]) - 4)
                    if (len(self._credentials["oauth"]["client_id"]) > 4)
                    else (len(self._credentials["oauth"]["client_id"]))
                )
                rdict["client_id"] = (
                    "*" * ix + self._credentials["oauth"]["client_id"][ix:]
                )
                rdict["client_secret"] = "*" * len(
                    self._credentials["oauth"]["client_secret"]
                )
            return rdict

    def cert(self, mask: bool = True):
        if "cert" not in self._credentials.keys():
            return None
        else:
            key: str = self._credentials["cert"]["key"]
            rdict: Dict[str, str] = {
                "username": self._credentials["cert"]["username"],
                "password": self._credentials["cert"]["password"],
                "crt": self._credentials["cert"]["crt"],
                "key": key,
            }
            if mask:
                ix: int = (
                    (len(self._credentials["cert"]["username"]) - 4)
                    if len(self._credentials["cert"]["username"]) > 4
                    else len(self._credentials["cert"]["username"])
                )
                rdict["username"] = (
                    "*" * ix + self._credentials["cert"]["username"][ix:]
                )
                for kx in ["password", "crt", "key"]:
                    rdict[kx] = "*" * len(rdict[kx])

            return rdict

    def proxy(self, mask: bool = False):
        if not ("proxy" in self._credentials):
            return None
        else:
            rdict: Dict[str, dict] = {}
            for kx in self._credentials["proxy"].keys():
                rdict[kx] = self._credentials["proxy"][kx]
                if mask:
                    rdict[kx] = "*" * len(rdict[kx])

            return rdict

    def credentials(
        self,
        mask: bool = False,
    ) -> Dict[str, dict]:
        rdict: Dict[str, dict] = {}
        for k in self._credentials.keys():
            rdict[k] = getattr(self, k)(mask=mask)

        return rdict

    def export_credentials(
        self,
        format: str = "json",
        export_file: str = None,
        mask: bool = False,
        oauth_only: bool = False,
        cert_only: bool = False,
        proxies: bool = True,
    ):
        if format not in ["json", "yaml", "yml"]:
            raise ValueError("Format must be either `json`, `yaml` or `yml`.")

        if oauth_only and cert_only:
            raise ValueError(
                "Only one of `oauth_only` or `cert_only` can be set to True."
            )

        if export_file is None:
            export_file = f"./jpmaqs_api_credentials_({datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}).{format}"

        output_dict: Dict[str, dict] = {}
        credentials: Dict[str, dict] = self.credentials(mask=mask)

        if (
            proxies
            and ("proxy" in credentials.keys())
            or ("proxies" in credentials.keys())
        ):
            output_dict["proxy"] = credentials["proxy"]
            output_dict["proxies"] = credentials["proxies"]

        if oauth_only:
            output_dict["oauth"] = credentials["oauth"]
        elif cert_only:
            output_dict["cert"] = credentials["cert"]

        output: str
        if format == "json":
            output = json.dumps(output_dict, indent=2)
        else:
            output = yaml.dump(output_dict)

        if not any([export_file.endswith(ext) for ext in [".json", ".yaml", ".yml"]]):
            print(
                f"Adding file extension to export file : {format}\n"
                f"Export file : {export_file}"
            )

            export_file = f"{export_file}.{format}"

        with open(export_file, "w") as f:
            f.write(output)

        return export_file
    
    def multi_credentials(self):
        assert (self.config_path is not None) and (self.config_path != "env"), \
            "MultiCredentials can only be used with a config file."
        rdict: Dict[str, dict] = {}
        """
        look for multi cred in the dict like :
        multi_cred:
            cred0:
                cl_id: "wfdfdfs"
                cl_sc: "fw;mfwlkfemw"
            cred1:
                cl_id: "wklfnwflknwe"
                cl_sc: "dsfnjlskfdn"
            cred2:...
        """
        new_dict : Dict[str, dict] = {}
        if self.config_path.endswith(".json"):
            with open(self.config_path, "r") as f:
                new_dict = json.load(f)
        elif self.config_path.endswith(".yaml") or self.config_path.endswith(".yml"):
            with open(self.config_path, "r") as f:
                new_dict = yaml.safe_load(f)
        else:
            raise ValueError(f"Unknown config file extension : {self.config_path}")
        
        if "multi_cred" not in new_dict.keys():
            raise ValueError("No multi_cred key in config file.")
        
        rs : List[Config] = [] 
        n : dict = new_dict.copy()
        for clid, clsc in [[n["multi_cred"][kx]["cl_id"], 
                            n["multi_cred"][kx]["cl_sc"]] for kx in n["multi_cred"].keys()]:
            rs.append(Config(client_id=clid, client_secret=clsc, proxy=self.proxy(mask=False)))

        return rs

    def __repr__(self):
        return f"JPMaQS API Config Object, methods : {list(self._credentials.keys())}"

    def __str__(self):
        creds_str: str
        if self._config_type == "json":
            creds_str = json.dumps(self.credentials(mask=True), indent=2)
        else:
            creds_str = yaml.dump(self.credentials(mask=True))
        return (
            f"JPMaQS API Config Object, methods : {list(self._credentials.keys())} \n"
            f"Credentials : {creds_str}"
        )
