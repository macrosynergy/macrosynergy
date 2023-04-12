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
from typing import Any, List, Dict, Optional, Callable
import itertools


##############################
#   Helpful Functions
##############################


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


def rec_search_dict(d: dict, key: str, match_substring: bool = False) -> str:
    """
    Recursively searches a dictionary for a key and returns the value
    associated with it.

    Parameters
    :param <dict> d: The dictionary to be searched.
    :param <str> key: The key to be searched for.
    :param <bool> match_substring: If True, the function will return
        the value of the first key that contains the substring
        specified by the key parameter. If False, the function will
        return the value of the first key that matches the key
        parameter exactly.

    Returns
    :return <str>: The value associated with the key.
    """
    result: Any = None
    for k, v in d.items():
        if match_substring:
            if key in k:
                return v
        else:
            if k == key:
                return v

        if isinstance(v, dict):
            item = rec_search_dict(v, key, match_substring)
            if item is not None:
                result = item
                break

    return result


def is_valid_iso_date(date: str) -> bool:
    try:
        datetime.datetime.strptime(date, "%Y-%m-%d")
        return True
    except ValueError:
        return False


def convert_to_iso_format(date: Any = None) -> str:
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
        # try and load the string as dd/mm/yyyy
        try:
            r = datetime.datetime.strptime(date, "%d/%m/%Y").strftime("%Y-%m-%d")
            return
        except ValueError:
            pass


def convert_iso_to_dq(date: str) -> str:
    if is_valid_iso_date(date):
        r = date.replace("-", "")
        assert len(r) == 8, "Date formatting failed"
        return r
    else:
        raise ValueError("Incorrect date format, should be YYYY-MM-DD")


def convert_dq_to_iso(date: str) -> str:
    if len(date) == 8:
        r = date[:4] + "-" + date[4:6] + "-" + date[6:]
        assert is_valid_iso_date(r), "Date format incorrect"
        return r
    else:
        raise ValueError("Incorrect date format, should be YYYYMMDD")


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


class JPMaQSAPIConfigObject(object):
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

        if not isinstance(config_path, (str, None)):
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
                for var in loaded_vars.keys():
                    loaded_vars[var] = rec_search_dict(config_dict, var, True)

                if loaded_vars["crt"] is None:
                    loaded_vars["crt"] = rec_search_dict(config_dict, "cert", True)

        all_args_present: Callable = lambda x: all([v is not None for v in x])

        # overwrite any loaded variables with any passed variables
        for pr_args in [oauth_vars, cert_vars, proxy_vars]:
            if all_args_present(pr_args):
                for var, val in zip(loaded_vars.keys(), pr_args):
                    loaded_vars[var] = val

        r_auth: Dict[str, Optional[str]] = {}
        # any complete set of credentials will now be in r_auth
        for vx, varsx in zip(
            [oauth_var_names, cert_var_names, proxy_var_names],
            ["oauth", "cert"] + proxy_var_names,
        ):
            if all_args_present([loaded_vars[v] for v in vx]):
                r_auth[varsx] = {v: loaded_vars[v] for v in vx}

        if not r_auth:
            raise ValueError(
                f"Failed to load authentication details from config file or environment variables."
                f" Please ensure the environment variables or config file path the specification"
                " in the package documentation."
            )

        if "cert" in r_auth.keys():
            for kx in ["crt", "key"]:
                if not os.path.isfile(r_auth["cert"][kx]):
                    raise ValueError(
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

                r_auth[pkx] = loaded_vars[pkx]


        self._credentials: Dict[str, dict] = r_auth

        self._config_type: Optional[str] = "yaml"  # default
        if config_path.endswith(".json"):
            self._config_type = "json"

    def oauth(self, mask: bool = True):
        if "oauth" not in self._credentials.keys():
            raise ValueError("OAuth credentials not found.")
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
    
    def proxy(self, mask : bool = True):
        if not(("proxy" in self._credentials) or ("proxies") in self._credentials):
            return None
        else:
            rdict: dict = {}

            for prx in ["proxy", "proxies"]:
                if prx in self._credentials.keys():
                    for kx in self._credentials[prx].keys():
                        rdict[kx] = self._credentials[prx][kx]
                        if mask:
                            rdict[kx] = "*" * len(rdict[kx])
            return rdict


    def credentials(self, mask: bool = True,
                    print_json: bool = False,
                    print_yaml: bool = False,
                    ) -> Dict[str, dict]:
        
        if not isinstance(mask, bool):
            raise ValueError("`mask` must be a boolean.")
        if not isinstance(print_json, bool):
            raise ValueError("`print_json` must be a boolean.")
        if not isinstance(print_yaml, bool):
            raise ValueError("`print_yaml` must be a boolean.")
        if print_json and print_yaml:
            raise ValueError("Only one of `print_json` or `print_yaml` can be True.")

        output_dict: Dict[str, dict] = {}
        if "oauth" in self._credentials.keys():
            output_dict["oauth"] = self.oauth(mask=mask)
        if "cert" in self._credentials.keys():
            output_dict["cert"] = self.cert(mask=mask)
        if "proxy" in self._credentials.keys():
            output_dict["proxy"] = self.proxy(mask=mask)

        if print_json:
            print(json.dumps(output_dict, indent=4))
            return

        if print_yaml:
            print(yaml.dump(output_dict, indent=4))
            return

        return output_dict

    def export_credentials(
        self,
        format: str = "yaml",
        export_file: str = None,
        mask: bool = False,
        oauth_only: bool = False,
        cert_only: bool = False,
        proxies: bool = True,
    ):
        if format not in ["json", "yaml", "yml"]:
            raise ValueError("Format must be either `json`, `yaml` or `yml`.")
        elif format == "yml":
            format = "yaml"

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
        if format == "yaml":
            output = json.dumps(output_dict, indent=2)
        else:
            output = yaml.dump(output_dict)

        if not any([export_file.endswith(ext) for ext in [".json", ".yaml", ".yml"]]):
            fm = ".yml" if format == "yaml" else ".json"
            export_file = f"{export_file}{fm}"
            print(
                f"Exporting to file : {export_file}"
            )

        with open(export_file, "w") as f:
            f.write(output)

        return export_file

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
