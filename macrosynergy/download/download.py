from typing import Any, Dict, List, Optional, Tuple, Union

from macrosynergy.download.aws_lambda import AWSLambdaInterface
from macrosynergy.download.dataquery import DataQueryInterface

CERT_BASE_URL: str = "https://platform.jpmorgan.com/research/dataquery/api/v2"
OAUTH_BASE_URL: str = (
    "https://api-developer.jpmorgan.com/research/dataquery-authe/api/v2"
)
OAUTH_TOKEN_URL: str = "https://authe.jpmchase.com/as/token.oauth2"
OAUTH_DQ_RESOURCE_ID: str = "JPMC:URI:RS-06785-DataQueryExternalApi-PROD"
JPMAQS_GROUP_ID: str = "JPMAQS"
API_DELAY_PARAM: float = 0.2  # 200ms delay between requests
TOKEN_EXPIRY_BUFFER: float = 0.9  # 90% of token expiry time.
API_RETRY_COUNT: int = 5  # retry count for transient errors
HL_RETRY_COUNT: int = 5  # retry count for "high-level" requests
MAX_CONTINUOUS_FAILURES: int = 5  # max number of continuous errors before stopping
HEARTBEAT_ENDPOINT: str = "/services/heartbeat"
TIMESERIES_ENDPOINT: str = "/expressions/time-series"
CATALOGUE_ENDPOINT: str = "/group/instruments"
HEARTBEAT_TRACKING_ID: str = "heartbeat"
OAUTH_TRACKING_ID: str = "oauth"
TIMESERIES_TRACKING_ID: str = "timeseries"
CATALOGUE_TRACKING_ID: str = "catalogue"


class DownloadInterface(DataQueryInterface, AWSLambdaInterface):
    """
    Routing class for downloading data from a source.
    """
    
    def __init__(
        self,
        oauth: bool = True,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        crt: Optional[str] = None,
        key: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        check_connection: bool = True,
        proxy: Optional[Dict] = None,
        suppress_warning: bool = True,
        debug: bool = False,
        print_debug_data: bool = False,
        dq_download_kwargs: dict = {},
        source="DataQuery",
        *args,
        **kwargs,
    ):
        """
        Initialize the class.
        """
        if source == "DataQuery":
            self.source = source
            DataQueryInterface.__init__(
                self,
                oauth=oauth,
                client_id=client_id,
                client_secret=client_secret,
                crt=crt,
                key=key,
                username=username,
                password=password,
                proxy=proxy,
                check_connection=check_connection,
                suppress_warning=suppress_warning,
                debug=debug,
                *args,
                **kwargs,
            )  # Initialize only DataQueryInterface
        elif source == "AWSLambda":
            self.source = source
            AWSLambdaInterface.__init__(
                self,
                access_key_id=client_id,
                secret_access_key=client_secret,
                aws_session_token=None,
                cred_file=None,
                debug=debug,
                batch_size=kwargs.get("batch_size", 15),
                check_connection=check_connection,
                base_url=kwargs.get("base_url", None),
                token_url=kwargs.get("token_url", None),
                suppress_warning=suppress_warning,
                region="eu-west-2",
                service="lambda",
            )  # Initialize only AWSLambdaInterface
        else:
            raise ValueError("Unsupported source")

    def _get_unavailable_expressions(
        self,
        expected_exprs: List[str] = None,
        dicts_list: List[Dict] = None,
    ) -> List[str]:
        """
        Get the list of unavailable expressions.
        """
        if self.source == "DataQuery":
            return DataQueryInterface._get_unavailable_expressions(
                self, expected_exprs, dicts_list
            )
        elif self.source == "AWSLambda":
            return AWSLambdaInterface._get_unavailable_expressions(
                self, expected_exprs, dicts_list
            )
        else:
            pass

    def check_connection(self, verbose=False, raise_error: bool = False) -> bool:
        """
        Check the connection to the source.
        """
        if self.source == "DataQuery":
            return DataQueryInterface.check_connection(self, verbose, raise_error)
        elif self.source == "AWSLambda":
            return AWSLambdaInterface.check_connection(self, verbose, raise_error)
        else:
            pass

    def _fetch(
        self,
        url: str,
        params: dict = None,
        tracking_id: Optional[str] = None,
    ) -> List[Dict]:
        """
        Fetch data from the source.
        """
        if self.source == "DataQuery":
            return DataQueryInterface._fetch(self, url, params, tracking_id)
        elif self.source == "AWSLambda":
            return AWSLambdaInterface._fetch(self, url, params, tracking_id)
        else:
            pass

    def _fetch_timeseries(
        self, url: str, params: dict, tracking_id: str = None, *args, **kwargs
    ) -> List[Dict]:
        """
        Fetch timeseries data from the source.
        """
        if self.source == "DataQuery":
            return DataQueryInterface._fetch_timeseries(
                self, url, params, tracking_id, *args, **kwargs
            )
        elif self.source == "AWSLambda":
            return AWSLambdaInterface._fetch_timeseries(
                self, url, params, tracking_id, *args, **kwargs
            )
        else:
            pass

    def get_catalogue(
        self,
        group_id: str = "",
        verbose: bool = True,
    ) -> List[str]:
        """
        Get the catalogue of available expressions.
        """
        if self.source == "DataQuery":
            return DataQueryInterface.get_catalogue(self, group_id, verbose)
        elif self.source == "AWSLambda":
            return AWSLambdaInterface.get_catalogue(self, group_id, verbose)
        else:
            pass

    def _concurrent_loop(
        self,
        expr_batches: List[List[str]],
        show_progress: bool,
        url: str,
        params: dict,
        tracking_id: str,
        delay_param: float,
        *args,
        **kwargs,
    ) -> Tuple[List[Union[Dict, Any]], List[List[str]]]:
        """
        Concurrent loop to fetch data.
        """
        if self.source == "DataQuery":
            return DataQueryInterface._concurrent_loop(
                self,
                expr_batches,
                show_progress,
                url,
                params,
                tracking_id,
                delay_param,
                *args,
                **kwargs,
            )
        elif self.source == "AWSLambda":
            return AWSLambdaInterface._concurrent_loop(
                self,
                expr_batches,
                show_progress,
                url,
                params,
                tracking_id,
                delay_param,
                *args,
                **kwargs,
            )
        else:
            pass

    def _chain_download_outputs(
        self,
        download_outputs: List[Union[Dict, Any]],
    ) -> List[Dict]:
        """
        Chain the download outputs.
        """
        if self.source == "DataQuery":
            return DataQueryInterface._chain_download_outputs(self, download_outputs)
        elif self.source == "AWSLambda":
            return AWSLambdaInterface._chain_download_outputs(self, download_outputs)
        else:
            pass

    def _download(
        self,
        expressions: List[str],
        params: dict,
        url: str,
        tracking_id: str,
        delay_param: float,
        show_progress: bool = False,
        retry_counter: int = 0,
        *args,
        **kwargs,
    ) -> List[dict]:
        """
        Backend method to download data from the source.
        """
        if self.source == "DataQuery":
            return DataQueryInterface._download(
                self,
                expressions,
                params,
                url,
                tracking_id,
                delay_param,
                show_progress,
                retry_counter,
                *args,
                **kwargs,
            )
        elif self.source == "AWSLambda":
            return AWSLambdaInterface._download(
                self,
                expressions,
                params,
                url,
                tracking_id,
                delay_param,
                show_progress,
                retry_counter,
                *args,
                **kwargs,
            )
        else:
            pass

    def download_data(
        self,
        expressions: List[str],
        start_date: str = "2000-01-01",
        end_date: str = None,
        show_progress: bool = False,
        endpoint: str = TIMESERIES_ENDPOINT,
        calender: str = "CAL_ALLDAYS",
        frequency: str = "FREQ_DAY",
        conversion: str = "CONV_LASTBUS_ABS",
        nan_treatment: str = "NA_NOTHING",
        reference_data: str = "NO_REFERENCE_DATA",
        retry_counter: int = 0,
        delay_param: float = API_DELAY_PARAM,
        batch_size: Optional[int] = None,
        *args,
        **kwargs,
    ) -> List[Dict]:
        """
        Download data from the source.
        """
        if self.source == "DataQuery":
            return DataQueryInterface.download_data(
                self,
                expressions,
                start_date,
                end_date,
                show_progress,
                endpoint,
                calender,
                frequency,
                conversion,
                nan_treatment,
                reference_data,
                retry_counter,
                delay_param,
                batch_size,
                *args,
                **kwargs,
            )
        elif self.source == "AWSLambda":
            return AWSLambdaInterface.download_data(
                self,
                expressions,
                start_date,
                end_date,
                show_progress,
                endpoint,
                calender,
                frequency,
                conversion,
                nan_treatment,
                reference_data,
                retry_counter,
                delay_param,
                batch_size,
                *args,
                **kwargs,
            )
        else:
            pass
