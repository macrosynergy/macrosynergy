CERT_BASE_URL: str = "https://platform.jpmorgan.com/research/dataquery/api/v2"
OAUTH_BASE_URL: str = (
    "https://api-developer.jpmorgan.com/research/dataquery-authe/api/v2"
)
OAUTH_TOKEN_URL: str = "https://authe.jpmchase.com/as/token.oauth2"
OAUTH_DQ_RESOURCE_ID: str = "JPMC:URI:RS-06785-DataQueryExternalApi-PROD"
JPMAQS_GROUP_ID: str = "JPMAQS"
API_DELAY_PARAM: float = 0.3  # 300ms delay between requests
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
