DQ_FILE_API_BASE_URL: str = (
    "https://api-dataquery.jpmchase.com/research/dataquery-authe/api/v2"
)
DQ_FILE_API_FALLBACK_BASE_URL: str = (
    "https://api-strm-gw01.jpmchase.com/research/dataquery-authe/api/v2"
)
DQ_FILE_API_SCOPE: str = "JPMC:URI:RS-06785-DataQueryExternalApi-PROD"
DQ_FILE_API_TIMEOUT: float = 300.0
DQ_FILE_API_HEADERS_TIMEOUT: float = 60.0
DQ_FILE_API_DELAY_PARAM: float = 0.04  # =1/25 ; 25 transactions per second
DQ_FILE_API_DELAY_MARGIN: float = 1.05  # 5% safety margin
DQ_FILE_API_SEGMENT_SIZE_MB: float = 8.0  # 8 MB
DQ_FILE_API_STREAM_CHUNK_SIZE: int = 8192  # 8 KB


JPMAQS_EARLIEST_FILE_DATE = "20220101"

JPMAQS_DATASET_THEME_MAPPING = {
    "Economic surprises": "JPMAQS_ECONOMIC_SURPRISES",
    "Financial conditions": "JPMAQS_FINANCIAL_CONDITIONS",
    "Generic returns": "JPMAQS_GENERIC_RETURNS",
    "Macroeconomic balance sheets": "JPMAQS_MACROECONOMIC_BALANCE_SHEETS",
    "Macroeconomic trends": "JPMAQS_MACROECONOMIC_TRENDS",
    "Shocks and risk measures": "JPMAQS_SHOCKS_RISK_MEASURES",
    "Stylized trading factors": "JPMAQS_STYLIZED_TRADING_FACTORS",
}
