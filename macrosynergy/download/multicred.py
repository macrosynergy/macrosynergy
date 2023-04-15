from typing import Dict, List, Optional, Union
from macrosynergy.download import JPMaQSDownload
from macrosynergy.download.dataquery import OAUTH_BASE_URL
from macrosynergy.management.utils import Config
import pandas as pd
from timeit import default_timer as timer
from tqdm import tqdm
from joblib import Parallel, delayed

###############
# Class MultiCredentials


class MultiCredentialDownload:
    def __init__(self, config: Config, base_url: str = OAUTH_BASE_URL):
        self.config = config
        self._base_url = base_url

    def _download(
        self,
        tickers: Optional[List[str]] = None,
        cids: Optional[List[str]] = None,
        xcats: Optional[List[str]] = None,
        metrics: List[str] = ["all"],
        start_date: str = "2000-01-01",
        end_date: Optional[str] = None,
        show_progress: bool = True,
        debug: bool = False,
        suppress_warning: bool = False,
        report_time_taken: bool = True,
    ) -> Dict[str, dict]:

        all_expressions = JPMaQSDownload.construct_expressions(
            tickers=tickers, cids=cids, xcats=xcats, metrics=metrics
        )

        # split the expressions into n chunks, where n is the number of credentials
        mcred: List[Config] = self.config.multi_credentials()

        chunk_size = len(all_expressions) // len(mcred)
        chunks = [
            all_expressions[i : i + chunk_size]
            for i in range(0, len(all_expressions), chunk_size)
        ]

        results = []
        start: float = timer()
        results = Parallel(n_jobs=4)(
            delayed(JPMaQSDownload(**cred.oauth(mask=False)).download)(
                expressions=chunk,
                start_date=start_date,
                end_date=end_date,
                as_dataframe=False,
                show_progress=show_progress,
            )
            for cred, chunk in tqdm(zip(mcred, chunks), total=len(mcred))
        )
        end: float = timer()
        if report_time_taken:
            print(f"MultiCred time taken: {end - start} seconds")

        # combine the results
        results = [item for sublist in results for item in sublist]
        return results


if __name__ == "__main__":
    from macrosynergy.management.utils import Config

    tickers: List[str] = [
        "USD_IMPINFB1Y_NSA",
        "SGD_GGPBGDPRATIO_NSA",
        "AUD_DU02YCRY_NSA",
        "AUD_DU02YCRY_VT10",
        "AUD_DU02YXR_NSA",
        "AUD_DU02YXR_VT10",
        "AUD_DU02YXRxEASD_NSA",
        "AUD_DU02YXRxLEV10_NSA",
        "AUD_DU05YCRY_NSA",
        "AUD_DU05YCRY_VT10",
        "AUD_DU05YXR_NSA",
        "AUD_DU05YXR_VT10",
        "AUD_DU05YXRxEASD_NSA",
        "AUD_DU05YXRxLEV10_NSA",
        "CHF_DU02YCRY_NSA",
        "CHF_DU02YCRY_VT10",
        "CHF_DU02YXR_NSA",
        "CHF_DU02YXR_VT10",
        "CHF_DU02YXRxEASD_NSA",
    ]

    mcd = MultiCredentialDownload(config=Config("./config_multicred.yml"))

    x = mcd._download(
        tickers=tickers,
        metrics=["all"],
        start_date="1990-01-01",
        end_date="2020-01-01",
        show_progress=True,
        report_time_taken=True,
    )
