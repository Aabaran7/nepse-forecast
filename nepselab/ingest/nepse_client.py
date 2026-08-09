"""Wrapper around nepse_scraper: retries, rate limiting, DataFrame returns.

NEPSE's endpoints are flaky and its TLS setup is non-standard (plain curl fails
where the library's session succeeds), so every call goes through _retry.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable

import pandas as pd
from nepse_scraper import Nepse_scraper

log = logging.getLogger(__name__)

NEPSE_INDEX_ID = 58
SENSITIVE_INDEX_ID = 57


class NepseClient:
    def __init__(self, min_interval: float = 0.7, max_retries: int = 4):
        self._api = Nepse_scraper()
        self.min_interval = min_interval
        self.max_retries = max_retries
        self._last_call = 0.0

    def _retry(self, fn: Callable[[], Any], what: str) -> Any:
        delay = 1.0
        for attempt in range(1, self.max_retries + 1):
            wait = self.min_interval - (time.monotonic() - self._last_call)
            if wait > 0:
                time.sleep(wait)
            try:
                out = fn()
                self._last_call = time.monotonic()
                return out
            except Exception as exc:  # noqa: BLE001 - upstream raises bare Exception
                self._last_call = time.monotonic()
                if attempt == self.max_retries:
                    raise
                log.warning("%s failed (%s), retry %d/%d in %.1fs",
                            what, type(exc).__name__, attempt, self.max_retries, delay)
                time.sleep(delay)
                delay *= 2

    # --- reference data -------------------------------------------------

    def securities(self) -> pd.DataFrame:
        return pd.DataFrame(self._retry(self._api.get_securities_list, "securities_list"))

    def company_info(self) -> pd.DataFrame:
        """Every listing with its SECTOR. Not the same call as securities().

        `get_securities_list` returns ids and names only. This one adds
        sectorName, companyName, instrumentType and regulatoryBody -- and the
        sector is the piece that makes the other 16 archived sector indices
        usable, and a P/E readable (19.6 means nothing until you know the
        banking median). 645 rows including debentures and mutual funds.
        """
        return pd.DataFrame(self._retry(self._api.get_all_securities, "all_securities"))

    def sectors(self) -> pd.DataFrame:
        return pd.DataFrame(self._retry(self._api.get_sectors, "sectors"))

    def disclosures(self) -> pd.DataFrame:
        return pd.DataFrame(self._retry(self._api.get_company_disclosures, "disclosures"))

    def sector_indices(self) -> pd.DataFrame:
        """The 17 tradable indices (ids 51-67), including NEPSE (58) and Sensitive (57)."""
        path = self._api.endpoints["sector_index_api"]["api"]
        return pd.DataFrame(self._retry(lambda: self._api.session.get(path).json(), "sector_index"))

    def market_summary_history(self, start: str, end: str) -> pd.DataFrame:
        """Market-wide turnover / traded shares / transactions / tradedScrips.

        Same rolling-window caveat as index_history: the date arguments are
        accepted and ignored. They are passed anyway so the call reads honestly
        and starts working if NEPSE ever honours them.
        """
        path = self._api.endpoints["market_summary_history_api"]["api"]
        rows = self._paged(path, {"startDate": start, "endDate": end}, "market_summary_history")
        return _normalise_dates(pd.DataFrame(rows))

    # --- price history --------------------------------------------------
    #
    # Both history endpoints return a Spring Data page envelope
    # ({content, totalPages, number, ...}), not the bare list the library's
    # type hints claim. get_indices_history() also exposes no page parameter,
    # so we drive the session directly and unwrap `content` ourselves.

    def _paged(self, path: str, params: dict, what: str, size: int = 500) -> list[dict]:
        rows: list[dict] = []
        page = 0
        while True:
            body = self._retry(
                lambda p=page: self._api.session.get(
                    path, params={**params, "page": p, "size": size}
                ).json(),
                f"{what}[p{page}]",
            )
            if isinstance(body, list):  # endpoint returned a bare list after all
                rows.extend(body)
                break
            chunk = body.get("content", [])
            rows.extend(chunk)
            total_pages = body.get("totalPages", 1)
            page += 1
            if page >= total_pages or not chunk:
                break
            if page > 200:
                log.warning("%s: pagination guard hit at page %d", what, page)
                break
        return rows

    def index_history(self, start: str, end: str, index_id: int = NEPSE_INDEX_ID) -> pd.DataFrame:
        """Daily index OHLC.

        WARNING: `start` and `end` are ignored by NEPSE. The endpoint returns a
        rolling ~225-session (~1 year) window no matter what range you request,
        and does not error on an out-of-range one -- ask for 2018 and you get
        last year's data with a 200. Verified in scripts/phase1_probe_depth.py.
        Anything needing a longer sample must come from data/archive/.
        """
        path = f"{self._api.endpoints['head_indices_api']['api']}/{index_id}"
        rows = self._paged(
            path, {"startDate": start, "endDate": end}, f"index_history({index_id})"
        )
        return _normalise_dates(pd.DataFrame(rows))

    def ticker_history(self, ticker: str, start: str, end: str, size: int = 500) -> pd.DataFrame:
        """Scrip daily history.

        Goes through the library method rather than the session: the security id
        is a *path* segment (`.../price/{security_id}`), and the library owns the
        symbol -> id resolution. Passing `symbol` as a query param returns 400.
        """
        rows: list[dict] = []
        page = 0
        while True:
            body = self._retry(
                lambda p=page: self._api.get_ticker_price_history(
                    ticker, start, end, page=p, size=size
                ),
                f"ticker_history({ticker})[p{page}]",
            )
            if isinstance(body, list):
                rows.extend(body)
                break
            chunk = body.get("content", [])
            rows.extend(chunk)
            page += 1
            if page >= body.get("totalPages", 1) or not chunk:
                break
            if page > 200:
                log.warning("%s: pagination guard hit", ticker)
                break
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        df["symbol"] = ticker
        return _normalise_dates(df)

    def today_price(self, business_date: str | None = None) -> pd.DataFrame:
        raw = self._retry(
            lambda: self._api.get_today_price(business_date),
            f"today_price({business_date})",
        )
        return _normalise_dates(pd.DataFrame(raw))


def _normalise_dates(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    for col in ("businessDate", "date", "publishedDate"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
            df = df.sort_values(col).reset_index(drop=True)
            break
    return df
