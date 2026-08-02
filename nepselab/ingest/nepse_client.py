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

    def sectors(self) -> pd.DataFrame:
        return pd.DataFrame(self._retry(self._api.get_sectors, "sectors"))

    def disclosures(self) -> pd.DataFrame:
        return pd.DataFrame(self._retry(self._api.get_company_disclosures, "disclosures"))

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
