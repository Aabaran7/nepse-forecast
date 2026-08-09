"""Daily headline scrape from Nepali financial news sites.

This is the one ingest path in the project with no API behind it. Prices come
from `nepse_client.py`, which wraps a maintained library and speaks JSON;
headlines exist only as HTML, so this module owns the parsing.

Three things make that harder than "call BeautifulSoup", and each is handled
here rather than left to callers:

  ATTRIBUTION, NOT TIMESTAMPS. A headline is scraped after the close, but the
  session it may explain is the NEXT one. Attributing a 17:00 NPT story to the
  same day's return is a look-ahead leak of exactly the kind plan §7 says the
  forward log exists to catch. `session_for()` maps every headline to the first
  session that could have traded on it, and that mapping -- not the scrape time
  -- is what any downstream feature must join on.

  LAYOUTS ROT. Each site is a `Source` with its own selector, and a source that
  returns zero rows raises rather than logging a shrug. A silent zero is
  indistinguishable from a quiet news day, and the archive would fill with
  holes nobody noticed -- the same failure mode as §3.4's missed sessions.

  THE PAGE IS NOT THE RECORD. Sites edit headlines in place. Storage is
  append-only via `nepselab.ingest.archive`, keyed on the article URL, so the
  first text we saw is the text we keep and a later revision surfaces as a
  recorded conflict instead of overwriting history.

Selectors verified against live HTML on 2026-08-09. All three sites are
server-rendered, so no browser is needed; `Source.render` is the escape hatch
for the day one of them goes client-side (see `render_with_playwright`).
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
import urllib.robotparser
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Callable, Iterable
from urllib.parse import urljoin, urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup

log = logging.getLogger(__name__)

# Kathmandu is UTC+5:45 and NEPSE closes at 15:00 NPT (configs/market_params.yaml,
# effective 2026-04-20). Anything published after this is tomorrow's information.
KTM = timezone(timedelta(hours=5, minutes=45))
MARKET_CLOSE_NPT = 15

# Identifies the scraper and gives site owners a route to complain, which is the
# point of the header. Deliberately a repo link rather than a personal address:
# this string is published in a public repo AND sent to every site we fetch.
USER_AGENT = (
    "nepselab-research/0.1 (personal NEPSE forecasting study; "
    "+https://github.com/Aabaran7/nepse-forecast)"
)

# Storage root. Deliberately NOT data/archive/ -- that directory holds the
# irreplaceable exchange data (§3.4) and headlines can be re-scraped from the
# sites' own archives if this is ever lost.
NEWS_DIR = "data/news"


class SourceEmpty(RuntimeError):
    """A source parsed cleanly but yielded nothing. Almost always a layout change."""


@dataclass
class Source:
    """One news site, reduced to the four things that differ between them."""

    name: str
    url: str
    item_selector: str          # CSS selector for one headline's container
    title_selector: str | None = None   # within the container; None = container text
    page_param: str | None = None       # e.g. "page" -> ?page=2 for pagination
    date_selector: str | None = None
    render: Callable[[str], str] | None = None  # None = plain HTTP GET
    encoding: str | None = None
    # Keep only articles whose own URL contains this. A section page is not a
    # section: the Himalayan Times /business page carries a sidebar of "latest
    # from everywhere", so a selector scoped to the page yielded sports results,
    # weather and crime alongside the business stories. Filtering on the
    # article's own path is also far more durable than CSS ancestry, which is
    # one redesign away from silently letting the sidebar back in.
    url_must_contain: str | None = None

    def page_url(self, page: int) -> str:
        if page <= 1 or not self.page_param:
            return self.url
        joiner = "&" if "?" in self.url else "?"
        return f"{self.url}{joiner}{self.page_param}={page}"


# --- the sites -------------------------------------------------------------
#
# ShareSansar is the primary: English, NEPSE-specific, ~10 stories/page, and its
# markup carries an explicit publication date. MeroLagani is Nepali-language and
# already trusted elsewhere in this project (it is the deep-history source, §3.5).
# The Himalayan Times is general business news -- broader, noisier, and included
# because a market-only feed cannot see a macro shock coming.

SOURCES: dict[str, Source] = {
    "sharesansar": Source(
        name="sharesansar",
        url="https://www.sharesansar.com/category/latest",
        item_selector="div.featured-news-list",
        title_selector="h4.featured-news-title",
        date_selector="span.text-org",
        # VERIFIED USELESS 2026-08-09: ?page=N is accepted and ignored -- pages
        # 1, 2 and 5 return byte-identical article sets, the same lie NEPSE tells
        # about startDate/endDate (§3.4). Left wired up because scrape_source
        # stops on the first page that adds nothing, so it costs one extra
        # request to find out if they ever fix it. There is NO history here:
        # headline backfill is impossible from this listing, which is why the
        # deep sentiment sample has to come from the Reddit dump instead.
        page_param="page",
    ),
    "merolagani": Source(
        name="merolagani",
        url="https://merolagani.com/NewsList.aspx",
        item_selector="h4.media-title",
        title_selector=None,
    ),
    "himalayantimes": Source(
        name="himalayantimes",
        url="https://thehimalayantimes.com/business",
        item_selector="h3.alith_post_title, h2.alith_post_title, h4.alith_post_title",
        title_selector=None,
        # Without this, 13 of 31 stored headlines came from /nepal, /sports,
        # /environment, /kathmandu and /opinion (measured 2026-08-09).
        url_must_contain="/business/",
    ),
}


# --- fetching --------------------------------------------------------------

class Fetcher:
    """Polite HTTP with one connection pool, a rate limit, and bounded retries.

    Mirrors `NepseClient`'s retry shape on purpose: same failure modes, same
    backoff, so there is one thing to reason about when a nightly run is slow.
    """

    def __init__(self, min_interval: float = 1.5, max_retries: int = 3,
                 timeout: float = 25.0, respect_robots: bool = True):
        self.min_interval = min_interval
        self.max_retries = max_retries
        self.timeout = timeout
        self.respect_robots = respect_robots
        self._last_call = 0.0
        self._robots: dict[str, urllib.robotparser.RobotFileParser] = {}
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml",
            "Accept-Language": "en-US,en;q=0.9,ne;q=0.8",
        })

    def allowed(self, url: str) -> bool:
        if not self.respect_robots:
            return True
        root = "{0.scheme}://{0.netloc}".format(urlparse(url))
        rp = self._robots.get(root)
        if rp is None:
            rp = urllib.robotparser.RobotFileParser()
            rp.set_url(root + "/robots.txt")
            try:
                rp.read()
            except Exception:  # noqa: BLE001
                # An unreachable robots.txt is not permission, but it is also not
                # a refusal. Default to allowed and say so, rather than silently
                # skipping a source for a reason nobody can see in the log.
                log.warning("%s: robots.txt unreachable, proceeding", root)
                rp = None
            self._robots[root] = rp
        return True if rp is None else rp.can_fetch(USER_AGENT, url)

    def get(self, url: str) -> str:
        if not self.allowed(url):
            raise PermissionError(f"robots.txt disallows {url}")
        delay = 2.0
        for attempt in range(1, self.max_retries + 1):
            wait = self.min_interval - (time.monotonic() - self._last_call)
            if wait > 0:
                time.sleep(wait)
            try:
                r = self.session.get(url, timeout=self.timeout)
                self._last_call = time.monotonic()
                r.raise_for_status()
                return r.text
            except Exception as exc:  # noqa: BLE001
                self._last_call = time.monotonic()
                if attempt == self.max_retries:
                    raise
                log.warning("GET %s failed (%s), retry %d/%d in %.0fs",
                            url, type(exc).__name__, attempt, self.max_retries, delay)
                time.sleep(delay)
                delay *= 2
        raise AssertionError("unreachable")


def render_with_playwright(url: str, wait_selector: str | None = None) -> str:
    """Fallback for a site that goes client-side. Not used by any source today.

    Kept as a function rather than a dependency: importing playwright costs a
    ~400 MB browser download in CI, and none of the three sources needs it. Wire
    it in per-source (`Source.render = render_with_playwright`) only when a
    parse starts returning zero rows *and* the HTML shows the list is missing.
    """
    from playwright.sync_api import sync_playwright  # imported late, on purpose

    with sync_playwright() as p:
        browser = p.chromium.launch()
        try:
            page = browser.new_page(user_agent=USER_AGENT)
            page.goto(url, wait_until="domcontentloaded", timeout=45_000)
            if wait_selector:
                page.wait_for_selector(wait_selector, timeout=20_000)
            return page.content()
        finally:
            browser.close()


# --- parsing ---------------------------------------------------------------

_WS = re.compile(r"\s+")


def clean(text: str) -> str:
    return _WS.sub(" ", text or "").strip()


def url_hash(url: str) -> str:
    """Stable 16-hex identity for an article.

    Hashing rather than storing the URL as the key keeps the archive key short
    and fixed-width, and strips query strings and fragments so the same article
    reached from two listing pages does not become two rows.
    """
    p = urlparse(url)
    canon = f"{p.netloc.lower().removeprefix('www.')}{p.path.rstrip('/')}"
    return hashlib.sha256(canon.encode()).hexdigest()[:16]


def parse_published(text: str | None) -> pd.Timestamp | None:
    """Best-effort publication date. None is an acceptable answer.

    ShareSansar renders "Sunday, August 9, 2026". MeroLagani and the Himalayan
    Times list markup carries no date at all on the index page. Rather than
    fetch every article to find out, an unknown date falls back to the scrape
    time in `session_for()` -- which is the conservative direction, because it
    can only ever push a headline FORWARD to a later session, never backward
    into one that had already traded.
    """
    if not text:
        return None
    t = clean(text)
    for fmt in ("%A, %B %d, %Y", "%B %d, %Y", "%Y-%m-%d", "%d %B %Y", "%b %d, %Y"):
        try:
            return pd.Timestamp(datetime.strptime(t, fmt))
        except ValueError:
            continue
    try:
        return pd.Timestamp(t)
    except Exception:  # noqa: BLE001
        return None


def parse(source: Source, html: str, base_url: str) -> list[dict]:
    soup = BeautifulSoup(html, "lxml")
    rows: list[dict] = []
    for node in soup.select(source.item_selector):
        title_node = node.select_one(source.title_selector) if source.title_selector else node
        if title_node is None:
            continue
        title = clean(title_node.get_text(" "))
        link = node.find("a", href=True)
        if not title or not link:
            continue
        url = urljoin(base_url, link["href"])
        if source.url_must_contain and source.url_must_contain not in url:
            continue
        published = None
        if source.date_selector:
            d = node.select_one(source.date_selector)
            published = parse_published(d.get_text(" ") if d else None)
        rows.append({
            "source": source.name,
            "url_hash": url_hash(url),
            "url": url,
            "title": title,
            "published": published,
        })
    return rows


def scrape_source(source: Source, fetcher: Fetcher, pages: int = 1) -> pd.DataFrame:
    """Scrape one source. Raises `SourceEmpty` rather than returning nothing.

    Pagination stops early on a page that adds no new URLs: some listings ignore
    an out-of-range page number and re-serve page 1, which would otherwise loop.
    """
    rows: list[dict] = []
    seen: set[str] = set()
    for page in range(1, pages + 1):
        url = source.page_url(page)
        html = source.render(url) if source.render else fetcher.get(url)
        page_rows = parse(source, html, url)
        fresh = [r for r in page_rows if r["url_hash"] not in seen]
        log.info("  %-16s page %d: %d items (%d new)",
                 source.name, page, len(page_rows), len(fresh))
        if not fresh:
            break
        seen.update(r["url_hash"] for r in fresh)
        rows.extend(fresh)

    if not rows:
        raise SourceEmpty(
            f"{source.name}: selector {source.item_selector!r} matched nothing at "
            f"{source.url}. Treat this as a layout change, not a quiet news day.")

    df = pd.DataFrame(rows)
    # Two of the three sources carry no date on the listing page, so `published`
    # arrives all-null for them. Fix the dtype here rather than letting concat
    # infer object and warn.
    df["published"] = pd.to_datetime(df["published"], errors="coerce")
    # Named for what it means once stored: merge() keeps the archived row on a
    # collision, so this is the first time we ever saw the article, not the last.
    df["first_seen_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return df


def scrape(names: Iterable[str] | None = None, pages: int = 1,
           fetcher: Fetcher | None = None) -> tuple[pd.DataFrame, dict[str, str]]:
    """Scrape every named source. One source failing never kills the others.

    Returns the combined frame and a per-source error map. Callers decide what a
    partial result means; `scripts/scrape_news.py` treats "every source failed"
    as an error and "some succeeded" as a warning, because the daily job must
    not go red over one site's outage.
    """
    fetcher = fetcher or Fetcher()
    frames, errors = [], {}
    for name in (names or SOURCES):
        src = SOURCES[name]
        try:
            frames.append(scrape_source(src, fetcher, pages=pages))
        except Exception as exc:  # noqa: BLE001
            errors[name] = f"{type(exc).__name__}: {exc}"
            log.error("%s failed: %s", name, exc)
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return combined, errors


# --- session attribution ---------------------------------------------------

def session_for(headline_time: pd.Timestamp, sessions: Iterable[date]) -> date | None:
    """The first trading session that could have traded on this headline.

    This is the module's only load-bearing piece of logic, and it exists because
    the alternative leaks. A story published at 17:00 NPT on a Thursday is not
    information the Thursday close reflects; it is information Sunday's -- now
    Friday's, after the 2026-04 week change -- open can act on. Feature code must
    join on this column, never on `scraped_utc`.

    Rule: same-day if published strictly before the 15:00 NPT close on a trading
    day, otherwise the next session in the calendar. `sessions` comes from the
    archived index history, which IS the trading calendar (see
    `scripts/archive_pull.py::sessions_to_pull`) -- so holidays and the 2022 and
    2026 trading-week changes are handled by construction rather than by a rule
    that would have to be maintained.
    """
    ts = pd.Timestamp(headline_time)
    ts = ts.tz_localize(timezone.utc) if ts.tz is None else ts
    local = ts.tz_convert(KTM)
    day, hour = local.date(), local.hour

    ordered = sorted(sessions)
    for s in ordered:
        if s > day or (s == day and hour < MARKET_CLOSE_NPT):
            return s
    return None  # the next session has not happened yet; resolve on a later run


def attribute(df: pd.DataFrame, sessions: Iterable[date]) -> pd.DataFrame:
    """Add `session` to a headline frame, preferring published time over scrape time.

    Call this at READ time, not before storing. A headline scraped after the
    close has no session yet -- the session it belongs to has not happened, so it
    is not in the calendar -- and it acquires one on the next trading day. Since
    the store is append-only, a `session` written as null would stay null for
    good. It is a pure function of (timestamp, calendar) and the calendar grows,
    so it is derived, never persisted.
    """
    if df.empty:
        return df
    df = df.copy()
    when = pd.to_datetime(df["published"], errors="coerce", utc=True)
    when = when.fillna(pd.to_datetime(df["first_seen_utc"], errors="coerce", utc=True))
    cal = sorted(sessions)
    df["session"] = [session_for(t, cal) for t in when]
    return df


def trading_sessions(index_df: pd.DataFrame | None = None) -> list[date]:
    """The trading calendar, read off the archived index history."""
    if index_df is None:
        from nepselab.ingest import archive
        index_df = archive.load("indices")
    if index_df.empty:
        return []
    return sorted(pd.to_datetime(index_df["businessDate"]).dt.date.unique())
