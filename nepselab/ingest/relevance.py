"""Which headlines could plausibly move NEPSE. A filter for cost, not for truth.

This exists for one practical reason: a general newspaper's business page runs
gold prices, phone tariffs, hotel awards and English-teaching conferences
alongside anything about the exchange. Sending all of it to a paid model costs
money to learn that a monsoon cashback offer is not a market signal.

Three rules govern what this is allowed to do.

  IT TAGS, IT NEVER DELETES. Every scraped headline is archived either way. The
  tag decides what gets *scored*, not what gets *kept* -- so if the rule here
  turns out to be wrong, the headlines it dismissed are still there to rescore.
  Dropping them at scrape time would make that mistake permanent, and the sites
  cannot be re-scraped for history (news.py: pagination is a lie).

  IT ERRS TOWARDS INCLUDING. A false positive costs a fraction of a cent. A
  false negative silently removes the one story that mattered, and nothing
  downstream can tell the difference between "filtered out" and "never
  happened". So the keyword list is broad and the company matcher is generous.

  IT IS NOT A SENTIMENT JUDGEMENT. "Relevant" here means "could touch the
  market", not "is bullish" or "is important". A story about a bank being fined
  is relevant. Whether that is bad news is the model's problem, not this file's.

Market-specific sources (ShareSansar, MeroLagani) are relevant by construction
and are not filtered at all -- their entire output is exchange news.
"""

from __future__ import annotations

import logging
import re

import pandas as pd

log = logging.getLogger(__name__)

# Sources whose every article is exchange news. Filtering these would only ever
# throw away signal.
ALWAYS_RELEVANT_SOURCES = {"sharesansar", "merolagani"}

# Deliberately broad. Grouped so the reason a headline survived is legible when
# someone asks why a story about remittances was scored.
MARKET_TERMS = {
    # the exchange and its plumbing
    "nepse", "sebon", "cdsc", "stock", "share", "shares", "equity", "equities",
    "bourse", "broker", "brokerage", "demat", "mero share", "meroshare",
    "floorsheet", "market cap", "listed", "delisting", "circuit",
    # corporate actions -- the most reliably price-moving category here
    "ipo", "fpo", "rights share", "right share", "bonus share", "dividend",
    "auction", "book clos", "aug­ment", "merger", "acquisition", "acquire",
    "buyback", "promoter share", "lock-in", "capital increase",
    # results and company finance
    "profit", "loss", "earnings", "quarter", "net profit", "revenue",
    "turnover", "npl", "non-performing", "provision", "capital adequacy",
    # policy and macro that reaches the index
    "nrb", "nepal rastra bank", "monetary policy", "interest rate",
    "lending rate", "liquidity", "cash reserve", "inflation", "remittance",
    "budget", "finance act", "capital gains tax", "cgt", "tax", "gdp",
    "foreign exchange", "forex", "reserves", "trade deficit",
    # sectors that dominate the listings
    "bank", "banking", "microfinance", "insurance", "hydropower", "hydro",
    "power company", "electricity", "mutual fund", "finance company",
    "development bank", "life insurance", "non-life",
}

# Nepali equivalents. MeroLagani is not filtered, but the Himalayan Times
# occasionally runs Devanagari, and a Nepali headline that scores zero against
# an English-only list would be dismissed for the wrong reason entirely.
MARKET_TERMS_NE = {
    "सेयर", "शेयर", "नेप्से", "लाभांश", "बोनस", "हकप्रद", "बैंक", "बीमा",
    "जलविद्युत", "लघुवित्त", "नाफा", "घाटा", "ब्याज", "राष्ट्र बैंक",
    "पुँजी", "कारोबार", "लगानी", "धितोपत्र",
}

# Reliably NOT market news, even inside a business section. Only used to break a
# tie when nothing above matched -- never to override a positive match, so
# "Nepal Premier League sponsor lists on NEPSE" still counts as relevant.
NOISE_HINTS = {
    "cricket", "football", "premier league", "match", "tournament", "goal",
    "weather", "rainfall", "temperature", "monsoon forecast", "earthquake",
    "arrested", "murder", "theft", "smuggl", "accident", "died", "death toll",
    "festival", "movie", "film", "actor", "singer", "concert",
}

_WORD = re.compile(r"[a-z]+")


def company_names(securities: pd.DataFrame | None = None) -> tuple[set[str], set[str]]:
    """(ticker symbols, two-word company phrases) for matching against a headline.

    The first version of this matched SINGLE tokens from company names and was
    useless in a way worth recording. Listed names decompose into ordinary
    English -- Premier Insurance, Central Finance, First Micro Finance, Himalayan
    Power -- so "Gold falls Rs 5,800" matched a company containing "Falls",
    "Three arrested" matched one containing "Three", and "the first few hours of
    a snakebite" matched "First". 12 of 14 supposed company hits were garbage.

    So: symbols only as whole uppercase words (a ticker is distinctive; the
    lowercase word "api" is not), and names only as two-word phrases, which is
    the shortest unit that actually identifies a company.
    """
    if securities is None:
        from nepselab.ingest import archive
        securities = archive.load("securities")
    if securities.empty:
        return set(), set()

    stop = {"limited", "ltd", "company", "co", "nepal", "nepalese", "the", "and",
            "public", "national", "development", "general", "investment",
            "holdings", "group", "industries", "industry", "corporation",
            "of", "for", "pvt", "private"}

    symbols = {str(s).strip().upper()
               for s in securities.get("symbol", pd.Series(dtype=str)).dropna().unique()
               if len(str(s).strip()) >= 3}

    phrases: set[str] = set()
    for name in securities.get("securityName", pd.Series(dtype=str)).dropna().unique():
        toks = [t for t in _WORD.findall(str(name).lower())
                if len(t) > 2 and t not in stop]
        for a, b in zip(toks, toks[1:]):
            phrases.add(f"{a} {b}")
    return symbols, phrases


def score(title: str, source: str = "",
          names: tuple[set[str], set[str]] | None = None) -> tuple[bool, str]:
    """(is_relevant, why). `why` is stored so a filtering decision is auditable."""
    if source in ALWAYS_RELEVANT_SOURCES:
        return True, "market-only source"

    t = (title or "").lower()
    if not t:
        return False, "empty title"

    hits = sorted({k for k in MARKET_TERMS if k in t})
    if hits:
        return True, "market term: " + ", ".join(hits[:3])

    ne_hits = sorted({k for k in MARKET_TERMS_NE if k in title})
    if ne_hits:
        return True, "market term (ne): " + ", ".join(ne_hits[:3])

    if names:
        symbols, phrases = names
        # Symbols must appear as uppercase words in the ORIGINAL title. Lowered,
        # tickers like API, BOK and SHINE are ordinary English.
        upper = set(re.findall(r"\b[A-Z]{3,}\b", title or ""))
        sym_hit = sorted(symbols & upper)
        if sym_hit:
            return True, "ticker: " + ", ".join(sym_hit[:3])
        phrase_hit = sorted(p for p in phrases if p in t)
        if phrase_hit:
            return True, "company: " + ", ".join(phrase_hit[:2])

    noise = sorted({k for k in NOISE_HINTS if k in t})
    if noise:
        return False, "no market term; looks like " + noise[0]
    return False, "no market term matched"


def tag(df: pd.DataFrame, securities: pd.DataFrame | None = None) -> pd.DataFrame:
    """Add `market_relevant` and `relevance_reason`. Adds columns, drops nothing."""
    if df.empty:
        return df
    names = company_names(securities)
    out = df.copy()
    scored = [score(t, s, names) for t, s in zip(out["title"], out.get("source", ""))]
    out["market_relevant"] = [r for r, _ in scored]
    out["relevance_reason"] = [w for _, w in scored]
    return out
