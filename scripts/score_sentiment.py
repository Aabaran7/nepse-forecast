"""Rate headlines bullish / bearish / neutral, and say which company they mention.

READ THIS BEFORE USING THE OUTPUT FOR ANYTHING
----------------------------------------------
Plan §5 gates sentiment behind validation against ~200 hand-labelled headlines.
That has not happened, so **these scores are not a model input**. They are
collected, stored and displayed, and nothing in nepselab/features/ reads them.
The gate is not bureaucratic: an unvalidated sentiment feature is indisting-
uishable from noise with a confident label on it, and §6 has already shown this
project will happily produce a rule that looks fine and is not.

WHY THE MODEL ALSO DOES ENTITY EXTRACTION
Matching headlines to tickers with string rules was tried and failed: 5 of 42
matched, and one of those was "Mero Share" (a CDSC website) matched to the
ticker MERO. A model that has read the sentence knows the difference. So it
returns the company as written, and THIS code validates that against the real
securities list -- the model proposes, the archive disposes. A ticker it invents
does not survive.

COST CONTROL, IN ORDER OF HOW MUCH THEY SAVE
  1. Only relevance-tagged headlines are sent (relevance.py). ~16 of 42 today.
  2. Already-scored headlines are never re-sent: the cache key is
     (url_hash, scorer_version), so a re-run costs nothing and a prompt change
     is an explicit version bump rather than a silent re-bill.
  3. Headlines are batched, so one request covers many.
  4. --dry-run and --limit exist, and --dry-run is free.

Usage:
    .venv/bin/python scripts/score_sentiment.py --dry-run
    .venv/bin/python scripts/score_sentiment.py --limit 20
    .venv/bin/python scripts/score_sentiment.py
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nepselab.ingest import archive, news, relevance  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("score_sentiment")

NEWS_ROOT = Path(news.NEWS_DIR)

# Bump this whenever the prompt or the model changes. It is half the cache key,
# so a bump means "score everything again under the new rules" -- and leaves the
# old scores in place rather than overwriting them, exactly like model_version
# in the forward log (§7).
# v2: v1's prompt asked the model to name "the listed company", which it cannot
# know and which cost real matches -- Standard Chartered came back null despite
# being SCB. v1's scores are kept, not overwritten; the version is half the cache
# key precisely so a prompt change is auditable rather than silent.
SCORER_VERSION = "gpt4omini-headline-v2"
DEFAULT_MODEL = "gpt-4o-mini"

BATCH = 15

SYSTEM = """You rate Nepali stock market news headlines for a research project.

For each headline, judge how a NEPSE investor would most likely read it:
  bullish  - suggests higher prices for a company, sector, or the market
  bearish  - suggests lower prices
  neutral  - factual, procedural, or genuinely ambiguous

Rules:
- Headlines may be in English or Nepali (Devanagari). Treat both the same.
- Rate the LIKELY MARKET READING, not whether the news is good for society.
  A bank fined by the regulator is bearish for that bank.
- Most headlines are neutral. Procedural notices, AGM dates, and routine
  announcements are neutral. Do not manufacture signal.
- If the headline names a specific company or organisation, return its name as
  written. Do NOT try to judge whether it is listed on NEPSE -- you cannot know
  that, and guessing costs real matches: asking for "the listed company" made
  you return null for Standard Chartered, which is listed as SCB. Name whoever
  is named; something downstream checks the exchange's own list.
  Return null only when no company is named at all -- market-wide, sector, or
  economy stories.
- confidence is your own certainty, 0 to 1.

Return ONLY a JSON object: {"results": [...]}, one entry per headline, in the
same order, each: {"i": <index>, "sentiment": "bullish|bearish|neutral",
"score": <-1..1>, "company": <string or null>, "confidence": <0..1>}"""


def load_env(path: Path = Path(".env")) -> None:
    """Minimal .env reader. Avoids a dependency for four lines of parsing."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def already_scored(version: str) -> set[str]:
    df = archive.load("sentiment", root=NEWS_ROOT)
    if df.empty:
        return set()
    return set(df.loc[df["scorer_version"] == version, "url_hash"])


def resolve_symbol(company: str | None, securities: pd.DataFrame) -> str | None:
    """Turn the model's company string into a real ticker, or nothing.

    The model is not trusted to know NEPSE's ticker set -- it is trusted to
    isolate the entity from the sentence, which is the part string rules could
    not do. Anything that does not match a listed name is dropped rather than
    guessed at, so a hallucinated symbol cannot reach the panel.
    """
    if not company or securities.empty:
        return None
    c = re.sub(r"[^a-z ]", " ", str(company).lower())
    c = re.sub(r"\b(limited|ltd|company|co|pvt|private|nepal)\b", " ", c)
    toks = [t for t in c.split() if len(t) > 2]
    if not toks:
        return None

    candidates: list[tuple[int, str]] = []
    for _, row in securities.iterrows():
        name = re.sub(r"[^a-z ]", " ", str(row.get("securityName", "")).lower())
        name_toks = set(name.split())
        n = sum(1 for t in toks if t in name_toks)
        # Two matching words is the floor. One is how "Premier Insurance"
        # matched a headline containing the word "premier" last time.
        if n >= 2:
            candidates.append((n, str(row.get("symbol"))))
    if not candidates:
        return None

    # One company name maps to many symbols: Kumari Bank Limited is KBL, KBLD86,
    # KBLD89, KBLD90, KEF, KDBY and KSY -- ordinary shares plus debentures and
    # mutual funds. News about "Kumari Bank" is about the equity, so prefer a
    # symbol with no digits (debentures carry the maturity year) and then the
    # shortest. Without this the answer depends on row order, which is a bug
    # that would look like it worked.
    best_n = max(n for n, _ in candidates)
    top = [s for n, s in candidates if n == best_n]
    return min(top, key=lambda s: (any(ch.isdigit() for ch in s), len(s), s))


def score_batch(client, model: str, rows: list[dict]) -> tuple[list[dict], dict]:
    listing = "\n".join(f'{i}. {r["title"]}' for i, r in enumerate(rows))
    resp = client.chat.completions.create(
        model=model,
        temperature=0,                       # same headline, same score
        response_format={"type": "json_object"},
        messages=[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": listing}],
    )
    body = json.loads(resp.choices[0].message.content)
    usage = {"prompt": resp.usage.prompt_tokens,
             "completion": resp.usage.completion_tokens}
    return body.get("results", []), usage


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true",
                    help="show what would be sent and roughly how many tokens. Free.")
    ap.add_argument("--limit", type=int, default=None,
                    help="score at most N headlines this run")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--version", default=SCORER_VERSION)
    args = ap.parse_args()

    load_env()

    path = NEWS_ROOT / "headlines.parquet"
    if not path.exists():
        log.error("no headlines yet; run scripts/scrape_news.py first")
        return 1

    securities = archive.load("securities")
    heads = relevance.tag(pd.read_parquet(path), securities)

    done = already_scored(args.version)
    todo = heads[heads["market_relevant"] & ~heads["url_hash"].isin(done)]
    todo = todo.sort_values("first_seen_utc", ascending=False)
    if args.limit:
        todo = todo.head(args.limit)

    log.info("%d headlines stored | %d market-relevant | %d already scored at %s "
             "| %d to score now",
             len(heads), int(heads["market_relevant"].sum()), len(done),
             args.version, len(todo))

    if todo.empty:
        log.info("nothing to do")
        return 0

    rows = todo.to_dict("records")

    if args.dry_run:
        chars = sum(len(r["title"]) for r in rows) + len(SYSTEM) * (
            (len(rows) + BATCH - 1) // BATCH)
        print(f"\n{'=' * 70}\nWOULD SEND {len(rows)} headline(s) in "
              f"{(len(rows) + BATCH - 1) // BATCH} request(s)\n{'=' * 70}")
        for r in rows[:10]:
            print(f"  [{r['source']:<14}] {r['title'][:72]}")
        if len(rows) > 10:
            print(f"  ... and {len(rows) - 10} more")
        print(f"\n~{chars // 4:,} input tokens (rough: 4 chars/token), plus "
              f"~{len(rows) * 40:,} output tokens.")
        print("Multiply by your provider's current per-million price. Nothing "
              "was sent and nothing was charged.")
        return 0

    if not os.environ.get("OPENAI_API_KEY"):
        log.error("OPENAI_API_KEY not set. Put it in .env (see .env.example) "
                  "or export it. --dry-run works without a key.")
        return 1

    from openai import OpenAI
    client = OpenAI()

    out, totals = [], {"prompt": 0, "completion": 0}
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")

    for start in range(0, len(rows), BATCH):
        chunk = rows[start:start + BATCH]
        try:
            results, usage = score_batch(client, args.model, chunk)
        except Exception as exc:  # noqa: BLE001
            log.error("batch %d failed (%s); keeping what scored so far",
                      start // BATCH + 1, exc)
            break
        totals["prompt"] += usage["prompt"]
        totals["completion"] += usage["completion"]

        by_i = {int(r["i"]): r for r in results if "i" in r}
        for i, src in enumerate(chunk):
            r = by_i.get(i)
            if r is None:
                log.warning("no result for %r; left unscored", src["title"][:50])
                continue
            company = r.get("company")
            out.append({
                "url_hash": src["url_hash"],
                "scorer_version": args.version,
                "source": src["source"],
                "title": src["title"],
                "sentiment": r.get("sentiment"),
                "score": r.get("score"),
                "confidence": r.get("confidence"),
                "company_raw": company,
                "symbol": resolve_symbol(company, securities),
                "model": args.model,
                "scored_utc": stamp,
            })
        log.info("  scored %d/%d", min(start + BATCH, len(rows)), len(rows))

    if not out:
        log.error("nothing scored")
        return 1

    res = archive.merge("sentiment", pd.DataFrame(out), root=NEWS_ROOT,
                        ignore=["scored_utc"])
    archive.record([res], root=NEWS_ROOT)

    df = pd.DataFrame(out)
    print(f"\n{'=' * 70}\nSCORED {len(df)} HEADLINE(S)\n{'=' * 70}")
    print(df["sentiment"].value_counts().to_string())
    linked = df["symbol"].notna().sum()
    print(f"\nlinked to a listed company: {linked}/{len(df)}")
    if linked:
        print(df[df["symbol"].notna()][["symbol", "sentiment", "title"]]
              .head(10).to_string(index=False))
    print(f"\ntokens: {totals['prompt']:,} in + {totals['completion']:,} out")
    print("Multiply by your provider's current per-million price for the cost.")
    print("\n§5: these scores are NOT a model input until validated against "
          "~200 hand-labelled headlines.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
