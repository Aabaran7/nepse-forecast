"""Phase 1c: does a usable NEPSE index history exist outside NEPSE's API?

This is §8.1 option D, the branch the plan said to resolve before choosing among
the others: "If a clean, verifiable daily index series back to 2020 can be
obtained, the original plan survives essentially intact."

The answer is a qualified yes, and the qualification is the interesting part.
Two independent sources both offer daily NEPSE index bars from 1997. Both match
the exchange almost perfectly over the 231 sessions we can actually check. But
against EACH OTHER, before 2016, they disagree about the sign of the daily return
on roughly one day in twenty-four -- and the sign of the daily return is the
target. Pre-2016 history is available and unusable; that distinction is what this
script exists to establish rather than assume.

It does three things and writes the accepted series to data/deep/:

  1. Scores each source against the archived exchange sessions (ground truth).
  2. Scores the sources against each other, per year, across 1997-2026.
  3. Derives the earliest year from which the series is contiguously trustworthy,
     then recomputes the §2 power table on the sample that survives.

Usage: .venv/bin/python scripts/phase1c_deep_history.py [--max-disagree 0.5]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nepselab.ingest import deep_history as dh  # noqa: E402
from phase1_power import P0, min_detectable, n_required  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("phase1c")

RULE = "=" * 78


def head(title: str) -> None:
    print(f"\n{RULE}\n{title}\n{RULE}")


def report_vs_archive(name: str, src: pd.DataFrame, arc: pd.DataFrame) -> None:
    a = dh.compare(arc, src)
    print(f"\n{name} vs archive: {a.n_common}/{len(arc)} archived sessions matched")
    if a.only_a:
        print(f"  MISSING from {name}: {[str(d.date()) for d in a.only_a]}")
    for c in ("open", "high", "low", "close"):
        if c in a.exact:
            flag = "" if a.exact[c] == a.n_common else "   <-- not exact"
            print(f"  {c:<6} exact {a.exact[c]:>3}/{a.n_common}"
                  f"  max abs diff {a.max_abs[c]:.4f}{flag}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-disagree", type=float, default=0.5,
                    help="max %% of days two sources may disagree on return sign")
    args = ap.parse_args()

    head("PHASE 1C: DEEP HISTORY SOURCING (plan §8.1 option D)")

    arc = dh.archive_index()
    print(f"archive (exchange, ground truth): {len(arc)} sessions, "
          f"{arc.date.min().date()} .. {arc.date.max().date()}")

    log.info("fetching merolagani ...")
    ml = dh.fetch_merolagani()
    log.info("fetching github dump ...")
    gh = dh.fetch_github()
    for n, s in (("merolagani", ml), ("github", gh)):
        print(f"{n:<12} {len(s):>5} sessions, {s.date.min().date()} .. {s.date.max().date()}")

    # --- 1. each source against the only rows we can actually verify ----------
    head("1. AGAINST THE EXCHANGE (the 231 sessions we archived ourselves)")
    report_vs_archive("merolagani", ml, arc)
    report_vs_archive("github", gh, arc)
    print("\nBoth reproduce the exchange essentially exactly over the verified")
    print("window. That is necessary and nowhere near sufficient: it says nothing")
    print("about 2016-2024, which is precisely the data NEPSE no longer serves.")

    # --- 2. the sources against each other, where nothing can adjudicate ------
    head("2. AGAINST EACH OTHER, 1997-2026 (no ground truth exists here)")
    yearly = dh.yearly_disagreement(ml, gh)
    show = yearly[["n", "close_mismatch", "rel_gt_10bp", "max_rel",
                   "sign_disagree", "sign_disagree_pct", "only_a", "only_b"]]
    print(show.to_string(float_format=lambda v: f"{v:.3f}"))
    print("\nonly_a = dates merolagani has and github lacks; only_b the reverse.")

    # --- 3. where does the series become trustworthy? -------------------------
    head(f"3. USABLE WINDOW (rule: sign disagreement <= {args.max_disagree}% "
         f"in every subsequent year)")
    start = dh.usable_start(yearly, args.max_disagree)
    if start is None:
        print("NO usable window. Neither source can be trusted at any depth.")
        return

    pre = yearly[yearly.index < start]
    post = yearly[yearly.index >= start]
    for label, blk in (("before", pre), (f"{start}+", post)):
        if not len(blk):
            continue
        n, sd = int(blk["n"].sum()), int(blk["sign_disagree"].sum())
        print(f"  {label:<8} n={n:>5}  sign disagreements={sd:>4} "
              f"({100 * sd / max(n, 1):.2f}%)")

    print(f"\nUSABLE FROM {start}-01-01.")

    # Independent second reason the same boundary falls where it does: the older
    # bars are close-only. Worth printing the actual counts rather than asserting
    # it, since the changeover is gradual across 2015-2016 rather than a switch.
    flat = ml.assign(
        flat=(ml.open == ml.high) & (ml.high == ml.low) & (ml.low == ml.close),
        year=ml.date.dt.year,
    ).groupby("year")["flat"].agg(["sum", "size"])
    before = flat[flat.index < start]
    after = flat[flat.index >= start]
    print(f"  flat bars (open=high=low=close, i.e. close-only):")
    print(f"    before {start}: {int(before['sum'].sum()):>5}/{int(before['size'].sum())} sessions")
    print(f"    {start}+      : {int(after['sum'].sum()):>5}/{int(after['size'].sum())} sessions")
    print("  So the pre-2016 era carries no intraday range at all, on top of the")
    print("  two sources disagreeing about its closes. Both reasons point here.")

    accepted = ml[ml.date >= f"{start}-01-01"].copy()

    # The close series being contiguous does not make every column contiguous.
    breaks = dh.turnover_scale_breaks(accepted)
    if len(breaks):
        print(f"\n  WARNING -- turnover jumps at {len(breaks)} month boundary(ies):")
        for when, ratio in breaks.items():
            print(f"    {when.date()}  x{ratio:,.0f}")
        print("    Prices are continuous across all of these. Judge each one:")
        print("    a units change moves the turnover start date, a real market")
        print("    event does not. TURNOVER FEATURES MUST NOT SPAN A UNITS BREAK.")

    path = dh.save(accepted, "nepse_index_deep")
    print(f"\nwrote {len(accepted)} rows -> {path}")
    print("  source: merolagani (231/231 exact closes vs exchange; github had 228/230")
    print("  and was missing one archived session outright)")
    print("  NOT written into data/archive/ -- that store is exchange-sourced and")
    print("  irreplaceable; this one is re-downloadable and only as good as its feed.")

    # --- 4. what this does to the §2 power table ------------------------------
    head("4. §2 POWER TABLE, RECOMPUTED ON THE SURVIVING SAMPLE")
    n_daily = len(accepted)
    n_train = 500
    scen = [
        ("h=1 daily, full usable sample", n_daily),
        ("h=1 daily, walk-forward test set", n_daily - n_train),
        ("h=5 weekly, non-overlapping blocks", n_daily // 5),
        ("h=5 weekly, walk-forward test blocks", (n_daily - n_train) // 5),
    ]
    print(f"baseline p0={P0:.3f} (majority class), alpha=0.05 one-sided, power=0.80\n")
    print(f"{'scenario':<44}{'n':>7}{'min edge':>11}{'was (225-session)':>20}")
    print("-" * 78)
    was = {0: "8.2pp", 1: "14.0pp", 2: "17.8pp", 3: "29.2pp"}
    for i, (label, n) in enumerate(scen):
        print(f"{label:<44}{n:>7}{min_detectable(P0, n) * 100:>10.1f}pp{was[i]:>20}")

    need = n_required(P0, 0.02)
    blocks = n_daily // 5
    print(f"\n§6 abandons on h=5 accuracy <= majority + 2pp.")
    print(f"  weekly blocks needed : {need:,.0f}")
    print(f"  weekly blocks now    : {blocks:,}")
    print(f"  shortfall            : {need / blocks:.1f}x  (was 85x)")
    print(f"\nThe h=1 daily test is now nearly resolvable "
          f"({min_detectable(P0, n_daily) * 100:.1f}pp vs a 2pp bar).")
    print("The h=5 weekly test §6 actually specifies is not, and dividing by 5 for")
    print("non-overlapping blocks is why. Deep history rescues the sample; it does")
    print("not rescue that particular threshold.")


if __name__ == "__main__":
    main()
