"""Sanity report for the deep index series (plan §3.3, extended to §3.5's data).

`scripts/phase0_quality.py` covers the Phase 0 pull in `data/raw`. That series
is 225 exchange-sourced sessions; this one is 2,434 sessions of third-party
history, and it is the sample every model will actually be fitted on. §8 says
sanity tests pass before any experiment runs, so this has to exist before
Phase 2 does.

Checks live in `nepselab/quality.py` so this script and `tests/test_quality.py`
run the same code rather than two descriptions of it.

Usage: .venv/bin/python scripts/phase1d_deep_quality.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nepselab import quality  # noqa: E402

DEEP = Path("data/deep/nepse_index_deep.parquet")
PARAMS = Path("configs/market_params.yaml")
RULE = "=" * 74


def hr(title: str) -> None:
    print(f"\n{RULE}\n{title}\n{RULE}")


def main() -> None:
    if not DEEP.exists():
        print(f"{DEEP} not found -- run scripts/phase1c_deep_history.py first")
        sys.exit(1)

    df = pd.read_parquet(DEEP).sort_values("date").reset_index(drop=True)
    eras = yaml.safe_load(PARAMS.read_text()).get("trading_week", [])

    hr("DEEP SERIES SANITY REPORT")
    print(f"{DEEP}: {len(df)} sessions, "
          f"{df.date.min().date()} .. {df.date.max().date()}")
    print(f"source(s): {', '.join(sorted(df['source'].unique()))}")

    hr("1. CHECKS")
    findings = quality.run_all(df, eras)
    worst = 0
    for f in findings:
        mark = "PASS" if f.passed else "FAIL"
        print(f"  [{mark}] {f.name:<28} {f.detail}")
        worst += 0 if f.passed else 1

    for f in findings:
        if f.passed or f.rows.empty:
            continue
        hr(f"DETAIL: {f.name}")
        cols = [c for c in ("date", "open", "high", "low", "close", "ret",
                            "weekday", "violation", "excess", "gap_days")
                if c in f.rows.columns]
        print(f.rows[cols].head(25).to_string(index=False))
        if len(f.rows) > 25:
            print(f"  ... and {len(f.rows) - 25} more")

    hr("2. BASELINE (the bar §6 measures against)")
    for label, sub in (("full 2016+", df),
                       ("mania 2020-06..2021-12",
                        df[(df.date >= "2020-06-01") & (df.date <= "2021-12-31")]),
                       ("chop 2022+", df[df.date >= "2022-01-01"]),
                       ("pre-mania 2016..2020-02",
                        df[(df.date >= "2016-01-01") & (df.date <= "2020-02-29")])):
        b = quality.direction_baseline(sub)
        print(f"  {label:<26} n={b['n']:>5}  up={b['up_share']:.3f}  "
              f"majority={b['majority_class']} at {b['majority_share']:.3f}")
    print("\n  Per §2 these differ by regime; do not assume a single baseline.")

    hr("3. CALENDAR PROFILE (discovery -- read it, don't assert on it)")
    prof = quality.weekday_profile(df, freq="Y")
    order = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday",
             "Friday", "Saturday"]
    prof = prof[[c for c in order if c in prof.columns]]
    prof.index = prof.index.year
    print(prof.to_string())
    print("\n  An era boundary shows up here and nowhere else -- the prices stay")
    print("  perfectly well-formed across it.")

    hr("4. CLOSURES")
    gaps = quality.calendar_gaps(df)
    print(f"  {len(gaps)} gap(s) > 4 calendar days")
    big = gaps[gaps.gap_days > 10]
    if len(big):
        print("  gaps > 10 days (resumption date, length):")
        for _, r in big.iterrows():
            print(f"    {r.date.date()}  {int(r.gap_days)} days")

    hr("VERDICT")
    if worst == 0:
        print("All checks pass. Safe to build features on.")
    else:
        print(f"{worst} check(s) failed. Per §8, experiments do not run until")
        print("each is either fixed or has a written policy in the plan.")
    sys.exit(1 if worst else 0)


if __name__ == "__main__":
    main()
