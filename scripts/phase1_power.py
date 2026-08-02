"""Phase 1: minimum detectable directional edge, given the sample we actually have.

Resolves the §2 TODO ("compute exactly and put the numbers in the results table
header"). §2 guessed ~4pp daily / ~7pp weekly off an assumed ~1500 observations.
The depth probe (scripts/phase1_probe_depth.py) showed the API serves ~225
sessions total, so those guesses need replacing with the real numbers.

One-sided test of a single proportion against a fixed baseline, alpha=0.05,
power=0.80. The baseline is the MAJORITY CLASS, not 50% -- per §2, only 45.5% of
index days were up in the verified window, so the majority class is `down` at
54.5% and that is the bar a directional model has to clear.

Usage: .venv/bin/python scripts/phase1_power.py
"""

from __future__ import annotations

from math import sqrt
from statistics import NormalDist

ALPHA = 0.05
POWER = 0.80
P0 = 0.545  # majority class (down) in the 2025-07..2026-07 verified window

Z_A = NormalDist().inv_cdf(1 - ALPHA)   # one-sided
Z_B = NormalDist().inv_cdf(POWER)


def n_required(p0: float, delta: float) -> float:
    """Sample size to detect p1 = p0 + delta at ALPHA/POWER, one-sided."""
    p1 = p0 + delta
    return (Z_A * sqrt(p0 * (1 - p0)) + Z_B * sqrt(p1 * (1 - p1))) ** 2 / delta ** 2


def min_detectable(p0: float, n: int) -> float:
    """Smallest delta detectable with n observations. Bisection on n_required."""
    lo, hi = 1e-4, 1 - p0 - 1e-6
    for _ in range(200):
        mid = (lo + hi) / 2
        if n_required(p0, mid) > n:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


SCENARIOS = [
    ("h=1 daily, full available sample",        225, "every session the API serves"),
    ("h=1 daily, walk-forward test set",         75, "225 minus a 150-session initial train window"),
    ("h=5 weekly, non-overlapping blocks",       45, "225 / 5; §2 requires non-overlapping for significance"),
    ("h=5 weekly, walk-forward test blocks",     15, "75 / 5"),
    ("-- what §2 ASSUMED --",                     0, ""),
    ("h=1 daily, assumed test set",            1000, "§2's assumption, not achievable"),
    ("h=5 weekly, assumed test blocks",         200, "§2's assumption, not achievable"),
]


def main() -> None:
    print(f"\n{'=' * 78}")
    print("MINIMUM DETECTABLE DIRECTIONAL EDGE")
    print(f"alpha={ALPHA} one-sided, power={POWER}, baseline p0={P0:.3f} (majority class = down)")
    print(f"{'=' * 78}")
    print(f"{'scenario':<40}{'n':>6}{'min edge':>11}   note")
    print("-" * 78)
    for label, n, note in SCENARIOS:
        if n == 0:
            print(f"{label:<40}")
            continue
        d = min_detectable(P0, n)
        print(f"{label:<40}{n:>6}{d * 100:>10.1f}pp   {note}")

    print(f"\n{'=' * 78}\nAGAINST THE §6 ABANDONMENT THRESHOLD\n{'=' * 78}")
    thresh = 0.02
    n_needed = n_required(P0, thresh)
    print(f"§6 abandons unless h=5 accuracy exceeds majority-class + {thresh * 100:.0f}pp")
    print(f"  = {(P0 + thresh) * 100:.1f}% accuracy")
    print(f"Detecting a {thresh * 100:.0f}pp edge at alpha={ALPHA}/power={POWER} needs "
          f"n = {n_needed:,.0f} observations.")
    print(f"  weekly blocks available : 45")
    print(f"  weekly blocks needed    : {n_needed:,.0f}")
    print(f"  shortfall               : {n_needed / 45:,.0f}x")
    print(f"\nAt 45 weekly blocks the smallest detectable edge is "
          f"{min_detectable(P0, 45) * 100:.1f}pp.")
    print("A 2pp edge is roughly an order of magnitude below the noise floor of")
    print("this sample. The §6 test cannot be run as written -- not because the")
    print("edge is absent, but because no result either way would be informative.")
    print("\nCalendar time to reach the needed sample, archiving forward from now")
    print("at ~225 sessions/yr:")
    for name, need in (("2pp edge, weekly blocks", n_needed),):
        yrs = need * 5 / 225
        print(f"  {name:<28}{yrs:>8.0f} years")


if __name__ == "__main__":
    main()
