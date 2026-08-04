"""One command from raw data to cost-adjusted, regime-split results (plan §1).

§1's first success criterion, stated literally. Everything below Phase 1 is
reproducible from this file; the only things it cannot do are the two that
depend on wall-clock time and external state -- the daily archive pull (§1b,
which must simply run every day forever) and the deep-history fetch (which
depends on MeroLagani being up).

Order matters and is enforced: §8's rule is that sanity tests pass before any
experiment runs, so a failing suite stops the pipeline rather than producing
results nobody should read.

    .venv/bin/python run_all.py              # sanity -> features -> models
    .venv/bin/python run_all.py --refresh    # also re-fetch the deep series
    .venv/bin/python run_all.py --quick      # fewer bootstrap reps and seeds
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = str(ROOT / ".venv/bin/python")
RULE = "=" * 78


def step(title: str, cmd: list[str], required: bool = True) -> bool:
    print(f"\n{RULE}\n>>> {title}\n{RULE}")
    t0 = time.time()
    r = subprocess.run(cmd, cwd=ROOT)
    ok = r.returncode == 0
    print(f"<<< {title}: {'OK' if ok else 'FAILED'} ({time.time() - t0:.0f}s)")
    if not ok and required:
        print(f"\nSTOPPING. §8: sanity and correctness gates run before "
              f"experiments, not after.")
        sys.exit(r.returncode)
    return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true",
                    help="re-fetch and re-reconcile the deep history first")
    ap.add_argument("--quick", action="store_true",
                    help="fewer bootstrap reps and seeds")
    args = ap.parse_args()

    boot = "500" if args.quick else "2000"
    seeds = "2" if args.quick else "5"

    if args.refresh:
        step("Phase 1c: deep history + two-source reconciliation",
             [PY, "scripts/phase1c_deep_history.py"])

    # §8: "sanity tests pass before any experiment runs." Both of these are
    # gates, not diagnostics -- a leak test that runs after the results are in
    # is a formality.
    step("Test suite (leak, embargo, cost, feature and log invariants)",
         [PY, "-m", "pytest", "tests/", "-q"])
    step("Sanity suite over the deep series (§3.3, §3.6)",
         [PY, "scripts/phase1d_deep_quality.py"])

    step("Phase 2a: walk-forward harness and baselines (h=1)",
         [PY, "scripts/phase2_walkforward.py", "--horizon", "1",
          "--n-boot", boot])
    step("Phase 2b: cost model and fills (§4)",
         [PY, "scripts/phase2b_costs.py"])

    for h in ("1", "5"):
        step(f"Phase 4: models, ablation, regimes, §6 thresholds (h={h})",
             [PY, "scripts/phase4_models.py", "--horizon", h,
              "--n-boot", boot, "--seeds", seeds])

    step("Phase 5: forward log status (§7)",
         [PY, "scripts/phase5_forward.py", "--score"], required=False)

    print(f"\n{RULE}\nDONE. See §6.2 of nepse-prediction-plan.md for the "
          f"recorded verdict.\n{RULE}")


if __name__ == "__main__":
    main()
