"""Date-indexed cost model and fill rules (plan §4).

§4's first sentence is "every constant is a date-indexed lookup, never a
scalar", and §4 turned out to be more right than it knew: sourcing the
constants (2026-08-04) found that the scrip circuit, the index circuit and the
trading hours ALL changed on 2026-04-20, and capital-gains rates changed on
2026-07-17. Three of those land inside the last eighty sessions of the sample.
A scalar cost model would be wrong for most of the history and wrong in a
different way for the end of it.

Three separate animals, never one round-trip percentage:

    per-trade  = notional x (commission_rate(notional, date) + sebon_fee(date))
               + dp_charge(date) x n_scrips        # FLAT. Does not scale.
    on sale    = max(0, realized_gain) x cgt_rate(holding_days, entity, date)

The flat DP charge is the one that decides this project. It does not scale with
notional, so at a small declared capital base it dominates, and it is charged
per scrip per settlement -- which is why §4 says results depend on the declared
capital base and must state it wherever a Sharpe appears.

Nothing here silently substitutes a default for a missing constant. A `null` in
market_params raises, because §6.1 is explicit that a backtest must not run
against placeholder costs and the failure mode of a quiet default is a Sharpe
that looks fine.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

PARAMS_PATH = Path("configs/market_params.yaml")


class MissingConstant(RuntimeError):
    """A cost constant needed for this trade is null or TODO in market_params."""


def _as_ts(v: Any) -> pd.Timestamp | None:
    return None if v in (None, "null") else pd.Timestamp(v)


class Params:
    """Date-indexed view over market_params.yaml."""

    def __init__(self, data: dict | None = None, path: Path = PARAMS_PATH,
                 validate: bool = True):
        self.data = data if data is not None else yaml.safe_load(path.read_text())
        if validate:
            problems = self.overlapping_eras()
            if problems:
                raise ValueError("market_params has overlapping eras:\n  "
                                 + "\n  ".join(problems))

    def overlapping_eras(self) -> list[str]:
        """Fields whose eras overlap, so the first match silently wins.

        Written because it happened: `settlement_cycle`'s T+3 entry was left
        open-ended, so it matched every date and T+2 was unreachable. The
        backtest ran perfectly and applied a three-day settlement lag to 2026.
        Nothing errored -- era lookup is first-match, and the first match was
        always there. An open-ended era anywhere but last is the bug.
        """
        out = []
        for field_name, entries in self.data.items():
            if not isinstance(entries, list):
                continue
            spans = []
            for e in entries:
                if not isinstance(e, dict) or "effective_from" not in e:
                    break
                spans.append((_as_ts(e.get("effective_from")),
                              _as_ts(e.get("effective_to"))))
            else:
                for i, (lo_a, hi_a) in enumerate(spans):
                    for j, (lo_b, hi_b) in enumerate(spans[i + 1:], start=i + 1):
                        a_lo = lo_a or pd.Timestamp.min
                        a_hi = hi_a or pd.Timestamp.max
                        b_lo = lo_b or pd.Timestamp.min
                        b_hi = hi_b or pd.Timestamp.max
                        if a_lo <= b_hi and b_lo <= a_hi:
                            out.append(
                                f"{field_name}: era {i} "
                                f"[{lo_a}..{hi_a}] overlaps era {j} "
                                f"[{lo_b}..{hi_b}] -- era {j} is unreachable")
        return out

    def era(self, field: str, date: pd.Timestamp) -> dict:
        """The entry in `field` covering `date`. Raises if none does."""
        date = pd.Timestamp(date)
        entries = self.data.get(field)
        if not entries:
            raise MissingConstant(f"market_params has no `{field}`")
        for e in entries:
            lo, hi = _as_ts(e.get("effective_from")), _as_ts(e.get("effective_to"))
            if (lo is None or date >= lo) and (hi is None or date <= hi):
                return e
        raise MissingConstant(f"no `{field}` era covers {date.date()}")

    def value(self, field: str, key: str, date: pd.Timestamp) -> Any:
        e = self.era(field, date)
        v = e.get(key)
        if v is None or v == "TODO":
            raise MissingConstant(
                f"`{field}.{key}` is {v!r} for {pd.Timestamp(date).date()} -- "
                f"plan §4 forbids guessing it and §6.1 forbids backtesting "
                f"against a placeholder. Source it or state an assumption "
                f"explicitly via CostModel(assume_...=...).")
        return v

    # --- individual constants ------------------------------------------------

    def commission_rate(self, notional: float, date: pd.Timestamp) -> float:
        """Tiered by the value of THIS trade, not by cumulative turnover."""
        tiers = self.value("broker_commission", "tiers", date)
        for t in tiers:
            up_to = t.get("up_to")
            if up_to is None or notional <= up_to:
                return float(t["rate"])
        return float(tiers[-1]["rate"])

    def sebon_fee_rate(self, date: pd.Timestamp) -> float:
        return float(self.value("sebon_fee_rate", "rate", date))

    def dp_charge(self, date: pd.Timestamp) -> float:
        return float(self.value("dp_charge_npr", "amount", date))

    def settlement_days(self, date: pd.Timestamp) -> int:
        return int(self.value("settlement_cycle", "days", date))

    def scrip_circuit(self, date: pd.Timestamp) -> float:
        return float(self.value("scrip_circuit_pct", "limit", date))

    def index_circuit(self, date: pd.Timestamp) -> float:
        return float(self.value("index_circuit", "daily_cap", date))

    def cgt_rate(self, holding_days: int, date: pd.Timestamp,
                 entity: str = "individual") -> float:
        if entity == "institutional":
            return float(self.value("cgt_rate", "institutional", date))
        key = "individual_long" if holding_days >= 365 else "individual_short"
        return float(self.value("cgt_rate", key, date))


@dataclass
class CostModel:
    """Applies §4's three cost components at a declared capital base.

    `n_scrips` is not cosmetic and defaults to the OPTIMISTIC value. The NEPSE
    index is not directly tradeable -- there is no index fund or ETF -- so
    holding "the index" means holding a basket, and the DP charge is per scrip.
    n_scrips=1 prices a hypothetical single instrument and is therefore a LOWER
    BOUND on real cost; a 20-name proxy basket multiplies the flat charge by 20.
    Any result must state which it used.

    `dp_charged_on` is likewise an assumption, not a fact: sources disagree
    (see market_params). Default "both" is the conservative reading.
    """

    params: Params
    capital: float                       # declared capital base, NPR
    n_scrips: int = 1
    dp_charged_on: str = "both"          # "both" | "sell" | "buy"
    entity: str = "individual"
    friction_multiplier: float = 1.0     # §4: report at 0x, 1x, 2x

    def trade_cost(self, notional: float, date: pd.Timestamp, side: str) -> float:
        """Cost of one buy or one sell, in NPR."""
        if notional <= 0:
            return 0.0
        rate = (self.params.commission_rate(notional, date)
                + self.params.sebon_fee_rate(date))
        variable = notional * rate
        dp = 0.0
        if self.dp_charged_on == "both" or self.dp_charged_on == side:
            dp = self.params.dp_charge(date) * self.n_scrips
        return (variable + dp) * self.friction_multiplier

    def capital_gains_tax(self, gain: float, holding_days: int,
                          date: pd.Timestamp) -> float:
        """On REALIZED GAINS only. A flat percentage would overcharge losses."""
        if gain <= 0:
            return 0.0
        rate = self.params.cgt_rate(holding_days, date, self.entity)
        return gain * rate * self.friction_multiplier

    def round_trip_bps(self, date: pd.Timestamp) -> float:
        """Round-trip cost in basis points of the declared capital base.

        The single most useful diagnostic here, because it makes the flat DP
        charge visible: halve the capital base and this number grows.
        """
        c = (self.trade_cost(self.capital, date, "buy")
             + self.trade_cost(self.capital, date, "sell"))
        return 1e4 * c / self.capital


def fill_blocked(row: pd.Series, direction: int, params: Params) -> bool:
    """True when the index sat at its circuit and a fill is not achievable.

    §3.2 calls this the single most likely source of phantom alpha: the days a
    signal fires hardest are the days you cannot get filled. Checked against the
    DATE-APPROPRIATE cap, which matters now that the index limit went 6% -> 8%
    on 2026-04-20.

    `direction` is +1 to buy, -1 to sell. A limit-up day blocks buying, not
    selling.
    """
    cap = params.index_circuit(row["date"])
    ret = row.get("day_return")
    if ret is None or pd.isna(ret):
        return False
    tol = 1e-4
    if direction > 0 and ret >= cap - tol:
        return True
    if direction < 0 and ret <= -cap + tol:
        return True
    return False
