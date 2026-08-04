"""Long-only index-vs-cash ledger with settlement-constrained exits (plan §4).

Long-only because NEPSE has no shorting, so a "down" prediction means hold cash,
not sell short. Exposure is a weight in {0, 1}: in the index or out of it.

Two rules that make the difference between a backtest and a fantasy:

**Settlement constrains exit.** A position bought on t cannot be sold until
t + settlement_cycle(t) TRADING days. §4 puts this in the ledger rather than the
strategy on purpose -- a strategy that "rebalances daily" simply cannot, and a
simulator that lets it will report an edge that no account could have captured.

**A limit day blocks the fill in the signal's direction.** The days a directional
signal fires hardest are exactly the days the market is locked, so this is where
phantom alpha comes from (§3.2).

The ledger reports PnL gross AND net, because the gap between them is the
finding -- §4 predicts flat DP charges will eat a daily-rebalancing signal, and
that prediction is only testable if both numbers are visible.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .costs import CostModel, fill_blocked


@dataclass
class Trade:
    date: pd.Timestamp
    side: str
    notional: float
    cost: float
    gain: float = 0.0
    tax: float = 0.0
    holding_days: int = 0


@dataclass
class BacktestResult:
    equity: pd.Series
    gross_equity: pd.Series
    trades: list[Trade]
    blocked_fills: int
    settlement_blocked: int
    exposure: pd.Series
    capital: float
    stats: dict = field(default_factory=dict)


def run_backtest(frame: pd.DataFrame, signal: np.ndarray, cost: CostModel,
                 price_col: str = "close") -> BacktestResult:
    """Walk a 0/1 exposure signal through the ledger, day by day.

    `frame` needs `date` and `close`, sorted ascending. `signal[i]` is the
    exposure DESIRED for the period after row i's close -- i.e. it is acted on
    at row i's close and held into row i+1. That alignment is the whole game:
    shifting it one row earlier is lookahead, and it is invisible in the output.
    """
    frame = frame.reset_index(drop=True)
    if len(signal) != len(frame):
        raise ValueError(f"signal has {len(signal)} rows, frame has {len(frame)}")

    px = frame[price_col].to_numpy(dtype=float)
    dates = pd.to_datetime(frame["date"])
    day_ret = frame["close"].pct_change().to_numpy()
    work = frame.assign(day_return=day_ret)

    cash = cost.capital
    units = 0.0
    entry_price = 0.0
    entry_idx: int | None = None
    bankrupt = False

    equity, gross_equity, exposure = [], [], []
    gross = cost.capital
    trades: list[Trade] = []
    blocked = 0
    settle_blocked = 0

    for i in range(len(frame)):
        want = int(signal[i])
        holding = units > 0
        row = work.iloc[i]

        if bankrupt:
            want = 0 if not holding else want

        if want == 1 and not holding and not bankrupt:
            fee_est = cost.trade_cost(cash, dates.iloc[i], "buy")
            if cash <= 0 or fee_est >= cash:
                # Ruined, or the flat charge alone exceeds the account. Without
                # this the ledger "buys" a negative number of units and equity
                # goes through zero into negative territory -- which is how the
                # first run of this reported a -100.1% drawdown and MORE trades
                # at 2x friction than at 1x. An account cannot spend money it
                # does not have, and a bankrupt one stops trading.
                bankrupt = True
            elif fill_blocked(row, +1, cost.params):
                blocked += 1
            else:
                notional = cash
                fee = cost.trade_cost(notional, dates.iloc[i], "buy")
                units = (notional - fee) / px[i]
                cash = 0.0
                entry_price, entry_idx = px[i], i
                trades.append(Trade(dates.iloc[i], "buy", notional, fee))

        elif want == 0 and holding:
            # Settlement: cannot sell until entry + settlement_cycle trading days.
            settle = cost.params.settlement_days(dates.iloc[entry_idx])
            if i - entry_idx < settle:
                settle_blocked += 1
            elif fill_blocked(row, -1, cost.params):
                blocked += 1
            else:
                notional = units * px[i]
                fee = cost.trade_cost(notional, dates.iloc[i], "sell")
                gain = (px[i] - entry_price) * units - fee
                hold_days = int((dates.iloc[i] - dates.iloc[entry_idx]).days)
                tax = cost.capital_gains_tax(gain, hold_days, dates.iloc[i])
                cash = max(0.0, notional - fee - tax)
                trades.append(Trade(dates.iloc[i], "sell", notional, fee,
                                    gain=gain, tax=tax, holding_days=hold_days))
                units, entry_price, entry_idx = 0.0, 0.0, None
                if cash <= 0:
                    bankrupt = True

        equity.append(cash + units * px[i])
        exposure.append(1.0 if units > 0 else 0.0)

        # Frictionless twin: same signal, same fills, no costs at all.
        if i > 0 and exposure[i - 1] > 0:
            gross *= px[i] / px[i - 1]
        gross_equity.append(gross)

    eq = pd.Series(equity, index=dates, name="equity")
    return BacktestResult(
        equity=eq,
        gross_equity=pd.Series(gross_equity, index=dates, name="gross_equity"),
        trades=trades,
        blocked_fills=blocked,
        settlement_blocked=settle_blocked,
        exposure=pd.Series(exposure, index=dates, name="exposure"),
        capital=cost.capital,
        stats=summarise(eq, pd.Series(gross_equity, index=dates), trades,
                        pd.Series(exposure, index=dates)),
    )


def _sessions_per_year(idx: pd.DatetimeIndex) -> float:
    """Estimated from the data, not assumed at 252.

    NEPSE runs ~230 sessions a year, ran 184 in 2020 (COVID) and traded a
    SIX-day week for four months in 2022 (§3.6). A hardcoded 252 would inflate
    every annualised figure here.
    """
    if len(idx) < 2:
        return 230.0
    years = (idx[-1] - idx[0]).days / 365.25
    return len(idx) / years if years > 0 else 230.0


def summarise(equity: pd.Series, gross: pd.Series, trades: list[Trade],
              exposure: pd.Series) -> dict:
    """The §2 metric set. Never RMSE, never a price-level number."""
    r = equity.pct_change().dropna()
    gr = gross.pct_change().dropna()
    ann = _sessions_per_year(pd.DatetimeIndex(equity.index))
    years = max((equity.index[-1] - equity.index[0]).days / 365.25, 1e-9)

    def sharpe(x: pd.Series) -> float:
        return float(x.mean() / x.std() * np.sqrt(ann)) if x.std() > 0 else 0.0

    dd = equity / equity.cummax() - 1.0
    buys = [t for t in trades if t.side == "buy"]
    turnover = sum(t.notional for t in trades) / equity.iloc[0] if len(equity) else 0.0

    # §2: what share of the profit came from the five best days? A strategy
    # whose edge lives in five sessions is not an edge, it is a lottery ticket.
    # Undefined when the strategy lost money -- a ratio to a negative total is
    # not a share of anything, and reporting one (the first run printed
    # "-22.9%") invites it to be read as if it meant something.
    top5 = float("nan")
    if len(r) > 5 and r.sum() > 0:
        top5 = float(r.nlargest(5).sum() / r.sum())

    return {
        "net_sharpe": sharpe(r),
        "gross_sharpe": sharpe(gr),
        "net_cagr": float((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1),
        "gross_cagr": float((gross.iloc[-1] / gross.iloc[0]) ** (1 / years) - 1),
        "max_drawdown": float(dd.min()),
        "total_return": float(equity.iloc[-1] / equity.iloc[0] - 1),
        "n_trades": len(trades),
        "n_round_trips": len(buys),
        "turnover_x_capital": float(turnover),
        "time_in_market": float(exposure.mean()),
        "pct_pnl_top5_days": top5,
        "ruined": bool(equity.iloc[-1] <= 0),
        "total_costs": float(sum(t.cost + t.tax for t in trades)),
        "sessions_per_year": float(ann),
    }


def buy_and_hold(frame: pd.DataFrame, cost: CostModel,
                 price_col: str = "close") -> BacktestResult:
    """The benchmark §1 demands alongside 50% and majority-class.

    Buys once, holds, sells at the end -- so it pays the flat DP charge twice in
    total, against a daily strategy paying it on every switch. That contrast is
    the entire argument of §4.
    """
    sig = np.ones(len(frame), dtype=int)
    sig[-1] = 0
    return run_backtest(frame, sig, cost, price_col=price_col)
