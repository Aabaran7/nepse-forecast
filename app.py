"""NEPSE dashboard.

Run locally:   .venv/bin/streamlit run app.py

WHAT THIS PAGE IS ALLOWED TO CLAIM
----------------------------------
Plan §6 tested three decision rules and abandoned all three; its conclusion is
that buy-and-hold is the tradeable answer. §7 is equally blunt about the forward
log: ~40 trading days cannot tell 55% accuracy from 50%, and the log exists for
leak detection and pipeline validation, not as evidence of an edge.

So this dashboard does not print a verdict. It shows the forward log's accuracy
NEXT TO the base rate you would get by always guessing the majority class,
because an accuracy figure alone is unreadable -- 55% looks like skill until you
notice the market rose on 55% of days. Every scrip view is descriptive: no
ranking by expected return, no "buy" column.

A dashboard that said BULLISH in large type would be quietly reversing the
project's own finding, and would be the single easiest way to fool its author.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nepselab.dashboard import data  # noqa: E402
from nepselab.features import scrip as scrip_features  # noqa: E402

st.set_page_config(page_title="NEPSE forward log", page_icon="📉", layout="wide")

# Palettes validated with the dataviz palette checker (all six checks pass in
# both modes). UP/DOWN is a diverging pair, not the conventional green/red:
# green-red is the one pairing red-green colourblind readers cannot separate,
# and it is never the only channel here -- every value ships with its sign.
UP, DOWN = "#0d7d9e", "#c2571a"
NEUTRAL, MUTED = "#6b7280", "#9ca3af"
CATEGORICAL = ["#3b6fd4", "#c2571a", "#8b5cf6", "#3f8f5f"]
GRID = "rgba(128,128,128,0.18)"

QUADRANT_LABEL = {
    "heavy_up": "Rose, heavy volume",
    "quiet_up": "Rose, normal volume",
    "heavy_down": "Fell, heavy volume",
    "quiet_down": "Fell, normal volume",
    "flat": "Unchanged",
}


def style(fig: go.Figure, height: int = 320) -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=8, r=8, t=28, b=8),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(size=13), hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    fig.update_xaxes(showgrid=False, showline=True, linecolor=GRID)
    fig.update_yaxes(showgrid=True, gridcolor=GRID, zeroline=False)
    return fig


@st.cache_data(ttl=600)
def load():
    return {
        "fresh": data.freshness(),
        "closes": data.index_closes(),
        "log": data.forward_log(),
        "panel": data.scrip_panel(),
        "news": data.headlines(),
        "sentiment": data.sentiment(),
    }


d = load()

# --- sidebar ---------------------------------------------------------------
with st.sidebar:
    st.subheader("Data freshness")
    st.caption("NEPSE serves a rolling year. A gap here is permanent (§3.4).")
    today = pd.Timestamp.today().normalize()
    for name, ts in d["fresh"].items():
        if ts is None:
            st.write(f"**{name}** — none")
            continue
        days = (today - pd.Timestamp(ts).normalize()).days
        st.write(f"**{name}** — {pd.Timestamp(ts).date()}"
                 + (f"  ·  {days}d ago" if days > 0 else "  ·  today"))

st.title("NEPSE forward log")
st.caption("Descriptive. No trading recommendation — see §6 and §7 in the plan.")

tab_log, tab_market, tab_stocks, tab_news = st.tabs(
    ["Forward log", "Market", "Stocks", "News"])

# --- forward log -----------------------------------------------------------
with tab_log:
    log = d["log"]
    if log.empty:
        st.info("No predictions logged yet. Run `scripts/phase5_forward.py`.")
    else:
        directional = data.directional_log(log)
        resolved = directional[directional["correct"].notna()]
        exposure = log[log["kind"] == "exposure"]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Directional predictions", len(directional))
        c2.metric("Resolved", len(resolved))
        c3.metric("Still open", len(directional) - len(resolved))
        c4.metric("Exposure rows", len(exposure),
                  help="Volatility-target rows (§6.6). Not direction forecasts, "
                       "so they are excluded from accuracy.")

        st.divider()

        if resolved.empty:
            st.warning(
                "Nothing has resolved yet, so there is no accuracy to report. "
                "That is the expected state early on — the target session for "
                "each prediction has not traded.")
        else:
            closes = d["closes"]
            rows = []
            for h in sorted(resolved["horizon"].unique()):
                sub = resolved[resolved["horizon"] == h]
                # The baseline that makes accuracy readable: always predict the
                # direction that happened more often. Beating 50% is not skill
                # in a market that rose on 55% of days.
                fwd = closes["close"].pct_change(int(h)).shift(-int(h)).dropna()
                base = max((fwd > 0).mean(), (fwd <= 0).mean())
                rows.append({"horizon": f"t+{int(h)}", "n": len(sub),
                             "accuracy": sub["correct"].mean(), "baseline": base})
            acc = pd.DataFrame(rows)

            fig = go.Figure()
            fig.add_bar(x=acc["horizon"], y=acc["accuracy"], name="Model",
                        marker_color=CATEGORICAL[0],
                        text=[f"{v:.0%}" for v in acc["accuracy"]],
                        textposition="outside")
            fig.add_bar(x=acc["horizon"], y=acc["baseline"], name="Always-guess baseline",
                        marker_color=MUTED,
                        text=[f"{v:.0%}" for v in acc["baseline"]],
                        textposition="outside")
            fig.update_layout(barmode="group", bargap=0.45, bargroupgap=0.08,
                              yaxis_tickformat=".0%", yaxis_range=[0, 1.05],
                              title="Accuracy vs. guessing the commoner direction")
            st.plotly_chart(style(fig), width="stretch")

            st.dataframe(
                acc.assign(accuracy=lambda x: (x["accuracy"] * 100).round(1),
                           baseline=lambda x: (x["baseline"] * 100).round(1)),
                hide_index=True, width="stretch")

        st.info(
            "**What this can and cannot show.** §7: the forward log is here to "
            "catch leaks and pipeline breakage — a feature that used future "
            "data, a source that changed shape. It is not evidence of an edge. "
            "About 40 trading days cannot separate 55% accuracy from 50%, and "
            "§6's capital gate needs 60+ forward days *and* forward accuracy "
            "within 5 points of the backtest before any money is involved.")

        with st.expander("Every logged prediction (never edited — §7)"):
            st.dataframe(log, hide_index=True, width="stretch")

# --- market ----------------------------------------------------------------
with tab_market:
    closes, panel = d["closes"], d["panel"]

    if not closes.empty:
        window = st.radio("Window", ["1 year", "3 years", "All"],
                          horizontal=True, index=0, key="win")
        cutoff = {"1 year": 365, "3 years": 1095, "All": None}[window]
        view = closes if cutoff is None else closes[
            closes["date"] >= closes["date"].max() - pd.Timedelta(days=cutoff)]

        fig = go.Figure()
        fig.add_scatter(x=view["date"], y=view["close"], mode="lines",
                        name="NEPSE", line=dict(color=CATEGORICAL[0], width=2))
        fig.update_layout(title="NEPSE index")
        st.plotly_chart(style(fig, 340), width="stretch")

    if not panel.empty:
        breadth = scrip_features.market_breadth(panel).tail(60)

        fig = go.Figure()
        fig.add_bar(x=breadth["businessDate"], y=breadth["advancers"],
                    name="Advancers", marker_color=UP)
        fig.add_bar(x=breadth["businessDate"], y=-breadth["decliners"],
                    name="Decliners", marker_color=DOWN)
        fig.update_layout(barmode="relative", bargap=0.25,
                          title="Breadth — traded scrips only, last 60 sessions")
        st.plotly_chart(style(fig), width="stretch")
        st.caption("A scrip that did not trade is excluded rather than counted "
                   "as unchanged: it did not decline, it did not participate.")

        conc = scrip_features.turnover_concentration(panel).tail(120)
        fig = go.Figure()
        fig.add_scatter(x=conc["businessDate"], y=conc["top10_share"],
                        mode="lines", name="Top 10 share",
                        line=dict(color=CATEGORICAL[2], width=2))
        fig.update_layout(title="Share of turnover taken by the 10 busiest scrips",
                          yaxis_tickformat=".0%")
        st.plotly_chart(style(fig), width="stretch")
        st.caption("A rising index carried by ten names is a different market "
                   "from the same index carried by two hundred.")

# --- stocks ----------------------------------------------------------------
with tab_stocks:
    panel = d["panel"]
    if panel.empty:
        st.info("No scrip data. Run `scripts/archive_pull.py`.")
    else:
        latest = scrip_features.latest_session(panel)
        session = pd.Timestamp(latest["businessDate"].iloc[0]).date()
        st.subheader(f"Session of {session}")

        st.warning(
            "**Volume is not direction.** The common advice — heavy volume on a "
            "falling price means big holders are selling — does not hold here. "
            "Over 228 sessions, scrips closing at their limit **up** traded "
            "3.8× their own normal volume; those closing at their limit **down** "
            "traded 2.3×, and were 5× rarer. A daily circuit truncates the fall "
            "and there is no short selling, so forced selling in this market "
            "looks like *low* volume at the limit. These columns describe what "
            "happened. They do not predict.")

        c1, c2, c3 = st.columns([2, 2, 3])
        hide_thin = c1.toggle("Hide thin scrips", value=True,
                              help=f"Fewer than {scrip_features.THIN_TRADES} "
                                   "transactions — the move may be one participant.")
        picked = c2.multiselect("Day type", list(QUADRANT_LABEL),
                                format_func=QUADRANT_LABEL.get)
        search = c3.text_input("Symbol contains", "")

        view = latest.copy()
        if hide_thin:
            view = view[~view["is_thin"]]
        if picked:
            view = view[view["quadrant"].isin(picked)]
        if search.strip():
            view = view[view["symbol"].str.contains(search.strip(), case=False, na=False)]

        counts = latest["quadrant"].value_counts()
        cols = st.columns(len(QUADRANT_LABEL))
        for col, key in zip(cols, QUADRANT_LABEL):
            col.metric(QUADRANT_LABEL[key], int(counts.get(key, 0)))

        show = view[[c for c in ["symbol", "closePrice", "ret", "vol_ratio",
                                 "totalTrades", "avg_trade_size", "is_thin",
                                 "quadrant", "at_limit_up", "at_limit_down",
                                 "totalTradedValue"] if c in view.columns]]
        st.dataframe(
            show, hide_index=True, width="stretch", height=520,
            column_config={
                "symbol": "Symbol",
                "closePrice": st.column_config.NumberColumn("Close", format="%.1f"),
                "ret": st.column_config.NumberColumn("Change", format="%.2f%%"),
                "vol_ratio": st.column_config.NumberColumn(
                    "Volume vs normal", format="%.1fx",
                    help="Versus this scrip's own 20-session median."),
                "totalTrades": st.column_config.NumberColumn("Trades", format="%d"),
                "avg_trade_size": st.column_config.NumberColumn("Avg trade", format="%.0f"),
                "is_thin": st.column_config.CheckboxColumn("Thin"),
                "quadrant": st.column_config.TextColumn("Day type"),
                "at_limit_up": st.column_config.CheckboxColumn("Limit up"),
                "at_limit_down": st.column_config.CheckboxColumn("Limit down"),
                "totalTradedValue": st.column_config.NumberColumn("Turnover", format="%.0f"),
            })
        st.caption(f"{len(view)} of {len(latest)} scrips shown. Sorted by turnover. "
                   "Change is shown as a fraction, not a percentage sign.")

# --- news ------------------------------------------------------------------
with tab_news:
    news, sent = d["news"], d["sentiment"]
    if news.empty:
        st.info("No headlines yet. Run `scripts/scrape_news.py`.")
    else:
        st.metric("Headlines stored", len(news))
        if sent.empty:
            st.warning(
                "**Not scored yet.** Sentiment needs `scripts/score_sentiment.py`, "
                "and plan §5 requires it to be checked against ~200 hand-labelled "
                "headlines before any score is treated as real. Until then this "
                "tab shows what was collected, not what it means.")

        view = news.copy()
        if "session" in view.columns:
            pending = int(view["session"].isna().sum())
            if pending:
                st.caption(
                    f"{pending} headlines are not yet assigned to a session. "
                    "That is correct, not a bug: a story filed after the 15:00 "
                    "close belongs to the next trading day, which has not "
                    "happened. They resolve on the next run.")
        st.dataframe(
            view[[c for c in ["session", "source", "title", "published", "url"]
                  if c in view.columns]],
            hide_index=True, width="stretch", height=560,
            column_config={
                "session": st.column_config.TextColumn(
                    "Trading session", help="The first session that could trade on this."),
                "source": "Source",
                "title": st.column_config.TextColumn("Headline", width="large"),
                "published": "Published",
                "url": st.column_config.LinkColumn("Link", display_text="open"),
            })
