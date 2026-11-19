"""Task 3 v9 ATC calendar check: is the closing-auction push bigger on S50 expiry and index-rebalance days?

Feature-only check (owner, 2026-10-27): looks at the SIZE of the push and the ATC volume share only, never
its sign and never the forecast targets. The push of day d is part of the afternoon target y_a, so the
window stops at the end of the select segment (2017-04 -> 2022-12) and the confirm segment stays unread.

    push_d  = log(ATC price / last continuous 1m close before the auction)   (in ticks of the lower price)
    share_d = ATC bar volume / total volume of day d

Day types (business days = SET trading days in the 1m data):
- s50_q:   S50 futures last trading day, Mar/Jun/Sep/Dec (business day before the last business day)
- s50_m:   same rule, other months
- set50:   last business day of Jun and Dec (SET50 semi-annual review takes effect the next day)
- msci:    last business day of Feb/May/Aug/Nov (approximation: MSCI sometimes picks another day)
- ftse:    third Friday of Mar/Jun/Sep/Dec, or the trading day before it

    .venv/bin/python src/task3/versions9/checks/atc_calendar_check.py
"""
import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src/task3/versions8"))
from pipeline import load  # noqa: E402

END = dt.date(2022, 12, 31)


def tick(p: float) -> float:
    for hi, t in ((2, .01), (5, .02), (10, .05), (25, .1), (100, .25), (200, .5), (400, 1.)):
        if p < hi:
            return t
    return 2.


def day_types(days: pd.DatetimeIndex) -> pd.Series:
    s = pd.Series(days, index=days)
    by_month = s.groupby([days.year, days.month])
    last, prev = by_month.transform("max"), by_month.transform(lambda x: x.iloc[-2] if len(x) > 1 else pd.NaT)
    kind = pd.Series("normal", index=days)
    q = days.month.isin([3, 6, 9, 12])
    kind[(days == prev) & q] = "s50_q"
    kind[(days == prev) & ~q] = "s50_m"
    kind[(days == last) & days.month.isin([6, 12])] = "set50"
    kind[(days == last) & days.month.isin([2, 5, 8, 11])] = "msci"
    for (y, m), g in s[q].groupby([days[q].year, days[q].month]):
        fridays = pd.date_range(dt.date(y, m, 1), periods=31, freq="D")
        third_fri = fridays[(fridays.weekday == 4) & (fridays.month == m)][2]
        ftse = g[g <= third_fri].max()
        if kind[ftse] == "normal":
            kind[ftse] = "ftse"
    return kind


def per_day(sym: str) -> pd.DataFrame:
    b = load.load_1m(sym, end=END)
    atc = b[b.session == "closing_label"].groupby("trade_date").agg(atc=("close", "last"), atc_vol=("volume", "sum"))
    cont = b[b.session == "afternoon"].groupby("trade_date").close.last().rename("cont")
    vol = b.groupby("trade_date").volume.sum().rename("vol")
    d = pd.concat([atc, cont, vol], axis=1, join="inner")
    d.index = pd.to_datetime(d.index)
    d["ticks"] = (d.atc - d.cont).abs() / np.minimum(d.atc, d.cont).map(tick)
    d["share"] = d.atc_vol / d.vol
    return d


def main() -> None:
    rows = []
    for sym in ("KBANK", "ADVANC"):
        d = per_day(sym)
        d["kind"] = day_types(d.index)
        for kind, g in d.groupby("kind"):
            rows.append({"sym": sym, "kind": kind, "n": len(g),
                         "pushed_%": 100 * (g.ticks > 0).mean(),
                         "ticks_mean": g.ticks.mean(), "ticks_se": g.ticks.std() / np.sqrt(len(g)),
                         ">=2ticks_%": 100 * (g.ticks >= 2).mean(),
                         "share_med_%": 100 * g.share.median()})
        print(sym, d.index.min().date(), "->", d.index.max().date(), "days", len(d))
    out = pd.DataFrame(rows).set_index(["sym", "kind"])
    print(out.round(2).to_string())


if __name__ == "__main__":
    main()
