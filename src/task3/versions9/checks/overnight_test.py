"""Task 3 v9 overnight test: do foreign markets known before 09:30 explain the morning opening gap?

Rule set before the run (owner, 2026-10-27): keep foreign data only if OLS R^2 of the opening gap on the
five series is >= 0.10 for each stock on the select segment (2019-2022); otherwise drop it.

Timing (Bangkok = UTC+7). SET day d, previous SET day p:
- ^GSPC, THD (New York): a session on NY date L ends 03:00-04:00 Bangkok on L+1, so every session with
  L <= d-1 is known at 09:30 on d. Feature = log(close of last L <= d-1 / close of last L <= p-1).
- THB=X: same rule (Yahoo's FX day ends before 07:00 Bangkok on L+1).
- ^N225 (opens 07:00 Bangkok), ^HSI (opens 08:30 Bangkok): only the OPEN of date d is known at 09:30.
  Feature = log(open on d / close of last date <= p); 0 if that market is shut on d.
Raw close only, never adjclose (restated by later dividends). Requested up to 2026-09-04 (K07 window).

    .venv/bin/python src/task3/versions9/checks/overnight_test.py
"""
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src/task3/versions8"))
from src.task3.versions5.pipeline.market_data import fetch_raw  # noqa: E402
from pipeline import load  # noqa: E402

RAW = ROOT / "data/raw/yahoo_chart/1d/_v9_overnight/task3-v9-overnight-20261027"
START, END = dt.date(2016, 1, 1), dt.date(2026, 9, 4)
CLOSE_SERIES = {"spx": "^GSPC", "thd": "THD", "thb": "THB=X"}   # close of last L <= d-1
OPEN_SERIES = {"n225": "^N225", "hsi": "^HSI"}                   # open on d only
SELECT = (dt.date(2019, 1, 1), dt.date(2022, 12, 31))


def raw_path(sym: str) -> Path:
    return RAW / f"{sym.replace('^', '_').replace('=', '_')}.json"


def fetch_all() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    for sym in [*CLOSE_SERIES.values(), *OPEN_SERIES.values()]:
        if not raw_path(sym).exists():
            raw_path(sym).write_bytes(fetch_raw(sym, START, END))


def series(sym: str) -> pd.DataFrame:
    """Daily bars indexed by each bar's local date in the exchange's own time zone."""
    r = json.loads(raw_path(sym).read_text())["chart"]["result"][0]
    q = r["indicators"]["quote"][0]
    # each bar's own local date (fixed 2026-10-28, M1 review): one gmtoffset put THB=X's UK-summer bars a day early
    local = pd.to_datetime(r["timestamp"], unit="s", utc=True).tz_convert(r["meta"]["exchangeTimezoneName"])
    df = pd.DataFrame({"date": local.date,
                       "open": q["open"], "close": q["close"]}).dropna()
    df = df[df["date"] <= END].drop_duplicates("date", keep="last").set_index("date").sort_index()
    return df


def last_on_or_before(s: pd.Series, days: pd.Index) -> np.ndarray:
    pos = s.index.searchsorted(days, side="right") - 1
    out = np.where(pos >= 0, s.to_numpy()[np.clip(pos, 0, None)], np.nan)
    return out


def features(days: list) -> pd.DataFrame:
    d = pd.Index(days)
    p = pd.Index([None, *days[:-1]])
    X = pd.DataFrame(index=d)
    for name, sym in CLOSE_SERIES.items():
        c = series(sym)["close"]
        dm1 = pd.Index([x - dt.timedelta(days=1) for x in d])
        pm1 = pd.Index([x - dt.timedelta(days=1) if x else dt.date(1900, 1, 1) for x in p])
        X[name] = np.log(last_on_or_before(c, dm1) / last_on_or_before(c, pm1))
    for name, sym in OPEN_SERIES.items():
        s = series(sym)
        base = last_on_or_before(s["close"], pd.Index([x if x else dt.date(1900, 1, 1) for x in p]))
        o = s["open"].reindex(d).to_numpy()
        X[name] = np.where(np.isnan(o), 0.0, np.log(o / base))
        X[f"{name}_open_eq_prev_close"] = np.isclose(o, base)
    return X.iloc[1:]


def leak_check(days: list) -> None:
    """Every value used for SET day d must be stamped before 09:30 Bangkok on d."""
    bkk = dt.timezone(dt.timedelta(hours=7))
    for sym, rule in [*[(s, "close") for s in CLOSE_SERIES.values()], *[(s, "open") for s in OPEN_SERIES.values()]]:
        r = json.loads(raw_path(sym).read_text())["chart"]["result"][0]
        tp = r["meta"]["currentTradingPeriod"]["regular"]
        start = dt.datetime.fromtimestamp(tp["start"], bkk).time()
        end = dt.datetime.fromtimestamp(tp["end"], bkk).time()
        print(f"  {sym}: regular session in Bangkok time {start}-{end} (exchange tz {r['meta']['exchangeTimezoneName']}),"
              f" uses {rule}")


def gap_table(sym: str) -> pd.DataFrame:
    b = load.load_1m(sym, end=SELECT[1])
    am = b[b["timestamp"].dt.strftime("%H:%M") < "12:30"]
    g = pd.DataFrame({"O": am.groupby("trade_date")["open"].first(), "P": am.groupby("trade_date")["close"].last()})
    off = load.load_official(sym, end=SELECT[1]).set_index("trade_date")["close"]
    g = g.join(off.rename("C"), how="inner")
    g["C_prev"] = g["C"].shift(1)
    led = load.load_ledger(sym).set_index("xd_date")["dividend_thb"]
    g["D"] = g.index.map(led).fillna(0.0)
    g["gap"] = np.log(g["O"] / (g["C_prev"] - g["D"]))
    g["morning"] = np.log(g["P"] / (g["C_prev"] - g["D"]))
    return g.dropna()


def r2(y: np.ndarray, X: np.ndarray) -> float:
    A = np.column_stack([np.ones(len(X)), X])
    beta = np.linalg.lstsq(A, y, rcond=None)[0]
    return 1 - ((y - A @ beta) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def oos_r2(y: np.ndarray, X: np.ndarray, split: int) -> float:
    A = np.column_stack([np.ones(len(X)), X])
    beta = np.linalg.lstsq(A[:split], y[:split], rcond=None)[0]
    e = y[split:] - A[split:] @ beta
    return 1 - (e ** 2).sum() / (y[split:] ** 2).sum()   # vs predicting zero gap (the baseline)


if __name__ == "__main__":
    fetch_all()
    print("Timing of each series:")
    leak_check([])
    cols = [*CLOSE_SERIES, *OPEN_SERIES]
    verdict = {}
    for sym in ["KBANK", "ADVANC"]:
        g = gap_table(sym)
        X = features(list(g.index)).reindex(g.index)
        g = g.join(X)
        g = g[(g.index >= SELECT[0]) & (g.index <= SELECT[1])].dropna(subset=cols)
        y, M = g["gap"].to_numpy(), g[cols].to_numpy()
        split = int((pd.Index(g.index) < dt.date(2021, 1, 1)).sum())
        res = {"days": len(g), "R2_gap": r2(y, M), "R2_morning": r2(g["morning"].to_numpy(), M),
               "OOS_R2_gap_2021_22": oos_r2(y, M, split), "OOS_R2_morning_2021_22": oos_r2(g["morning"].to_numpy(), M, split)}
        verdict[sym] = res["R2_gap"] >= 0.10
        print(f"\n{sym} select 2019-2022: " + " · ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}" for k, v in res.items()))
        print("  corr with gap:", {c: round(float(np.corrcoef(g[c], y)[0, 1]), 3) for c in cols})
        print("  single-series R2 of gap:", {c: round(r2(y, g[[c]].to_numpy()), 3) for c in cols})
        print("  days with open == previous close:", {c: int(g[f"{c}_open_eq_prev_close"].sum()) for c in OPEN_SERIES})
        print("  days the market was shut on d (feature 0):", {c: int((g[c] == 0).sum()) for c in cols})
    print("\nRule R2_gap >= 0.10 on each stock:", verdict, "->", "KEEP" if all(verdict.values()) else "DROP")
