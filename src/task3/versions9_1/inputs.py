"""Task 3 v9.1 inputs shared by every group (tickets/T00 §3). Reads the v9 feature store and raw bars only.

Every builder looks back from day d and never uses a row dated d or later unless the ticket says the round may see it
(the afternoon round sees d's morning). Windows and candles are point-in-time: tests/test_task3_v9_1.py rebuilds them
from truncated data and checks nothing changes.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from src.task3.versions9 import features as F
from src.task3.versions9 import ladder as L

END = dt.date(2026, 9, 4)          # last day before the F11 seal; T06 is the only reader past it
L_WIN = 20                         # T00 §3: one window length, never tuned


def table(rnd: str, end: dt.date = END) -> pd.DataFrame:
    """v9's round table (with `seq_i` for the afternoon) plus both rounds' targets and a global row id."""
    t = L.table(rnd, end=end)
    a = pd.read_parquet(F.STORE / "afternoon.parquet")[["date", "stock", "y_a", "P", "C_prev", "D"]]
    a["y_m_"] = np.log(a["P"] / (a["C_prev"] - a["D"]))
    # left merge keeps v9's row order: pooled HGB/MLP/CNN fits depend on it, and R2-R4 must reproduce v9 bit for bit
    t = t.merge(a[["date", "stock", "y_a", "y_m_"]].rename(columns={"y_a": "y_a_"}), on=["date", "stock"], how="left")
    t["row_i"] = np.arange(len(t), dtype=float)
    t["day_i"] = [float(d.toordinal()) for d in t["date"]]      # date order for models that cap their context (A14)
    return t


def cols(rnd: str, stock: str | None, kind: str) -> list:
    """kind="own": v9 ridge's columns for `stock` · kind="pooled": v9 HGB's columns (+ is_kbank)."""
    if kind == "own":
        own = F.feature_cols(rnd, stock)
        return own if rnd == "morning" else [c for c in own if c not in L.LATE]
    pooled = F.feature_cols(rnd, None) + ["is_kbank"]
    return pooled if rnd == "morning" else [c for c in pooled if c not in L.LATE]


def windows(t: pd.DataFrame, cols_: list, n: int = L_WIN) -> np.ndarray:
    """(rows, n, len(cols_) + 2): the stock's previous n rows of `cols_`, y_m, y_a (row d itself excluded).
    A window with any missing value is all NaN, so the row is dropped (no fill)."""
    v = t[[*cols_, "y_m_", "y_a_"]].to_numpy(float)
    out = np.full((len(t), n, v.shape[1]), np.nan)
    for _, idx in t.groupby("stock").indices.items():
        idx = np.sort(idx)
        assert (np.diff(t["date"].to_numpy()[idx]) > dt.timedelta(0)).all()
        for k, i in enumerate(idx):
            if k >= n:
                out[i] = v[idx[k - n:k]]
    out[~np.isfinite(out).all(axis=(1, 2))] = np.nan
    return out


def minute_seq() -> np.ndarray:
    """v9's (rows, 2, 155) morning minute sequences; a table row points at its sequence through `seq_i`."""
    return np.load(F.STORE / "sequences.npz")["x"]


def session_candles(raw: dict) -> pd.DataFrame:
    """Half-day candles of one stock (T03 §1.1), ordered in time: morning (close = P_d, known 12:30) and afternoon
    (close = official C_d, known 17:00). `amount` = sum(close x volume) of the 1m bars."""
    b = raw["m1"]
    b = b.assign(amount=b["close"] * b["volume"])
    am = F.morning_bars(b)
    pm = b[b["session"].isin(["afternoon", "closing_label"])]
    off = raw["official"].set_index("trade_date")

    def agg(x: pd.DataFrame) -> pd.DataFrame:
        g = x.sort_values("bar_end").groupby("trade_date")
        return pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(),
                             "close": g["close"].last(), "volume": g["volume"].sum(), "amount": g["amount"].sum()})

    m = agg(am).assign(half="am")
    m["known_at"] = F.stamp(m.index, dt.time(12, 30))
    p = agg(pm)
    p = p[p.index.isin(off.index)]
    c = off["close"].reindex(p.index)
    p["close"] = c                                         # official close (ATC), known with the official row
    p["high"], p["low"] = np.maximum(p["high"], c), np.minimum(p["low"], c)
    p = p.assign(half="pm")
    p["known_at"] = F.stamp(p.index, dt.time(17, 0))
    out = pd.concat([m, p]).rename_axis("date").reset_index()
    return out.sort_values("known_at").reset_index(drop=True)


def daily_ohlc(raw: dict) -> pd.DataFrame:
    """Official daily OHLCV of one stock (E06 morning chart)."""
    return raw["official"][["trade_date", "open", "high", "low", "close", "volume", "known_at"]].rename(
        columns={"trade_date": "date"}).sort_values("date").reset_index(drop=True)


def xd_log_factor(t: pd.DataFrame, stock: str) -> pd.Series:
    """G(d) = sum over XD days x <= d of log((C_{x-1} - D_x) / C_{x-1}), by date, for one stock."""
    s = t[t["stock"] == stock].set_index("date")
    lf = np.log((s["C_prev"] - s["D"]) / s["C_prev"]).where(s["D"] > 0, 0.0).fillna(0.0)
    return lf.cumsum()


def adjust_xd(dates, prices: np.ndarray, G: pd.Series, asof: dt.date) -> np.ndarray:
    """Back-adjust prices dated before each XD day x <= asof by that day's factor, so the last close before
    forecasting `asof` equals its anchor C_{d-1} - D_d. Only uses D of days up to `asof` (known before it)."""
    g = G.reindex(pd.Index(list(dates))).to_numpy()
    return prices * np.exp(G.loc[asof] - g)[:, None] if prices.ndim == 2 else prices * np.exp(G.loc[asof] - g)
