"""Vendored copy of the locked v9 feature code (src/task3/versions9/features.py @ a7e3bdf, tag v9-prereg).

Generated, do not edit: the constants and functions below are copied verbatim so the live side computes features with
the same code the models were trained on, without the research repo. `python live.py --selftest` checks, when
src/task3/versions9 is importable, that every function here is still byte-identical to the locked file.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

BKK = "Asia/Bangkok"
STOCKS = ("KBANK", "ADVANC")
PEERS = {"KBANK": ["BBL", "KTB", "SCB", "TTB", "KKP", "TISCO"], "ADVANC": []}   # ADVANC's one peer TRUE starts 2023-03-03 (owner 2026-10-28: drop)
SESSION_CHANGE = dt.date(2024, 3, 25)        # afternoon opens 14:00 instead of 14:30 (Q07)
FOREIGN = {"spx": ("^GSPC", "America/New_York", dt.time(16, 0), "close"),
           "thd": ("THD", "America/New_York", dt.time(16, 0), "close"),
           "thb": ("THB=X", "Europe/London", dt.time(23, 59), "close"),
           "n225": ("^N225", "Asia/Tokyo", dt.time(15, 30), "open"),
           "hsi": ("^HSI", "Asia/Hong_Kong", dt.time(16, 10), "open")}
OPEN_TIME = {"Asia/Tokyo": dt.time(9, 0), "Asia/Hong_Kong": dt.time(9, 30)}
MORNING = ["spx", "thd", "thb", "n225", "hsi", "fut_post", "on_prev", "id_prev", "range_prev", "vol_prev",
           "peer_prev", "monday", "gap_days", "long_break", "xd", "atc_push"]
AFTERNOON = [*MORNING, "r_morning", "jump", "r_last30", "r_last60", "vwap_gap", "range_am", "vol_am", "peer_am",
             "set50_am", "set_am", "sector_am", "fut_am", "basis_1230", "post_2024", "tick_dir"]


def feature_cols(rnd: str, stock: str | None = None) -> list:
    """Columns a model may use. Peer returns only for a stock with peers; pooled models (stock=None) never get them."""
    cols = MORNING if rnd == "morning" else AFTERNOON
    return [c for c in cols if not c.startswith("peer_") or (stock is not None and PEERS[stock])]


def stamp(dates, t: dt.time, tz: str = BKK) -> pd.DatetimeIndex:
    return pd.DatetimeIndex([pd.Timestamp(dt.datetime.combine(d, t), tz=tz).tz_convert(BKK) for d in dates])


def truncate(raw: dict, cutoff: pd.Timestamp) -> dict:
    cut = lambda f: f[f["known_at"] <= cutoff]  # noqa: E731
    return {k: ({p: cut(f) for p, f in v.items()} if isinstance(v, dict) else cut(v)) for k, v in raw.items()}


def last_on_or_before(s: pd.Series, when) -> np.ndarray:
    s = s.sort_index()
    pos = s.index.searchsorted(when, side="right") - 1
    return np.where(pos >= 0, s.to_numpy()[np.clip(pos, 0, None)], np.nan)


def morning_bars(b: pd.DataFrame) -> pd.DataFrame:
    t = b["bar_end"].dt.tz_convert(BKK).dt.time
    return b[(b["session"] == "morning") & (t <= dt.time(12, 30))]


def day_close(bars_: pd.DataFrame, upto: dt.time | None = None) -> pd.Series:
    """Close of the last bar per trade_date (with bar_end <= `upto` if given)."""
    if upto is not None:
        bars_ = bars_[bars_["bar_end"].dt.tz_convert(BKK).dt.time <= upto]
    return bars_.sort_values("bar_end").groupby("trade_date")["close"].last()


def tick_dir(closes: pd.Series) -> float:
    c = closes.to_numpy()
    c = c[np.r_[True, c[1:] != c[:-1]]]
    return float(np.sign(c[-1] - c[-2])) if len(c) > 1 else 0.0


def build(raw: dict, days: list) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Morning and afternoon tables for one stock on `days` (the full SET calendar, ascending)."""
    idx = pd.Index(days, name="date")
    prev = pd.Index([None, *days[:-1]])
    prev2 = pd.Index([None, None, *days[:-2]])
    f = pd.DataFrame(index=idx)

    off = raw["official"].set_index("trade_date")
    D = raw["ledger"].groupby("xd_date")["dividend_thb"].sum().reindex(idx).fillna(0.0)
    O, H, L, C, V = (off[c].reindex(idx) for c in ("open", "high", "low", "close", "volume"))
    f["C_prev"] = C.shift(1)
    f["D"] = D
    anchor = f["C_prev"] - D

    y = raw["yahoo"]
    for name, (_, _, _, field) in FOREIGN.items():
        close = y[(y["name"] == name) & (y["field"] == "close")].set_index("date")["value"]
        if field == "close":
            dm1 = [d - dt.timedelta(days=1) for d in days]
            pm1 = [p - dt.timedelta(days=1) if p else dt.date(1900, 1, 1) for p in prev]
            f[name] = np.log(last_on_or_before(close, dm1) / last_on_or_before(close, pm1))
        else:
            o = y[(y["name"] == name) & (y["field"] == "open")].set_index("date")["value"].reindex(idx).to_numpy()
            base = last_on_or_before(close, [p if p else dt.date(1900, 1, 1) for p in prev])
            f[name] = np.where(np.isnan(o), 0.0, np.log(o / base))
    f.iloc[0, f.columns.get_indexer(list(FOREIGN))] = np.nan

    f["fut_post"] = raw["fut_post"].set_index("trade_date")["post_close_logret"].reindex(prev).to_numpy()
    f["on_prev"] = np.log(O / (C.shift(1) - D)).shift(1)
    f["id_prev"] = np.log(C / O).shift(1)
    f["range_prev"] = np.log(H / L).shift(1)
    f["vol_prev"] = np.log(V / V.rolling(20, min_periods=15).mean()).shift(1)
    atc = pd.DataFrame({p: b[b["session"] == "closing_label"].groupby("trade_date")["close"].last().reindex(idx)
                        for p, b in raw["peers"].items()}, index=idx)
    f["peer_prev"] = np.log(atc / atc.shift(1)).mean(axis=1).shift(1)
    f["monday"] = [float(d.weekday() == 0) for d in days]
    f["gap_days"] = [float((d - p).days) if p else np.nan for d, p in zip(days, prev)]
    f["long_break"] = (f["gap_days"] >= 4).astype(float).where(f["gap_days"].notna())
    f["xd"] = D / f["C_prev"]
    cont = day_close(raw["m1"][raw["m1"]["session"] == "afternoon"]).reindex(idx)
    f["atc_push"] = np.log(C / cont).shift(1)

    am = morning_bars(raw["m1"])
    g = am.groupby("trade_date")
    P = g["close"].last().reindex(idx)
    f["P"] = P
    f["C"] = C
    f["y_m"] = np.log(P / anchor)
    f["y_a"] = np.log(C / P)
    morning = f[["C_prev", "D", "P", "y_m", *MORNING]].copy()

    a = f
    a["r_morning"] = np.log(P / anchor)
    a["jump"] = np.log(g["open"].first().reindex(idx) / anchor)
    a["r_last30"] = np.log(P / day_close(am, dt.time(12, 0)).reindex(idx))
    a["r_last60"] = np.log(P / day_close(am, dt.time(11, 30)).reindex(idx))
    vwap = (am["close"] * am["volume"]).groupby(am["trade_date"]).sum() / g["volume"].sum()
    a["vwap_gap"] = np.log(P / vwap.reindex(idx))
    a["range_am"] = np.log(g["high"].max() / g["low"].min()).reindex(idx)
    vam = g["volume"].sum().reindex(idx)
    a["vol_am"] = np.log(vam / vam.rolling(20, min_periods=15).mean().shift(1))
    pam = pd.DataFrame({p: day_close(morning_bars(b)).reindex(idx) for p, b in raw["peers"].items()}, index=idx)
    a["peer_am"] = np.log(pam / atc.shift(1)).mean(axis=1)
    for name in ("set50", "set", "sector"):
        b = raw[name]
        a[f"{name}_am"] = np.log(day_close(b, dt.time(12, 30)).reindex(idx) / day_close(b).reindex(idx).shift(1))
    fut = raw["fut"]
    fut_am = day_close(fut[fut["bar_start"].dt.time < dt.time(12, 30)], dt.time(13, 0)).reindex(idx)
    a["fut_am"] = np.log(fut_am / day_close(fut).reindex(idx).shift(1))
    a["basis_1230"] = np.log(fut_am / day_close(raw["set50"], dt.time(12, 30)).reindex(idx))
    a["post_2024"] = [float(d >= SESSION_CHANGE) for d in days]
    a["tick_dir"] = g["close"].apply(tick_dir).reindex(idx)
    afternoon = a[["C_prev", "D", "P", "C", "y_a", *AFTERNOON]].copy()
    return morning, afternoon
