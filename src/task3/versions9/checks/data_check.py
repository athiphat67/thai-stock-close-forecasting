"""Task 3 v9 data check: is the local data enough to build both rounds (morning close P_d, official close C_d)?

Reads through the v8 loaders (read-only), so the sealed window (trade_date >= 2026-09-05) is never returned.
Run from the repo root:  .venv/bin/python src/task3/versions9/checks/data_check.py
"""
import sys
from datetime import date, time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src/task3/versions8"))
from pipeline import load  # noqa: E402

EXPORT = ROOT / "data/final_data/task3-v8-set-export"
SYMBOLS = ["KBANK", "ADVANC"]
SEALED = date(2026, 9, 5)


def tick(p: float) -> float:
    for lim, t in [(2, .01), (5, .02), (10, .05), (25, .1), (100, .25), (200, .5), (400, 1)]:
        if p < lim:
            return t
    return 2.0


def export(group: str, name: str) -> pd.DataFrame:
    d = pd.read_parquet(EXPORT / group / f"{name}.parquet")
    d["trade_date"] = pd.to_datetime(d["trade_date"]).dt.date
    return d[d["trade_date"] < SEALED]


def check_stock(sym: str) -> None:
    b = load.load_1m(sym)
    t = b["timestamp"].dt.time
    off = load.load_official(sym).set_index("trade_date")["close"]
    days_1m, days_off = set(b["trade_date"]), set(off.index)
    print(f"\n## {sym}")
    print(f"1m: {b['trade_date'].min()} -> {b['trade_date'].max()}, {len(days_1m)} days · official: {len(days_off)} days")
    print(f"  official days with no 1m bars: {len(days_off - days_1m)} · 1m days with no official close: {len(days_1m - days_off)}")
    print(f"  bars before 09:30: {(t < time(9, 30)).sum()} · bars 12:30-13:00: {((t >= time(12, 30)) & (t < time(13, 0))).sum()}")

    am = b[(t >= time(9, 30)) & (t < time(12, 30))]
    last = am.groupby("trade_date")["timestamp"].max().dt.time
    P = am.groupby("trade_date")["close"].last()
    print(f"  days with morning bars: {len(P)} · official days with no morning bar: {len(days_off - set(P.index))}")
    print(f"  last morning bar at 12:29: {(last == time(12, 29)).mean():.1%} · at or before 12:25: {(last <= time(12, 25)).mean():.2%}"
          f" · earliest last bar: {last.min()}")
    per_year = pd.Series(sorted(P.index)).map(lambda d: d.year).value_counts().sort_index()
    print("  morning-close days per year:", per_year.to_dict())

    # second source for P_d: TradingView 30m bar 12:00-12:30 (2019+)
    tv = export("stock", f"{sym}_30m")
    tv = tv[pd.to_datetime(tv["bar_end"]).dt.strftime("%H:%M") == "12:30"].set_index("trade_date")["close"]
    j = pd.concat([P.rename("mt5"), tv.rename("tv")], axis=1, join="inner")
    diff = (j.mt5 - j.tv).abs()
    tk = j.mt5.map(tick)
    print(f"  P_d MT5 vs TradingView 30m (12:30 close), {len(j)} days {j.index.min()} -> {j.index.max()}: "
          f"exact {(diff < 1e-9).mean():.1%} · within 1 tick {(diff <= tk + 1e-9).mean():.1%} · max {diff.max():.2f} baht")
    bad = j[diff > tk + 1e-9]
    if len(bad):
        print("  days off by more than 1 tick (first 5):", bad.head().round(2).to_dict("index"))

    # official close vs SETSMART close
    ss = load.load_setsmart(sym).set_index("trade_date")["close"].astype(float)
    k = pd.concat([off.rename("off"), ss.rename("ss")], axis=1, join="inner")
    print(f"  official vs SETSMART close, {len(k)} days: exact {((k.off - k.ss).abs() < 1e-9).mean():.1%}")

    led = load.load_ledger(sym)
    print(f"  XD events: {len(led)} ({led['xd_date'].min()} -> {led['xd_date'].max()}),"
          f" on official days: {led['xd_date'].isin(days_off).sum()}")


def check_others(days_off: set) -> None:
    print("\n## Other local inputs (rows before 2026-09-05 only)")
    fut = pd.read_parquet(EXPORT / "futures_post_close.parquet")
    fut["trade_date"] = pd.to_datetime(fut["trade_date"]).dt.date
    print(f"futures post-close 16:45-16:55: {fut['trade_date'].min()} -> {fut['trade_date'].max()}, {len(fut)} days,"
          f" official days missing: {len({d for d in days_off if d >= fut['trade_date'].min()} - set(fut['trade_date']))}")
    for group, name in [("index", "SET50_30m"), ("index", "SET50_10m"), ("index", "SET_30m"), ("sector", "BANK_30m"),
                        ("sector", "ICT_30m"), ("tfex", "S501_30m"), ("tfex", "S501_10m")]:
        d = export(group, name)
        ends = pd.to_datetime(d["bar_end"]).dt.strftime("%H:%M")
        am_ok = (ends <= "12:30").groupby(d["trade_date"]).any().mean()
        print(f"{group}/{name}: {d['trade_date'].min()} -> {d['trade_date'].max()}, {d['trade_date'].nunique()} days,"
              f" first bar end {ends.min()}, bars ending <= 09:30: {(ends <= '09:30').sum()},"
              f" days with a bar ending <= 12:30: {am_ok:.0%}")


if __name__ == "__main__":
    all_days = set()
    for s in SYMBOLS:
        check_stock(s)
        all_days |= set(load.load_official(s)["trade_date"])
    check_others(all_days)
    assert tick(150) == .5 and tick(250) == 1, "tick table"
