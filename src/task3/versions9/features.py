"""Task 3 v9 M1 feature tables (03_FEATURES §1-2): one row per stock per SET day for each round.

    .venv/bin/python -m src.task3.versions9.features      # writes data/final_data/task3-v9-feature-store/

Every raw row carries `known_at`, the Bangkok time it became public (bars: `bar_end`). `build()` only reads `raw`,
so `checks/m1_check.py` can drop every row with known_at > cutoff and recompute: the values must not change
(03_FEATURES header). The trading calendar is the one input not truncated, since SET publishes it ahead.

Targets (02_PROTOCOL §3): y_m = log(P_d / (C_{d-1} - D_d)), y_a = log(C_d / P_d), P_d = close of the last morning
1m bar with bar_end <= 12:30 (Q03). Not clipped or filled here: clipping is per training window, missing stays NaN.
The scaled foreign composite (03_FEATURES §1 row 2) needs the training window's SD, so models build it, not this file.
"""

from __future__ import annotations

import datetime as dt
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from src.task3.versions8.pipeline import load

BKK = "Asia/Bangkok"
STOCKS = ("KBANK", "ADVANC")
PEERS = {"KBANK": ["BBL", "KTB", "SCB", "TTB", "KKP", "TISCO"], "ADVANC": []}   # ADVANC's one peer TRUE starts 2023-03-03 (owner 2026-10-28: drop)
SECTOR = {"KBANK": "BANK", "ADVANC": "ICT"}
CUTOFF = {"morning": dt.time(9, 30), "afternoon": dt.time(13, 0)}
SESSION_CHANGE = dt.date(2024, 3, 25)        # afternoon opens 14:00 instead of 14:30 (Q07)
YAHOO = load.ROOT / "data/raw/yahoo_chart/1d/_v9_overnight/task3-v9-overnight-20261027"
SET_EXPORT = load.ROOT / "data/final_data/task3-v8-set-export"
STORE = load.ROOT / "data/final_data/task3-v9-feature-store"
# name: (Yahoo symbol, exchange tz, local close time, field used). Close series: last close with L <= d-1.
# Open series: open of d itself (0 if shut). THB=X's Yahoo day is a London day, so its close lands at 23:59 GMT.
FOREIGN = {"spx": ("^GSPC", "America/New_York", dt.time(16, 0), "close"),
           "thd": ("THD", "America/New_York", dt.time(16, 0), "close"),
           "thb": ("THB=X", "Europe/London", dt.time(23, 59), "close"),
           "n225": ("^N225", "Asia/Tokyo", dt.time(15, 30), "open"),
           "hsi": ("^HSI", "Asia/Hong_Kong", dt.time(16, 10), "open")}
HK_PM_ONLY = {dt.date(2020, 8, 19), dt.date(2023, 10, 9)}   # M1 review 2026-10-28; Yahoo still has a row
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


def yahoo() -> pd.DataFrame:
    """Long frame (name, date, field, value, known_at): opens and closes become public at different times."""
    out = []
    for name, (sym, tz, close_t, _) in FOREIGN.items():
        r = json.loads((YAHOO / f"{sym.replace('^', '_').replace('=', '_')}.json").read_text())["chart"]["result"][0]
        q = r["indicators"]["quote"][0]
        # each bar's own local date: one fixed gmtoffset put THB=X's UK-summer bars (23:00 UTC) on the day before
        local = pd.to_datetime(r["timestamp"], unit="s", utc=True).tz_convert(r["meta"]["exchangeTimezoneName"])
        d = pd.DataFrame({"date": local.date, "open": q["open"], "close": q["close"]}).drop_duplicates("date", keep="last")
        if name == "hsi":   # typhoon signal 8 cancelled these mornings: the "open" came at 13:00 HKT, after 09:30
            d.loc[d["date"].isin(HK_PM_ONLY), "open"] = np.nan
        for field, t in (("close", close_t), ("open", OPEN_TIME.get(tz, close_t))):
            x = d[["date", field]].dropna().rename(columns={field: "value"})
            out.append(x.assign(name=name, field=field, known_at=stamp(x["date"], t, tz)))
    return pd.concat(out, ignore_index=True)


def bars(path: Path) -> pd.DataFrame:
    b = pd.read_parquet(path)
    b = b[pd.to_datetime(b["trade_date"]).dt.date < load.blocked_from()].copy()   # K07 guard, as load.py does
    b["trade_date"] = pd.to_datetime(b["trade_date"]).dt.date
    return b.assign(known_at=b["bar_end"])


def m1(sym: str) -> pd.DataFrame:
    b = load.load_1m(sym)[["trade_date", "timestamp", "session", "open", "high", "low", "close", "volume"]]
    b = b.assign(bar_end=b.pop("timestamp") + pd.Timedelta(minutes=1))   # MT5 labels a bar by its start
    return b.assign(known_at=b["bar_end"])


def raw_inputs(sym: str) -> dict:
    off = load.load_official(sym)
    led = load.load_ledger(sym)
    full = pd.read_csv(load.path("ledger"))
    ann = pd.to_datetime(full[full["symbol"] == sym].set_index("xd_date")["announce_datetime"])
    ann.index = pd.to_datetime(ann.index).date
    # 19 early rows have no announcement time: the XD date is fixed days ahead, so use 17:00 of the day before
    known = [ann.get(x) if pd.notna(ann.get(x)) else pd.Timestamp(x - dt.timedelta(days=1)) + pd.Timedelta(hours=17)
             for x in led["xd_date"]]
    fp = load.load_futures_daily(30)[["trade_date", "post_close_logret"]]
    return {
        "official": off.assign(known_at=stamp(off["trade_date"], dt.time(17, 0))),
        "ledger": led.assign(known_at=pd.DatetimeIndex(known).tz_localize(BKK)),
        "m1": m1(sym),
        "peers": {p: m1(p) for p in PEERS[sym]},
        "yahoo": yahoo(),
        "fut_post": fp.assign(known_at=stamp(fp["trade_date"], dt.time(17, 15))),   # 16:45 bar ends 17:15
        "fut": bars(SET_EXPORT / "tfex/S501_30m.parquet"),
        "set50": bars(SET_EXPORT / "index/SET50_30m.parquet"),
        "set": bars(SET_EXPORT / "index/SET_30m.parquet"),
        "sector": bars(SET_EXPORT / f"sector/{SECTOR[sym]}_30m.parquet"),
    }


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


def sequences(raw: dict, days: list, anchor: pd.Series) -> np.ndarray:
    """1D-CNN input (02_PROTOCOL §3 step 6), shape (days, 2, 155): per morning minute (bar_end 09:56 → 12:30),
    log return (the first minute vs the anchor C_{d-1} - D_d) and log(1 + volume / mean volume per minute over the
    20 days before d). A minute with no trade has return 0 and volume 0 (it happened, it is not a fill).
    A day with no morning bar or no anchor is all NaN."""
    am = morning_bars(raw["m1"])
    t = am["bar_end"].dt.tz_convert(BKK).dt.strftime("%H:%M")
    close = am.pivot_table(index="trade_date", columns=t, values="close", aggfunc="last").reindex(index=days, columns=MINUTES)
    vol = am.pivot_table(index="trade_date", columns=t, values="volume", aggfunc="sum").reindex(index=days, columns=MINUTES)
    traded = close.notna().any(axis=1).to_numpy()
    a = anchor.reindex(days).to_numpy()
    c = close.ffill(axis=1).to_numpy()
    c = np.where(np.isnan(c), a[:, None], c)                       # before the first trade the price is the anchor
    r = np.diff(np.log(np.column_stack([a, c])), axis=1)
    v = vol.fillna(0.0)
    per_min = v.sum(axis=1).rolling(20, min_periods=15).mean().shift(1) / len(MINUTES)
    q = np.log1p(v.to_numpy() / per_min.to_numpy()[:, None])
    out = np.stack([r, q], axis=1)
    out[~(traded & np.isfinite(a) & np.isfinite(per_min.to_numpy()) & (per_min.to_numpy() > 0))] = np.nan
    return out


MINUTES = [f"{h:02d}:{m:02d}" for h, m in ((x // 60, x % 60) for x in range(9 * 60 + 56, 12 * 60 + 31))]


def calendar(sym: str) -> list:
    return sorted(load.load_official(sym)["trade_date"])


def write() -> dict:
    out = {"morning": [], "afternoon": []}
    seq, seq_keys = [], []
    for sym in STOCKS:
        raw, days = raw_inputs(sym), calendar(sym)
        m, a = build(raw, days)
        out["morning"].append(m.assign(stock=sym))
        out["afternoon"].append(a.assign(stock=sym))
        seq.append(sequences(raw, days, a["C_prev"] - a["D"]))
        seq_keys += [(sym, str(d)) for d in days]
    STORE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(STORE / "sequences.npz", x=np.concatenate(seq).astype(np.float32),
                        stock=np.array([k[0] for k in seq_keys]), date=np.array([k[1] for k in seq_keys]))
    summary = {}
    for rnd, parts in out.items():
        t = pd.concat(parts).reset_index()
        t.to_parquet(STORE / f"{rnd}.parquet", index=False)
        summary[rnd] = {"rows": len(t), "first": str(t["date"].min()), "last": str(t["date"].max()),
                        "non_null_from": {c: str(t.loc[t[c].notna(), "date"].min()) for c in t.columns if c not in ("date", "stock")}}
    for p in YAHOO.glob("*.json"):
        load.sha256(p)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=load.ROOT).stdout.strip()
    manifest = {"git_commit": commit, "summary": summary, **load.manifest()}
    (STORE / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str))
    return summary


if __name__ == "__main__":
    print(json.dumps(write(), indent=1))
