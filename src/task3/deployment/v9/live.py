"""Live v9 run: raw input tables -> features (locked v9 code, v9_features.py) -> prediction -> one log line per stock.

    python live.py --inputs DIR --round morning|afternoon --date YYYY-MM-DD [--log predictions.jsonl]
    python live.py --selftest        # replays inputs_example/ and checks the locked code is unchanged

Input tables in DIR (CSV, one header row; times are ISO with offset, e.g. 2026-11-10T09:58:00+07:00):
  daily.csv     symbol, trade_date, open, high, low, close, volume     official daily bars of KBANK and ADVANC,
                                                                        >= 21 trading days up to the day before
  bars_1m.csv   symbol, bar_end, open, high, low, close, volume [, session]
                complete 1m bars labelled by END time: KBANK and ADVANC for >= 21 trading days up to the cutoff,
                the six bank peers for the last 2 days (+ today's morning for the afternoon fallback).
                session (morning / intermission / afternoon / closing_label) is derived from bar_end if absent.
  yahoo.csv     name (spx, thd, thb, n225, hsi), date (the exchange's own date), open, close
  futures.csv   trade_date, fut_1645, fut_1655   (optional) S50 front month: close of the 30m bar starting 16:15
                                                  and of the bar starting 16:45 -> fut_post = log(fut_1655 / fut_1645)
  xd.csv        symbol, xd_date, dividend_thb, announce_datetime          (optional) XD dividends
Rows that would be known only after the cutoff (09:30 / 13:00 of --date) are dropped before the features are built,
with the same `known_at` rules as the training data, so passing extra rows can never leak.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import predict as P  # noqa: E402
import v9_features as F  # noqa: E402

CUTOFF = {"morning": dt.time(9, 30), "afternoon": dt.time(13, 0)}
BAR_COLS = ["open", "high", "low", "close", "volume"]


def session_of(bar_end: pd.Series) -> pd.Series:
    """MT5 session labels from bar END time (SET: morning <= 12:30, afternoon opens 14:00 since 2024-03-25)."""
    t = bar_end.dt.tz_convert(F.BKK).dt.strftime("%H:%M")
    s = pd.Series("afternoon", index=bar_end.index)
    s[t <= "12:30"] = "morning"
    s[(t > "12:30") & (t <= "14:00")] = "intermission"
    s[t > "16:30"] = "closing_label"
    return s


def read(inputs: Path, name: str, required: bool = True) -> pd.DataFrame | None:
    p = inputs / f"{name}.csv"
    if not p.exists():
        if required:
            raise FileNotFoundError(p)
        return None
    return pd.read_csv(p)


def bars(b: pd.DataFrame) -> pd.DataFrame:
    b = b.copy()
    b["bar_end"] = pd.to_datetime(b["bar_end"], utc=True).dt.tz_convert(F.BKK)
    if "session" not in b or b["session"].isna().any():
        b["session"] = session_of(b["bar_end"])
    b["trade_date"] = b["bar_end"].dt.date
    for c in BAR_COLS:
        b[c] = b[c].astype(float)
    return b.sort_values("bar_end").assign(known_at=b["bar_end"])


def raw_for(inputs: Path, sym: str) -> tuple[dict, list]:
    """The `raw` dict features.build expects, from the input tables. Same known_at rules as features.raw_inputs."""
    daily = read(inputs, "daily")
    d = daily[daily["symbol"] == sym].copy()
    d["trade_date"] = pd.to_datetime(d["trade_date"]).dt.date
    for c in BAR_COLS:
        d[c] = d[c].astype(float)
    official = d.sort_values("trade_date").assign(known_at=F.stamp(d["trade_date"], dt.time(17, 0)))

    m = bars(read(inputs, "bars_1m"))
    peers = {p: m[m["symbol"] == p] for p in F.PEERS[sym]}

    y = read(inputs, "yahoo")
    yrows = []
    for name, (_, tz, close_t, _) in F.FOREIGN.items():
        x = y[y["name"] == name].copy()
        x["date"] = pd.to_datetime(x["date"]).dt.date
        for field, t in (("close", close_t), ("open", F.OPEN_TIME.get(tz, close_t))):
            z = x[["date", field]].dropna().rename(columns={field: "value"})
            yrows.append(z.assign(name=name, field=field, known_at=F.stamp(z["date"], t, tz)))
    yahoo = pd.concat(yrows, ignore_index=True)

    fut = read(inputs, "futures", required=False)
    if fut is None:
        fut = pd.DataFrame({"trade_date": [], "fut_1645": [], "fut_1655": []})
    fut["trade_date"] = pd.to_datetime(fut["trade_date"]).dt.date
    fut_post = pd.DataFrame({"trade_date": fut["trade_date"],
                             "post_close_logret": np.log(fut["fut_1655"].astype(float) / fut["fut_1645"].astype(float))})
    fut_post = fut_post.assign(known_at=F.stamp(fut_post["trade_date"], dt.time(17, 15)))

    xd = read(inputs, "xd", required=False)
    if xd is None:
        xd = pd.DataFrame({"symbol": [], "xd_date": [], "dividend_thb": [], "announce_datetime": []})
    xd = xd[xd["symbol"] == sym].copy()
    xd["xd_date"] = pd.to_datetime(xd["xd_date"]).dt.date
    known = [pd.Timestamp(a).tz_localize(F.BKK) if pd.notna(a) and pd.Timestamp(a).tzinfo is None
             else pd.Timestamp(a).tz_convert(F.BKK) if pd.notna(a)
             else (pd.Timestamp(x - dt.timedelta(days=1)) + pd.Timedelta(hours=17)).tz_localize(F.BKK)
             for x, a in zip(xd["xd_date"], xd["announce_datetime"])]
    ledger = xd.assign(dividend_thb=xd["dividend_thb"].astype(float), known_at=pd.DatetimeIndex(known))

    empty = pd.DataFrame({"trade_date": [], "bar_start": pd.DatetimeIndex([], tz=F.BKK),
                          "bar_end": pd.DatetimeIndex([], tz=F.BKK), "close": [], "known_at": pd.DatetimeIndex([], tz=F.BKK)})
    raw = {"official": official, "ledger": ledger, "m1": m[m["symbol"] == sym], "peers": peers, "yahoo": yahoo,
           "fut_post": fut_post, "fut": empty, "set50": empty, "set": empty, "sector": empty}

    def tz(f: pd.DataFrame) -> pd.DataFrame:   # an empty table comes out tz-naive and cannot be compared to the cutoff
        k = pd.DatetimeIndex(f["known_at"])
        return f.assign(known_at=k.tz_localize(F.BKK) if k.tz is None else k.tz_convert(F.BKK))
    raw = {k: ({p: tz(f) for p, f in v.items()} if isinstance(v, dict) else tz(v)) for k, v in raw.items()}
    return raw, sorted(official["trade_date"])


def features(inputs: Path, sym: str, day: dt.date, rnd: str) -> tuple[dict, dict]:
    raw, days = raw_for(inputs, sym)
    cutoff = pd.Timestamp(dt.datetime.combine(day, CUTOFF[rnd]), tz=F.BKK)
    days = [d for d in days if d < day] + [day]
    m, a = F.build(F.truncate(raw, cutoff), days)
    row = (m if rnd == "morning" else a).loc[day]
    used = {c for k, m in P.models().items() if k.startswith(rnd + "/") for c in m["cols"]}   # index/futures-at-noon
    want = [c for c in (F.MORNING if rnd == "morning" else F.AFTERNOON) if c in used]       # features: no model uses
    feats = {c: (None if pd.isna(row[c]) else float(row[c])) for c in want}
    anchors = {"C_prev": row["C_prev"], "D": row["D"]} if rnd == "morning" else {"P": row["P"]}
    return feats, {k: (None if pd.isna(v) else float(v)) for k, v in anchors.items()}


def run(inputs: Path, rnd: str, day: dt.date, log: Path | None) -> list[dict]:
    created = pd.Timestamp.now(tz=F.BKK)
    cutoff = pd.Timestamp(dt.datetime.combine(day, CUTOFF[rnd]), tz=F.BKK)
    bundle_sha = hashlib.sha256((HERE / "bundle" / "models.joblib").read_bytes()).hexdigest()
    out = []
    for sym in F.STOCKS:
        line = {"round": rnd, "symbol": sym, "date": str(day), "cutoff": cutoff.isoformat(),
                "created_at": created.isoformat(timespec="seconds"), "on_time": bool(created < cutoff),
                "bundle_sha256": bundle_sha}
        try:
            feats, anchors = features(inputs, sym, day, rnd)
            if None in anchors.values():
                raise ValueError(f"missing anchor {anchors}: need yesterday's official close (morning) or today's morning bars (afternoon)")
            res = P.predict({"round": rnd, "symbol": sym, "date": str(day), **anchors, "features": feats})
            line.update(prediction=res["prediction"], prediction_raw=res["prediction_raw"], model=res["model"], yhat=res["yhat"], anchor=res["anchor"],
                        target=res["target"], missing_for_primary=res.get("missing_for_primary", []), features=feats)
        except Exception as e:  # noqa: BLE001  one stock failing must not stop the other
            line.update(error=f"{type(e).__name__}: {e}")
        out.append(line)
    if log:
        with open(log, "a") as f:
            for line in out:
                f.write(json.dumps(line) + "\n")
    return out


def selftest() -> None:
    try:   # on the research branch: the vendored code must still be the locked code
        sys.path.insert(0, str(HERE.parents[3]))
        from src.task3.versions9 import features as LOCKED
        import inspect
        for name in ("feature_cols", "stamp", "truncate", "last_on_or_before", "morning_bars", "day_close", "tick_dir", "build"):
            assert inspect.getsource(getattr(F, name)) == inspect.getsource(getattr(LOCKED, name)), f"{name} drifted"
        for name in ("PEERS", "SESSION_CHANGE", "FOREIGN", "OPEN_TIME", "MORNING", "AFTERNOON"):
            assert getattr(F, name) == getattr(LOCKED, name), f"{name} drifted"
        print("ok: v9_features.py is identical to src/task3/versions9/features.py")
    except ImportError:
        print("skip: src/task3/versions9 not here, vendored-code check not run")
    exp = json.loads((HERE / "inputs_example" / "expected.json").read_text())
    for e in exp:
        got = {r["symbol"]: r for r in run(HERE / "inputs_example", e["round"], dt.date.fromisoformat(e["date"]), None)}[e["symbol"]]
        assert "error" not in got, got
        assert got["model"] == e["model"], (e, got["model"])
        assert abs(got["prediction_raw"] - e["prediction"]) < 1e-9, (e, got["prediction_raw"])   # expected.json holds the raw value
        assert got["prediction"] == P.to_tick(e["prediction"]), (e, got["prediction"])
        for k, v in e["features"].items():
            g = got["features"][k]
            assert (v is None and g is None) or (v is not None and g is not None and abs(g - v) < 1e-12), (e["round"], e["symbol"], k, v, g)
    print(f"ok: {len(exp)} live runs from inputs_example/ match the training features and predictions")


if __name__ == "__main__":
    if sys.argv[1:] == ["--selftest"]:
        selftest()
        sys.exit()
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", type=Path, required=True)
    ap.add_argument("--round", choices=list(CUTOFF), required=True)
    ap.add_argument("--date", type=dt.date.fromisoformat, required=True)
    ap.add_argument("--log", type=Path, default=None, help="append one JSON line per stock")
    a = ap.parse_args()
    print(json.dumps(run(a.inputs, a.round, a.date, a.log), indent=1))
