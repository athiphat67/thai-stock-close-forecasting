"""Historical replay of live.py: for past days, write the live input CSVs from the training data (no session column,
so live.py derives it), run live.features at the cutoff, and require the training feature store values and the bundle
predictions. Also writes inputs_example/ (one real day) + expected.json for `live.py --selftest`.

    OMP_NUM_THREADS=1 .venv/bin/python -W ignore -m src.task3.deployment.v9.replay_check [--from 2026-06-01] [--example 2026-08-31]
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import tempfile
from pathlib import Path

os.environ.setdefault("TASK3_V8_CONFIG", "src/task3/deployment/v9/data_config.yaml")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.task3.deployment.v9 import live as LV  # noqa: E402
from src.task3.deployment.v9 import predict as P  # noqa: E402
from src.task3.versions8.pipeline import load  # noqa: E402
from src.task3.versions9 import features as F  # noqa: E402

HERE = Path(__file__).resolve().parent
STORE = load.ROOT / "data/final_data/task3-v9-deploy-store"
YAHOO = load.ROOT / "data/raw/yahoo_chart/1d/_v9_m5/task3-v9-m5-20261028"
HIST = 30   # trading days of history written per run (features need <= 21)


def sources() -> dict:
    daily = pd.concat([load.load_official(s).assign(symbol=s) for s in F.STOCKS])
    m = []
    for s in [*F.STOCKS, *F.PEERS["KBANK"]]:
        b = load.load_1m(s)
        b = b.assign(bar_end=b["timestamp"] + pd.Timedelta(minutes=1))
        m.append(b[["symbol", "trade_date", "bar_end", "session", *LV.BAR_COLS]])
    m = pd.concat(m, ignore_index=True)
    lab = LV.session_of(m["bar_end"])
    mism = m[lab != m["session"]]
    y = []
    for name, (sym, *_ ) in F.FOREIGN.items():
        r = json.loads((YAHOO / f"{sym.replace('^', '_').replace('=', '_')}.json").read_text())["chart"]["result"][0]
        q = r["indicators"]["quote"][0]
        local = pd.to_datetime(r["timestamp"], unit="s", utc=True).tz_convert(r["meta"]["exchangeTimezoneName"])
        y.append(pd.DataFrame({"name": name, "date": local.date, "open": q["open"], "close": q["close"]})
                 .drop_duplicates("date", keep="last"))
    fut = pd.read_parquet(load.path("futures") / "futures_post_close.parquet")
    fut = fut[fut["tf_minutes"] == 30][["trade_date", "fut_1645", "fut_1655"]]
    fut["trade_date"] = pd.to_datetime(fut["trade_date"]).dt.date
    led = pd.read_csv(load.path("ledger"))[["symbol", "xd_date", "dividend_thb", "announce_datetime"]]
    return {"daily": daily, "m1": m, "yahoo": pd.concat(y), "fut": fut, "xd": led, "session_mismatch": mism}


def write_inputs(src: dict, day: dt.date, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    cal = sorted(src["daily"]["trade_date"].unique())
    hist = [d for d in cal if d < day][-HIST:]
    lo = hist[0]
    d = src["daily"]
    d[(d["trade_date"] >= lo) & (d["trade_date"] < day)][["symbol", "trade_date", *LV.BAR_COLS]].to_csv(out / "daily.csv", index=False)
    m = src["m1"]
    keep = m["symbol"].isin(F.STOCKS) & (m["trade_date"] >= lo) | (m["trade_date"] >= hist[-2])
    m = m[keep & (m["trade_date"] <= day)]
    m.assign(bar_end=m["bar_end"].map(lambda t: t.isoformat()))[["symbol", "bar_end", *LV.BAR_COLS]].to_csv(out / "bars_1m.csv", index=False)
    y = src["yahoo"]
    y[(y["date"] >= lo - dt.timedelta(days=7)) & (y["date"] <= day)].to_csv(out / "yahoo.csv", index=False)
    f = src["fut"]
    f[(f["trade_date"] >= lo) & (f["trade_date"] < day)].to_csv(out / "futures.csv", index=False)
    x = src["xd"]
    x[(pd.to_datetime(x["xd_date"]).dt.date >= lo)].to_csv(out / "xd.csv", index=False)


def main(start: dt.date, example: dt.date) -> dict:
    src = sources()
    store = {r: pd.read_parquet(STORE / f"{r}.parquet").set_index(["date", "stock"]) for r in ("morning", "afternoon")}
    days = sorted(d for d in store["morning"].index.get_level_values(0).unique() if d >= start)
    worst, n, models = 0.0, 0, {}
    tmp = Path(tempfile.mkdtemp())
    try:
        for day in days:
            write_inputs(src, day, tmp)
            for rnd in ("morning", "afternoon"):
                for sym in F.STOCKS:
                    feats, anchors = LV.features(tmp, sym, day, rnd)
                    row = store[rnd].loc[(day, sym)]
                    for c, v in feats.items():
                        s = row[c]
                        if (v is None) != pd.isna(s) or (v is not None and abs(v - s) > 1e-12):
                            if c.startswith("peer_") and day > dt.date(2026, 9, 4):
                                continue   # deploy store blanks 2-bank peer averages after 09-04 (build_data.py)
                            raise AssertionError(f"{day} {rnd} {sym} {c}: live {v} vs store {s}")
                        if v is not None:
                            worst = max(worst, abs(v - s))
                    n += 1
                    res = P.predict({"round": rnd, "symbol": sym, **anchors, "features": feats})
                    models[res["model"]] = models.get(res["model"], 0) + 1
    finally:
        shutil.rmtree(tmp)

    ex = HERE / "inputs_example"
    shutil.rmtree(ex, ignore_errors=True)
    write_inputs(src, example, ex)
    expected = []
    for rnd in ("morning", "afternoon"):
        for r in LV.run(ex, rnd, example, None):
            expected.append({**{k: r[k] for k in ("round", "symbol", "date", "model", "features")}, "prediction": r["prediction_raw"]})
    (ex / "expected.json").write_text(json.dumps(expected, indent=1))
    out = {"days": len(days), "first": str(days[0]), "last": str(days[-1]), "runs": n, "max_abs_diff": worst,
           "models_used": models, "session_label_mismatches": len(src["session_mismatch"]),
           "example_day": str(example), "example_models": {f"{e['round']}/{e['symbol']}": e["model"] for e in expected}}
    print(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", type=dt.date.fromisoformat, default=dt.date(2026, 6, 1))
    ap.add_argument("--example", type=dt.date.fromisoformat, default=dt.date(2026, 8, 31))
    a = ap.parse_args()
    main(a.start, a.example)
