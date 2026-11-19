"""RMSE (main) and MAE of the v9 deploy pipeline, raw and rounded to the SET tick, against "price does not move".

    OMP_NUM_THREADS=1 .venv/bin/python -W ignore -m src.task3.deployment.v9.evaluate [--from 2026-09-07] [--to 2026-10-21]

1. walk-forward on the months opened for the refit: every week the four models are refit with fit.fit_one on rows
   dated before the week's first day (same rule as ladder.walk), then each day is scored through predict.predict,
   so primary/fallback choice and tick rounding are exactly what live.py does.
2. the locked confirm run (v9-confirm-20261028-a, 2023-01 -> 2026-09-04), same metrics with tick rounding added.
Writes eval_metrics.json next to this file.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.task3.deployment.v9 import fit as FT
from src.task3.deployment.v9 import predict as P
from src.task3.versions9 import features as F
from src.task3.versions9 import ladder as L

HERE = Path(__file__).resolve().parent
CONFIRM = F.load.ROOT / "artifacts/task3/v9/v9-confirm-20261028-a/daily.parquet"


def fit_all(tables: dict, before: dt.date) -> dict:
    models = {}
    for rnd, name, kind, pooled in FT.MODELS:
        t, target = tables[rnd], L.ROUND[rnd][0]
        spec = FT.spec_of(rnd, name)
        for stock in ([None] if pooled else list(F.STOCKS)):
            cols = spec["cols"](stock or F.STOCKS[0])
            g = t if pooled else t[t["stock"] == stock]
            tr = g[(g["date"] >= L.TRAIN_FROM) & (g["date"] < before)].dropna(subset=[*cols, target])
            hp = FT.HP_2026["hgb"] if kind == "hgb" else FT.HP_2026["huber"][stock] if kind == "huber" else None
            m = FT.fit_one(kind, tr[cols].to_numpy(float), tr[target].to_numpy(float), hp)
            m["cols"], m["train_last"] = cols, str(tr["date"].max())
            models[f"{rnd}/{name}/{stock or 'pooled'}"] = m
    return models


def walk(start: dt.date, end: dt.date) -> pd.DataFrame:
    F.STORE = F.load.ROOT / "data/final_data/task3-v9-deploy-store"
    tables = {r: L.table(r, end=end) for r in ("morning", "afternoon")}
    days = sorted({d for t in tables.values() for d in t["date"] if start <= d <= end})
    out = []
    for _, week in pd.Series(days).groupby(lambda i: days[i].isocalendar()[:2]):
        P._MODELS = fit_all(tables, week.iloc[0])
        for rnd, t in tables.items():
            cols = sorted({c for k, m in P._MODELS.items() if k.startswith(rnd) for c in m["cols"]} - {"is_kbank"})
            for _, r in t[t["date"].isin(week)].iterrows():
                truth = r["P"] if rnd == "morning" else r["C"]
                if not np.isfinite(truth):
                    continue
                req = {"round": rnd, "symbol": r["stock"], "date": str(r["date"]),
                       "features": {c: (None if not np.isfinite(r[c]) else float(r[c])) for c in cols}}
                req.update({"C_prev": float(r["C_prev"]), "D": float(r["D"])} if rnd == "morning" else {"P": float(r["P"])})
                res = P.predict(req)
                out.append({"round": rnd, "stock": r["stock"], "date": r["date"], "y": float(truth), "model": res["model"],
                            "raw": res["prediction_raw"], "tick": res["prediction"], "base": res["anchor"]})
    P._MODELS = None
    return pd.DataFrame(out)


def confirm() -> pd.DataFrame:
    d = pd.read_parquet(CONFIRM)
    d = d[d["role"] == "chosen"]
    d = d.assign(y=np.where(d["round"] == "morning", d["P"], d["C"]),
                 base=np.where(d["round"] == "morning", d["anchor_m"], d["P"]), raw=d["hat"], model=d["spec"])
    d = d.dropna(subset=["y", "raw", "base"])
    return d.assign(tick=d["raw"].map(P.to_tick))[["round", "stock", "date", "y", "model", "raw", "tick", "base"]]


def score(d: pd.DataFrame) -> dict:
    out = {}
    for (rnd, sym), g in d.groupby(["round", "stock"]):
        base_tick = g["base"].map(P.to_tick)   # morning anchor C_prev - D can be off-grid on XD days
        row = {"days": len(g), "first": str(g["date"].min()), "last": str(g["date"].max()),
               "models_used": g["model"].value_counts().to_dict()}
        for name, p in (("raw", g["raw"]), ("tick", g["tick"]), ("no_change", base_tick)):
            e = p - g["y"]
            row[name] = {"rmse": float(np.sqrt((e ** 2).mean())), "mae": float(e.abs().mean()),
                         "exact_hit": float((e.abs() < 1e-9).mean())}
        for name in ("raw", "tick"):
            row[name]["rmse_vs_no_change_pct"] = 100 * (1 - row[name]["rmse"] / row["no_change"]["rmse"])
            row[name]["mae_vs_no_change_pct"] = 100 * (1 - row[name]["mae"] / row["no_change"]["mae"])
        out[f"{rnd}/{sym}"] = row
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", type=dt.date.fromisoformat, default=dt.date(2026, 9, 7))
    ap.add_argument("--to", dest="end", type=dt.date.fromisoformat, default=dt.date(2026, 10, 21))
    a = ap.parse_args()
    w = walk(a.start, a.end)
    res = {"walk_forward_new_months": score(w), "confirm_locked": score(confirm()),
           "note": "RMSE is the main metric, MAE second; prices in baht; tick = rounded to the SET price spread"}
    (HERE / "eval_metrics.json").write_text(json.dumps(res, indent=1, default=str))
    w.to_csv(HERE / "eval_walk_forward.csv", index=False)
    print(json.dumps(res, indent=1, default=str))
