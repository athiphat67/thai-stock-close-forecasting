"""Task 3 v9.1 scoring of one run (tickets/T00 §5 items 1-5). Cross-run tables, MCS and the multiple-trial haircut
are T05's (they need every group's preds on one set of common days).

Each window (selection 2019-2022, confirm 2023 -> 2026-09-04) is scored by v9's confirm.score on the days every spec
of the run forecasts, so all specs share one baseline. The paired difference with R3 uses ladder.block_bootstrap,
whose fixed seed draws the same day blocks for every spec (pick.py's method).
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from src.task3.versions9 import confirm as C
from src.task3.versions9 import ladder as L

D = dt.date
WINDOWS = {"select": (D(2019, 1, 1), D(2022, 12, 31), C.DRY["periods"]),
           "confirm": (D(2023, 1, 1), D(2026, 9, 4), C.CONFIRM["periods"])}


def _keys(p: pd.DataFrame, t: pd.DataFrame, rnd: str) -> set:
    _, anchor, truth = L.ROUND[rnd]
    x = p.dropna(subset=["yhat"]).merge(t[["date", "stock", anchor, truth]], on=["date", "stock"])
    x = x.dropna(subset=[anchor, truth])
    return set(zip(x["date"], x["stock"]))


def collapse(x: pd.DataFrame, rnd: str) -> dict:
    """sd(yhat) / sd(y) per year (T00 §5 item 4): below 0.05 the model forecasts its training mean."""
    _, anchor, truth = L.ROUND[rnd]
    y = np.log(x[truth] / x[anchor])
    yr = pd.to_datetime(x["date"]).dt.year
    return {str(k): float(x.loc[i, "yhat"].std() / y.loc[i].std()) for k, i in x.groupby(yr).groups.items()}


def score_round(preds: dict, t: pd.DataFrame, rnd: str) -> tuple[dict, pd.DataFrame, dict]:
    keys = {k: _keys(p, t, rnd) for k, p in preds.items()}
    common = set.intersection(*keys.values())
    res, daily, diag = {}, [], {}
    for w, (a, b, periods) in WINDOWS.items():
        on = {(d_, s) for d_, s in common if a <= d_ <= b}
        if not on:
            continue
        res[w], day = {}, {}
        for k, p in preds.items():
            q = p[[(d_, s) in on for d_, s in zip(p["date"], p["stock"])]]
            r, x = C.score(q, t, rnd, periods)
            r["own_days"] = len({kk for kk in keys[k] if a <= kk[0] <= b})
            res[w][k] = r
            day[k] = x.groupby("date")[["se", "se0"]].sum()
            daily.append(x.assign(spec=k, window=w))
            diag.setdefault(k, {})[f"sd_ratio_{w}"] = collapse(x, rnd)
        if "R3" in day:
            ref = L.block_bootstrap(day["R3"]["se"].to_numpy(), day["R3"]["se0"].to_numpy())
            for k in day:
                boot = L.block_bootstrap(day[k]["se"].to_numpy(), day[k]["se0"].to_numpy())
                res[w][k]["vs_R3"] = {"diff": res[w][k]["skill"] - res[w]["R3"]["skill"], "se": float((boot - ref).std())}
    res["common_days"] = {w: len({d_ for d_, s in common if a <= d_ <= b}) for w, (a, b, _) in WINDOWS.items()}
    return res, pd.concat(daily, ignore_index=True), diag


def pc_gain(pc_run, main_run, span=(D(2021, 1, 1), D(2022, 12, 31)), need=0.01) -> dict:
    """T00 §7: skill on the planted target minus skill on the real one, per spec, on the 2021-2022 days both runs
    scored. Each skill is against its own target's baseline. Pass = gain >= 1% (half the 2% planted)."""
    a, b = (pd.read_parquet(r / "daily.parquet") for r in (pc_run, main_run))
    out = {}
    for k in sorted(set(a["spec"]) & set(b["spec"])):
        x, y = a[a["spec"] == k], b[(b["spec"] == k) & (b["window"] == "select")]
        keys = set(zip(x["date"], x["stock"])) & {kk for kk in zip(y["date"], y["stock"]) if span[0] <= kk[0] <= span[1]}
        sk = lambda z: 1 - z["se"].sum() / z["se0"].sum()  # noqa: E731
        on = lambda z: z[[kk in keys for kk in zip(z["date"], z["stock"])]]  # noqa: E731
        g = float(sk(on(x)) - sk(on(y)))
        out[k] = {"days": len({kk[0] for kk in keys}), "gain": g, "pass": g >= need}
    return out


def _f(v, fmt: str) -> str:
    return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else format(v, fmt)


def print_table(res: dict) -> None:
    for w, r in res.items():
        if w == "common_days":
            continue
        print(f"\n{w} · common days {res['common_days'][w]}")
        print("| spec | skill | SE | vs R3 (SE) | CW p | KBANK | ADVANC | trim 1% | periods |")
        for k, m in r.items():
            v = m.get("vs_R3")
            print(f"| {k} | {_f(m['skill'], '+.2%')} | {_f(m['skill_se'], '.2%')} | "
                  + (f"{_f(v['diff'], '+.2%')} ({_f(v['se'], '.2%')})" if v else "—")
                  + f" | {_f(m['cw_p'], '.4f')} | {_f(m['KBANK']['skill'], '+.2%')} | {_f(m['ADVANC']['skill'], '+.2%')} | "
                  f"{_f(m['skill_trim1'], '+.2%')} | "
                  + ", ".join(f"{p} {_f(s, '+.2%')}" for p, s in m["periods"].items()) + " |")
