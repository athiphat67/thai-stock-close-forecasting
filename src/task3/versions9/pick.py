"""Task 3 v9 M3: pick one model per round on the selection period (02_PROTOCOL §5 item 1) and the fallback κ (§3).

    .venv/bin/python -W ignore -m src.task3.versions9.pick morning artifacts/task3/v9/v9-morning-20261028-a

Rule: take the highest pooled skill on the days every (non-late) spec forecasts; a simpler rung (§3 table) that is
behind by less than 1 SE of the paired difference (block bootstrap, 5-day blocks, 10,000 draws, seed 340, same days
for both stocks) wins instead, the simplest such rung first. Fallback = κ · ŷ of the round's step-1 OLS, κ in
{0, 0.1, ..., 1} with the lowest pooled scaled MSE. Writes <run>/pick.json. Reads only the run's selection rows.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.task3.versions9.ladder import ROUND, block_bootstrap, specs

RUNG = {"ols": 1, "ridge": 2, "huber": 3, "hgb": 4, "mlp": 5, "cnn": 6, "ensemble": 7}
FALLBACK = {"morning": "ols_atc", "afternoon": "ols_small"}   # §3: morning = spec #2; afternoon's base OLS


def main(rnd: str, run: Path) -> dict:
    d = pd.read_parquet(run / "daily.parquet")
    sp = {s["name"]: s for s in specs(rnd) if s["n"] is not None}
    late = {k for k, s in sp.items() if s.get("late")}
    keys = d.groupby("spec").apply(lambda x: set(zip(x["date"], x["stock"])))
    common = set.intersection(*[keys[k] for k in sp if k not in late])
    on = d[[(a, b) in common for a, b in zip(d["date"], d["stock"])]].copy()
    # daily.parquet scales by each spec's own days: rescale with one mean truth² per stock so specs share a baseline
    _, anchor, truth = ROUND[rnd]
    m = on.assign(m=on.groupby("stock")[truth].transform(lambda s: (s ** 2).mean()))["m"]
    on["se"], on["se0"] = on["e"] ** 2 / m, on["e0"] ** 2 / m
    day = on.groupby(["spec", "date"])[["se", "se0"]].sum()
    se0 = day.xs(next(iter(sp)))["se0"].to_numpy()
    assert all(np.allclose(day.xs(k)["se0"].to_numpy(), se0) for k in sp if k not in late), "baselines differ"
    point = {k: 1 - day.xs(k)["se"].sum() / se0.sum() for k in sp if k not in late}
    boot = {k: block_bootstrap(day.xs(k)["se"].to_numpy(), se0) for k in point}
    best = max(point, key=point.get)
    gap = {k: {"behind": point[best] - point[k], "se_diff": float((boot[best] - boot[k]).std())} for k in point}
    within = [k for k in point if gap[k]["behind"] < gap[k]["se_diff"] or k == best]
    chosen = min(within, key=lambda k: (RUNG[sp[k]["kind"]], -point[k]))

    f = d[d["spec"] == FALLBACK[rnd]]
    m = f.groupby("stock")[truth].transform(lambda s: (s ** 2).mean())
    mse = {round(k, 1): float((((f[anchor] * np.exp(k * f["yhat"]) - f[truth]) ** 2) / m).sum())
           for k in np.arange(0, 1.01, 0.1)}
    kappa = min(mse, key=mse.get)

    out = {"round": rnd, "run": str(run), "common_days": len({c[0] for c in common}),
           "skill": point, "behind_best": gap, "within_1se": within, "best": best, "chosen": chosen,
           "late_specs_not_in_rule": sorted(late),
           "fallback": {"spec": FALLBACK[rnd], "kappa": kappa, "scaled_sse": mse}}
    (run / "pick.json").write_text(json.dumps(out, indent=1, default=str))
    print(f"{rnd}: best {best} {point[best]:+.2%} · within 1 SE {within} · chosen {chosen} {point[chosen]:+.2%}"
          f" · fallback {FALLBACK[rnd]} kappa {kappa}")
    for k in sorted(point, key=point.get, reverse=True):
        print(f"  {k:12s} rung {RUNG[sp[k]['kind']]} skill {point[k]:+.2%} behind {gap[k]['behind']:+.2%}"
              f" SE diff {gap[k]['se_diff']:.2%}")
    return out


if __name__ == "__main__":
    main(sys.argv[1], Path(sys.argv[2]))
