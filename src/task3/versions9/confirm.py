"""Task 3 v9 M4: run the picked model and the fallback of each round on the confirm period once (02_PROTOCOL §5).

    OMP_NUM_THREADS=1 .venv/bin/python -W ignore -m src.task3.versions9.confirm --dry   # 2019-2022, must reproduce M2/M3
    OMP_NUM_THREADS=1 .venv/bin/python -W ignore -m src.task3.versions9.confirm         # 2023-01 -> 2026-09-04, ONCE

Same walk-forward code as the selection runs (ladder.walk): weekly refit, yearly hyperparameter pick, training on
everything before each week. Pass (§5 item 3, all three, per round):
  1. pooled one-sided Clark-West vs the baseline passes Holm over the two rounds at alpha 0.05;
  2. pooled skill > 0 in >= 2 of the 3 periods, and each stock's skill over the whole period >= -0.1%;
  3. pooled skill stays > 0 after dropping the 1% of days with the largest gain.
A round that fails uses its fallback κ·ŷ (§5 item 4); nothing is re-picked. Numbers on the confirm period are
labelled "seen before" (v5-v8 looked at these prices for a different target, §1).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess

import numpy as np
import pandas as pd

from src.task3.versions9 import features as F
from src.task3.versions9 import ladder as L

D = dt.date
RUNS = {"morning": L.OUT / "v9-morning-20261028-a", "afternoon": L.OUT / "v9-afternoon-20261028-a"}
CONFIRM = {"window": (D(2023, 1, 1), D(2026, 9, 4)),
           "periods": {"2023": (D(2023, 1, 1), D(2023, 12, 31)), "2024": (D(2024, 1, 1), D(2024, 12, 31)),
                       "2025-01..2026-09-04": (D(2025, 1, 1), D(2026, 9, 4))}}
DRY = {"window": L.SELECT,
       "periods": {"2019-2020": (D(2019, 1, 1), D(2020, 12, 31)), "2021": (D(2021, 1, 1), D(2021, 12, 31)),
                   "2022": (D(2022, 1, 1), D(2022, 12, 31))}}
SPLITS = {"seen_by_v8 (2025-08-25+)": (D(2025, 8, 25), D(2026, 9, 4)),
          "before 2024-03-25": (D(2000, 1, 1), D(2024, 3, 24)), "from 2024-03-25": (D(2024, 3, 25), D(2100, 1, 1))}
SECOND = {"morning": "C_prev", "afternoon": "anchor_m"}     # §4.1 secondary baseline


def run_round(rnd: str, cfg: dict) -> dict:
    pick = json.loads((RUNS[rnd] / "pick.json").read_text())
    sp = {s["name"]: s for s in L.specs(rnd)}
    t = L.table(rnd, end=cfg["window"][1])
    weeks = sorted(set(t.loc[(t["date"] >= cfg["window"][0]) & (t["date"] <= cfg["window"][1]), "week"]))
    hps = {}
    out = {}
    for role, name, kappa in (("chosen", pick["chosen"], 1.0), ("fallback", pick["fallback"]["spec"], pick["fallback"]["kappa"])):
        p = L.walk(sp[name], t, rnd, weeks, hps)
        out[role] = {"spec": name, "kappa": kappa, "preds": p.assign(yhat=kappa * p["yhat"])}
    out["hyperparameters"] = hps
    out["table"] = t
    return out


def score(p: pd.DataFrame, t: pd.DataFrame, rnd: str, periods: dict) -> tuple[dict, pd.DataFrame]:
    _, anchor, truth = L.ROUND[rnd]
    x = p.merge(t[["date", "stock", anchor, truth, SECOND[rnd], "D"]], on=["date", "stock"]).dropna(subset=[anchor, truth])
    x["hat"] = x[anchor] * np.exp(x["yhat"])
    m = x.groupby("stock")[truth].transform(lambda s: (s ** 2).mean())     # one scaling per stock for the whole window
    x["e"], x["e0"], x["e2"] = x["hat"] - x[truth], x[anchor] - x[truth], x[SECOND[rnd]] - x[truth]
    x["se"], x["se0"], x["se2"] = x["e"] ** 2 / m, x["e0"] ** 2 / m, x["e2"] ** 2 / m
    x["cw"] = (x["e0"] ** 2 - x["e"] ** 2 + (x[anchor] - x["hat"]) ** 2) / m
    day = x.groupby("date")[["se", "se0", "se2", "cw"]].sum()
    sk = lambda g: float(1 - g["se"].sum() / g["se0"].sum()) if len(g) else None  # noqa: E731
    dd = pd.Series(day.index, index=day.index)
    gain = day["se0"] - day["se"]
    boot = L.block_bootstrap(day["se"].to_numpy(), day["se0"].to_numpy())
    r = {"days": len(day), "first": str(day.index.min()), "last": str(day.index.max()), "skill": sk(day),
         "skill_se": float(boot.std()), "skill_ci95": [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))],
         "cw_p": L.nw_p(day["cw"].to_numpy()),
         "skill_trim1": sk(day[gain < gain.quantile(0.99)]),
         "skill_vs_second_baseline": float(1 - day["se"].sum() / day["se2"].sum()),
         "periods": {k: sk(day[(dd >= a) & (dd <= b)]) for k, (a, b) in periods.items()},
         "splits": {k: sk(day[(dd >= a) & (dd <= b)]) for k, (a, b) in SPLITS.items()},
         "skill_by_year": {str(y): sk(g) for y, g in day.groupby(pd.to_datetime(day.index).year)},
         "xd_days": sk(x[x["D"] > 0]) if (x["D"] > 0).any() else None}
    for s, g in x.groupby("stock"):
        mv, mt = g["hat"] - g[anchor], g[truth] - g[anchor]
        r[s] = {"days": len(g), "skill": float(1 - (g["e"] ** 2).mean() / (g["e0"] ** 2).mean()),
                "rmse": float(np.sqrt((g["e"] ** 2).mean())), "rmse0": float(np.sqrt((g["e0"] ** 2).mean())),
                "mae": float(g["e"].abs().mean()), "mae0": float(g["e0"].abs().mean()),
                "mz_slope": float(np.polyfit(mv, mt, 1)[0]) if mv.std() > 0 else None}
    return r, x


def judge(res: dict) -> dict:
    """§5 item 3 for both rounds; Holm over the family {morning, afternoon} at 0.05."""
    order = sorted(res, key=lambda k: res[k]["cw_p"])
    holm, still = {}, True
    for i, k in enumerate(order):
        still = still and res[k]["cw_p"] <= 0.05 / (len(order) - i)
        holm[k] = still
    out = {}
    for k, r in res.items():
        per = [v for v in r["periods"].values() if v is not None]
        c = {"1_cw_holm": holm[k], "2a_periods_positive": f"{sum(v > 0 for v in per)}/{len(per)}",
             "2b_each_stock_ge_-0.1%": all(r[s]["skill"] >= -0.001 for s in F.STOCKS),
             "3_trim1_positive": r["skill_trim1"] > 0}
        c["pass"] = c["1_cw_holm"] and sum(v > 0 for v in per) >= 2 and c["2b_each_stock_ge_-0.1%"] and c["3_trim1_positive"]
        out[k] = c
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true", help="2019-2022: must reproduce the selection runs exactly")
    a = ap.parse_args()
    cfg = DRY if a.dry else CONFIRM
    runs = {rnd: run_round(rnd, cfg) for rnd in ("morning", "afternoon")}
    res, fb, daily = {}, {}, []
    for rnd, r in runs.items():
        res[rnd], x = score(r["chosen"]["preds"], r["table"], rnd, cfg["periods"])
        fb[rnd], xf = score(r["fallback"]["preds"], r["table"], rnd, cfg["periods"])
        daily += [x.assign(round=rnd, role="chosen", spec=r["chosen"]["spec"]),
                  xf.assign(round=rnd, role="fallback", spec=r["fallback"]["spec"])]
        if a.dry:   # same code path as M2/M3: forecasts must match the selection runs to the last bit
            old = pd.read_parquet(RUNS[rnd] / "preds.parquet")
            for role in ("chosen", "fallback"):
                new = r[role]["preds"]
                o = old[old["spec"] == r[role]["spec"]].merge(new, on=["date", "stock"], suffixes=("_old", ""))
                diff = float(np.max(np.abs(o["yhat_old"] * r[role]["kappa"] - o["yhat"])))
                assert len(o) == len(new) == (old["spec"] == r[role]["spec"]).sum() and diff == 0, (rnd, role, diff)
                print(f"dry {rnd} {role} {r[role]['spec']}: {len(o)} forecasts identical to the selection run")
    verdict = judge(res)
    for rnd in res:
        r, f = res[rnd], fb[rnd]
        print(f"\n{rnd} chosen {runs[rnd]['chosen']['spec']}: days {r['days']} skill {r['skill']:+.2%} (SE {r['skill_se']:.2%})"
              f" CW p {r['cw_p']:.4f} KBANK {r['KBANK']['skill']:+.2%} ADVANC {r['ADVANC']['skill']:+.2%}"
              f" trim1 {r['skill_trim1']:+.2%} periods " + ", ".join(f"{k} {v:+.2%}" for k, v in r["periods"].items()))
        print(f"  splits " + ", ".join(f"{k} {v:+.2%}" for k, v in r["splits"].items() if v is not None)
              + f" · vs secondary baseline {r['skill_vs_second_baseline']:+.2%}")
        print(f"  verdict {verdict[rnd]}")
        print(f"  fallback {runs[rnd]['fallback']['spec']} x{runs[rnd]['fallback']['kappa']}: skill {f['skill']:+.2%}"
              f" CW p {f['cw_p']:.4f} periods " + ", ".join(f"{k} {v:+.2%}" for k, v in f["periods"].items()))
    if a.dry:
        return
    run = f"v9-confirm-{dt.date.today():%Y%m%d}-a"
    out = L.OUT / run
    assert not out.exists(), f"{out} exists: the confirm period is opened once (§5 item 2)"
    out.mkdir(parents=True)
    pd.concat(daily).to_parquet(out / "daily.parquet", index=False)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=F.load.ROOT).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "src/task3/versions9"], capture_output=True, text=True,
                           cwd=F.load.ROOT).stdout.strip()
    (out / "metrics.json").write_text(json.dumps({
        "run_id": run, "git_commit": commit, "uncommitted_changes": dirty, "label": "seen before (v5-v8, other target)",
        "chosen": res, "fallback": fb, "verdict": verdict,
        "hyperparameters": {rnd: r["hyperparameters"] for rnd, r in runs.items()},
        "finished": dt.datetime.now().isoformat(timespec="seconds")}, indent=1, default=str))
    print(f"\nwrote {out.relative_to(F.load.ROOT)}")


if __name__ == "__main__":
    main()
