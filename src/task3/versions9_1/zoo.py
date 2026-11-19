"""Task 3 v9.1 model zoo runner (tickets/T00 §4). Same walk-forward as v9, one registry of every spec.

    OMP_NUM_THREADS=1 .venv/bin/python -W ignore -m src.task3.versions9_1.zoo <round> --group A|B|C|E|R [--only ID ...]
        [--smoke] [--to YYYY-MM-DD] [--pc]

Expanding window from 2017-04, scored weeks 2019-01-01 -> 2026-09-04 in one pass (an expanding window makes this the
same as running the selection and confirm periods apart), training on rows dated before each week, >= 120 training
days. Hyperparameters re-picked once a year on 3 time-ordered 10% blocks (v9 T1-T5). `walk` and `select` are
ladder.walk / ladder.select with three changes only: train/grid come from MODELS, the refit key follows the spec's
cadence, and which models pick an iteration count comes from MODELS (`iterative`). R0-R4 rerun v9's own models and
must reproduce v9 bit for bit (tests/test_task3_v9_1.py).
--pc plants y_c = y + b*x (T00 §7, rho² 2%) with x a fake feature column, scored on 2021-2022 only.
Writes artifacts/task3/v9_1/<run id>/: preds, daily, metrics.json, hp.json, diag.json, run_manifest.json.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
import time
from importlib import metadata

import numpy as np
import pandas as pd
import torch

from src.task3.versions9 import features as F
from src.task3.versions9 import ladder as L
from src.task3.versions9_1 import inputs as I
from src.task3.versions9_1 import models_cnn, models_fm, models_seq, models_tab
from src.task3.versions9_1 import score as S

OUT = F.load.ROOT / "artifacts/task3/v9_1"
START = dt.date(2019, 1, 1)
PC_RHO2, PC_SEED, PC_SPAN = 0.02, 340, (dt.date(2021, 1, 1), dt.date(2022, 12, 31))
PICK = {"morning": "huber", "afternoon": "hgb"}            # v9-prereg picks (R3)
TABPFN_ROWS = {"morning": None, "afternoon": None}   # T01 §2.1: idle timing 2026-10-30 = 11-14 s / 24-56 s <= 60 s, full window


def _v9(name: str, rnd: str) -> dict:
    """A v9 ladder spec, run through v9's own train()."""
    s = {x["name"]: x for x in L.specs(rnd)}[name]
    kind = s["kind"]
    return dict(group="R", cols=s["cols"], pooled=s.get("pooled", False), refit="month" if kind == "cnn" else "week",
                grid=L.grid(kind), iterative=kind in ("hgb", "mlp", "cnn"),
                train=lambda X, y, hp, n_iter=None, val=None, k=kind: L.train(k, X, y, hp, n_iter, val))


def _zero(X, y, hp, n_iter=None, val=None):
    f = lambda Z: np.zeros(len(Z))  # noqa: E731
    return (f, None, float((val[1] ** 2).mean())) if val else f


def _todo(X, y, hp, n_iter=None, val=None):
    raise NotImplementedError("model not written yet: see its ticket")


def _meta(group, cols, inp, refit, grid, iterative=False, rounds=("morning", "afternoon")):
    return dict(group=group, cols=cols, input=inp, refit=refit, grid=grid, iterative=iterative, rounds=rounds,
                pooled=cols == "pooled", train=_todo)


WD = [1e-4, 1e-3, 1e-2]
# T01-T04 lock every value below; a train function replaces _todo when its ticket is written
MODELS = {
    "A01": _meta("A", "own", "tab", "week", [0.01, 0.03, 0.1]),
    "A02": _meta("A", "own", "tab", "week", [0.01, 0.03, 0.1]),
    "A03": _meta("A", "own", "tab", "week", [1, 2, 3]),
    "A04": _meta("A", "own", "tab", "week", [1, 3, 5]),
    "A05": _meta("A", "own", "tab", "week", [0.1, 1, 10]),
    "A06": _meta("A", "own", "tab", "week", [20, 50, 100]),
    "A07": _meta("A", "pooled", "tab", "week", [50, 100, 200]),
    "A08": _meta("A", "pooled", "tab", "week", [50, 100, 200]),
    "A09": _meta("A", "pooled", "tab", "week", [2, 3, 4], iterative=True),
    "A10": _meta("A", "pooled", "tab", "week", [4, 8, 16], iterative=True),
    "A11": _meta("A", "pooled", "tab", "week", [2, 3, 4], iterative=True),
    "A12": _meta("A", "pooled", "tab", "month", WD, iterative=True),
    "A13": _meta("A", "pooled", "tab", "month", WD, iterative=True),
    "A14": {**_meta("A", "pooled", "tab", "week", [None]), "pre": ["day_i"]},
    # pre: the row pointer the model reads its y / window / minute sequence through (models_seq.bind)
    "B01": {**_meta("B", "own", "tab", "week", [(1, 0), (0, 1), (1, 1)]), "pre": ["row_i"]},
    **{f"B{i:02d}": {**_meta("B", "pooled", "win", "month", WD, iterative=True), "pre": ["win_i"]} for i in range(2, 12)},
    "B02m": {**_meta("B", "pooled", "min", "month", WD, iterative=True, rounds=("afternoon",)), "pre": ["seq_i"]},
    "B06m": {**_meta("B", "pooled", "min", "month", WD, iterative=True, rounds=("afternoon",)), "pre": ["seq_i"]},
    **{f"C0{i}": {**_meta("C", "none", "fm", "none", [None]), "pre": ["row_i"]} for i in range(1, 5)},
    # T04 §1.3: g sees only its own input and is_kbank (models_cnn.bind sets e_i / img_i)
    # r2_ok: residual specs use R2's rows only (models_cnn.bind)
    **{e: {**_meta("E", "none", "win|min", "month", WD, iterative=True), "pre": ["e_i", "row_i", "is_kbank", "r2_ok"]}
       for e in ("E01", "E02", "E03", "E04", "E05")},
    "E02p": {**_meta("E", "none", "win|min", "month", WD, iterative=True), "pre": ["e_i", "row_i", "is_kbank"]},
    "E06": {**_meta("E", "none", "img", "year", [None], iterative=True), "pre": ["img_i", "row_i", "is_kbank", "r2_ok"]},
    "E06p": {**_meta("E", "none", "img", "year", [None], iterative=True), "pre": ["img_i", "row_i", "is_kbank"]},
    "E07": {**_meta("E", "none", "min", "month", WD, iterative=True, rounds=("afternoon",)),
            "pre": ["seq_i", "row_i", "is_kbank", "r2_ok"]},
}
for _k in ("E01", "E02", "E03", "E04", "E05", "E02p", "E06", "E06p", "E07"):
    MODELS[_k]["pooled"] = True       # T04 §1.3: one g for both stocks, is_kbank at the head (ridge stays per stock)
for _k, _f in {**models_tab.TRAIN, **models_seq.TRAIN, **models_fm.TRAIN, **models_cnn.TRAIN}.items():
    MODELS[_k]["train"] = _f


def registry(rnd: str) -> dict:
    """Specs of one round: R0-R4 (v9 references, not counted) + the zoo."""
    r = {"R0": dict(group="R", cols=lambda s: [], pooled=False, refit="week", grid=[None], iterative=False, train=_zero),
         "R1": _v9("mean", rnd), "R2": _v9("ridge", rnd), "R3": _v9(PICK[rnd], rnd)}
    if rnd == "afternoon":
        r["R4"] = _v9("cnn", rnd)
    for k, m in MODELS.items():
        if rnd in m["rounds"]:
            c, pre = m["cols"], m.get("pre", [])
            r[k] = {**m, "cols": (lambda s, c=c, pre=pre: [*pre, *I.cols(rnd, s, c)]) if c in ("own", "pooled")
                    else (lambda s, pre=pre: [*pre])}
    if "A14" in r:
        r["A14"]["train"] = models_tab.tabpfn(TABPFN_ROWS[rnd])
    return r


def select(m: dict, X, y, dates) -> tuple:
    """ladder.select with the grid, the iteration flag and train() taken from the registry."""
    g = m["grid"]
    if len(g) == 1 and not m["iterative"]:
        return g[0], None, None
    days = np.sort(np.unique(dates))
    n = len(days)
    bounds = [days[int(n * q)] for q in (0.7, 0.8, 0.9)] + [days[-1] + dt.timedelta(days=1)]
    res = {}
    for hp in g:
        errs, iters = [], []
        for a, b in zip(bounds[:-1], bounds[1:]):
            tr, va = dates < a, (dates >= a) & (dates < b)
            _, it, v = m["train"](X[tr], y[tr], hp, val=(X[va], y[va]))
            errs.append(v)
            iters.append(it)
        res[hp] = (float(np.mean(errs)), iters)
    hp = min(res, key=lambda k: res[k][0])
    it = res[hp][1]
    return hp, (int(round(np.mean(it))) if it[0] is not None else None), {str(k): v[0] for k, v in res.items()}


def refit_key(cadence: str, first: dt.date, wk):
    return {"week": wk, "month": (first.year, first.month), "year": first.year, "none": wk}[cadence]


def walk(name: str, m: dict, t: pd.DataFrame, rnd: str, weeks: list, log: dict) -> pd.DataFrame:
    """ladder.walk with the registry's train/select and the spec's refit cadence."""
    target = L.ROUND[rnd][0]
    groups = [(None, t)] if m.get("pooled") else [(s, t[t["stock"] == s]) for s in F.STOCKS]
    out = []
    for stock, g in groups:
        cols = m["cols"](stock or F.STOCKS[0])
        rows = g[(g["date"] >= L.TRAIN_FROM)].dropna(subset=[*cols, target])
        hp, n_iter, year, fit_key, f = None, None, None, None, None
        for wk in weeks:
            tr = rows[rows["week"] < wk]
            te = g[(g["week"] == wk)].dropna(subset=cols)
            if tr["date"].nunique() < L.MIN_DAYS or te.empty:
                continue
            y_year = te["date"].min().year
            models_cnn.CTX["year_now"] = y_year      # E residual specs refit ridge with R2's alpha of this year
            X, y, d = tr[cols].to_numpy(float), tr[target].to_numpy(float), tr["date"].to_numpy()
            if y_year != year:
                hp, n_iter, scores = select(m, X, y, d)
                year = y_year
                log.setdefault(name, {})[f"{stock or 'pooled'} {year}"] = {"hp": hp, "n_iter": n_iter, "val_mse": scores}
            key = refit_key(m["refit"], te["date"].min(), wk)
            if key != fit_key:
                f, fit_key = m["train"](X, y, hp, n_iter), key
            out.append(pd.DataFrame({"date": te["date"], "stock": te["stock"], "yhat": f(te[cols].to_numpy(float))}))
    out = out or [pd.DataFrame(columns=["date", "stock", "yhat"])]
    return pd.concat(out, ignore_index=True).assign(spec=name)


def plant(t: pd.DataFrame, rnd: str) -> pd.DataFrame:
    """T00 §7: y_c = y + b*x, x ~ N(0,1) as a new feature column `pc_x`; truth moved to match y_c."""
    target, anchor, truth = L.ROUND[rnd]
    t = t.copy()
    x = np.random.default_rng(PC_SEED).standard_normal(len(t))
    b = np.nanstd(t[target]) * np.sqrt(PC_RHO2 / (1 - PC_RHO2))
    t["pc_x"] = x
    t[target] = t[target] + b * x
    t[truth] = t[anchor] * np.exp(t[target])
    return t


def plant_ar(t: pd.DataFrame, rnd: str) -> pd.DataFrame:
    """T03 §5 (group C sees no features): y_c(d) = y(d) + phi*y_c(d-1) per stock, phi = sqrt(rho²) so the predictable
    part is ~2% of the variance; a missing day restarts the chain. `pc_dy` = y_c - y moves C04's candles."""
    target, anchor, truth = L.ROUND[rnd]
    t = t.copy()
    y = t[target].to_numpy(float)
    yc = y.copy()
    phi = np.sqrt(PC_RHO2)
    for _, idx in t.groupby("stock").indices.items():
        prev = 0.0
        for i in np.sort(idx):
            if np.isfinite(y[i]):
                yc[i] = y[i] + phi * prev
                prev = yc[i]
            else:
                prev = 0.0
    t["pc_dy"] = np.nan_to_num(yc - y)
    t[target] = yc
    t[truth] = t[anchor] * np.exp(t[target])
    return t


def _git(*a) -> str:
    return subprocess.run(["git", *a], capture_output=True, text=True, cwd=F.load.ROOT).stdout.strip()


def plant_z(t: pd.DataFrame, rnd: str) -> pd.DataFrame:
    """T04 §6: y_c = y + b*z with z seen by the CNN inputs but not by ridge's columns, standardised (missing -> 0).
    Morning: the XD-adjusted 5-day return to d-1 (sum of y_m + y_a over the 5 previous rows, all in W and the chart).
    Afternoon: d's 10:00 -> 10:30 return (minutes 10:01-10:30 of S, also in the 5-minute chart)."""
    target, anchor, truth = L.ROUND[rnd]
    t = t.copy()
    if rnd == "morning":
        z = (t["y_m_"] + t["y_a_"]).groupby(t["stock"]).transform(lambda v: v.shift(1).rolling(5).sum()).to_numpy()
    else:
        S, i = I.minute_seq(), t["seq_i"].to_numpy()
        z = np.full(len(t), np.nan)
        ok = np.isfinite(i)
        z[ok] = S[i[ok].astype(int), 0, 5:35].sum(1)
    z = np.nan_to_num((z - np.nanmean(z)) / np.nanstd(z))
    t[target] = t[target] + np.nanstd(t[target]) * np.sqrt(PC_RHO2 / (1 - PC_RHO2)) * z
    t[truth] = t[anchor] * np.exp(t[target])
    return t


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("round", choices=["morning", "afternoon"])
    ap.add_argument("--group", required=True, choices=["A", "B", "C", "E", "R"])
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--smoke", action="store_true", help="Jul-Aug 2019, forecast counts only, nothing written")
    ap.add_argument("--to", type=dt.date.fromisoformat, default=I.END)
    ap.add_argument("--pc", action="store_true", help="positive control, scored 2021-2022")
    a = ap.parse_args(argv)
    rnd = a.round
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    t = I.table(rnd)
    lo, hi = (PC_SPAN if a.pc else (START, a.to))
    if a.smoke:                    # July-August 2019: the first weeks where both rounds' v9 specs forecast
        lo, hi = dt.date(2019, 7, 1), dt.date(2019, 8, 31)
    if a.pc:
        t = {"C": plant_ar, "E": plant_z}.get(a.group, plant)(t, rnd)
    t = models_seq.bind(t, rnd)
    weeks = sorted(set(t.loc[(t["date"] >= lo) & (t["date"] <= hi), "week"]))
    reg = registry(rnd)
    names = [k for k, m in reg.items() if m["group"] == a.group and (not a.only or k in a.only)]
    if not names or (a.only and len(names) != len(set(a.only))):
        ap.error(f"--only {a.only} does not match group {a.group} in the {rnd} registry")
    preds, hps, secs = {}, {}, {}
    if a.group == "E":                 # residual specs add R2's own walk-forward forecast (T04 §1.1)
        preds["R2"] = walk("R2", reg["R2"], t, rnd, weeks, hps)
        t = models_cnn.bind(t, rnd, preds["R2"], hps, reg["R2"]["cols"], names)
    if a.group == "C":                 # no training: forecast every row once, the walk looks them up
        secs = {f"{k} prepare": v for k, v in models_fm.prepare(t, rnd, names, lo, hi).items()}
        print(secs, flush=True)
    for k in names:
        m = reg[k]
        if a.pc and a.group not in ("C", "E"):
            base = m["cols"]
            m = {**m, "cols": lambda s, base=base: [*base(s), "pc_x"]}
        t0 = time.time()
        preds[k] = walk(k, m, t, rnd, weeks, hps)
        secs[k] = round(time.time() - t0, 1)
        print(f"{k}: {len(preds[k])} forecasts in {secs[k]} s", flush=True)
    if a.smoke:
        return {"counts": {k: len(p) for k, p in preds.items()}}

    res, daily, diag = S.score_round(preds, t, rnd)
    diag["train"] = models_seq.diag()
    if a.group == "E":                 # T04 §1.6: size of the CNN part, sd(yhat - yhat_R2) / sd(y), on scored rows
        y = t.set_index(["date", "stock"])[L.ROUND[rnd][0]]
        r2 = preds["R2"].set_index(["date", "stock"])["yhat"]
        for k in names:
            p = preds[k].set_index(["date", "stock"])["yhat"]
            on = p.index.intersection(r2.index)
            diag["train"].setdefault(k, {})["cnn_part_sd_ratio"] = float((p[on] - r2[on]).std() / y[on].std())
    tag = "pc" if a.pc else a.group
    run = f"v91-{rnd}-{tag}-{dt.date.today():%Y%m%d}-a"
    out = OUT / run
    while out.exists():
        run = run[:-1] + chr(ord(run[-1]) + 1)
        out = OUT / run
    out.mkdir(parents=True)
    pd.concat(preds.values()).to_parquet(out / "preds.parquet", index=False)
    daily.to_parquet(out / "daily.parquet", index=False)
    (out / "metrics.json").write_text(json.dumps(res, indent=1, default=str))
    (out / "hp.json").write_text(json.dumps(hps, indent=1, default=str))
    (out / "diag.json").write_text(json.dumps(diag, indent=1, default=str))
    pk = ["numpy", "pandas", "scikit-learn", "torch", "xgboost", "lightgbm", "catboost", "statsmodels", "xlstm", "chronos-forecasting", "timesfm", "transformers"]
    (out / "run_manifest.json").write_text(json.dumps({
        "run_id": run, "argv": sys.argv[1:] if argv is None else argv, "git_commit": _git("rev-parse", "HEAD"),
        "uncommitted_changes": _git("status", "--porcelain", "src/task3/versions9_1", "src/task3/versions9"),
        "store_commit": json.loads((F.STORE / "manifest.json").read_text())["git_commit"],
        "scored": [str(lo), str(hi)], "train_from": str(L.TRAIN_FROM), "seconds": secs,
        "packages": {p: metadata.version(p) for p in pk}, "python": sys.version.split()[0],
        **({"weights": models_fm.WEIGHTS, "uni2ts": models_fm.UNI2TS, "device": "cpu"} if a.group == "C" else {}),
        **({"device": {"E06": models_cnn.JKX["device"], "other": "cpu"}} if a.group == "E" else {}),
        "finished": dt.datetime.now().isoformat(timespec="seconds")}, indent=1))
    S.print_table(res)
    print(f"\nwrote {out.relative_to(F.load.ROOT)}")
    return {"run": out, "preds": preds}


if __name__ == "__main__":
    main()
