"""Task 3 v9 model ladder on the selection period 2019-2022 (02_PROTOCOL §1, §3, §4). M2 = morning, M3 = afternoon.

    OMP_NUM_THREADS=1 .venv/bin/python -W ignore -m src.task3.versions9.ladder morning [--smoke]

Every spec: log-return target, expanding window from 2017-04, weekly refit on rows dated before the week's first
day, features clipped at the window's 1%/99% (and standardised for ridge/Huber/MLP), rows with a missing feature
dropped, >= 120 training days before the first forecast. Hyperparameters are re-picked once a year (first scored
week of the year) on 3 consecutive 10% blocks at the end of the window, each trained on everything before it
(T1-T5); HGB/MLP then refit all year with the mean best iteration/epoch. Pooled models split blocks by date.
Writes artifacts/task3/v9/<run id>/: preds.parquet, daily.parquet (per day, per stock squared errors / mean P²
for paired SEs in M3), metrics.json, run_manifest.json. Never reads a row after 2022-12-31.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import time
from math import erf, sqrt

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.neural_network import MLPRegressor
import torch


from src.task3.versions9 import features as F

SELECT = (dt.date(2019, 1, 1), dt.date(2022, 12, 31))
TRAIN_FROM = dt.date(2017, 4, 1)
MIN_DAYS = 120
ALPHAS = list(10.0 ** np.arange(-2, 4.01, 0.5))          # 13 values, ridge and Huber
HGB_DEPTHS = [2, 3]
MLP_DECAYS = [1e-4, 1e-3, 1e-2]
MLP_SEEDS = range(340, 345)
CNN_DECAYS = [1e-4, 1e-3, 1e-2]
CNN_SEEDS = range(340, 343)
LATE = ["fut_post", "fut_am", "set50_am", "set_am", "sector_am", "basis_1230"]   # start 2019/2020, not 2017
SEQ: np.ndarray | None = None   # (rows, 2, 155) minute sequences, row = `seq_i` in the table (1D-CNN only)
FOREIGN = ["spx", "thd", "thb", "n225", "hsi"]
OUT = F.load.ROOT / "artifacts/task3/v9"
ROUND = {"morning": ("y_m", "anchor_m", "P"), "afternoon": ("y_a", "P", "C")}   # target, anchor, truth


def specs(rnd: str) -> list[dict]:
    """The ladder of one round, in run order. `n` = spec number in the EXPERIMENT_LOG count (None = not a spec)."""
    own = lambda s: F.feature_cols(rnd, s)  # noqa: E731
    pooled = F.feature_cols(rnd, None) + ["is_kbank"]
    if rnd == "morning":
        return [
            dict(name="mean", kind="mean", cols=lambda s: [], n=None),
            dict(name="ols_foreign", kind="ols", cols=lambda s: FOREIGN, n=1),
            dict(name="ols_atc", kind="ols", cols=lambda s: [*FOREIGN, "atc_push"], n=2),
            dict(name="ols_futures", kind="ols", cols=lambda s: [*FOREIGN, "atc_push", "fut_post"], n=4),
            dict(name="ridge", kind="ridge", cols=own, n=5),
            dict(name="huber", kind="huber", cols=own, n=6),
            dict(name="hgb", kind="hgb", cols=lambda s: pooled, pooled=True, n=7),
            dict(name="mlp", kind="mlp", cols=lambda s: pooled, pooled=True, n=8),
            dict(name="ensemble", kind="ensemble", parts=("ridge", "hgb"), n=9),
        ]
    early = lambda s: [c for c in own(s) if c not in LATE]  # noqa: E731
    pooled_early = [c for c in pooled if c not in LATE]
    peer = lambda s: ["peer_am"] if F.PEERS[s] else []  # noqa: E731
    return [
        dict(name="mean", kind="mean", cols=lambda s: [], n=None),
        dict(name="ols_small", kind="ols", cols=lambda s: ["r_morning", "r_last30", *peer(s)], n=1),
        dict(name="ols_range", kind="ols", cols=lambda s: ["r_morning", "r_last30", *peer(s), "r_x_range"], n=2),
        dict(name="ridge", kind="ridge", cols=early, n=3),
        dict(name="ridge_index", kind="ridge", cols=own, late=True, n=4),
        dict(name="huber", kind="huber", cols=early, n=5),
        dict(name="hgb", kind="hgb", cols=lambda s: pooled_early, pooled=True, n=6),
        dict(name="mlp", kind="mlp", cols=lambda s: pooled_early, pooled=True, n=7),
        dict(name="cnn", kind="cnn", cols=lambda s: [*pooled_early, "seq_i"], pooled=True, n=8),
        dict(name="ensemble", kind="ensemble", parts=("ridge", "hgb"), n=9),
    ]


class Net(torch.nn.Module):
    """Step 6: conv 2 x (8 channels, kernel 5) over the minutes -> mean over time -> with the tabular features -> linear."""

    def __init__(self, p: int):
        super().__init__()
        nn = torch.nn
        self.conv = nn.Sequential(nn.Conv1d(2, 8, 5), nn.ReLU(), nn.Conv1d(8, 8, 5), nn.ReLU())
        self.head = nn.Linear(8 + p, 1)

    def forward(self, s, x):
        return self.head(torch.cat([self.conv(s).mean(-1), x], 1)).squeeze(1)


def train_cnn(X, y, hp, n_iter=None, val=None):
    """1D-CNN, pooled. X's last column is the row in SEQ. AdamW lr 1e-3, batch 64, <= 100 epochs with patience 10
    on validation, 3 seeds averaged. Tabular features clipped + standardised and each sequence channel
    standardised on the training window; target fit as y / sd(y)."""
    tab = X[:, :-1]
    lo, hi = np.quantile(tab, .01, axis=0), np.quantile(tab, .99, axis=0)
    c = np.clip(tab, lo, hi)
    mu, sd = c.mean(0), c.std(0)
    sd[sd == 0] = 1.0
    s_tr = SEQ[X[:, -1].astype(int)]
    s_mu, s_sd = s_tr.mean(axis=(0, 2))[None, :, None], s_tr.std(axis=(0, 2))[None, :, None]

    def prep(Z):
        return (torch.tensor((SEQ[Z[:, -1].astype(int)] - s_mu) / s_sd, dtype=torch.float32),
                torch.tensor((np.clip(Z[:, :-1], lo, hi) - mu) / sd, dtype=torch.float32))

    ys = y.std() or 1.0
    S, T = prep(X)
    Y = torch.tensor(y / ys, dtype=torch.float32)
    Sv, Tv = prep(val[0]) if val else (None, None)
    nets, opts, gens = [], [], []
    for seed in CNN_SEEDS:
        torch.manual_seed(seed)
        net = Net(T.shape[1])
        nets.append(net)
        opts.append(torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=hp))
        gens.append(torch.Generator().manual_seed(seed))

    def predict(S_, T_):
        with torch.no_grad():
            return np.mean([net(S_, T_).numpy() for net in nets], axis=0) * ys

    curve = []
    for _ in range(100 if val else n_iter):
        for net, opt, gen in zip(nets, opts, gens):
            net.train()
            for b in torch.randperm(len(Y), generator=gen).split(64):
                opt.zero_grad()
                torch.nn.functional.mse_loss(net(S[b], T[b]), Y[b]).backward()
                opt.step()
            net.eval()
        if val:
            curve.append(float(((val[1] - predict(Sv, Tv)) ** 2).mean()))
            if len(curve) - 1 - int(np.argmin(curve)) >= 10:
                break
    if val:
        best_iter, v = best_with_patience(curve, 10)
        return None, best_iter, v
    return lambda Z: predict(*prep(Z))


# ---------------------------------------------------------------- models

def best_with_patience(curve: list[float], patience: int) -> tuple[int, float]:
    """(iterations, loss) at the minimum, scanning until `patience` steps pass without improvement."""
    best, k = np.inf, 0
    for i, v in enumerate(curve):
        if v < best:
            best, k = v, i
        elif i - k >= patience:
            break
    return k + 1, best


def train(kind: str, X, y, hp, n_iter=None, val=None):
    """Fit one model. With `val=(Xv, yv)`, returns (predict, best_iter, val_mse) for hyperparameter selection."""
    scale = kind in ("ridge", "huber", "mlp")
    if kind == "cnn":
        return train_cnn(X, y, hp, n_iter, val)
    if kind == "mean":
        m = float(y.mean())
        f = lambda Z: np.full(len(Z), m)  # noqa: E731
        return (f, None, float(((val[1] - m) ** 2).mean())) if val else f
    lo, hi = np.quantile(X, .01, axis=0), np.quantile(X, .99, axis=0)
    mu, sd = np.zeros(X.shape[1]), np.ones(X.shape[1])
    if scale:
        c = np.clip(X, lo, hi)
        mu, sd = c.mean(0), c.std(0)
        sd[sd == 0] = 1.0
    apply = lambda Z: (np.clip(Z, lo, hi) - mu) / sd  # noqa: E731   window's clip + scale, reused for any rows
    Xtr, Xv = apply(X), (apply(val[0]) if val else None)
    best_iter = None
    if kind == "ols":
        A = np.column_stack([np.ones(len(Xtr)), Xtr])
        beta = np.linalg.lstsq(A, y, rcond=None)[0]
        f = lambda Z: np.column_stack([np.ones(len(Z)), apply(Z)]) @ beta  # noqa: E731
    elif kind in ("ridge", "huber"):
        # Huber's data-term gradient is O(1) whatever the units of y, so its alpha only matches ridge's (whose
        # shrinkage ignores y's scale) when y has unit SD: fit both on y / sd(y) and scale the forecast back
        ys = y.std() or 1.0
        mdl = Ridge(alpha=hp) if kind == "ridge" else HuberRegressor(epsilon=1.35, alpha=hp, max_iter=2000)
        mdl.fit(Xtr, y / ys)
        f = lambda Z: mdl.predict(apply(Z)) * ys  # noqa: E731
    elif kind == "hgb":
        mdl = HistGradientBoostingRegressor(learning_rate=0.05, min_samples_leaf=50, l2_regularization=1.0,
                                            max_depth=hp, max_iter=500 if val else n_iter, early_stopping=False,
                                            random_state=340)
        mdl.fit(Xtr, y)
        if val:
            curve = [float(((val[1] - p) ** 2).mean()) for p in mdl.staged_predict(Xv)]
            best_iter, v = best_with_patience(curve, 20)
            return None, best_iter, v
        f = lambda Z: mdl.predict(apply(Z))  # noqa: E731
    elif kind == "mlp":
        ys = y.std() or 1.0          # Adam needs O(1) targets; predictions are scaled back
        nets = [MLPRegressor(hidden_layer_sizes=(16,), activation="relu", solver="adam", learning_rate_init=1e-3,
                             batch_size=64, alpha=hp, random_state=s) for s in MLP_SEEDS]
        curve = []
        for _ in range(200 if val else n_iter):
            for net in nets:
                net.partial_fit(Xtr, y / ys)
            if val:
                p = np.mean([net.predict(Xv) for net in nets], axis=0) * ys
                curve.append(float(((val[1] - p) ** 2).mean()))
                if len(curve) - 1 - int(np.argmin(curve)) >= 20:
                    break
        if val:
            best_iter, v = best_with_patience(curve, 20)
            return None, best_iter, v
        f = lambda Z: np.mean([net.predict(apply(Z)) for net in nets], axis=0) * ys  # noqa: E731
    else:
        raise ValueError(kind)
    if val:
        return f, best_iter, float(((val[1] - f(val[0])) ** 2).mean())
    return f


def grid(kind: str) -> list:
    return {"ridge": ALPHAS, "huber": ALPHAS, "hgb": HGB_DEPTHS, "mlp": MLP_DECAYS, "cnn": CNN_DECAYS}.get(kind, [None])


def select(kind: str, X, y, dates) -> tuple:
    """Yearly pick (T2-T3): mean validation MSE over 3 time-ordered 10% blocks at the end of the window."""
    g = grid(kind)
    if len(g) == 1 and kind not in ("hgb", "mlp", "cnn"):
        return g[0], None, None
    days = np.sort(np.unique(dates))
    n = len(days)
    bounds = [days[int(n * q)] for q in (0.7, 0.8, 0.9)] + [days[-1] + dt.timedelta(days=1)]
    res = {}
    for hp in g:
        errs, iters = [], []
        for a, b in zip(bounds[:-1], bounds[1:]):
            tr, va = dates < a, (dates >= a) & (dates < b)
            _, it, v = train(kind, X[tr], y[tr], hp, val=(X[va], y[va]))
            errs.append(v)
            iters.append(it)
        res[hp] = (float(np.mean(errs)), iters)
    hp = min(res, key=lambda k: res[k][0])
    it = res[hp][1]
    return hp, (int(round(np.mean(it))) if it[0] is not None else None), {str(k): v[0] for k, v in res.items()}


# ---------------------------------------------------------------- walk-forward

def table(rnd: str, end: dt.date = SELECT[1]) -> pd.DataFrame:
    t = pd.read_parquet(F.STORE / f"{rnd}.parquet")
    t = t[t["date"] <= end].copy()          # selection runs never read the confirm period; only confirm.py passes `end`
    t["anchor_m"] = t["C_prev"] - t["D"]
    t["is_kbank"] = (t["stock"] == "KBANK").astype(float)
    t["week"] = pd.to_datetime(t["date"]).dt.to_period("W-SUN")
    if rnd == "afternoon":
        global SEQ
        t["r_x_range"] = t["r_morning"] * t["range_am"]          # Gao et al.: the move carries on more on wide days
        z = np.load(F.STORE / "sequences.npz")
        SEQ = z["x"]
        pos = pd.Series(np.arange(len(SEQ)), index=pd.MultiIndex.from_arrays([z["stock"], z["date"]]))
        i = pos.reindex(pd.MultiIndex.from_arrays([t["stock"], t["date"].astype(str)])).to_numpy().astype(float)
        good = np.zeros(len(i), bool)
        good[~np.isnan(i)] = np.isfinite(SEQ[i[~np.isnan(i)].astype(int)]).all(axis=(1, 2))
        t["seq_i"] = np.where(good, i, np.nan)                     # NaN = no complete sequence, row dropped
    return t


def walk(spec: dict, t: pd.DataFrame, rnd: str, weeks: list, log: dict) -> pd.DataFrame:
    target = ROUND[rnd][0]
    groups = [(None, t)] if spec.get("pooled") else [(s, t[t["stock"] == s]) for s in F.STOCKS]
    out = []
    for stock, g in groups:
        cols = spec["cols"](stock or F.STOCKS[0])
        rows = g[(g["date"] >= TRAIN_FROM)].dropna(subset=[*cols, target])
        hp, n_iter, year, fit_key, f = None, None, None, None, None
        for wk in weeks:
            tr = rows[rows["week"] < wk]
            te = g[(g["week"] == wk)].dropna(subset=cols)
            if tr["date"].nunique() < MIN_DAYS or te.empty:
                continue
            y_year = te["date"].min().year
            X, y, d = tr[cols].to_numpy(float), tr[target].to_numpy(float), tr["date"].to_numpy()
            if y_year != year:
                hp, n_iter, scores = select(spec["kind"], X, y, d)
                year = y_year
                log.setdefault(spec["name"], {})[f"{stock or 'pooled'} {year}"] = {"hp": hp, "n_iter": n_iter,
                                                                                   "val_mse": scores}
            first = te["date"].min()
            key = (first.year, first.month) if spec["kind"] == "cnn" else wk   # 1D-CNN refits monthly (§1)
            if key != fit_key:
                f, fit_key = train(spec["kind"], X, y, hp, n_iter), key
            out.append(pd.DataFrame({"date": te["date"], "stock": te["stock"], "yhat": f(te[cols].to_numpy(float))}))
    out = out or [pd.DataFrame(columns=["date", "stock", "yhat"])]   # a late spec may not start inside a short window
    return pd.concat(out, ignore_index=True).assign(spec=spec["name"])


# ---------------------------------------------------------------- scoring

def nw_p(x: np.ndarray, lag: int = 5) -> float:
    """One-sided p of mean(x) > 0 with a Newey-West variance (02_PROTOCOL §4.3)."""
    n, u = len(x), x - x.mean()
    v = u @ u / n + 2 * sum((1 - k / (lag + 1)) * (u[k:] @ u[:-k]) / n for k in range(1, lag + 1))
    return 0.5 * (1 - erf(x.mean() / sqrt(v / n) / sqrt(2)))


def daily(p: pd.DataFrame, t: pd.DataFrame, rnd: str) -> pd.DataFrame:
    """Per (date, stock): squared errors of the model and the baseline, scaled by the stock's mean truth² (K11)."""
    _, anchor, truth = ROUND[rnd]
    x = p.merge(t[["date", "stock", anchor, truth]], on=["date", "stock"]).dropna(subset=[anchor, truth])
    x["hat"] = x[anchor] * np.exp(x["yhat"])
    m = x.groupby("stock")[truth].transform(lambda s: (s ** 2).mean())
    x["e"], x["e0"] = x["hat"] - x[truth], x[anchor] - x[truth]
    x["se"], x["se0"] = x["e"] ** 2 / m, x["e0"] ** 2 / m
    x["cw"] = (x["e0"] ** 2 - x["e"] ** 2 + (x[anchor] - x["hat"]) ** 2) / m
    return x


def block_bootstrap(se: np.ndarray, se0: np.ndarray, reps=10_000, block=5, seed=340) -> np.ndarray:
    """Pooled skill on resampled days (rows are days, both stocks summed), moving blocks of 5."""
    n = len(se)
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n - block + 1, size=(reps, -(-n // block)))
    idx = (starts[:, :, None] + np.arange(block)).reshape(reps, -1)[:, :n]
    return 1 - se[idx].sum(1) / se0[idx].sum(1)


def metrics(x: pd.DataFrame, truth: str) -> dict:
    by_day = x.groupby("date")[["se", "se0", "cw"]].sum()
    gain = by_day["se0"] - by_day["se"]
    keep = gain < gain.quantile(0.99)
    boot = block_bootstrap(by_day["se"].to_numpy(), by_day["se0"].to_numpy())
    out = {"days": len(by_day), "skill": 1 - by_day["se"].sum() / by_day["se0"].sum(),
           "skill_se": float(boot.std()), "skill_ci95": [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))],
           "cw_p": nw_p(by_day["cw"].to_numpy()),
           "skill_trim1": 1 - by_day.loc[keep, "se"].sum() / by_day.loc[keep, "se0"].sum(),
           "skill_by_year": {str(y): 1 - g["se"].sum() / g["se0"].sum()
                             for y, g in by_day.groupby(pd.to_datetime(by_day.index).year)}}
    for s, g in x.groupby("stock"):
        mv, mt = g["hat"] - g[ROUND_ANCHOR], g[truth] - g[ROUND_ANCHOR]
        out[s] = {"days": len(g), "rmse": float(np.sqrt((g["e"] ** 2).mean())), "rmse0": float(np.sqrt((g["e0"] ** 2).mean())),
                  "skill": 1 - (g["e"] ** 2).mean() / (g["e0"] ** 2).mean(),
                  "mae": float(g["e"].abs().mean()), "mae0": float(g["e0"].abs().mean()),
                  "mz_slope": float(np.polyfit(mv, mt, 1)[0]) if mv.std() > 0 else None}
    return out


ROUND_ANCHOR = None


def main() -> None:
    global ROUND_ANCHOR
    ap = argparse.ArgumentParser()
    ap.add_argument("round", choices=["morning", "afternoon"])
    ap.add_argument("--smoke", action="store_true", help="Jan-Feb 2019 only, forecast counts only, no scores")
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    rnd = a.round
    ROUND_ANCHOR = ROUND[rnd][1]
    t = table(rnd)
    hi = dt.date(2019, 2, 28) if a.smoke else SELECT[1]     # M2's smoke run scored 2022; M3's only checks it runs
    weeks = sorted(set(t.loc[(t["date"] >= SELECT[0]) & (t["date"] <= hi), "week"]))
    preds, hps, secs = {}, {}, {}
    for spec in specs(rnd):
        if a.only and spec["name"] not in a.only and spec["kind"] != "ensemble":
            continue
        t0 = time.time()
        if spec["kind"] == "ensemble":
            if not all(p in preds for p in spec["parts"]):
                continue
            j = preds[spec["parts"][0]].merge(preds[spec["parts"][1]], on=["date", "stock"])
            preds[spec["name"]] = pd.DataFrame({"date": j["date"], "stock": j["stock"], "spec": spec["name"],
                                                "yhat": (j["yhat_x"] + j["yhat_y"]) / 2})
        else:
            preds[spec["name"]] = walk(spec, t, rnd, weeks, hps)
        secs[spec["name"]] = round(time.time() - t0, 1)
        print(f"{spec['name']}: {len(preds[spec['name']])} forecasts in {secs[spec['name']]} s", flush=True)

    if a.smoke:
        return
    truth = ROUND[rnd][2]
    d = {k: daily(p, t, rnd) for k, p in preds.items()}
    late = {s["name"] for s in specs(rnd) if s.get("late")}
    keys = {k: set(zip(x["date"], x["stock"])) for k, x in d.items()}
    common = set.intersection(*[v for k, v in keys.items() if k not in late])
    on = lambda x, c: x[[(a_, b_) in c for a_, b_ in zip(x["date"], x["stock"])]]  # noqa: E731
    res = {}
    for k, x in d.items():   # a late spec is compared with ridge on its own (shorter) days, §1
        res[k] = {"common_days": metrics(on(x, common & keys[k]), truth), "own_days": metrics(x, truth)}
        if k in late and "ridge" in d:
            res[f"ridge@{k}"] = {"common_days": metrics(on(d["ridge"], common & keys[k]), truth)}
    n = {s["name"]: s["n"] for s in specs(rnd)}
    n.update({k: None for k in res if k.startswith("ridge@")})
    print(f"\ncommon days: {len({c[0] for c in common})} ({min(common)[0]} -> {max(common)[0]})")
    print("| spec | # | skill (pooled) | SE | CW p | KBANK | ADVANC | trim 1% | " +
          " | ".join(str(y) for y in res[next(iter(res))]["common_days"]["skill_by_year"]) + " |")
    for k, r in res.items():
        c = r["common_days"]
        print(f"| {k} | {n[k] or '—'} | {c['skill']:+.2%} | {c['skill_se']:.2%} | {c['cw_p']:.4f} | "
              f"{c['KBANK']['skill']:+.2%} | {c['ADVANC']['skill']:+.2%} | {c['skill_trim1']:+.2%} | " +
              " | ".join(f"{v:+.2%}" for v in c["skill_by_year"].values()) + " |")
    run = f"v9-{rnd}-{dt.date.today():%Y%m%d}-a"
    out = OUT / run
    while out.exists():
        run = run[:-1] + chr(ord(run[-1]) + 1)
        out = OUT / run
    out.mkdir(parents=True)
    pd.concat(preds.values()).to_parquet(out / "preds.parquet", index=False)
    pd.concat(x.assign(spec=k) for k, x in d.items()).to_parquet(out / "daily.parquet", index=False)
    (out / "metrics.json").write_text(json.dumps({"hyperparameters": hps, "results": res}, indent=1, default=str))
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=F.load.ROOT).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "src/task3/versions9"], capture_output=True, text=True,
                           cwd=F.load.ROOT).stdout.strip()
    store = json.loads((F.STORE / "manifest.json").read_text())
    (out / "run_manifest.json").write_text(json.dumps({
        "run_id": run, "git_commit": commit, "uncommitted_changes": dirty, "store_commit": store["git_commit"],
        "store_files": store["files"], "select": [str(x) for x in SELECT], "train_from": str(TRAIN_FROM),
        "seeds": {"mlp": list(MLP_SEEDS), "hgb": 340, "bootstrap": 340}, "seconds": secs,
        "finished": dt.datetime.now().isoformat(timespec="seconds")}, indent=1))
    print(f"\nwrote {out.relative_to(F.load.ROOT)}")


if __name__ == "__main__":
    main()
