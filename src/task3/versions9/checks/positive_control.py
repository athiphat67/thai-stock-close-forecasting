"""Task 3 v9 positive control: how large a signal can the morning pipeline (ridge + Clark-West) see?

Design (04_REFERENCES §8.1 item 1, not a spec): plant a fake feature x ~ N(0,1) (seeded, independent of
everything) into the real morning target, y_c = y + b*x with b = sd(y) * sqrt(rho2 / (1 - rho2)), so x explains
about rho2 of the planted target's variance. Then run the protocol's ridge (02_PROTOCOL §3: standardise and clip
on the training window, alpha in 10^{-2..4} picked on the last 20% of the window by time, refit, weekly refit)
twice: with and without x, on spec #2's columns (5 foreign series + atc_push). Detection = pooled one-sided
Clark-West p of "with x" over "without x" below 0.05 (and 0.025, the first Holm step).
x carries no information about y, so this spends nothing of the selection period: it only measures the design.

    .venv/bin/python src/task3/versions9/checks/positive_control.py            # full run, ~minutes
    .venv/bin/python src/task3/versions9/checks/positive_control.py --check    # self-check only
"""
import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import atc_push_test as apt  # noqa: E402

ALPHAS = 10.0 ** np.arange(-2, 4.01, 0.5)          # 02_PROTOCOL §3, 13 values
RHO2 = [0.0, 0.001, 0.0025, 0.005, 0.01, 0.02, 0.05]
SEEDS = range(340, 360)
TRAIN_FROM, SCORE_FROM, END = dt.date(2017, 1, 1), dt.date(2019, 1, 1), dt.date(2022, 12, 31)
BASE = [*apt.FOREIGN, "atc_push"]


def ridge_fit(X: np.ndarray, y: np.ndarray, alpha: float):
    """Ridge with an unpenalised intercept on already standardised X (sklearn's summed loss)."""
    xm, ym = X.mean(0), y.mean()
    Xc = X - xm
    b = np.linalg.solve(Xc.T @ Xc + alpha * np.eye(X.shape[1]), Xc.T @ (y - ym))
    return ym - xm @ b, b


def ridge_week(tr_X: np.ndarray, tr_y: np.ndarray, te_X: np.ndarray):
    """Protocol ridge for one weekly fit: clip + standardise on the window, pick alpha on its last 20%."""
    lo, hi = np.quantile(tr_X, .01, axis=0), np.quantile(tr_X, .99, axis=0)
    tr_X, te_X = np.clip(tr_X, lo, hi), np.clip(te_X, lo, hi)
    mu, sd = tr_X.mean(0), tr_X.std(0)
    sd[sd == 0] = 1.0
    tr_X, te_X = (tr_X - mu) / sd, (te_X - mu) / sd
    cut = int(len(tr_y) * 0.8)
    err = []
    for al in ALPHAS:
        a, b = ridge_fit(tr_X[:cut], tr_y[:cut], al)
        err.append(((tr_y[cut:] - a - tr_X[cut:] @ b) ** 2).mean())
    k = int(np.argmin(err))
    a, b = ridge_fit(tr_X, tr_y, ALPHAS[k])
    return a + te_X @ b, k


def walk(g: pd.DataFrame, cols: list, target: str):
    """Weekly expanding refit; returns predictions and the index of the alpha picked each week."""
    d = pd.to_datetime(pd.Index(g.index))
    week = d.to_period("W-SUN")
    rows = g[g.index >= TRAIN_FROM]
    rd = pd.to_datetime(pd.Index(rows.index))
    preds, picks = {}, []
    for wk in sorted(set(week[(d >= pd.Timestamp(SCORE_FROM)) & (g.index <= END)])):
        tr = rows[rd < wk.start_time]
        te = g[week == wk]
        if len(tr) < len(cols) + 10 or te.empty:
            continue
        p, k = ridge_week(tr[cols].to_numpy(), tr[target].to_numpy(), te[cols].to_numpy())
        preds.update(zip(te.index, p))
        picks.append(k)
    return pd.Series(preds), np.array(picks)


def one(tabs: dict, rho2: float, seed: int):
    """Pooled CW p and per-stock skill gain of 'with x' over 'without x' for one planted signal."""
    rng = np.random.default_rng(seed)
    cw, gains, edge = [], {}, []
    for sym, g in tabs.items():
        g = g.copy()
        g["x"] = rng.standard_normal(len(g))
        b = g["morning"].std() * np.sqrt(rho2 / (1 - rho2))
        g["y_c"] = g["morning"] + b * g["x"]
        anchor = g["C_prev"] - g["D"]
        g["P_c"] = anchor * np.exp(g["y_c"])
        pb, _ = walk(g, BASE, "y_c")
        pm, picks = walk(g, [*BASE, "x"], "y_c")
        idx = pb.index.intersection(pm.index)
        t = g.loc[idx]
        hb, hm = anchor[idx] * np.exp(pb[idx]), anchor[idx] * np.exp(pm[idx])
        eb, em = hb - t["P_c"], hm - t["P_c"]
        gains[sym] = 1 - (em ** 2).mean() / (eb ** 2).mean()
        edge.append(float(np.mean(picks == len(ALPHAS) - 1)))
        cw.append(((eb ** 2 - em ** 2 + (hb - hm) ** 2) / (t["P_c"] ** 2).mean()).rename(sym))
    pooled = pd.concat(cw, axis=1).sum(axis=1, min_count=1).dropna()
    return apt.nw_p(pooled.to_numpy()), gains, float(np.mean(edge))


def check() -> None:
    rng = np.random.default_rng(0)
    X, y = rng.standard_normal((500, 4)), rng.standard_normal(500)
    a, b = ridge_fit(X, y, 1e-10)                       # alpha -> 0 is OLS with intercept
    ols = np.linalg.lstsq(np.column_stack([np.ones(500), X]), y, rcond=None)[0]
    assert np.allclose([a, *b], ols, atol=1e-8)
    assert np.abs(ridge_fit(X, y, 1e12)[1]).max() < 1e-8  # huge alpha shrinks to zero
    y2 = y + 0.5 * X[:, 0]                              # strong planted signal is found, alpha stays low
    p, k = ridge_week(X[:400], y2[:400], X[400:])
    assert np.corrcoef(p, X[400:, 0])[0, 1] > 0.9 and k < len(ALPHAS) - 1
    assert apt.nw_p(np.full(100, 1.0) + rng.standard_normal(100) * 0.1) < 1e-6
    print("check ok")


def main() -> None:
    check()
    tabs = {s: apt.table(s).dropna(subset=[*BASE, "morning"]) for s in ("KBANK", "ADVANC")}
    for s, g in tabs.items():
        sc = g[(g.index >= SCORE_FROM) & (g.index <= END)]
        print(f"{s}: rows {len(g)} · scored days {len(sc)} ({sc.index.min()} -> {sc.index.max()})")
    rows = []
    for rho2 in RHO2:
        for seed in SEEDS:
            p, gains, edge = one(tabs, rho2, seed)
            rows.append({"rho2": rho2, "seed": seed, "p": p, **{f"gain_{k}": v for k, v in gains.items()},
                         "alpha_at_top": edge})
        r = pd.DataFrame([x for x in rows if x["rho2"] == rho2])
        print(f"rho2 {rho2:.2%}: detect p<0.05 {np.mean(r.p < .05):.0%} · p<0.025 {np.mean(r.p < .025):.0%}"
              f" · median p {r.p.median():.3f} · skill gain KBANK {r.gain_KBANK.mean():+.3%} ADVANC {r.gain_ADVANC.mean():+.3%}"
              f" · weeks alpha at top {r.alpha_at_top.mean():.0%}", flush=True)
    out = Path(__file__).resolve().parents[4] / "artifacts/task3/v9/positive_control"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "positive_control.csv", index=False)
    print(f"saved {out / 'positive_control.csv'}")


if __name__ == "__main__":
    check() if "--check" in sys.argv else main()
