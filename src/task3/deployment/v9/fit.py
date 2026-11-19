"""Fit the four v9 deploy models with the locked v9 code and save them as one bundle.

    OMP_NUM_THREADS=1 .venv/bin/python -W ignore -m src.task3.deployment.v9.fit [--train-end YYYY-MM-DD] [--store DIR]

Reads a v9 feature store: --store, default the deploy store build_data.py writes (data/final_data/task3-v9-deploy-store). Same rules as ladder.train / ladder.walk:
rows from 2017-04-01, rows with a missing feature or target dropped, clip at the window's 1%/99%, Huber and OLS
per stock, HGB pooled. Hyperparameters are the 2026 picks of the confirm run (v9-confirm-20261028-a), which
v9 keeps for the whole year. Writes bundle/models.joblib + bundle/manifest.json and checks every model against
ladder.train on the training rows (must match to 1e-12).
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import HuberRegressor

from src.task3.versions9 import features as F
from src.task3.versions9 import ladder as L

HERE = Path(__file__).resolve().parent
BUNDLE = HERE / "bundle"
HP_2026 = {"huber": {"KBANK": 1000.0, "ADVANC": 100.0}, "hgb": {"max_depth": 2, "max_iter": 28}}
KAPPA = {"morning": 0.7, "afternoon": 0.5}
# v9.2 (2026-11-11): the morning primary is v9's Huber without fut_post (S50 futures) and peer_prev (bank ATC), the two
# inputs the live feed cannot get. Same kind, alphas and features otherwise; walk-forward skill 2023-01 -> 2026-08-28
# +6.49% vs the locked Huber's +6.46% on the same days (REPORT_V92.md).
DEPLOY_SPECS = {("morning", "huber_lite"): dict(name="huber_lite", kind="huber", cols=lambda s: [
    c for c in F.feature_cols("morning", s) if c not in ("fut_post", "peer_prev")])}


def spec_of(rnd: str, name: str) -> dict:
    return DEPLOY_SPECS.get((rnd, name)) or next(s for s in L.specs(rnd) if s["name"] == name)


# (round, model name, kind, pooled)
MODELS = [("morning", "huber_lite", "huber", False), ("morning", "ols_atc", "ols", False),
          ("afternoon", "hgb", "hgb", True), ("afternoon", "ols_small", "ols", False)]


def fit_one(kind: str, X: np.ndarray, y: np.ndarray, hp) -> dict:
    """Same arithmetic as ladder.train, but keeps the pieces so the model can be saved."""
    lo, hi = np.quantile(X, .01, axis=0), np.quantile(X, .99, axis=0)
    mu, sd, ys = np.zeros(X.shape[1]), np.ones(X.shape[1]), 1.0
    if kind == "huber":
        c = np.clip(X, lo, hi)
        mu, sd = c.mean(0), c.std(0)
        sd[sd == 0] = 1.0
    Z = (np.clip(X, lo, hi) - mu) / sd
    m = {"kind": kind, "lo": lo, "hi": hi, "mu": mu, "sd": sd}
    if kind == "ols":
        m["beta"] = np.linalg.lstsq(np.column_stack([np.ones(len(Z)), Z]), y, rcond=None)[0]
    elif kind == "huber":
        ys = y.std() or 1.0
        m["est"] = HuberRegressor(epsilon=1.35, alpha=hp, max_iter=2000).fit(Z, y / ys)
    else:
        m["est"] = HistGradientBoostingRegressor(learning_rate=0.05, min_samples_leaf=50, l2_regularization=1.0,
                                                 max_depth=hp["max_depth"], max_iter=hp["max_iter"],
                                                 early_stopping=False, random_state=340).fit(Z, y)
    m["ys"] = ys
    return m


def predict_one(m: dict, X: np.ndarray) -> np.ndarray:
    Z = (np.clip(X, m["lo"], m["hi"]) - m["mu"]) / m["sd"]
    if m["kind"] == "ols":
        return np.column_stack([np.ones(len(Z)), Z]) @ m["beta"]
    return m["est"].predict(Z) * m["ys"]


def main(train_end: dt.date | None, store: Path | None = None) -> dict:
    F.STORE = store or F.load.ROOT / "data/final_data/task3-v9-deploy-store"
    tables = {r: L.table(r, end=train_end or dt.date(2100, 1, 1)) for r in ("morning", "afternoon")}
    models, rows_info = {}, {}
    for rnd, name, kind, pooled in MODELS:
        t, target = tables[rnd], L.ROUND[rnd][0]
        spec = spec_of(rnd, name)
        for stock in ([None] if pooled else list(F.STOCKS)):
            cols = spec["cols"](stock or F.STOCKS[0])
            g = t if pooled else t[t["stock"] == stock]
            tr = g[g["date"] >= L.TRAIN_FROM].dropna(subset=[*cols, target])
            X, y = tr[cols].to_numpy(float), tr[target].to_numpy(float)
            hp = HP_2026["hgb"] if kind == "hgb" else HP_2026["huber"][stock] if kind == "huber" else None
            m = fit_one(kind, X, y, hp)
            ref = L.train(kind, X, y, hp["max_depth"] if kind == "hgb" else hp,
                          hp["max_iter"] if kind == "hgb" else None)
            assert np.allclose(predict_one(m, X), ref(X), rtol=0, atol=1e-12), f"{name} {stock} differs from ladder"
            m["cols"] = cols
            key = f"{rnd}/{name}/{stock or 'pooled'}"
            models[key] = m
            rows_info[key] = {"rows": len(tr), "first": str(tr["date"].min()), "last": str(tr["date"].max()),
                              "features": cols, "hp": hp}
    BUNDLE.mkdir(exist_ok=True)
    joblib.dump(models, BUNDLE / "models.joblib")
    store = F.STORE
    manifest = {
        "bundle": "task3-v9-deploy",
        "fitted_at": dt.datetime.now(dt.timezone(dt.timedelta(hours=7))).isoformat(timespec="seconds"),
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
        "locked_tag": "v9-prereg (8021a49); v9.2 morning primary huber_lite (no fut_post, no peer_prev)",
        "train_end": str(max(v["last"] for v in rows_info.values())),
        "hyperparameters": "2026 picks of the confirm run v9-confirm-20261028-a",
        "kappa": KAPPA,
        "models": rows_info,
        "feature_store": str(store.relative_to(F.load.ROOT)),
        "feature_store_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in sorted(store.glob("*.parquet"))},
        "models_joblib_sha256": hashlib.sha256((BUNDLE / "models.joblib").read_bytes()).hexdigest(),
        "python": sys.version.split()[0],
        "versions": {"scikit-learn": sklearn.__version__, "numpy": np.__version__, "joblib": joblib.__version__},
        "checked_against_ladder_train": True,
    }
    (BUNDLE / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str))
    if BUNDLE == HERE / "bundle":
        write_examples(tables)
    return manifest


def write_examples(tables: dict, days: int = 5) -> None:
    """Last `days` store days per round and stock as requests + expected outputs, for parity tests on the live side.
    These rows are inside the training window (in-sample): they test the plumbing, not the skill."""
    from src.task3.deployment.v9 import predict as P
    P._MODELS = None
    out = []
    for rnd in ("morning", "afternoon"):
        t = tables[rnd]
        cols = sorted({c for k, m in joblib.load(BUNDLE / "models.joblib").items() if k.startswith(rnd) for c in m["cols"]}
                      - {"is_kbank"}, key=lambda c: F.AFTERNOON.index(c) if c in F.AFTERNOON else 99)
        for _, r in t[t["date"].isin(sorted(t["date"].unique())[-days:])].sort_values(["date", "stock"]).iterrows():
            feats = {c: (None if not np.isfinite(r[c]) else float(r[c])) for c in cols}
            req = {"round": rnd, "symbol": r["stock"], "date": str(r["date"]), "features": feats}
            req.update({"C_prev": float(r["C_prev"]), "D": float(r["D"])} if rnd == "morning" else {"P": float(r["P"])})
            got = P.predict(req)
            truth = float(r["P"] if rnd == "morning" else r["C"])
            out.append({"request": req, "expected": {"model": got["model"], "yhat": got["yhat"], "prediction": got["prediction_raw"]},
                        "actual_price": truth})
    (HERE / "examples.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-end", type=dt.date.fromisoformat, default=None, help="last training date (default: all)")
    ap.add_argument("--store", type=Path, default=None, help="feature store dir (default: the deploy store)")
    a = ap.parse_args()
    print(json.dumps(main(a.train_end, a.store), indent=1, default=str))
