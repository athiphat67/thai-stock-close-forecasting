"""Task 3 v9.1 group A: tabular models A01-A14 (tickets/T01).

Every train() follows ladder.train: train(X, y, hp, n_iter=None, val=None) returns predict(Z), or with
val=(Xv, yv) returns (predict or None, best_iter, val_mse) for the yearly pick. Clip at the training window's
1%/99% for all; standardise X for A01-A06 and A12-A13; fit on y/sd(y) for A01-A06 and A12-A13 (trees use raw y).
"""

from __future__ import annotations

import os

import numpy as np
import torch
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVR
from xgboost import XGBRegressor

from src.task3.versions9.ladder import best_with_patience

SEEDS = range(340, 345)
MAX_ROUNDS, TREE_PATIENCE = 500, 20
MAX_EPOCHS, NN_PATIENCE = 200, 20


def prep(X, scale: bool):
    """The training window's clip (and standardisation), reused for any rows."""
    lo, hi = np.quantile(X, .01, axis=0), np.quantile(X, .99, axis=0)
    mu, sd = np.zeros(X.shape[1]), np.ones(X.shape[1])
    if scale:
        c = np.clip(X, lo, hi)
        mu, sd = c.mean(0), c.std(0)
        sd[sd == 0] = 1.0
    return lambda Z: (np.clip(Z, lo, hi) - mu) / sd


def _done(f, val):
    return (f, None, float(((val[1] - f(val[0])) ** 2).mean())) if val else f


def sk_model(make):
    """A01-A06: an sklearn regressor on standardised X and y / sd(y)."""
    def train(X, y, hp, n_iter=None, val=None):
        apply = prep(X, True)
        ys = y.std() or 1.0
        mdl = make(hp, *X.shape[::-1]).fit(apply(X), y / ys)
        return _done(lambda Z: np.ravel(mdl.predict(apply(Z))) * ys, val)
    return train


def forest(cls, bootstrap: bool):
    """A07-A08: 500 trees, a third of the features per split, leaves of >= hp rows; clip only, raw y."""
    def train(X, y, hp, n_iter=None, val=None):
        apply = prep(X, False)
        mdl = cls(n_estimators=500, max_features=0.33, min_samples_leaf=hp, bootstrap=bootstrap,
                  random_state=340, n_jobs=1).fit(apply(X), y)
        return _done(lambda Z: mdl.predict(apply(Z)), val)
    return train


def booster(kind: str):
    """A09-A11: lr 0.05, <= 500 rounds. With val, the validation MSE after every round (from the library's own
    eval log) gives the best round with patience 20, as v9's HGB; the refit then uses the mean best round."""
    def make(hp, rounds):
        if kind == "xgb":
            return XGBRegressor(learning_rate=0.05, n_estimators=rounds, max_depth=hp, min_child_weight=50,
                                reg_lambda=1.0, subsample=0.8, colsample_bytree=0.8, tree_method="hist", nthread=1,
                                random_state=340, eval_metric="rmse")
        if kind == "lgbm":
            return LGBMRegressor(learning_rate=0.05, n_estimators=rounds, num_leaves=hp, min_child_samples=50,
                                 reg_lambda=1.0, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, num_threads=1,
                                 deterministic=True, force_row_wise=True, random_state=340, verbose=-1)
        return CatBoostRegressor(learning_rate=0.05, iterations=rounds, depth=hp, l2_leaf_reg=3.0, thread_count=1,
                                 random_seed=340, verbose=0, allow_writing_files=False, use_best_model=False)

    def train(X, y, hp, n_iter=None, val=None):
        apply = prep(X, False)
        mdl = make(hp, MAX_ROUNDS if val else n_iter)
        if not val:
            mdl.fit(apply(X), y)
            return lambda Z: mdl.predict(apply(Z))
        ev = [(apply(val[0]), val[1])]
        if kind == "xgb":
            mdl.fit(apply(X), y, eval_set=ev, verbose=False)
            curve = np.square(mdl.evals_result()["validation_0"]["rmse"])
        elif kind == "lgbm":
            mdl.fit(apply(X), y, eval_set=ev, eval_metric="l2")
            curve = np.asarray(mdl.evals_result_["valid_0"]["l2"])
        else:
            mdl.fit(apply(X), y, eval_set=ev)
            curve = np.square(mdl.get_evals_result()["validation"]["RMSE"])
        best_iter, v = best_with_patience(list(map(float, curve)), TREE_PATIENCE)
        return None, best_iter, v
    return train


class NN(torch.nn.Module):
    """GKX feed-forward net: ReLU hidden layers with batch norm, Kaiming init, linear output."""

    def __init__(self, p: int, widths: tuple):
        super().__init__()
        layers = []
        for w in widths:
            lin = torch.nn.Linear(p, w)
            torch.nn.init.kaiming_normal_(lin.weight, nonlinearity="relu")
            torch.nn.init.zeros_(lin.bias)
            layers += [lin, torch.nn.BatchNorm1d(w), torch.nn.ReLU()]
            p = w
        self.net = torch.nn.Sequential(*layers, torch.nn.Linear(p, 1))

    def forward(self, x):
        return self.net(x).squeeze(1)


def nn(widths: tuple):
    """A12-A13: Adam lr 1e-3 with weight decay hp, batch 64, <= 200 epochs, patience 20 on the 5-seed mean."""
    def train(X, y, hp, n_iter=None, val=None):
        apply = prep(X, True)
        ys = y.std() or 1.0
        T, Y = torch.tensor(apply(X), dtype=torch.float32), torch.tensor(y / ys, dtype=torch.float32)
        nets, opts, gens = [], [], []
        for s in SEEDS:
            torch.manual_seed(s)
            nets.append(NN(T.shape[1], widths))
            opts.append(torch.optim.Adam(nets[-1].parameters(), lr=1e-3, weight_decay=hp))
            gens.append(torch.Generator().manual_seed(s))

        def predict(Z):
            Zt = torch.tensor(apply(Z), dtype=torch.float32)
            with torch.no_grad():
                return np.mean([net.eval()(Zt).numpy() for net in nets], axis=0) * ys

        curve = []
        for _ in range(MAX_EPOCHS if val else n_iter):
            for net, opt, gen in zip(nets, opts, gens):
                net.train()
                for b in torch.randperm(len(Y), generator=gen).split(64):
                    if len(b) < 2:            # batch norm needs two rows
                        continue
                    opt.zero_grad()
                    torch.nn.functional.mse_loss(net(T[b]), Y[b]).backward()
                    opt.step()
            if val:
                curve.append(float(((val[1] - predict(val[0])) ** 2).mean()))
                if len(curve) - 1 - int(np.argmin(curve)) >= NN_PATIENCE:
                    break
        if val:
            best_iter, v = best_with_patience(curve, NN_PATIENCE)
            return None, best_iter, v
        return predict
    return train


def _mps(fn):
    """Run fn with torch's determinism check off: MPS kernels have no deterministic versions (T01 §2.1 measures
    the run-to-run difference instead)."""
    prev = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(False)
    try:
        return fn()
    finally:
        torch.use_deterministic_algorithms(prev)


def tabpfn(max_rows: int | None):
    """A14: TabPFN v2 weights (Prior Labs License), 8 estimators, mean forecast, MPS. No training: each refit puts
    the window in as context. X's first column is `day_i` (date ordinal), used only to keep the latest `max_rows`
    rows when T01 §2.1's timing rule caps the context; it is not a feature."""
    os.environ.setdefault("TABPFN_NO_BROWSER", "1")
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion

    def train(X, y, hp, n_iter=None, val=None):
        day, X = X[:, 0], X[:, 1:]
        if max_rows and len(X) > max_rows:
            keep = day >= np.sort(day)[-max_rows]          # whole days: both stocks of the boundary day stay
            X, y = X[keep], y[keep]
        apply = prep(X, False)
        ys = y.std() or 1.0
        mdl = TabPFNRegressor.create_default_for_version(ModelVersion.V2, device="mps", random_state=340,
                                                         n_estimators=8)
        _mps(lambda: mdl.fit(apply(X), y / ys))
        f = lambda Z: _mps(lambda: mdl.predict(apply(Z[:, 1:]), output_type="mean")) * ys  # noqa: E731
        return _done(f, val)
    return train


TRAIN = {
    "A01": sk_model(lambda hp, p, n: Lasso(alpha=hp, max_iter=10000, tol=1e-6, selection="cyclic")),
    "A02": sk_model(lambda hp, p, n: ElasticNet(alpha=hp, l1_ratio=0.5, max_iter=10000, tol=1e-6)),
    "A03": sk_model(lambda hp, p, n: PLSRegression(n_components=min(hp, p), scale=False)),
    "A04": sk_model(lambda hp, p, n: make_pipeline(PCA(n_components=min(hp, p), svd_solver="full"), LinearRegression())),
    "A05": sk_model(lambda hp, p, n: SVR(kernel="rbf", C=hp, epsilon=0.1, gamma="scale", cache_size=500)),
    # k <= rows: the first morning selection blocks (2019) train on ~86 rows
    "A06": sk_model(lambda hp, p, n: KNeighborsRegressor(n_neighbors=min(hp, n), weights="uniform")),
    "A07": forest(RandomForestRegressor, True),
    "A08": forest(ExtraTreesRegressor, False),
    "A09": booster("xgb"),
    "A10": booster("lgbm"),
    "A11": booster("cat"),
    "A12": nn((32, 16, 8)),
    "A13": nn((32, 16, 8, 4, 2)),
}
