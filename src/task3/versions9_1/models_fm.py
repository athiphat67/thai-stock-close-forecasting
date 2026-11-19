"""Task 3 v9.1 group C: zero-shot foundation models C01-C04 (tickets/T03).

Nothing is trained, so prepare() forecasts every scored row once, before the walk, and the walk's weekly "fit" only
looks the forecasts up (X's only column is `row_i`). C01-C03 see the round's own y series of the stock, the last 512
values dated before d. C04 Kronos sees the last 512 half-day candles known at the round's cutoff, XD-adjusted so the
last close is the anchor; its forecast close c gives yhat = log(c / anchor).

C03 Moirai runs in .venv-moirai (uni2ts pins torch 2.4 / numpy 1 / pandas 2 / JAX; the side venv has torch 2.14,
numpy 2, pandas 2.3, no JAX) through moirai_side.py. Weights load from the local Hugging Face cache at pinned
revisions; nothing is downloaded during a run.
"""

from __future__ import annotations

import datetime as dt
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.task3.versions9 import features as F
from src.task3.versions9 import ladder as L
from src.task3.versions9_1 import inputs as I

N_CTX, SEED, BATCH = 512, 340, 64
ROOT = F.load.ROOT
KRONOS_DIR = ROOT / "external/Kronos"            # shiyu-coder/Kronos, commit below
MOIRAI_PY = ROOT / ".venv-moirai/bin/python"
# released = the Hugging Face repo's creation date; rows after it are the clean window (T03 §2 (b))
WEIGHTS = {
    "C01": dict(repo="amazon/chronos-bolt-small", revision="772f3d25d38aec6d914c8949dab4462e2d46f5d8",
                sha256="06a6a19bbe74bc10a9cd193bd4bf2bf638ae07f7e0d51653ae7ab8ea968a21dd", released="2024-11-25"),
    "C02": dict(repo="google/timesfm-2.5-200m-pytorch", revision="1d952420fba87f3c6dee4f240de0f1a0fbc790e3",
                sha256="2f776efe6245e42b24bc4153ffdf61810140210e4bd3b01fb21f7aa779ab6ce8", released="2025-09-02"),
    "C03": dict(repo="Salesforce/moirai-2.0-R-small", revision="30f43ff08c8494f4943ae1521e9d4e94a0fbb389",
                sha256="fb5652a3db8ea572606221b7cb1e77bb8962b168e4d4cc752cf31ceb04074669", released="2025-08-06"),
    "C04": dict(repo="NeoQuasar/Kronos-small", revision="901c26c1332695a2a8f243eb2f37243a37bea320",
                sha256="b082dfcbd8e8c142a725c8bbb99781802f38fec81210e13479effb32b3c3e020", released="2025-06-30",
                tokenizer="NeoQuasar/Kronos-Tokenizer-base", tokenizer_revision="0e0117387f39004a9016484a186a908917e22426",
                tokenizer_sha256="59d85f6af76a2c3b8240ea06cb21db4213b4eeca053f246b23e29cf832fc6bee",
                code="shiyu-coder/Kronos@67b630e67f6a18c9e9be918d9b4337c960db1e9a"),
}
UNI2TS = "SalesforceAIResearch/uni2ts@cfd46d4510ed8896f263116f32928eede05b0a75"
FC: dict = {}         # prepare(): spec -> forecast per table row (NaN where not asked)


def _path(repo: str, revision: str) -> str:
    from huggingface_hub import snapshot_download
    return snapshot_download(repo, revision=revision, local_files_only=True)


def contexts(y: np.ndarray, stock: np.ndarray, rows: np.ndarray, n: int = N_CTX) -> list:
    """For each row i: the last n finite y of the same stock on earlier rows. Rows of one stock are in date order
    (inputs.windows asserts it), so earlier row index = earlier day; row i's own y is never included."""
    out = {}
    for s in np.unique(stock[rows]):
        ok = np.flatnonzero((stock == s) & np.isfinite(y))
        for i in rows[stock[rows] == s]:
            k = np.searchsorted(ok, i)
            out[i] = y[ok[max(0, k - n):k]]
    return [out[i] for i in rows]


def _chunks(x: list, n: int = BATCH):
    return (x[i:i + n] for i in range(0, len(x), n))


def chronos(ctx: list) -> np.ndarray:
    """C01: Chronos-Bolt's 0.5 quantile, one step (the pipeline returns it as its mean)."""
    from chronos import BaseChronosPipeline
    w = WEIGHTS["C01"]
    pipe = BaseChronosPipeline.from_pretrained(_path(w["repo"], w["revision"]), device_map="cpu", torch_dtype=torch.float32)
    out = []
    for b in _chunks(ctx):
        _, mean = pipe.predict_quantiles([torch.tensor(c, dtype=torch.float32) for c in b], prediction_length=1,
                                         quantile_levels=[0.5])
        out.append(mean[:, 0].numpy())
    return np.concatenate(out)


def timesfm(ctx: list) -> np.ndarray:
    """C02: TimesFM 2.5 point forecast, one step. infer_is_positive off: returns can be negative."""
    import timesfm as tf
    w = WEIGHTS["C02"]
    m = tf.TimesFM_2p5_200M_torch.from_pretrained(_path(w["repo"], w["revision"]), torch_compile=False)
    m.compile(tf.ForecastConfig(max_context=N_CTX, max_horizon=128, normalize_inputs=True,
                                use_continuous_quantile_head=True, force_flip_invariance=True, infer_is_positive=False,
                                fix_quantile_crossing=True, per_core_batch_size=BATCH))
    point, _ = m.forecast(horizon=1, inputs=[np.asarray(c, float) for c in ctx])   # forecast() pads its list in place
    return point[:, 0]


def moirai(ctx: list) -> np.ndarray:
    """C03: Moirai 2.0's 0.5 quantile, one step, computed in .venv-moirai by moirai_side.py."""
    w = WEIGHTS["C03"]
    with tempfile.TemporaryDirectory() as d:
        a = np.full((len(ctx), N_CTX), np.nan)
        for i, c in enumerate(ctx):
            a[i, N_CTX - len(c):] = c
        np.save(Path(d) / "ctx.npy", a)
        subprocess.run([str(MOIRAI_PY), "-I", str(Path(__file__).with_name("moirai_side.py")), str(Path(d) / "ctx.npy"),
                        str(Path(d) / "out.npy"), _path(w["repo"], w["revision"])], check=True)
        return np.load(Path(d) / "out.npy")


# ---------------------------------------------------------------- C04 Kronos on half-day candles

END_TIME = {"am": dt.time(12, 30), "pm": dt.time(16, 30)}       # T03 §1.1: bar-end times sent to Kronos


def kronos_inputs(t: pd.DataFrame, rnd: str, rows: np.ndarray, n: int = N_CTX) -> list:
    """Per row: (context candles, target bar's timestamp). Morning: candles up to d-1's afternoon; afternoon: up to
    d's morning. Prices before each XD day <= d are back-adjusted (inputs.adjust_xd). `pc_dy` (positive control)
    scales the round's candle of each day by exp(y_c - y)."""
    half, out = ("am" if rnd == "morning" else "pm"), {}
    for s in np.unique(t["stock"].to_numpy()[rows]):
        c = I.session_candles(F.raw_inputs(s))
        if "pc_dy" in t:
            dy = t[t["stock"] == s].set_index("date")["pc_dy"]
            on = c["half"] == half
            c.loc[on, ["open", "high", "low", "close"]] *= np.exp(dy.reindex(c.loc[on, "date"]).fillna(0).to_numpy())[:, None]
        c["ts"] = [pd.Timestamp(dt.datetime.combine(d, END_TIME[h])) for d, h in zip(c["date"], c["half"])]
        G = I.xd_log_factor(t, s)
        for i in rows[t["stock"].to_numpy()[rows] == s]:
            d = t.at[i, "date"]
            last = c[(c["date"] < d) if rnd == "morning" else ((c["date"] < d) | ((c["date"] == d) & (c["half"] == "am")))]
            x = last.iloc[-n:].copy()
            px = ["open", "high", "low", "close"]
            x[px] = I.adjust_xd(x["date"], x[px].to_numpy(float), G, d)
            out[i] = (x, pd.Timestamp(dt.datetime.combine(d, END_TIME[half])))
    return [out[i] for i in rows]


def kronos(inp: list) -> np.ndarray:
    """C04: mean close of 20 samples (T 1.0, top_p 0.9), torch.manual_seed(340) before each batch; yhat = log(c / last
    close), the last close being the XD-adjusted anchor."""
    sys.path.insert(0, str(KRONOS_DIR))
    from model import Kronos, KronosPredictor, KronosTokenizer
    w = WEIGHTS["C04"]
    tok = KronosTokenizer.from_pretrained(_path(w["tokenizer"], w["tokenizer_revision"]))
    mdl = Kronos.from_pretrained(_path(w["repo"], w["revision"]))
    pred = KronosPredictor(mdl, tok, device="cpu", max_context=N_CTX)
    cols = ["open", "high", "low", "close", "volume", "amount"]
    out = []
    for b in _chunks(inp, 16):
        torch.manual_seed(SEED)
        res = pred.predict_batch([x[cols].reset_index(drop=True) for x, _ in b], [x["ts"].reset_index(drop=True) for x, _ in b],
                                 [pd.Series([ts]) for _, ts in b], pred_len=1, T=1.0, top_p=0.9, sample_count=20, verbose=False)
        out.append(np.log([r["close"].iloc[0] / x["close"].iloc[-1] for r, (x, _) in zip(res, b)]))
    return np.concatenate(out)


SERIES = {"C01": chronos, "C02": timesfm, "C03": moirai}


def prepare(t: pd.DataFrame, rnd: str, names: list, lo: dt.date, hi: dt.date) -> dict:
    """Forecast every row dated lo..hi once per spec; returns seconds per spec. Kronos needs all 512 candles, so a
    row with fewer is left unforecast (none from 2019 on)."""
    rows = np.flatnonzero(((t["date"] >= lo) & (t["date"] <= hi)).to_numpy())
    y, stock = t[L.ROUND[rnd][0]].to_numpy(float), t["stock"].to_numpy()
    secs = {}
    for k in names:
        t0 = __import__("time").time()
        fc = np.full(len(t), np.nan)
        if k == "C04":
            inp = kronos_inputs(t, rnd, rows)
            keep = [j for j, (x, _) in enumerate(inp) if len(x) == N_CTX]
            fc[rows[keep]] = kronos([inp[j] for j in keep])
        else:
            torch.manual_seed(SEED)
            fc[rows] = SERIES[k](contexts(y, stock, rows))
        FC[k] = fc
        secs[k] = round(__import__("time").time() - t0, 1)
    return secs


def lookup(name: str):
    """The walk's train(): no fit, predict(Z) returns prepare()'s forecasts for Z's rows (X = [row_i])."""
    def train(X, y, hp, n_iter=None, val=None):
        f = lambda Z: FC[name][Z[:, 0].astype(int)]  # noqa: E731
        return (f, None, float("nan")) if val else f
    return train


TRAIN = {k: lookup(k) for k in ("C01", "C02", "C03", "C04")}
