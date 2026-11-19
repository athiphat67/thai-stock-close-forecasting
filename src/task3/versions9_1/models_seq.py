"""Task 3 v9.1 group B: sequence models B01-B11, B02m, B06m (tickets/T02). Shared with T04.

Torch specs see two inputs: W, the stock's 20 previous rows (inputs.windows: pooled columns + y_m + y_a) or, for B02m
and B06m, the afternoon's minute sequence S; and x, today's pooled columns. Every spec is encoder(W) -> h (16) ->
dropout 0.1 -> yhat = Linear([h, x]), so they differ only in the encoder. X's first column points at W / S (`win_i`,
`seq_i`) and is not a feature; bind() sets it for the run's table. B01 ARIMAX is per stock on the own columns, refit
weekly, and filters through each week's known y one day at a time (`row_i` finds that y).
"""

from __future__ import annotations

import warnings

import numpy as np
import torch
import torch.nn.functional as Fn
from statsmodels.tsa.statespace.sarimax import SARIMAX
from torch import nn

from src.task3.versions9 import ladder as L
from src.task3.versions9_1 import inputs as I

SEEDS = range(340, 345)
MAX_EPOCHS, PATIENCE, BATCH, H = 100, 10, 64, 16
CTX: dict = {}        # bind(): the run's windows, sequences, target and dates
DIAG: dict = {}       # per spec: best epoch of every selection fold, ARIMAX fits that did not converge, B10 VSN weights


def bind(t, rnd: str):
    """Point the models at this run's table (after the positive control plants its target). Adds `win_i`."""
    c = I.cols(rnd, None, "pooled")
    W = I.windows(t, c)
    ok = np.isfinite(W).all(axis=(1, 2))
    t = t.assign(win_i=np.where(ok, t["row_i"], np.nan))
    CTX.clear()
    CTX.update(W=W.astype(np.float32), p=len(c), chan=[*c, "y_m", "y_a"], y=t[L.ROUND[rnd][0]].to_numpy(float),
               stock=t["stock"].to_numpy(), year=np.array([d.year for d in t["date"]]))
    if "seq_i" in t:
        CTX["S"] = I.minute_seq().transpose(0, 2, 1).astype(np.float32)     # (rows, 155, 2): time first, as W
    DIAG.clear()
    return t


def _stats(a):
    lo, hi = np.quantile(a, .01, axis=0), np.quantile(a, .99, axis=0)
    c = np.clip(a, lo, hi)
    sd = c.std(0)
    sd[sd == 0] = 1.0
    return lo, hi, c.mean(0), sd


def _scale(a, st):
    lo, hi, mu, sd = st
    return ((np.clip(a, lo, hi) - mu) / sd).astype(np.float32)


# ---------------------------------------------------------------- encoders: (B, L, C) [+ x] -> (B, 16)

class Last(nn.Module):
    """B02/B03 (B02m): the recurrent net's hidden state at the last step."""

    def __init__(self, rnn, c, L_, px):
        super().__init__()
        self.rnn, self.out = rnn(c, 32, batch_first=True), nn.Linear(32, H)

    def forward(self, w, x):
        return self.out(self.rnn(w)[0][:, -1])


class TCN(nn.Module):
    """B04: 4 causal residual blocks, kernel 3, dilation 1/2/4/8 (receptive field 31), 16 channels, last step."""

    def __init__(self, c, L_, px):
        super().__init__()
        self.convs = nn.ModuleList(nn.Conv1d(c if i == 0 else H, H, 3, dilation=d) for i, d in enumerate((1, 2, 4, 8)))
        self.skip = nn.Conv1d(c, H, 1)

    def forward(self, w, x):
        h = w.transpose(1, 2)
        for i, conv in enumerate(self.convs):
            r = self.skip(h) if i == 0 else h
            h = torch.relu(conv(Fn.pad(h, (2 * conv.dilation[0], 0))) + r)
        return h[:, :, -1]


def _layer(d, heads, ff):
    # dropout lives after the encoder only (T02 §2)
    return nn.TransformerEncoderLayer(d, heads, ff, dropout=0.0, batch_first=True)


class Trans(nn.Module):
    """B05: one token per day, learned position embedding, 1 layer d 32 / 2 heads / FF 64, last token."""

    def __init__(self, c, L_, px):
        super().__init__()
        self.inp, self.pos, self.enc, self.out = nn.Linear(c, 32), nn.Embedding(L_, 32), _layer(32, 2, 64), nn.Linear(32, H)

    def forward(self, w, x):
        return self.out(self.enc(self.inp(w) + self.pos.weight)[:, -1])


class PatchTST(nn.Module):
    """B06/B06m: channel-independent patches through one shared encoder (d 16, 2 heads, FF 32), flattened.
    The most recent n*patch steps are used (B06m: the last 150 of 155 minutes)."""

    def __init__(self, c, L_, px, patch=5):
        super().__init__()
        self.patch, self.n = patch, L_ // patch
        self.emb, self.pos, self.enc = nn.Linear(patch, H), nn.Embedding(self.n, H), _layer(H, 2, 2 * H)
        self.out = nn.Linear(c * self.n * H, H)

    def forward(self, w, x):
        B, L_, C = w.shape
        z = w[:, L_ - self.n * self.patch:].transpose(1, 2).reshape(B * C, self.n, self.patch)
        return self.out(self.enc(self.emb(z) + self.pos.weight).reshape(B, -1))


class ITrans(nn.Module):
    """B07: each channel's 20-day series is one token (Linear 20 -> 32), attention across channels, mean."""

    def __init__(self, c, L_, px):
        super().__init__()
        self.inp, self.enc, self.out = nn.Linear(L_, 32), _layer(32, 2, 64), nn.Linear(32, H)

    def forward(self, w, x):
        return self.out(self.enc(self.inp(w.transpose(1, 2))).mean(1))


def decompose(w, k=5):
    """DLinear's split of (B, L, C) into a moving-average trend (ends repeated) and the remainder, as (B, C, L)."""
    z = w.transpose(1, 2)
    trend = Fn.avg_pool1d(Fn.pad(z, ((k - 1) // 2, k // 2), mode="replicate"), k, stride=1)
    return trend, z - trend


class DLinear(nn.Module):
    """B08: Linear 20 -> 1 on the trend and on the remainder, weights shared by every channel."""

    def __init__(self, c, L_, px):
        super().__init__()
        self.t, self.s, self.out = nn.Linear(L_, 1), nn.Linear(L_, 1), nn.Linear(2 * c, H)

    def forward(self, w, x):
        tr, se = decompose(w)
        return self.out(torch.cat([self.t(tr), self.s(se)], -1).flatten(1))


class NHiTS(nn.Module):
    """B09: 3 blocks on the max-pooled residual (kernel 4, 2, 1), MLP 2 x 64, backcast subtracted, forecasts summed."""

    def __init__(self, c, L_, px):
        super().__init__()
        self.k = (4, 2, 1)
        self.mlp = nn.ModuleList(nn.Sequential(nn.Linear(c * (L_ // k), 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU())
                                 for k in self.k)
        self.back = nn.ModuleList(nn.Linear(64, c * L_) for _ in self.k)
        self.fore = nn.ModuleList(nn.Linear(64, H) for _ in self.k)

    def forward(self, w, x):
        r, f = w.transpose(1, 2), 0
        for k, mlp, back, fore in zip(self.k, self.mlp, self.back, self.fore):
            h = mlp(Fn.max_pool1d(r, k).flatten(1))
            r, f = r - back(h).view_as(r), f + fore(h)
        return f


class VarLinear(nn.Module):
    """One Linear per variable, (..., C, i) -> (..., C, o), each initialised as PyTorch's Linear."""

    def __init__(self, c, i, o):
        super().__init__()
        lin = [nn.Linear(i, o) for _ in range(c)]
        self.w = nn.Parameter(torch.stack([m.weight.detach().T for m in lin]))
        self.b = nn.Parameter(torch.stack([m.bias.detach() for m in lin]))

    def forward(self, a):
        return torch.einsum("...ci,cio->...co", a, self.w) + self.b


class GRN(nn.Module):
    """Lim et al. 2021 eq. 2-4: LayerNorm(skip(a) + GLU(W1 ELU(W2 a)))."""

    def __init__(self, lin, i, h, o):
        super().__init__()
        self.w2, self.w1, self.glu = lin(i, h), lin(h, h), lin(h, 2 * o)
        self.skip = lin(i, o) if i != o else nn.Identity()
        self.norm = nn.LayerNorm(o)

    def forward(self, a):
        return self.norm(self.skip(a) + Fn.glu(self.glu(self.w1(Fn.elu(self.w2(a))))))


class VSN(nn.Module):
    """Variable selection: a GRN per variable (scalar -> 16), softmax weights from a GRN over all of them."""

    def __init__(self, c):
        super().__init__()
        self.var = GRN(lambda i, o: VarLinear(c, i, o), 1, H, H)
        self.sel = GRN(nn.Linear, c, H, c)

    def forward(self, a):
        v = torch.softmax(self.sel(a), -1)
        return (v.unsqueeze(-1) * self.var(a.unsqueeze(-1))).sum(-2), v


class TFT(nn.Module):
    """B10 TFT-lite: VSN at every step -> LSTM 32; x through its own VSN; no attention, no quantile loss."""

    def __init__(self, c, L_, px):
        super().__init__()
        self.vw, self.vx, self.lstm, self.out = VSN(c), VSN(px), nn.LSTM(H, 32, batch_first=True), nn.Linear(32 + H, H)
        self.weights = None

    def forward(self, w, x):
        e, self.weights = self.vw(w)
        return self.out(torch.cat([self.lstm(e)[0][:, -1], self.vx(x)[0]], 1))


class XLSTM(nn.Module):
    """B11: Linear C -> 32, xLSTM stack of 2 blocks (mLSTM, sLSTM; pure-torch backends, the package README's block
    settings), last step."""

    def __init__(self, c, L_, px):
        super().__init__()
        from xlstm import (FeedForwardConfig, mLSTMBlockConfig, mLSTMLayerConfig, sLSTMBlockConfig, sLSTMLayerConfig,
                           xLSTMBlockStack, xLSTMBlockStackConfig)
        cfg = xLSTMBlockStackConfig(
            mlstm_block=mLSTMBlockConfig(mlstm=mLSTMLayerConfig(conv1d_kernel_size=4, qkv_proj_blocksize=4, num_heads=4)),
            slstm_block=sLSTMBlockConfig(slstm=sLSTMLayerConfig(backend="vanilla", num_heads=4, conv1d_kernel_size=4,
                                                                bias_init="powerlaw_blockdependent"),
                                         feedforward=FeedForwardConfig(proj_factor=1.3, act_fn="gelu")),
            context_length=L_, num_blocks=2, embedding_dim=32, slstm_at=[1])
        self.inp, self.stack, self.out = nn.Linear(c, 32), xLSTMBlockStack(cfg), nn.Linear(32, H)

    def forward(self, w, x):
        return self.out(self.stack(self.inp(w))[:, -1])


class Net(nn.Module):
    def __init__(self, enc, px):
        super().__init__()
        self.enc, self.drop, self.head = enc, nn.Dropout(0.1), nn.Linear(H + px, 1)

    def forward(self, w, x):
        return self.head(torch.cat([self.drop(self.enc(w, x)), x], 1)).squeeze(1)


def seq_model(name: str, make, src: str = "W"):
    """B02-B11 (src W) and B02m/B06m (src S): AdamW lr 1e-3, weight decay hp, batch 64, <= 100 epochs, patience 10 on
    the 5-seed mean, fit on y / sd(y). W's feature channels use x's clip and scale; the rest use their own."""
    def train(X, y, hp, n_iter=None, val=None):
        i, x = X[:, 0].astype(int), X[:, 1:]
        A = CTX[src][i]
        sx = _stats(x)
        if src == "W":
            p = CTX["p"]
            sy = _stats(A[..., p:].reshape(-1, A.shape[-1] - p))
            sw = tuple(np.concatenate([a[:p], b]) for a, b in zip(sx, sy))
        else:
            sw = _stats(A.reshape(-1, A.shape[-1]))
        ys = y.std() or 1.0
        Wt, Xt = torch.tensor(_scale(A, sw)), torch.tensor(_scale(x, sx))
        Y = torch.tensor(y / ys, dtype=torch.float32)
        nets, opts, gens = [], [], []
        for s in SEEDS:
            torch.manual_seed(s)
            nets.append(Net(make(A.shape[2], A.shape[1], x.shape[1]), x.shape[1]))
            opts.append(torch.optim.AdamW(nets[-1].parameters(), lr=1e-3, weight_decay=hp))
            gens.append(torch.Generator().manual_seed(s))

        def predict(Z):
            j = Z[:, 0].astype(int)
            w, z = torch.tensor(_scale(CTX[src][j], sw)), torch.tensor(_scale(Z[:, 1:], sx))
            with torch.no_grad():
                return np.mean([net.eval()(w, z).numpy() for net in nets], axis=0) * ys

        curve = []
        for _ in range(MAX_EPOCHS if val else n_iter):
            for net, opt, gen in zip(nets, opts, gens):
                net.train()
                for b in torch.randperm(len(Y), generator=gen).split(BATCH):
                    opt.zero_grad()
                    Fn.mse_loss(net(Wt[b], Xt[b]), Y[b]).backward()
                    opt.step()
            if val:
                curve.append(float(((val[1] - predict(val[0])) ** 2).mean()))
                if len(curve) - 1 - int(np.argmin(curve)) >= PATIENCE:
                    break
        if val:
            best, v = L.best_with_patience(curve, PATIENCE)
            DIAG.setdefault(name, {}).setdefault("best_epoch", []).append(best)
            return None, best, v
        if name == "B10":           # mean VSN weight per W channel over the training windows, by refit year
            with torch.no_grad():
                wts = np.mean([(net.eval()(Wt, Xt), net.enc.weights.mean((0, 1)).numpy())[1] for net in nets], axis=0)
            DIAG.setdefault(name, {}).setdefault("vsn", {}).setdefault(int(CTX["year"][i].max()), []).append(wts)
        return predict
    return train


# ---------------------------------------------------------------- B01 ARIMAX

def arimax(name: str = "B01"):
    """SARIMAX(p, 0, q) with a constant on the round's log return, exog = standardised own columns, per stock.
    Forecasts are one-step-ahead through the rows asked for, each using y only up to the day before (y of the asked
    rows comes from CTX, i.e. the week's days already settled). A fit that does not converge keeps the stock's
    last converged parameters for that (p, q) (T02 §4.5)."""
    last: dict = {}

    def train(X, y, hp, n_iter=None, val=None):
        i, x = X[:, 0].astype(int), X[:, 1:]
        sx = _stats(x)
        mod = SARIMAX(y, exog=_scale(x, sx).astype(float), order=(hp[0], 0, hp[1]), trend="c")
        with warnings.catch_warnings():             # convergence is read from mle_retvals below
            warnings.simplefilter("ignore")
            res = mod.fit(disp=False)
        key = (CTX["stock"][i[0]], tuple(hp))
        if res.mle_retvals.get("converged", True):
            last[key] = res.params
        else:
            DIAG.setdefault(name, {}).setdefault("not_converged", 0)
            DIAG[name]["not_converged"] += 1
            if key in last:
                res = mod.filter(last[key])

        def predict(Z, yz=None):
            j = Z[:, 0].astype(int)
            yz = CTX["y"][j] if yz is None else yz
            ext = res.append(yz, exog=_scale(Z[:, 1:], sx).astype(float), refit=False)
            return np.asarray(ext.predict(start=len(y), end=len(y) + len(Z) - 1))

        if val:
            return None, None, float(((val[1] - predict(val[0], val[1])) ** 2).mean())
        predict.res, predict.scale = res, sx           # for tests/test_task3_v9_1.py
        return predict
    return train


def diag() -> dict:
    """DIAG for diag.json: share of selection folds whose best epoch is 1, and B10's VSN weights averaged per year."""
    out = {}
    for k, d in DIAG.items():
        o = dict(d)
        if "best_epoch" in d:
            o["epoch1_share"] = float(np.mean(np.array(d["best_epoch"]) == 1))
        if "vsn" in d:
            o["vsn"] = {yr: dict(zip(CTX["chan"], np.mean(v, axis=0).round(4).tolist())) for yr, v in d["vsn"].items()}
        out[k] = o
    return out


TRAIN = {
    "B01": arimax(),
    "B02": seq_model("B02", lambda c, L_, px: Last(nn.LSTM, c, L_, px)),
    "B03": seq_model("B03", lambda c, L_, px: Last(nn.GRU, c, L_, px)),
    "B04": seq_model("B04", TCN),
    "B05": seq_model("B05", Trans),
    "B06": seq_model("B06", PatchTST),
    "B07": seq_model("B07", ITrans),
    "B08": seq_model("B08", DLinear),
    "B09": seq_model("B09", NHiTS),
    "B10": seq_model("B10", TFT),
    "B11": seq_model("B11", XLSTM),
    "B02m": seq_model("B02m", lambda c, L_, px: Last(nn.LSTM, c, L_, px), "S"),
    "B06m": seq_model("B06m", lambda c, L_, px: PatchTST(c, L_, px, patch=15), "S"),
}
