"""Task 3 v9.1 group E: CNNs E01-E07, E02p, E06p (tickets/T04).

Every spec is g(input) -> features -> dropout -> Linear([features, is_kbank]). Residual specs (all but E02p, E06p)
forecast yhat = yhat_R2 + g, where yhat_R2 is R2's own walk-forward forecast (run first by zoo.main) and g is fit on
y - ridge, the ridge refit per stock on g's training rows with R2's alpha for the year being forecast (T04 §1.1).
X = [input pointer, row_i, is_kbank (, r2_ok for residual specs: rows where R2's columns exist)]; the pointer is `e_i` (morning: the 20-day window W, afternoon: the minute
sequence S), `img_i` (E06 chart images) or `seq_i` (E07 spectrogram of S).
E01-E05, E07 train as group B (T02 §2: AdamW 1e-3, batch 64, patience 10, 5 seeds, dropout 0.1, CPU); E06/E06p as
JKX (Adam 1e-5, batch 128, patience 2, 3 seeds, dropout 0.5, Xavier init, MPS when available).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as Fn
from torch import nn

from src.task3.versions9 import ladder as L
from src.task3.versions9_1 import images as G
from src.task3.versions9_1 import models_seq as B

H = 16
CTX: dict = {}        # bind(): round, R2 forecasts and alphas, ridge columns, images, spectrograms; walk sets year_now
SEQ = dict(opt=torch.optim.AdamW, lr=1e-3, batch=64, patience=10, seeds=range(340, 345), drop=0.1, device="cpu")
JKX = dict(opt=torch.optim.Adam, lr=1e-5, batch=128, patience=2, seeds=range(340, 343), drop=0.5,
           device="mps" if torch.backends.mps.is_available() else "cpu")
# T04 §4.3 training rule, decided on the positive control 2026-11-01 (EXPERIMENT_LOG): morning E06p failed (+0.11%) and
# stopped at epoch 1 in 5/6 fits -> lr 1e-4; afternoon E06p passed (+1.33%) -> stays 1e-5
JKX_P = {**JKX, "lr": {"morning": 1e-4, "afternoon": 1e-5}}


def bind(t, rnd: str, r2, r2_log: dict, r2_cols, specs: list):
    """After models_seq.bind: point the E specs at their inputs. `r2` = R2's walk-forward preds (date, stock, yhat),
    `r2_log` its hp log, `r2_cols(stock)` its columns. Images / spectrograms are built only if a spec needs them."""
    yhat = t[["date", "stock"]].merge(r2[["date", "stock", "yhat"]], on=["date", "stock"], how="left")["yhat"]
    CTX.clear()
    CTX.update(rnd=rnd, r2=yhat.to_numpy(float), stock=t["stock"].to_numpy(),
               alpha={tuple(k.rsplit(" ", 1)): v["hp"] for k, v in r2_log.get("R2", {}).items()},
               own={s: t[r2_cols(s)].to_numpy(float) for s in np.unique(t["stock"])})
    own = np.zeros(len(t), bool)
    for s, a in CTX["own"].items():
        own |= (CTX["stock"] == s) & np.isfinite(a).all(1)
    # a residual spec trains and forecasts on R2's rows only (morning: fut_post starts 2019-01)
    t = t.assign(e_i=t["win_i"] if rnd == "morning" else t["seq_i"], r2_ok=np.where(own, 1.0, np.nan))
    if {"E06", "E06p"} & set(specs):
        CTX["IMG"], img_i = G.build(t, rnd)
        t = t.assign(img_i=img_i)
    if "E07" in specs:
        CTX["SPEC"] = G.spectrogram(B.CTX["S"])
    return t


def ridge_base(r, y):
    """Per stock ridge on R2's columns, fit on rows r with R2's alpha for the year being forecast; returns a function
    rows -> fitted values (rows are r2_ok, so every column exists)."""
    st, yr, fits = CTX["stock"], str(CTX["year_now"]), {}
    for s in np.unique(st[r]):
        m = st[r] == s
        fits[s] = L.train("ridge", CTX["own"][s][r[m]], y[m], CTX["alpha"][(s, yr)])

    def base(rr):
        out = np.full(len(rr), np.nan)
        for s, f in fits.items():
            m = st[rr] == s
            out[m] = f(CTX["own"][s][rr[m]])
        return out
    return base


def _source(src: str, i):
    """Raw input rows for pointer i, in the shape the encoder takes."""
    if src == "e":
        return B.CTX["W" if CTX["rnd"] == "morning" else "S"][i]          # (n, L, C)
    if src == "img":
        return CTX["IMG"][i][:, None].astype(np.float32) / 255            # (n, 1, 64, W)
    return CTX["SPEC"][i]                                                 # (n, 2, 17, 16)


def _scaler(src: str, A):
    """Fit on the training rows: per-channel clip + standardise (last axis of a sequence, axis 1 of a spectrogram).
    Images go in as 0..1."""
    if src == "img":
        return lambda a: a
    if src == "e":
        st = B._stats(A.reshape(-1, A.shape[-1]))
        return lambda a: B._scale(a, st)
    st = [B._stats(A[:, c].reshape(-1, 1)) for c in range(A.shape[1])]
    return lambda a: np.stack([B._scale(a[:, c], s) for c, s in enumerate(st)], 1)


# ---------------------------------------------------------------- encoders: input -> (B, out) flat features

def _causal(conv, h):
    return conv(Fn.pad(h, ((conv.kernel_size[0] - 1) * conv.dilation[0], 0)))


def _pool(h, k):
    """MaxPool(k) over time after dropping the oldest steps, so the latest minute is always in the last window."""
    return Fn.max_pool1d(h[..., h.shape[-1] % k:], k)


class E01(nn.Module):
    """1D-CNN. Morning: two causal k3 convs, last 5 steps. Afternoon: k5 conv + MaxPool(5) twice (155 -> 5 steps)."""

    def __init__(self, c, L_):
        super().__init__()
        self.long = L_ > 20
        k = 5 if self.long else 3
        self.c1, self.c2, self.out = nn.Conv1d(c, H, k), nn.Conv1d(H, H, k), 5 * H

    def forward(self, w):
        h = w.transpose(1, 2)
        if self.long:
            h = _pool(torch.relu(self.c1(h)), 5)              # 155 -> 151 -> 30
            return _pool(torch.relu(self.c2(h)), 5).flatten(1)  # 30 -> 26 -> 5
        h = torch.relu(_causal(self.c2, torch.relu(_causal(self.c1, h))))
        return h[:, :, -5:].flatten(1)


class E02(nn.Module):
    """CNN-LSTM: Conv1d k3 (+ MaxPool(5) on the afternoon) -> LSTM 32 -> last hidden state."""

    def __init__(self, c, L_):
        super().__init__()
        self.long, self.conv, self.lstm, self.out = L_ > 20, nn.Conv1d(c, H, 3), nn.LSTM(H, 32, batch_first=True), 32

    def forward(self, w):
        h = torch.relu(self.conv(w.transpose(1, 2)))
        if self.long:
            h = _pool(h, 5)
        return self.lstm(h.transpose(1, 2))[0][:, -1]


class E03(nn.Module):
    """WaveNet: 1x1 in, causal k2 dilated gated layers (tanh x sigmoid) with residual and skip 1x1 convs, skips summed
    at the last step -> ReLU. Dilations 1..16 (morning) or 1..128 (afternoon)."""

    def __init__(self, c, L_):
        super().__init__()
        dil = [2 ** i for i in range(8 if L_ > 20 else 5)]
        self.inp = nn.Conv1d(c, H, 1)
        self.conv = nn.ModuleList(nn.Conv1d(H, 2 * H, 2, dilation=d) for d in dil)
        self.res = nn.ModuleList(nn.Conv1d(H, H, 1) for _ in dil)
        self.skip = nn.ModuleList(nn.Conv1d(H, H, 1) for _ in dil)
        self.out = H

    def forward(self, w):
        h, s = self.inp(w.transpose(1, 2)), 0
        for conv, res, skip in zip(self.conv, self.res, self.skip):
            a, b = _causal(conv, h).chunk(2, 1)
            z = torch.tanh(a) * torch.sigmoid(b)
            h, s = h + res(z), s + skip(z)
        return torch.relu(s[:, :, -1])


class Inception(nn.Module):
    """TimesNet's Inception_Block_V1: mean of 2D convs with kernels 1, 3, 5 ('same' padding)."""

    def __init__(self, c):
        super().__init__()
        self.k = nn.ModuleList(nn.Conv2d(c, c, k, padding=k // 2) for k in (1, 3, 5))

    def forward(self, x):
        return torch.stack([k(x) for k in self.k]).mean(0)


class E04(nn.Module):
    """TimesNet, one TimesBlock: per sample, the 2 strongest FFT frequencies (DC excluded) give periods L // f; the
    series is folded to (L / p, p), passed through Inception -> GELU -> Inception, unfolded, and the two views are
    combined with softmax weights of their amplitudes, plus the input (residual). Last 5 steps."""

    def __init__(self, c, L_):
        super().__init__()
        self.emb, self.conv, self.out = nn.Linear(c, H), nn.Sequential(Inception(H), nn.GELU(), Inception(H)), 5 * H

    def forward(self, w):
        x = self.emb(w)
        Bn, L_, C = x.shape
        amp = torch.fft.rfft(x, dim=1).abs().mean(-1)
        amp[:, 0] = 0
        top, f = torch.topk(amp, 2, dim=1)
        period = L_ // f
        res = torch.zeros(Bn, 2, L_, C)
        for k in range(2):
            for p in torch.unique(period[:, k]).tolist():
                m = period[:, k] == p
                n = -(-L_ // p) * p
                z = Fn.pad(x[m], (0, 0, 0, n - L_)).reshape(-1, n // p, p, C).permute(0, 3, 1, 2)
                res[m, k] = self.conv(z).permute(0, 2, 3, 1).reshape(-1, n, C)[:, :L_]
        x = x + (torch.softmax(top, 1)[:, :, None, None] * res).sum(1)
        return x[:, -5:].flatten(1)


class E05(nn.Module):
    """ModernTCN, one block: per-variable patch embedding (patch 4 / stride 2 morning, 10 / 5 afternoon, 16 dims),
    depthwise conv along patches (kernel 5 / 13) + BatchNorm, ConvFFN within each variable (16 -> 32 -> 16) and across
    variables (C -> 2C -> C), GELU, residual; flattened."""

    def __init__(self, c, L_):
        super().__init__()
        long = L_ > 20
        p, s, k = (10, 5, 13) if long else (4, 2, 5)
        self.c, self.n = c, (L_ - p) // s + 1
        self.emb = nn.Conv1d(1, H, p, stride=s)
        self.dw, self.bn = nn.Conv1d(c * H, c * H, k, padding=k // 2, groups=c * H), nn.BatchNorm1d(c * H)
        self.ffn1 = nn.Sequential(nn.Conv1d(c * H, c * 2 * H, 1, groups=c), nn.GELU(), nn.Conv1d(c * 2 * H, c * H, 1, groups=c))
        self.ffn2 = nn.Sequential(nn.Conv1d(H * c, H * 2 * c, 1, groups=H), nn.GELU(), nn.Conv1d(H * 2 * c, H * c, 1, groups=H))
        self.out = c * H * self.n

    def forward(self, w):
        Bn, L_, C = w.shape
        x = self.emb(w.transpose(1, 2).reshape(Bn * C, 1, L_)).reshape(Bn, C * H, self.n)     # variable-major
        z = self.ffn1(self.bn(self.dw(x)))
        z = z.reshape(Bn, C, H, self.n).transpose(1, 2).reshape(Bn, H * C, self.n)          # dim-major for ffn2
        z = self.ffn2(z).reshape(Bn, H, C, self.n).transpose(1, 2).reshape(Bn, C * H, self.n)
        return (x + z).flatten(1)


class E07(nn.Module):
    """Spectrogram (2, 17 freq, 16 frames): two 3x3 convs, each ReLU + MaxPool over time only (16 -> 4 frames)."""

    def __init__(self, c, L_):
        super().__init__()
        self.c1, self.c2, self.out = nn.Conv2d(2, 8, 3, padding=1), nn.Conv2d(8, 8, 3, padding=1), 8 * 17 * 4

    def forward(self, s):
        h = Fn.max_pool2d(torch.relu(self.c1(s)), (1, 2))
        return Fn.max_pool2d(torch.relu(self.c2(h)), (1, 2)).flatten(1)


class JKXNet(nn.Module):
    """E06: JKX 20-day CNN. 3 blocks Conv2d 5x3 -> BN -> LeakyReLU(0.01) -> MaxPool(2,1), 64/128/256 channels, first
    conv stride (3,1) dilation (2,1) padding (12,1); flat 256 x 3 x width (46,080 for 64 x 60). Xavier init."""

    def __init__(self, c, width):
        super().__init__()
        layers = []
        for i, (a, b) in enumerate(((1, 64), (64, 128), (128, 256))):
            conv = nn.Conv2d(a, b, (5, 3), stride=(3, 1), dilation=(2, 1), padding=(12, 1)) if i == 0 else \
                nn.Conv2d(a, b, (5, 3), padding=(2, 1))
            layers += [conv, nn.BatchNorm2d(b), nn.LeakyReLU(0.01), nn.MaxPool2d((2, 1))]
        self.net, self.out = nn.Sequential(*layers), 256 * 3 * width

    def forward(self, x):
        return self.net(x).flatten(1)


class Net(nn.Module):
    def __init__(self, enc, drop):
        super().__init__()
        self.enc, self.drop, self.head = enc, nn.Dropout(drop), nn.Linear(enc.out + 1, 1)

    def forward(self, a, k):
        return self.head(torch.cat([self.drop(self.enc(a)), k[:, None]], 1)).squeeze(1)


def _xavier(net):
    for m in net.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(m.weight)
    return net


# ---------------------------------------------------------------- training

def cnn_model(name: str, make, src: str = "e", residual: bool = True, cfg: dict = SEQ):
    """MSE on y / sd(y) (y - ridge for a residual spec), patience on the seed-mean validation MSE in units of y."""
    def train(X, y, hp, n_iter=None, val=None):
        i, r, k = X[:, 0].astype(int), X[:, 1].astype(int), X[:, 2]
        base = ridge_base(r, y) if residual else None
        if residual:
            y = y - base(r)
        A = _source(src, i)
        scale, dev = _scaler(src, A), cfg["device"]
        At, Kt = torch.tensor(scale(A)), torch.tensor(k, dtype=torch.float32)
        ys = y.std() or 1.0
        Y = torch.tensor(y / ys, dtype=torch.float32)
        nets, opts, gens = [], [], []
        for s in cfg["seeds"]:
            torch.manual_seed(s)
            net = Net(make(*((A.shape[1], A.shape[-1]) if src == "img" else (A.shape[-1], A.shape[1]))), cfg["drop"])
            nets.append((_xavier(net) if src == "img" else net).to(dev))
            lr = cfg["lr"][CTX["rnd"]] if isinstance(cfg["lr"], dict) else cfg["lr"]
            opts.append(cfg["opt"](net.parameters(), lr=lr, **({"weight_decay": hp} if hp is not None else {})))
            gens.append(torch.Generator().manual_seed(s))

        def g(j, kk):
            a, kt = torch.tensor(scale(_source(src, j))), torch.tensor(kk, dtype=torch.float32)
            with torch.no_grad():
                return np.mean([np.concatenate([net.eval()(a[q].to(dev), kt[q].to(dev)).cpu().numpy()
                                                for q in torch.arange(len(a)).split(512)]) for net in nets], 0) * ys

        if val:
            vi, vr, vk = val[0][:, 0].astype(int), val[0][:, 1].astype(int), val[0][:, 2]
            vb = base(vr) if residual else 0.0
        curve = []
        for _ in range(MAX_EPOCHS if val else n_iter):
            for net, opt, gen in zip(nets, opts, gens):
                net.train()
                for q in torch.randperm(len(Y), generator=gen).split(cfg["batch"]):
                    opt.zero_grad()
                    Fn.mse_loss(net(At[q].to(dev), Kt[q].to(dev)), Y[q].to(dev)).backward()
                    opt.step()
            if val:
                curve.append(float(((val[1] - vb - g(vi, vk)) ** 2).mean()))
                if len(curve) - 1 - int(np.argmin(curve)) >= cfg["patience"]:
                    break
        if val:
            best, v = L.best_with_patience(curve, cfg["patience"])
            B.DIAG.setdefault(name, {}).setdefault("best_epoch", []).append(best)
            return None, best, v

        def predict(Z):
            j, rr, kk = Z[:, 0].astype(int), Z[:, 1].astype(int), Z[:, 2]
            return g(j, kk) + (CTX["r2"][rr] if residual else 0.0)
        return predict
    return train


MAX_EPOCHS = 100
TRAIN = {
    "E01": cnn_model("E01", E01),
    "E02": cnn_model("E02", E02),
    "E02p": cnn_model("E02p", E02, residual=False),
    "E03": cnn_model("E03", E03),
    "E04": cnn_model("E04", E04),
    "E05": cnn_model("E05", E05),
    "E06": cnn_model("E06", JKXNet, "img", cfg=JKX),
    "E06p": cnn_model("E06p", JKXNet, "img", residual=False, cfg=JKX_P),
    "E07": cnn_model("E07", E07, "spec"),
}
