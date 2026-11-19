"""Task 3 v9.1 group E inputs (tickets/T04 §4.1, §5): JKX OHLC chart images (E06/E06p) and minute spectrograms (E07).

Images are drawn with numpy (no anti-aliasing): 3 px per bar (open left, high-low line centre, close right), a 1 px
moving-average line through the centre columns, volume bars in the bottom 12 rows under a 1-row gap, price scaled
from the image's lowest low to its highest high (moving average included) over the top 51 rows. Black 0, white 255.
  morning   64 x 60: daily OHLC of d-20 .. d-1 (XD-adjusted to d's anchor), 20-day moving average of the close.
  afternoon 64 x 90: d's thirty 5-minute bars, bar_end 10:05 .. 12:30 (09:56-10:00 joins the first), with the
            running mean of the bar closes since 10:00. A bar with no trade is a blank column.
build() draws every row of a round once and caches it in artifacts/task3/v9_1/images_<round>.npz.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
from scipy.signal import get_window

from src.task3.versions9 import features as F
from src.task3.versions9_1 import inputs as I

H, PRICE_ROWS, VOL_ROWS = 64, 51, 12
N_BARS = {"morning": 20, "afternoon": 30}
MA_DAYS = 20
CACHE = F.load.ROOT / "artifacts/task3/v9_1"


def _line(img, r0, c0, r1, c1):
    """Pixels from (r0, c0) to (r1, c1), one per column and enough per row that the line has no gaps."""
    n = max(abs(r1 - r0), abs(c1 - c0)) + 1
    img[np.rint(np.linspace(r0, r1, n)).astype(int), np.rint(np.linspace(c0, c1, n)).astype(int)] = 255


def draw(o, h, l, c, v, ma) -> np.ndarray:
    """One image from per-bar arrays (NaN = no bar / no MA point). Width = 3 x bars."""
    n = len(c)
    img = np.zeros((H, 3 * n), np.uint8)
    ok = np.isfinite(c)
    if not ok.any():
        return img
    hi, lo = np.nanmax(np.r_[h[ok], ma]), np.nanmin(np.r_[l[ok], ma])
    span = hi - lo if hi > lo else 1.0
    row = lambda p: np.rint((hi - p) / span * (PRICE_ROWS - 1)).astype(int)  # noqa: E731   top row = highest price
    vmax = np.nanmax(v[ok]) if np.nanmax(v[ok]) > 0 else 1.0
    for j in np.flatnonzero(ok):
        img[row(o[j]), 3 * j] = 255
        img[row(h[j]):row(l[j]) + 1, 3 * j + 1] = 255
        img[row(c[j]), 3 * j + 2] = 255
        k = int(np.rint(v[j] / vmax * VOL_ROWS))
        if k:
            img[H - k:, 3 * j + 1] = 255
    pts = [(row(m), 3 * j + 1) for j, m in enumerate(ma) if np.isfinite(m)]
    for a, b in zip(pts, pts[1:]):
        _line(img, *a, *b)
    if len(pts) == 1:
        img[pts[0]] = 255
    return img


def daily_image(day: pd.DataFrame, G: pd.Series, d: dt.date) -> np.ndarray | None:
    """Morning image for day d from `day` (official OHLCV, date order) or None without 39 earlier days."""
    past = day[day["date"] < d].tail(MA_DAYS + N_BARS["morning"] - 1)
    if len(past) < MA_DAYS + N_BARS["morning"] - 1:
        return None
    px = I.adjust_xd(past["date"], past[["open", "high", "low", "close"]].to_numpy(float), G, d)
    if not np.isfinite(px).all():
        return None
    ma = pd.Series(px[:, 3]).rolling(MA_DAYS).mean().to_numpy()[-N_BARS["morning"]:]
    o, h, l, c = px[-N_BARS["morning"]:].T
    return draw(o, h, l, c, past["volume"].to_numpy(float)[-N_BARS["morning"]:], ma)


def five_minute(am: pd.DataFrame) -> pd.DataFrame:
    """Thirty 5-minute bars of one morning's 1m bars (bar_end 09:56-12:30), indexed 0..29; missing bars absent."""
    t = am["bar_end"].dt.tz_convert(F.BKK)
    m = (t.dt.hour * 60 + t.dt.minute - 600).to_numpy()                 # minutes after 10:00
    b = np.maximum(0, -(-m // 5) - 1)                                    # 10:01-10:05 (and 09:56-10:00) -> 0
    g = am.assign(bin=b).sort_values("bar_end").groupby("bin")
    return pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(),
                         "close": g["close"].last(), "volume": g["volume"].sum()})


def intraday_image(am: pd.DataFrame) -> np.ndarray | None:
    """Afternoon image from one day's morning 1m bars, or None if it has none."""
    if am.empty:
        return None
    x = five_minute(am).reindex(range(N_BARS["afternoon"]))
    c = x["close"].to_numpy(float)
    ma = np.where(np.isfinite(c), pd.Series(c).expanding().mean().to_numpy(), np.nan)   # NaN closes are skipped
    return draw(*(x[k].to_numpy(float) for k in ("open", "high", "low")), c, x["volume"].to_numpy(float), ma)


def build(t: pd.DataFrame, rnd: str) -> tuple[np.ndarray, np.ndarray]:
    """(images, img_i): every table row's image (uint8) and its index into them (NaN = none). Cached per round,
    keyed by (stock, date)."""
    path = CACHE / f"images_{rnd}.npz"
    if path.exists():
        z = np.load(path, allow_pickle=False)
        x, keys = z["x"], pd.MultiIndex.from_arrays([z["stock"], z["date"]])
    else:
        imgs, ks = [], []
        for s in F.STOCKS:
            raw = F.raw_inputs(s)
            days = t.loc[t["stock"] == s, "date"]
            if rnd == "morning":
                day, G = I.daily_ohlc(raw), I.xd_log_factor(t, s)
                made = ((d, daily_image(day, G, d)) for d in days)
            else:
                am = F.morning_bars(raw["m1"])
                by = dict(tuple(am.groupby("trade_date")))
                made = ((d, intraday_image(by[d]) if d in by else None) for d in days)
            for d, im in made:
                if im is not None:
                    imgs.append(im)
                    ks.append((s, str(d)))
        x = np.stack(imgs)
        CACHE.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, x=x, stock=np.array([k[0] for k in ks]), date=np.array([k[1] for k in ks]))
        keys = pd.MultiIndex.from_tuples(ks)
    pos = pd.Series(np.arange(len(x)), index=keys)
    i = pos.reindex(pd.MultiIndex.from_arrays([t["stock"], t["date"].astype(str)])).to_numpy().astype(float)
    return x, i


# ---------------------------------------------------------------- E07 spectrogram (T04 §5)

WIN, HOP = 32, 8


def spectrogram(S: np.ndarray) -> np.ndarray:
    """(rows, 155, 2) minute sequences -> (rows, 2, 17, 16): channel 0 = log power of the return channel's STFT
    (Hann 32, hop 8, no padding), channel 1 = the frame's summed return (signed), repeated along frequency."""
    r = S[:, :, 0].astype(float)
    n = (r.shape[1] - WIN) // HOP + 1
    seg = np.stack([r[:, k * HOP:k * HOP + WIN] for k in range(n)], axis=1)          # (rows, 16, 32)
    p = np.abs(np.fft.rfft(seg * get_window("hann", WIN), axis=-1)) ** 2              # (rows, 16, 17)
    power = np.log(p + 1e-12).transpose(0, 2, 1)
    move = np.repeat(seg.sum(-1)[:, None, :], power.shape[1], axis=1)
    return np.stack([power, move], axis=1).astype(np.float32)
