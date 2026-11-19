"""Score one v9 request with the fitted bundle. Needs only numpy, scikit-learn and joblib (versions: bundle/manifest.json).

    python predict.py request.json                # prints the prediction JSON
    python predict.py --selftest                  # replays examples.json, must print "ok"

Request (features are built by the live adapter with the formulas in README.md §6):
    {"round": "morning" | "afternoon", "symbol": "KBANK" | "ADVANC", "date": "YYYY-MM-DD",
     "C_prev": 158.5, "D": 0.0,          # morning: anchor = C_prev - D
     "P": 159.0,                         # afternoon: anchor = P (morning close, bar_end <= 12:30)
     "features": {"spx": 0.0012, ...}}   # every feature of the primary model; missing / null -> fallback
"""

from __future__ import annotations

import json
import math
import sys
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import joblib
import numpy as np

HERE = Path(__file__).resolve().parent
PRIMARY = {"morning": "huber_lite", "afternoon": "hgb"}   # v9.2: huber_lite needs no S50 futures / bank peers
FALLBACK = {"morning": "ols_atc", "afternoon": "ols_small"}
KAPPA = {"morning": 0.7, "afternoon": 0.5}
_MODELS: dict | None = None
# SET price spread table (same as src/data_ingestion/finalize_master_dataset_v1_1.py SET_TICK_SIZES): (upper bound, tick)
SET_TICKS = ((Decimal("2"), Decimal("0.01")), (Decimal("5"), Decimal("0.02")), (Decimal("10"), Decimal("0.05")),
             (Decimal("25"), Decimal("0.10")), (Decimal("100"), Decimal("0.25")), (Decimal("200"), Decimal("0.50")),
             (Decimal("400"), Decimal("1.00")), (None, Decimal("2.00")))


def tick_size(price: Decimal) -> Decimal:
    return next(t for hi, t in SET_TICKS if hi is None or price < hi)


def to_tick(price: float) -> float:
    """Nearest price the SET can trade at (half up). Band edges (2, 5, ..., 400) sit on both grids, so the band of the
    raw price always gives the nearest valid price."""
    p = Decimal(repr(price))
    t = tick_size(p)
    return float((p / t).quantize(Decimal(1), rounding=ROUND_HALF_UP) * t)


def models() -> dict:
    global _MODELS
    if _MODELS is None:
        _MODELS = joblib.load(HERE / "bundle" / "models.joblib")
    return _MODELS


def _yhat(m: dict, x: np.ndarray) -> float:
    z = (np.clip(x, m["lo"], m["hi"]) - m["mu"]) / m["sd"]
    if m["kind"] == "ols":
        return float(np.r_[1.0, z] @ m["beta"])
    return float(m["est"].predict(z[None, :])[0] * m["ys"])


def _row(m: dict, feats: dict, symbol: str) -> np.ndarray | None:
    f = {**feats, "is_kbank": 1.0 if symbol == "KBANK" else 0.0}
    v = [f.get(c) for c in m["cols"]]
    if any(x is None or not math.isfinite(float(x)) for x in v):
        return None
    return np.array(v, float)


def predict(req: dict) -> dict:
    rnd, sym = req["round"], req["symbol"]
    if rnd not in PRIMARY or sym not in ("KBANK", "ADVANC"):
        raise ValueError(f"bad round/symbol: {rnd} {sym}")
    anchor = float(req["C_prev"]) - float(req.get("D") or 0.0) if rnd == "morning" else float(req["P"])
    if not (math.isfinite(anchor) and anchor > 0):
        raise ValueError(f"bad anchor {anchor}")
    feats = req.get("features", {})
    out = {"round": rnd, "symbol": sym, "date": req.get("date"), "anchor": anchor}
    m = models()[f"{rnd}/{PRIMARY[rnd]}/{'pooled' if rnd == 'afternoon' else sym}"]
    x = _row(m, feats, sym)
    if x is not None:
        y, used = _yhat(m, x), PRIMARY[rnd]
    else:
        fb = models()[f"{rnd}/{FALLBACK[rnd]}/{sym}"]
        xf = _row(fb, feats, sym)
        if xf is None:
            missing = [c for c in fb["cols"] if feats.get(c) is None]
            raise ValueError(f"fallback {FALLBACK[rnd]} also lacks features: {missing}")
        y, used = KAPPA[rnd] * _yhat(fb, xf), FALLBACK[rnd]
        out["missing_for_primary"] = [c for c in m["cols"] if c != "is_kbank"
                                      and (feats.get(c) is None or not math.isfinite(float(feats[c])))]
    raw = anchor * math.exp(y)
    out.update(model=used, yhat=y, prediction=to_tick(raw), prediction_raw=raw,
               target="P_d (morning close, bar_end <= 12:30)" if rnd == "morning" else "C_d (official close)")
    return out


def selftest() -> None:
    ex = json.loads((HERE / "examples.json").read_text())
    for e in ex:
        got = predict(e["request"])
        want = e["expected"]
        assert got["model"] == want["model"], (e["request"]["date"], got["model"], want["model"])
        # examples.json stores the raw model value under "prediction" (written before tick rounding existed)
        assert abs(got["prediction_raw"] - want["prediction"]) < 1e-9, (e["request"]["date"], got, want)
        assert got["prediction"] == to_tick(want["prediction"]), (e["request"]["date"], got)
    cases = {1.994: 1.99, 1.996: 2.0, 4.99: 5.0, 4.975: 4.98, 9.97: 9.95, 9.98: 10.0, 24.94: 24.9, 99.8: 99.75, 99.9: 100.0,
             100.2: 100.0, 100.25: 100.5, 158.74: 158.5, 199.76: 200.0, 200.4: 200.0, 241.6374: 242.0,
             338.6767: 339.0, 338.5: 339.0, 399.4: 399.0, 399.6: 400.0, 400.9: 400.0, 401.0: 402.0}
    for raw, want in cases.items():
        assert to_tick(raw) == want, (raw, to_tick(raw), want)
    print(f"ok: {len(ex)} examples match, {len(cases)} tick cases pass")


if __name__ == "__main__":
    if sys.argv[1:] == ["--selftest"]:
        selftest()
    else:
        print(json.dumps(predict(json.loads(Path(sys.argv[1]).read_text())), indent=1))
