"""Task 3 v9 M1 done-check (02_PROTOCOL §8): the feature store neither leaks past the cutoff nor drifts from spec #2.

1. Cutoff test (03_FEATURES header): for sampled days d, drop every raw row with known_at > cutoff(d) (09:30 / 13:00),
   rebuild, and require row d to equal the full-data row in every feature (NaN == NaN). The truncation must also
   bite: P_d is unknown at 09:30 and C_d at 13:00.
2. Spec #2 (foreign 5 + atc_push, OLS, weekly refit, 2019-2022) from the store must give the same skill vs spec #1
   as checks/atc_push_test.py (EXPERIMENT_LOG 2026-10-27: +0.78% / +1.68%).

    .venv/bin/python -W ignore src/task3/versions9/checks/m1_check.py
"""
import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import atc_push_test as apt  # noqa: E402
from src.task3.versions9 import features as F  # noqa: E402

N_RANDOM = 20


def sample_days(m: pd.DataFrame) -> list:
    ok = m[m["P"].notna() & (m.index >= dt.date(2017, 6, 1))]
    rng = np.random.default_rng(340)
    edge = [ok.index[ok["D"] > 0][-1], ok.index[ok["long_break"] == 1][-1], ok.index[ok.index >= F.SESSION_CHANGE][0],
            ok.index[ok.index >= dt.date(2023, 3, 3)][0], ok.index[-1]]
    return sorted({*edge, *rng.choice(ok.index, N_RANDOM, replace=False)})


def same(x: pd.Series, y: pd.Series) -> pd.Series:
    x, y = x.astype(float), y.astype(float)
    return (np.isclose(x, y, rtol=0, atol=1e-12) | (x.isna() & y.isna()))


def cutoff_test() -> None:
    # known_at is stamped from the date label, so a mislabeled date hides from truncation (THB=X, M1 review):
    # every foreign daily bar must sit on a local weekday
    y = F.yahoo()
    assert (pd.to_datetime(y["date"]).dt.dayofweek < 5).all(), y[pd.to_datetime(y["date"]).dt.dayofweek >= 5].head()
    for sym in F.STOCKS:
        raw, days = F.raw_inputs(sym), F.calendar(sym)
        m, a = F.build(raw, days)
        checked = 0
        for d in sample_days(m):
            for rnd, full, cols, hidden in (("morning", m, ["C_prev", "D", *F.MORNING], "P"),
                                            ("afternoon", a, ["C_prev", "D", "P", *F.AFTERNOON], "C")):
                cut = pd.Timestamp(dt.datetime.combine(d, F.CUTOFF[rnd]), tz=F.BKK)
                tm, ta = F.build(F.truncate(raw, cut), days)
                t = tm if rnd == "morning" else ta
                bad = [c for c in cols if not same(full.loc[[d], c], t.loc[[d], c]).all()]
                assert not bad, f"{sym} {d} {rnd}: {bad} change when data after {cut} is dropped: " \
                                f"{full.loc[d, bad].to_dict()} vs {t.loc[d, bad].to_dict()}"
                assert pd.isna(t.loc[d, hidden]) and pd.notna(full.loc[d, hidden]), f"{sym} {d} {rnd}: truncation did not bite"
                if rnd == "afternoon":   # 1D-CNN minute sequences (M3) must not change either
                    k = days.index(d)
                    s_full = F.sequences(raw, days, a["C_prev"] - a["D"])[k]
                    s_cut = F.sequences(F.truncate(raw, cut), days, ta["C_prev"] - ta["D"])[k]
                    assert np.allclose(s_full, s_cut, equal_nan=True, rtol=0, atol=1e-12), f"{sym} {d}: sequence changes"
                    assert np.isfinite(s_full).all(), f"{sym} {d}: sequence has NaN on a traded day"
                checked += 1
        print(f"cutoff test {sym}: {checked} day-rounds, every feature equal after truncation")


def spec2_from_store() -> None:
    store = pd.read_parquet(F.STORE / "morning.parquet")
    base_cols, cols, train_from, score_from, min_rows, _, _ = apt.SPECS["A"]
    for sym in F.STOCKS:
        res = {}
        for name, g in (("atc_push_test", apt.table(sym)),
                        ("store", store[store["stock"] == sym].set_index("date").rename(columns={"y_m": "morning"}))):
            pb, _ = apt.walk(g, base_cols, train_from, score_from, min_rows)
            pm, _ = apt.walk(g, cols, train_from, score_from, min_rows)
            idx = pb.index.intersection(pm.index)
            t = g.loc[idx]
            anchor = t["C_prev"] - t["D"]
            eb, em = anchor * np.exp(pb[idx]) - t["P"], anchor * np.exp(pm[idx]) - t["P"]
            res[name] = (len(idx), 1 - (em**2).mean() / (eb**2).mean())
        print(f"spec #2 {sym}: " + " · ".join(f"{k} {n} days skill {s:+.4%}" for k, (n, s) in res.items()))
        assert abs(res["store"][1] - res["atc_push_test"][1]) < 0.0005, f"{sym}: store drifts from atc_push_test"


def coverage() -> None:
    for rnd in ("morning", "afternoon"):
        t = pd.read_parquet(F.STORE / f"{rnd}.parquet")
        has = {s: g.set_index("date").drop(columns="stock").notna() for s, g in t.groupby("stock")}
        first = pd.DataFrame({s: h.idxmax().where(h.any(), "never") for s, h in has.items()})
        print(f"\n{rnd}: first date with a value, per stock (only columns that differ)")
        print(first[first["KBANK"] != first["ADVANC"]].to_string())


if __name__ == "__main__":
    cutoff_test()
    spec2_from_store()
    coverage()
