"""Task 3 v9 morning specs #2 (A) and #3 (B): is yesterday's close pushed off the "true" price by the auction?

Rules and keep criteria were committed before the run (EXPERIMENT_LOG 2026-10-27, commit 8db5665):
- A = spec #1 (5 foreign series) + atc_push, trained 2017-04 on, scored 2019-2022, vs spec #1.
- B = A + mid_gap, both B and its comparator A trained on rows with mid_gap (2021-01-05 on), scored from the
  first week with >= 120 training rows to 2022-12.
- OLS + constant per stock, refit weekly on rows d < first day of the week, features clipped at the 1%/99%
  quantiles of the training window. Keep if skill > 0 on both stocks, pooled one-sided CW p < 0.05 and the new
  coefficient has the mechanism's sign in >= 90% of weekly fits on both stocks.

    atc_push_d = log(C_{d-1} / last continuous 1m close of d-1)        (known after 16:40 on d-1)
    mid_gap_d  = log(((bid + offer)/2 of d-1, SETSMART) / C_{d-1})     (book left after the auction on d-1)

    .venv/bin/python src/task3/versions9/checks/atc_push_test.py
"""
import datetime as dt
import sys
from math import erf, sqrt
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import overnight_test as ot  # noqa: E402  (also puts versions8 on the path)
from pipeline import load  # noqa: E402

END = dt.date(2022, 12, 31)
FOREIGN = [*ot.CLOSE_SERIES, *ot.OPEN_SERIES]
SPECS = {  # name: (comparator cols, model cols, train_from, score_from, min_rows, new col, expected sign)
    "A": (FOREIGN, [*FOREIGN, "atc_push"], dt.date(2017, 1, 1), dt.date(2019, 1, 1), 0, "atc_push", -1),
    "B": ([*FOREIGN, "atc_push"], [*FOREIGN, "atc_push", "mid_gap"], dt.date(2021, 1, 1), dt.date(2021, 1, 1), 120,
          "mid_gap", +1),
}


def table(sym: str) -> pd.DataFrame:
    g = ot.gap_table(sym)
    g = g.join(ot.features(list(g.index)).reindex(g.index)[FOREIGN])
    b = load.load_1m(sym, end=END)
    cont = b[b["session"] == "afternoon"].groupby("trade_date")["close"].last()
    s = load.load_setsmart(sym, end=END).set_index("trade_date")
    f = load.load_official(sym, end=END).set_index("trade_date")[["close"]].rename(columns={"close": "C"})
    f["atc_push"] = np.log(f["C"] / cont.reindex(f.index))
    mid = ((s["bid"] + s["offer"]) / 2).where((s["bid"] > 0) & (s["offer"] > 0))
    f["mid_gap"] = np.log(mid.reindex(f.index) / f["C"])
    return g.join(f[["atc_push", "mid_gap"]].shift(1))   # value of the previous official trading day


def walk(g: pd.DataFrame, cols: list, train_from, score_from, min_rows: int):
    d = pd.to_datetime(pd.Index(g.index))
    week = d.to_period("W-SUN")
    rows = g[(g.index >= train_from)].dropna(subset=[*cols, "morning"])
    rd = pd.to_datetime(pd.Index(rows.index))
    preds, betas = {}, []
    for wk in sorted(set(week[(d >= pd.Timestamp(score_from)) & (g.index <= END)])):
        tr = rows[rd < wk.start_time]
        if len(tr) < max(min_rows, len(cols) + 2):
            continue
        lo, hi = tr[cols].quantile(.01), tr[cols].quantile(.99)
        A = np.column_stack([np.ones(len(tr)), tr[cols].clip(lo, hi, axis=1)])
        beta = np.linalg.lstsq(A, tr["morning"].to_numpy(), rcond=None)[0]
        te = g[(week == wk)].dropna(subset=cols)
        preds.update(zip(te.index, np.column_stack([np.ones(len(te)), te[cols].clip(lo, hi, axis=1)]) @ beta))
        betas.append(beta)
    return pd.Series(preds), np.array(betas)


def nw_p(x: np.ndarray, lag: int = 5) -> float:
    """One-sided p of mean(x) > 0 with a Newey-West variance."""
    n, u = len(x), x - x.mean()
    v = u @ u / n + 2 * sum((1 - k / (lag + 1)) * (u[k:] @ u[:-k]) / n for k in range(1, lag + 1))
    t = x.mean() / sqrt(v / n)
    return 0.5 * (1 - erf(t / sqrt(2)))


def main() -> None:
    tabs = {s: table(s) for s in ("KBANK", "ADVANC")}
    for name, (base_cols, cols, train_from, score_from, min_rows, new, sign) in SPECS.items():
        print(f"\n=== spec {name}: + {new} · train from {train_from} · score {score_from} -> {END}")
        cw = []
        for sym, g in tabs.items():
            pb, _ = walk(g, base_cols, train_from, score_from, min_rows)
            pm, bm = walk(g, cols, train_from, score_from, min_rows)
            idx = pb.index.intersection(pm.index)
            t = g.loc[idx]
            anchor = t["C_prev"] - t["D"]
            hat_b, hat_m = anchor * np.exp(pb[idx]), anchor * np.exp(pm[idx])
            eb, em, e0 = hat_b - t["P"], hat_m - t["P"], anchor - t["P"]
            k = cols.index(new) + 1
            right = float(np.mean(np.sign(bm[:, k]) == sign))
            print(f"{sym}: days {len(idx)} ({min(idx)} -> {max(idx)}) · weekly fits {len(bm)}"
                  f" · skill vs comparator {1 - (em**2).mean() / (eb**2).mean():+.4%}"
                  f" · vs B0 model {1 - (em**2).mean() / (e0**2).mean():+.4%} comparator {1 - (eb**2).mean() / (e0**2).mean():+.4%}"
                  f" · coef {new} last {bm[-1, k]:+.3f} median {np.median(bm[:, k]):+.3f} right sign {right:.0%}")
            cw.append(((eb**2 - em**2 + (hat_b - hat_m)**2) / (t["P"]**2).mean()).rename(sym))
        pooled = pd.concat(cw, axis=1).sum(axis=1, min_count=1).dropna()
        print(f"pooled CW one-sided p = {nw_p(pooled.to_numpy()):.4f} on {len(pooled)} days")
        if name == "B":   # context only: A trained from 2017 on the same scored days
            for sym, g in tabs.items():
                pa, _ = walk(g, cols[:-1], dt.date(2017, 1, 1), score_from, 0)
                pm, _ = walk(g, cols, train_from, score_from, min_rows)
                idx = pa.index.intersection(pm.index)
                t = g.loc[idx]
                anchor = t["C_prev"] - t["D"]
                ea, em = anchor * np.exp(pa[idx]) - t["P"], anchor * np.exp(pm[idx]) - t["P"]
                print(f"  context {sym}: B vs A trained from 2017 skill {1 - (em**2).mean() / (ea**2).mean():+.4%}")


if __name__ == "__main__":
    main()
