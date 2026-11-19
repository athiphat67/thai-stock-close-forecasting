"""Build the v9 deploy feature store from newer exports, with the locked v9 feature code (features.build).

    .venv/bin/python -W ignore -m src.task3.deployment.v9.build_data \
        [--mt5 data/raw/mt5_export_20261021/1m] [--yahoo data/raw/yahoo_chart/1d/_v9_m5/task3-v9-m5-20261028]

Uses data_config.yaml (the v8 loader config with the date block opened, owner 2026-11-09). Writes
  data/final_data/task3-v9-deploy-data/1m/        KBANK/ADVANC from --mt5, the v8 peer files linked as they are
  data/final_data/task3-v9-deploy-data/official/  v5 official snapshot + daily bars built from MT5 1m after its
                                                  last date (source "mt5_1m_derived"; no SETSMART EOD after 09-04)
  data/final_data/task3-v9-deploy-store/          morning/afternoon parquet + manifest (never the locked v9 store)
Refuses if the derived daily bars stop matching the official ones on 2023+ (OHLC exact on >= 99% of days).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("TASK3_V8_CONFIG", "src/task3/deployment/v9/data_config.yaml")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.task3.versions8.pipeline import load  # noqa: E402
from src.task3.versions9 import features as F  # noqa: E402

V8_1M = load.ROOT / "data/final_data/master_dataset_1min_&_5min/1m"
V5_OFFICIAL = load.ROOT / "data/final_data/task3-v5-pit-snapshot"
OUT = load.ROOT / "data/final_data/task3-v9-deploy-data"
STORE = load.ROOT / "data/final_data/task3-v9-deploy-store"


def daily_from_1m(b: pd.DataFrame) -> pd.DataFrame:
    b = b[b["is_complete"]].sort_values("timestamp")
    g = b.groupby("trade_date")
    return pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(),
                         "close": g["close"].last(), "volume": g["volume"].sum()}).astype(float)


def link(src: Path, dst: Path) -> None:
    dst.unlink(missing_ok=True)
    dst.symlink_to(src.resolve())


def main(mt5: Path, yahoo: Path) -> dict:
    assert load.CONFIG.name == "data_config.yaml", f"wrong loader config {load.CONFIG}"
    (OUT / "1m").mkdir(parents=True, exist_ok=True)
    for p in (OUT / "1m").glob("*"):
        p.unlink()
    new = {s: next(mt5.glob(f"{s}_1M_*.parquet")) for s in F.STOCKS}
    for s, p in new.items():
        link(p, OUT / "1m" / p.name)
    for p in V8_1M.glob("*_1M_*.parquet"):
        if p.name.split("_")[0] not in F.STOCKS:
            link(p, OUT / "1m" / p.name)

    off = pd.read_parquet(V5_OFFICIAL / "sealed" / "official_daily.parquet")
    dev = pd.read_parquet(V5_OFFICIAL / "development" / "official_daily.parquet")
    both = pd.concat([dev, off])
    both["trade_date"] = pd.to_datetime(both["trade_date"]).dt.date
    added, report = [], {}
    for s, p in new.items():
        b = pd.read_parquet(p, columns=["timestamp", "trade_date", "is_complete", "open", "high", "low", "close", "volume"])
        b["trade_date"] = pd.to_datetime(b["trade_date"]).dt.date
        d = daily_from_1m(b)
        o = both[both["symbol"] == s].set_index("trade_date")[["open", "high", "low", "close", "volume"]].astype(float)
        j = d.join(o, rsuffix="_off", how="inner")
        j = j[j.index >= pd.Timestamp("2023-01-01").date()]
        match = float(np.mean([np.isclose(j[c], j[f"{c}_off"]).mean() for c in ("open", "high", "low", "close")]))
        vr = (j["volume"] / j["volume_off"])
        if match < 0.99:
            raise RuntimeError(f"{s}: 1m daily bars match official OHLC on only {match:.3f} of 2023+ days")
        last = o.index.max()
        ext = d[d.index > last].reset_index().rename(columns={"index": "trade_date"})
        ext["symbol"], ext["source"] = s, "mt5_1m_derived"
        added.append(ext)
        report[s] = {"official_last": str(last), "added_days": len(ext),
                     "added_first": str(ext["trade_date"].min()) if len(ext) else None,
                     "added_last": str(ext["trade_date"].max()) if len(ext) else None,
                     "check_2023_on": {"days": len(j), "ohlc_exact_share": round(match, 4),
                                       "volume_ratio_median": round(float(vr.median()), 4),
                                       "volume_within_1pct_share": round(float(((vr > .99) & (vr < 1.01)).mean()), 3)}}
    for part in ("development", "sealed"):
        (OUT / "official" / part).mkdir(parents=True, exist_ok=True)
    link(V5_OFFICIAL / "development" / "official_daily.parquet", OUT / "official" / "development" / "official_daily.parquet")
    cols = ["trade_date", "symbol", "source", *load.PRICE_COLS]     # the columns load_official reads
    sealed = pd.concat([off[cols].astype({c: float for c in load.PRICE_COLS}), *[a[cols] for a in added]], ignore_index=True)
    sealed["trade_date"] = pd.to_datetime(sealed["trade_date"])
    target = OUT / "official" / "sealed" / "official_daily.parquet"
    target.unlink(missing_ok=True)
    sealed.to_parquet(target, index=False)

    F.YAHOO, F.STORE = yahoo, STORE
    summary = F.write()
    # features.build averages whichever peers have data. On new rows (after the v5 official snapshot) that becomes a
    # 2-bank average, so blank it: peer_am needs day d, peer_prev needs d-1. Rows up to the snapshot stay as the
    # locked v9 store has them (KTB/SCB already end 2026-08-28 there).
    peer_end = max(min(load.load_1m(p)["trade_date"].max() for p in F.PEERS["KBANK"]),
                   max(pd.to_datetime(off["trade_date"]).dt.date))
    for rnd in ("morning", "afternoon"):
        t = pd.read_parquet(STORE / f"{rnd}.parquet").sort_values(["stock", "date"])
        prev = t.groupby("stock")["date"].shift(1)
        t.loc[prev > peer_end, "peer_prev"] = np.nan
        if "peer_am" in t:
            t.loc[t["date"] > peer_end, "peer_am"] = np.nan
        t.to_parquet(STORE / f"{rnd}.parquet", index=False)
    out = {"official_extension": report, "yahoo": str(yahoo.relative_to(load.ROOT)),
           "mt5": {s: str(p.relative_to(load.ROOT)) for s, p in new.items()}, "store": str(STORE.relative_to(load.ROOT)),
           "peer_features_blank_after": str(peer_end),
           "store_last_day": {k: v["last"] for k, v in summary.items()}}
    (OUT / "data_report.json").write_text(json.dumps(out, indent=1, default=str))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mt5", type=Path, default=load.ROOT / "data/raw/mt5_export_20261021/1m")
    ap.add_argument("--yahoo", type=Path, default=load.ROOT / "data/raw/yahoo_chart/1d/_v9_m5/task3-v9-m5-20261028")
    a = ap.parse_args()
    print(json.dumps(main(a.mt5.resolve(), a.yahoo.resolve()), indent=1, default=str))
