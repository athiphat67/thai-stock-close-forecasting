"""Free live inputs for live.py from Yahoo chart data: no S50 futures, no bank peers (v9.2 morning needs neither).

    python fetch_yahoo.py --out DIR          # writes DIR/daily.csv, bars_1m.csv, yahoo.csv, xd.csv
    python live.py --inputs DIR --round morning --date YYYY-MM-DD --log predictions.jsonl

daily.csv   KBANK.BK / ADVANC.BK daily bars (Yahoo close = official close; the newest day, if Yahoo has not filled its
            close yet, takes meta.regularMarketPrice when regularMarketTime is after the 16:30 ATC that day)
bars_1m.csv the same stocks' 1m bars for the last 7 days (Yahoo's limit), 5m bars for the 60 days before that,
            labelled by END time (Yahoo labels by start)
yahoo.csv   ^GSPC, THD, THB=X, ^N225, ^HSI daily open/close, dated in each exchange's own time zone
xd.csv      Yahoo dividend events (announce time unknown -> live.py treats it as known 17:00 the day before XD)
Rows after the cutoff are dropped by live.py itself, so fetching more than needed cannot leak.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import urllib.request
from pathlib import Path

import pandas as pd

BKK = "Asia/Bangkok"
STOCKS = {"KBANK": "KBANK.BK", "ADVANC": "ADVANC.BK"}
FOREIGN = {"spx": "^GSPC", "thd": "THD", "thb": "THB=X", "n225": "^N225", "hsi": "^HSI"}


def chart(sym: str, interval: str, rng: str) -> dict:
    u = f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}?interval={interval}&range={rng}&events=div"
    req = urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0"})
    return json.load(urllib.request.urlopen(req, timeout=30))["chart"]["result"][0]


def frame(r: dict) -> pd.DataFrame:
    q = r["indicators"]["quote"][0]
    t = pd.to_datetime(r["timestamp"], unit="s", utc=True).tz_convert(r["meta"]["exchangeTimezoneName"])
    return pd.DataFrame({"t": t, **{c: q[c] for c in ("open", "high", "low", "close", "volume")}})


def main(out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    daily, m1, xd = [], [], []
    for s, y in STOCKS.items():
        r = chart(y, "1d", "6mo")
        d = frame(r).assign(symbol=s)
        d["trade_date"] = d["t"].dt.date
        meta = r["meta"]
        last = pd.Timestamp(meta["regularMarketTime"], unit="s", tz="UTC").tz_convert(BKK)
        i = d.index[-1]
        if pd.isna(d.at[i, "close"]) and last.date() == d.at[i, "trade_date"] and last.time() >= dt.time(16, 30):   # ATC match 16:35-16:40
            d.at[i, "close"] = meta["regularMarketPrice"]
        daily.append(d.dropna(subset=["open", "high", "low", "close", "volume"]))
        for e in (r.get("events") or {}).get("dividends", {}).values():
            xd.append({"symbol": s, "xd_date": pd.Timestamp(e["date"], unit="s", tz="UTC").tz_convert(BKK).date(),
                       "dividend_thb": e["amount"], "announce_datetime": None})
        b = frame(chart(y, "1m", "7d")).dropna(subset=["close"]).assign(symbol=s, step=1)
        # ponytail: Yahoo keeps 1m for 7 days only; older days come as 5m bars so the afternoon `vol_am` (20-morning
        # volume mean) exists. Yahoo's 5m morning volume runs 0.73-1.09x its own 1m sum, so vol_am is approximate.
        old = frame(chart(y, "5m", "60d")).dropna(subset=["close"]).assign(symbol=s, step=5)
        b = pd.concat([old[old["t"].dt.date < b["t"].dt.date.min()], b])
        b["bar_end"] = (b["t"] + pd.to_timedelta(b["step"], unit="min")).map(lambda x: x.isoformat())
        m1.append(b)
    pd.concat(daily)[["symbol", "trade_date", "open", "high", "low", "close", "volume"]].to_csv(out / "daily.csv", index=False)
    pd.concat(m1)[["symbol", "bar_end", "open", "high", "low", "close", "volume"]].to_csv(out / "bars_1m.csv", index=False)
    pd.DataFrame(xd, columns=["symbol", "xd_date", "dividend_thb", "announce_datetime"]).to_csv(out / "xd.csv", index=False)
    yrows = []
    for name, sym in FOREIGN.items():
        f = frame(chart(sym, "1d", "3mo"))
        yrows.append(pd.DataFrame({"name": name, "date": f["t"].dt.date, "open": f["open"], "close": f["close"]})
                     .drop_duplicates("date", keep="last"))
    pd.concat(yrows).to_csv(out / "yahoo.csv", index=False)
    return {"out": str(out), "daily_last": str(max(d["trade_date"].max() for d in daily)),
            "bar_days": len({x[:10] for x in pd.concat(m1)["bar_end"]}), "xd_rows": len(xd)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    print(json.dumps(main(ap.parse_args().out), indent=1))
