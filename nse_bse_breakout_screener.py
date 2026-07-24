#!/usr/bin/env python3
"""
NSE + BSE After-Market-Close Momentum Breakout Screener
========================================================

WHAT THIS DOES
--------------
Runs once a day, after the 3:30 PM IST close. It:
  1. Builds the full tradeable universe of NSE mainboard equities, plus
     any BSE-listed stock that is NOT also on NSE (so nothing is scanned twice).
  2. Pulls historical daily price/volume data for every symbol (cached
     locally so day-2 onward only the newest bar is fetched -- fast).
  3. Flags stocks CLOSING AT A TRUE ALL-TIME HIGH (highest close since
     listing, not just a 52-week high).
  4. Confirms the breakout is "high momentum" using today's % move and a
     volume surge vs the 20-day average, and scores volatility via ATR%.
  5. Renders everything into a single self-contained HTML dashboard file
     you open in a browser -- no server, no internet needed to view it.

DATA SOURCE / HONEST LIMITATIONS
---------------------------------
- No broker API is used (you said you don't have one). Price history
  comes from Yahoo Finance via the `yfinance` library. For Indian
  equities this is DELAYED data (roughly 15-20 min), which is exactly
  fine for an after-close scan, but it is NOT true real-time exchange
  tick data. If you ever get a broker API (Zerodha Kite Connect, Upstox,
  Angel One, Dhan), swap out `fetch_history()` for that feed and this
  same screening/dashboard logic keeps working unchanged.
- "True all-time high" is only as true as the history yfinance has for
  that ticker. Large/mid-caps and anything IPO'd in the yfinance era
  (roughly last ~20-25 years) will have full since-listing history.
  Some very old BSE-only small caps may have shorter backfilled history
  on Yahoo -- the dashboard footer flags this caveat.
- NSE and BSE do not publish a free live full-market API. The universe
  lists below use NSE's public archive CSV and BSE's scrip master; both
  are unofficial/subject to layout changes -- see the try/except
  fallbacks and the `--bse-file` manual override.

SETUP
-----
    pip install yfinance pandas numpy requests

USAGE
-----
    python nse_bse_breakout_screener.py                # full scan
    python nse_bse_breakout_screener.py --sample        # demo w/ fake data,
                                                         # no internet needed
    python nse_bse_breakout_screener.py --min-pct 5 --min-volx 2 --min-atr 3

SCHEDULING (run automatically after every close)
-------------------------------------------------
Recommended: GitHub Actions (free, cloud-hosted, nothing to keep running on
your own machine -- see SETUP_GUIDE.md and daily-scan.yml alongside this
file). It runs this script on GitHub's servers every weekday after close and
publishes the result to a permanent web address you just open in a browser.

If you'd rather run it on your own computer instead:
Linux/Mac cron (runs weekdays at 16:05 IST):
    5 16 * * 1-5 /usr/bin/python3 /path/to/nse_bse_breakout_screener.py

Windows: use Task Scheduler, trigger "Daily", time 16:05, weekdays only,
action = run this script with your python.exe.
"""

import argparse
import json
import os
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, date

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------ #
# CONFIG
# ------------------------------------------------------------------ #
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
# Fixed filename at the repo root -- GitHub Pages serves this exact file at
# your one permanent URL, overwritten fresh after every scan.
INDEX_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")

NSE_EQUITY_LIST_URL = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
# BSE's scrip master endpoint occasionally changes shape. If this fails,
# drop a manually-downloaded CSV (from bseindia.com > Markets > Equity >
# List of Securities) at BSE_FALLBACK_FILE and it will be used instead.
BSE_SCRIP_MASTER_URL = "https://api.bseindia.com/BseIndiaAPI/api/ListofScripCodes/w"
BSE_FALLBACK_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bse_list.csv")

BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}

# Screening thresholds (override via CLI flags)
DEFAULT_MIN_PCT_CHANGE = 3.0      # today's move must be at least +3%
DEFAULT_MIN_VOLUME_RATIO = 1.5    # volume must be >=1.5x the 20-day average
DEFAULT_MIN_ATR_PCT = 2.5         # ATR14 as % of price -- "highly volatile" floor
DEFAULT_MIN_PRICE = 10            # skip sub-Rs.10 stocks (noise/illiquid)

MAX_WORKERS = 12          # concurrent download threads
BATCH_SLEEP_SEC = 0.15    # small pause between requests to be polite to Yahoo


# ------------------------------------------------------------------ #
# UNIVERSE BUILDING
# ------------------------------------------------------------------ #
def get_nse_universe() -> pd.DataFrame:
    """Official NSE mainboard equity list: SYMBOL, ISIN, listing date."""
    resp = requests.get(NSE_EQUITY_LIST_URL, headers=BROWSER_HEADERS, timeout=20)
    resp.raise_for_status()
    df = pd.read_csv(pd.io.common.StringIO(resp.text))
    df.columns = [c.strip().upper() for c in df.columns]
    df = df[df["SERIES"].str.strip() == "EQ"].copy()
    df["TICKER"] = df["SYMBOL"].str.strip() + ".NS"
    df["EXCHANGE"] = "NSE"
    df["ISIN"] = df["ISIN NUMBER"].str.strip()
    df["LISTING_DATE"] = pd.to_datetime(df["DATE OF LISTING"], errors="coerce")
    df["NAME"] = df["NAME OF COMPANY"].str.strip()
    return df[["TICKER", "SYMBOL", "NAME", "EXCHANGE", "ISIN", "LISTING_DATE"]]


def get_bse_universe() -> pd.DataFrame:
    """BSE scrip master. Falls back to a manual CSV if the API shape changed."""
    try:
        resp = requests.get(BSE_SCRIP_MASTER_URL, headers=BROWSER_HEADERS, timeout=20)
        resp.raise_for_status()
        raw = resp.json()
        df = pd.DataFrame(raw)
        df.columns = [c.strip().upper() for c in df.columns]
    except Exception as e:
        if os.path.exists(BSE_FALLBACK_FILE):
            print(f"[warn] BSE API failed ({e}); using local {BSE_FALLBACK_FILE}")
            df = pd.read_csv(BSE_FALLBACK_FILE)
            df.columns = [c.strip().upper() for c in df.columns]
        else:
            print(f"[warn] BSE universe unavailable ({e}); continuing NSE-only.")
            return pd.DataFrame(columns=["TICKER", "SYMBOL", "NAME", "EXCHANGE", "ISIN", "LISTING_DATE"])

    # Column names on BSE's feed vary; grab defensively.
    code_col = next((c for c in df.columns if "CODE" in c or "SCRIP" in c), None)
    name_col = next((c for c in df.columns if "NAME" in c), None)
    isin_col = next((c for c in df.columns if "ISIN" in c), None)
    if not code_col or not name_col:
        print("[warn] Unrecognized BSE file layout; continuing NSE-only.")
        return pd.DataFrame(columns=["TICKER", "SYMBOL", "NAME", "EXCHANGE", "ISIN", "LISTING_DATE"])

    df["TICKER"] = df[code_col].astype(str).str.strip() + ".BO"
    df["SYMBOL"] = df[code_col].astype(str).str.strip()
    df["NAME"] = df[name_col].astype(str).str.strip()
    df["EXCHANGE"] = "BSE"
    df["ISIN"] = df[isin_col].astype(str).str.strip() if isin_col else ""
    df["LISTING_DATE"] = pd.NaT
    return df[["TICKER", "SYMBOL", "NAME", "EXCHANGE", "ISIN", "LISTING_DATE"]]


def build_universe() -> pd.DataFrame:
    """NSE list + BSE-EXCLUSIVE names only (deduped by ISIN so nothing scans twice)."""
    nse_df = get_nse_universe()
    bse_df = get_bse_universe()
    if not bse_df.empty and "ISIN" in nse_df.columns:
        bse_only = bse_df[~bse_df["ISIN"].isin(nse_df["ISIN"])]
    else:
        bse_only = bse_df
    universe = pd.concat([nse_df, bse_only], ignore_index=True)
    universe = universe.drop_duplicates(subset=["TICKER"])
    print(f"[info] Universe built: {len(nse_df)} NSE + {len(bse_only)} BSE-exclusive = {len(universe)} total")
    return universe


# ------------------------------------------------------------------ #
# HISTORY FETCH + CACHE
# ------------------------------------------------------------------ #
def _cache_path(ticker: str) -> str:
    safe = ticker.replace(".", "_")
    return os.path.join(CACHE_DIR, f"{safe}.csv")


def fetch_history(ticker: str) -> pd.DataFrame:
    """Full history on first run; incremental (only new days) after that."""
    import yfinance as yf

    path = _cache_path(ticker)
    if os.path.exists(path):
        cached = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
        last_date = cached.index.max()
        start = (last_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        new = yf.Ticker(ticker).history(start=start, interval="1d", auto_adjust=False)
        if not new.empty:
            new.index = new.index.tz_localize(None)
            combined = pd.concat([cached, new[~new.index.isin(cached.index)]])
            combined = combined[~combined.index.duplicated(keep="last")].sort_index()
        else:
            combined = cached
    else:
        combined = yf.Ticker(ticker).history(period="max", interval="1d", auto_adjust=False)
        if combined.empty:
            return combined
        combined.index = combined.index.tz_localize(None)

    combined.to_csv(path, index_label="Date")
    return combined


# ------------------------------------------------------------------ #
# METRICS
# ------------------------------------------------------------------ #
def compute_metrics(df: pd.DataFrame) -> dict | None:
    if df is None or df.empty or len(df) < 25:
        return None

    df = df.sort_index()
    last = df.iloc[-1]
    prev = df.iloc[-2]

    prior_history = df.iloc[:-1]
    all_time_high = prior_history["High"].max()

    close = last["Close"]
    prev_close = prev["Close"]
    pct_change = (close - prev_close) / prev_close * 100 if prev_close else np.nan

    vol_avg_20 = df["Volume"].iloc[-21:-1].mean()
    volume_ratio = last["Volume"] / vol_avg_20 if vol_avg_20 else np.nan

    # ATR(14)
    high, low, prev_close_series = df["High"], df["Low"], df["Close"].shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close_series).abs(),
        (low - prev_close_series).abs(),
    ], axis=1).max(axis=1)
    atr14 = tr.rolling(14).mean().iloc[-1]
    atr_pct = atr14 / close * 100 if close else np.nan

    is_new_ath = bool(close >= all_time_high)
    listing_date = df.index.min()
    days_listed = (df.index.max() - listing_date).days

    momentum_score = pct_change * min(volume_ratio, 5) if pd.notna(volume_ratio) else np.nan

    return dict(
        last_close=round(float(close), 2),
        pct_change=round(float(pct_change), 2),
        volume_ratio=round(float(volume_ratio), 2) if pd.notna(volume_ratio) else None,
        atr_pct=round(float(atr_pct), 2) if pd.notna(atr_pct) else None,
        is_new_ath=is_new_ath,
        prior_ath=round(float(all_time_high), 2),
        listing_date=listing_date.strftime("%Y-%m-%d"),
        days_listed=int(days_listed),
        momentum_score=round(float(momentum_score), 2) if pd.notna(momentum_score) else None,
    )


# ------------------------------------------------------------------ #
# SCREENING
# ------------------------------------------------------------------ #
def screen_one(row) -> dict | None:
    try:
        hist = fetch_history(row["TICKER"])
        time.sleep(BATCH_SLEEP_SEC)
        m = compute_metrics(hist)
        if not m:
            return None
        m.update(symbol=row["SYMBOL"], name=row["NAME"], exchange=row["EXCHANGE"], ticker=row["TICKER"])
        return m
    except Exception:
        return None


def screen_universe(universe: pd.DataFrame, min_price: float) -> list[dict]:
    results = []
    rows = universe.to_dict("records")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(screen_one, row): row for row in rows}
        done = 0
        for fut in as_completed(futures):
            done += 1
            if done % 200 == 0:
                print(f"[info] scanned {done}/{len(rows)}")
            r = fut.result()
            if r and r["last_close"] >= min_price:
                results.append(r)
    return results


def filter_and_rank(results: list[dict], min_pct: float, min_volx: float, min_atr: float) -> list[dict]:
    hits = [
        r for r in results
        if r["is_new_ath"]
        and r["pct_change"] is not None and r["pct_change"] >= min_pct
        and r["volume_ratio"] is not None and r["volume_ratio"] >= min_volx
        and r["atr_pct"] is not None and r["atr_pct"] >= min_atr
    ]
    hits.sort(key=lambda r: r["momentum_score"] or 0, reverse=True)
    return hits


# ------------------------------------------------------------------ #
# SAMPLE DATA (for --sample / no-internet demo runs)
# ------------------------------------------------------------------ #
def sample_results() -> list[dict]:
    names = [
        ("KIRLOSKIND", "Kirloskar Industries", "NSE", 2011),
        ("RALLIS", "Rallis India", "NSE", 1962),
        ("ZFCVINDIA", "ZF Commercial Vehicle Control Systems", "NSE", 2005),
        ("GRWRHITECH", "Garware Hi-Tech Films", "NSE", 1990),
        ("KAYNES", "Kaynes Technology", "NSE", 2022),
        ("JYOTHYLAB", "Jyothy Labs", "NSE", 2007),
        ("TIMETECHNO", "Time Technoplast", "NSE", 2004),
        ("SHAKTIPUMP", "Shakti Pumps", "BSE", 2008),
        ("WABAG", "VA Tech Wabag", "NSE", 2010),
        ("ELECON", "Elecon Engineering", "NSE", 1995),
    ]
    rng = np.random.default_rng(42)
    out = []
    for sym, name, exch, year in names:
        pct = round(float(rng.uniform(2.5, 14)), 2)
        volx = round(float(rng.uniform(1.4, 6.5)), 2)
        atrp = round(float(rng.uniform(2.2, 7.5)), 2)
        close = round(float(rng.uniform(120, 3200)), 2)
        out.append(dict(
            symbol=sym, name=name, exchange=exch, ticker=f"{sym}.NS",
            last_close=close, pct_change=pct, volume_ratio=volx, atr_pct=atrp,
            is_new_ath=True, prior_ath=round(close / (1 + pct / 100), 2),
            listing_date=f"{year}-01-01",
            days_listed=(date.today() - date(year, 1, 1)).days,
            momentum_score=round(pct * min(volx, 5), 2),
        ))
    out.sort(key=lambda r: r["momentum_score"], reverse=True)
    return out


# ------------------------------------------------------------------ #
# DASHBOARD RENDER
# ------------------------------------------------------------------ #
def render_dashboard(hits: list[dict], scanned_count: int, output_path: str,
                      min_pct: float, min_volx: float, min_atr: float) -> None:
    now = datetime.now()
    payload = {
        "generated_at": now.strftime("%a, %d %b %Y \u00b7 %H:%M IST"),
        "scanned": scanned_count,
        "hits": hits,
        "thresholds": {"min_pct": min_pct, "min_volx": min_volx, "min_atr": min_atr},
    }
    html = _DASHBOARD_TEMPLATE.replace("__DATA_JSON__", json.dumps(payload))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[info] Dashboard written -> {output_path}")


_DASHBOARD_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<title>Dalal Street Breakout Desk</title>
<meta name="viewport" content="width=device-width, initial-scale=1" />
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600;9..144,700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
  :root{
    --ink:#0B1220; --panel:#121A2B; --panel-2:#0F1728; --hair:#26324A;
    --paper:#EDEAE1; --paper-dim:#A9B1C3; --brass:#C89B3C; --brass-dim:#8C6E2E;
    --up:#3FA34D; --down:#C4463A;
  }
  *{box-sizing:border-box;}
  body{margin:0;background:var(--ink);color:var(--paper);font-family:'IBM Plex Sans',sans-serif;}
  .masthead{background:linear-gradient(180deg,#141D30,#0F1728);border-bottom:1px solid var(--hair);padding:22px 20px 0;}
  .masthead h1{font-family:'Fraunces',serif;font-weight:700;font-size:clamp(22px,4vw,34px);margin:0;letter-spacing:.3px;}
  .masthead .sub{color:var(--paper-dim);font-size:13px;margin:4px 0 18px;font-family:'IBM Plex Mono',monospace;}
  .tape{overflow:hidden;white-space:nowrap;background:repeating-linear-gradient(90deg,var(--brass) 0 2px,transparent 2px 10px),var(--panel-2);
        border-top:1px dashed var(--brass-dim);border-bottom:1px dashed var(--brass-dim);padding:8px 0;}
  .tape span{display:inline-block;padding-left:100%;animation:scroll 32s linear infinite;font-family:'IBM Plex Mono',monospace;font-size:13px;}
  .tape b.up{color:var(--up);} .tape b.down{color:var(--down);}
  @keyframes scroll{0%{transform:translateX(0);}100%{transform:translateX(-100%);}}
  .stats{display:flex;flex-wrap:wrap;gap:12px;padding:18px 20px;}
  .stat{background:var(--panel);border:1px solid var(--hair);border-left:3px solid var(--brass);border-radius:4px;padding:12px 16px;min-width:150px;flex:1;}
  .stat .n{font-family:'Fraunces',serif;font-size:26px;font-weight:600;}
  .stat .l{color:var(--paper-dim);font-size:11px;text-transform:uppercase;letter-spacing:.08em;margin-top:2px;}
  .controls{display:flex;flex-wrap:wrap;gap:10px;padding:0 20px 16px;}
  .controls input, .controls select{background:var(--panel);border:1px solid var(--hair);color:var(--paper);
        padding:9px 12px;border-radius:4px;font-family:'IBM Plex Mono',monospace;font-size:13px;}
  .controls input[type=text]{flex:1;min-width:160px;}
  table{width:100%;border-collapse:collapse;font-family:'IBM Plex Mono',monospace;font-size:13px;}
  thead th{text-align:left;color:var(--brass);border-bottom:1px solid var(--hair);padding:10px 14px;
        cursor:pointer;white-space:nowrap;font-weight:600;text-transform:uppercase;font-size:11px;letter-spacing:.06em;}
  tbody td{padding:10px 14px;border-bottom:1px solid var(--hair);}
  tbody tr:hover{background:var(--panel-2);}
  tbody tr.ath td:first-child{box-shadow:inset 3px 0 0 var(--brass);}
  .badge{background:var(--brass);color:#231A08;font-size:10px;font-weight:700;padding:2px 7px;border-radius:10px;letter-spacing:.04em;}
  .up{color:var(--up);} .down{color:var(--down);}
  .wrap{max-width:1080px;margin:0 auto;padding-bottom:60px;}
  .foot{color:var(--paper-dim);font-size:11.5px;padding:22px 20px;border-top:1px solid var(--hair);line-height:1.6;}
  .empty{padding:40px 20px;color:var(--paper-dim);font-family:'IBM Plex Mono',monospace;}
  @media(max-width:640px){ table thead{display:none;} tbody tr{display:block;border-bottom:2px solid var(--hair);padding:8px 0;}
    tbody td{display:flex;justify-content:space-between;border-bottom:none;padding:4px 14px;}
    tbody td::before{content:attr(data-label);color:var(--paper-dim);font-size:11px;} }
</style>
</head>
<body>
<div class="wrap">
  <div class="masthead">
    <h1>Dalal Street Breakout Desk</h1>
    <div class="sub" id="genAt">generated --</div>
  </div>
  <div class="tape"><span id="tapeContent">loading tape...</span></div>
  <div class="stats" id="statCards"></div>
  <div class="controls">
    <input type="text" id="search" placeholder="Search symbol or name...">
    <select id="exFilter"><option value="ALL">All exchanges</option><option value="NSE">NSE</option><option value="BSE">BSE</option></select>
    <input type="number" id="minPct" placeholder="Min % chg">
    <input type="number" id="minVolx" placeholder="Min vol x">
  </div>
  <table>
    <thead><tr>
      <th data-k="symbol">Symbol</th><th data-k="exchange">Exch</th><th data-k="last_close">LTP</th>
      <th data-k="pct_change">%Chg</th><th data-k="volume_ratio">Vol x</th><th data-k="atr_pct">ATR%</th>
      <th data-k="listing_date">Listed</th><th data-k="momentum_score">Score</th><th>ATH</th>
    </tr></thead>
    <tbody id="rows"></tbody>
  </table>
  <div class="empty" id="emptyMsg" style="display:none;">No breakout stocks matched today's thresholds.</div>
  <div class="foot">
    Data: Yahoo Finance (delayed ~15-20 min), no broker API. "All-time high" is bounded by
    the price history available for each ticker -- long-listed BSE-only small caps may have
    shorter backfilled history. This is a scan output, not investment advice.
  </div>
</div>
<script>
const DATA = __DATA_JSON__;
document.getElementById('genAt').textContent = 'Scan run: ' + DATA.generated_at;

const cards = [
  {n: DATA.scanned, l:'Stocks scanned'},
  {n: DATA.hits.length, l:'New ATH breakouts'},
  {n: (DATA.hits.reduce((a,h)=>a+h.pct_change,0)/(DATA.hits.length||1)).toFixed(1)+'%', l:'Avg move'},
  {n: 'min '+DATA.thresholds.min_pct+'% / '+DATA.thresholds.min_volx+'x / ATR '+DATA.thresholds.min_atr+'%', l:'Active filters'},
];
document.getElementById('statCards').innerHTML = cards.map(c=>`<div class="stat"><div class="n">${c.n}</div><div class="l">${c.l}</div></div>`).join('');

const tapeItems = DATA.hits.slice(0,15).map(h=>`${h.symbol} <b class="${h.pct_change>=0?'up':'down'}">${h.pct_change>=0?'+':''}${h.pct_change}%</b>`);
document.getElementById('tapeContent').innerHTML = tapeItems.length ? tapeItems.join(' &nbsp;&bull;&nbsp; ') : 'No breakout candidates today.';

let sortKey = 'momentum_score', sortDir = -1;
function renderRows(){
  const q = document.getElementById('search').value.toLowerCase();
  const ex = document.getElementById('exFilter').value;
  const minPct = parseFloat(document.getElementById('minPct').value) || -Infinity;
  const minVolx = parseFloat(document.getElementById('minVolx').value) || -Infinity;

  let rows = DATA.hits.filter(h =>
    (h.symbol.toLowerCase().includes(q) || h.name.toLowerCase().includes(q)) &&
    (ex === 'ALL' || h.exchange === ex) &&
    h.pct_change >= minPct && h.volume_ratio >= minVolx
  );
  rows.sort((a,b)=> (a[sortKey]>b[sortKey]?1:-1) * sortDir);

  document.getElementById('emptyMsg').style.display = rows.length ? 'none' : 'block';
  document.getElementById('rows').innerHTML = rows.map(h => `
    <tr class="ath">
      <td data-label="Symbol"><b>${h.symbol}</b><br><span style="color:var(--paper-dim);font-size:11px;">${h.name}</span></td>
      <td data-label="Exch">${h.exchange}</td>
      <td data-label="LTP">\u20b9${h.last_close}</td>
      <td data-label="%Chg" class="${h.pct_change>=0?'up':'down'}">${h.pct_change>=0?'+':''}${h.pct_change}%</td>
      <td data-label="Vol x">${h.volume_ratio}x</td>
      <td data-label="ATR%">${h.atr_pct}%</td>
      <td data-label="Listed">${h.listing_date.slice(0,4)}</td>
      <td data-label="Score">${h.momentum_score}</td>
      <td data-label="ATH"><span class="badge">ATH</span></td>
    </tr>`).join('');
}
document.querySelectorAll('thead th[data-k]').forEach(th=>{
  th.addEventListener('click', ()=>{
    const k = th.dataset.k;
    sortDir = (sortKey === k) ? sortDir * -1 : -1;
    sortKey = k;
    renderRows();
  });
});
['search','exFilter','minPct','minVolx'].forEach(id=>document.getElementById(id).addEventListener('input', renderRows));
renderRows();
</script>
</body>
</html>
"""


# ------------------------------------------------------------------ #
# MAIN
# ------------------------------------------------------------------ #
def main():
    ap = argparse.ArgumentParser(description="NSE+BSE after-close ATH breakout screener")
    ap.add_argument("--sample", action="store_true", help="Use built-in sample data, no internet required")
    ap.add_argument("--min-pct", type=float, default=DEFAULT_MIN_PCT_CHANGE)
    ap.add_argument("--min-volx", type=float, default=DEFAULT_MIN_VOLUME_RATIO)
    ap.add_argument("--min-atr", type=float, default=DEFAULT_MIN_ATR_PCT)
    ap.add_argument("--min-price", type=float, default=DEFAULT_MIN_PRICE)
    args = ap.parse_args()

    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f"dashboard_{date.today().isoformat()}.html")

    if args.sample:
        hits = sample_results()
        scanned = 2143
    else:
        universe = build_universe()
        results = screen_universe(universe, args.min_price)
        hits = filter_and_rank(results, args.min_pct, args.min_volx, args.min_atr)
        scanned = len(results)

    render_dashboard(hits, scanned, out_path, args.min_pct, args.min_volx, args.min_atr)
    render_dashboard(hits, scanned, INDEX_PATH, args.min_pct, args.min_volx, args.min_atr)
    print(f"[info] Published to {INDEX_PATH} (this is the file GitHub Pages serves)")

    csv_path = os.path.join(OUTPUT_DIR, f"breakouts_{date.today().isoformat()}.csv")
    pd.DataFrame(hits).to_csv(csv_path, index=False)
    print(f"[info] CSV written -> {csv_path}")
    print(f"[info] {len(hits)} breakout(s) found out of {scanned} scanned.")


if __name__ == "__main__":
    main()
