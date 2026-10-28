#!/usr/bin/env python3
"""
flow_scan.py - one-table forced order flow scan.

Run it twice a day:
  * before the open   -> forced flow from options hedging through today's close
  * ~30-60 min before the close -> adds leveraged-ETF rebalancing based on today's move

    python flow_scan.py

Each ticker gets a BUY / SELL rating, the estimated forced flow in dollars, that flow as a
% of a normal day's volume, and a rough estimated price push.

Tickers come from tickers.txt next to this script ([index] and [stocks] sections).
Requires: pip install numpy pandas scipy yfinance

These are estimates from public data and assumptions about who holds the options.
Not predictions, not investment advice.
"""
from __future__ import annotations

import argparse
import datetime as dt
import math
import os
import re
import sys
import time
import traceback
import zlib
from zoneinfo import ZoneInfo

try:
    import numpy as np
    import pandas as pd
    from scipy.stats import norm
except ImportError as _e:
    sys.exit(f"Missing library: {_e.name}\nInstall with:  pip install numpy pandas scipy yfinance")

VERSION = "flow_scan 2026-10-27f"
ET = ZoneInfo("America/New_York")
YEAR = 365.0 * 86400.0
HERE = os.path.dirname(os.path.abspath(__file__))
TICKERS_FILE = os.path.join(HERE, "tickers.txt")
CACHE_DIR = os.path.join(HERE, "oi_saved")  # last good open interest per ticker (overwritten each time)

# =============================================================================
# SETTINGS
# =============================================================================
SETTINGS = {
    # Used only if tickers.txt doesn't exist yet (it is created from these on first run).
    "index": ["SPY", "QQQ", "IWM", "DIA", "SMH", "SOXX", "DRAM", "IGV", "XLK", "XLE", "XLF",
              "XLV", "XLI", "XLY", "XLU", "GLD", "GDX"],
    "stocks": ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "NFLX", "BRK-B",
               "JPM", "LLY", "COST", "AMD", "MU", "TSM", "ARM", "MRVL", "ALAB", "CRDO", "SMCI",
               "DELL", "ANET", "AAOI", "COHR", "LITE", "IREN", "CRWV", "NBIS", "VRT", "CIFR",
               "APLD", "PLTR", "ORCL", "CRM", "NOW", "SNOW", "PATH", "SOFI", "HOOD", "XOM",
               "RKLB", "BE"],

    # Leveraged / inverse ETFs by the ticker they track (leverage: 3 = 3x long, -3 = 3x inverse).
    "leveraged_etfs": {
        "SPY": {"SPXL": 3, "UPRO": 3, "SPXS": -3, "SPXU": -3, "SSO": 2, "SDS": -2, "SH": -1},
        "QQQ": {"TQQQ": 3, "SQQQ": -3, "QLD": 2, "QID": -2, "PSQ": -1},
        "IWM": {"TNA": 3, "TZA": -3, "UWM": 2, "TWM": -2},
        "NVDA": {"NVDL": 2},
        "TSLA": {"TSLL": 2},
        "MSTR": {"MSTU": 2},
    },

    # Rating thresholds: net forced flow as % of the ticker's normal daily $ volume.
    "lean_pct": 1.0,    # >= 1%  -> LEAN BUY / LEAN SELL  (+ / -)
    "rating_pct": 3.0,  # >= 3%  -> BUY / SELL            (++ / --)
    "strong_pct": 8.0,  # >= 8%  -> STRONG BUY / SELL     (+++ / ---)

    # Assumed dealer positioning:
    #   index ETFs: customers buy puts and sell calls (dealers short puts, long calls)
    #   stocks:     customers buy calls and puts (dealers short both)
    "max_expiry_days": 45,
    "strike_window": 0.20,
    "risk_free": 0.04,
    "request_pause": 0.4,   # seconds between tickers, to avoid Yahoo rate limits
}

# Tickers treated as index/sector ETFs (customers mostly buy puts, sell calls) when a plain
# watchlist is passed in. Everything else is treated as a single stock.
KNOWN_ETFS = {
    "SPY", "QQQ", "IWM", "DIA", "VOO", "IVV", "VTI", "RSP", "MDY", "SMH", "SOXX", "DRAM", "IGV",
    "XLK", "XLE", "XLF", "XLV", "XLI", "XLY", "XLU", "XLP", "XLB", "XLRE", "XLC", "KRE", "KBE",
    "XBI", "IBB", "XHB", "ITB", "XRT", "XME", "XOP", "OIH", "GLD", "SLV", "GDX", "GDXJ", "USO",
    "UNG", "TLT", "IEF", "HYG", "LQD", "EEM", "EFA", "FXI", "KWEB", "EWZ", "EWJ", "ARKK",
    "IBIT", "ETHA", "BITO", "URA", "TAN", "ICLN", "JETS", "VNQ", "SOXL", "TQQQ", "SQQQ",
}

HOLIDAYS = {dt.date.fromisoformat(d) for d in [
    "2026-01-01", "2026-01-19", "2026-02-16", "2026-04-03", "2026-05-25", "2026-06-19",
    "2026-07-03", "2026-09-07", "2026-11-26", "2026-12-25",
    "2027-01-01", "2027-01-18", "2027-02-15", "2027-03-26", "2027-05-31", "2027-06-18",
    "2027-07-05", "2027-09-06", "2027-11-25", "2027-12-24"]}
EARLY_CLOSE = {dt.date.fromisoformat(d) for d in ["2026-11-27", "2026-12-24", "2027-11-26"]}


# =============================================================================
# Calendar
# =============================================================================
def is_trading_day(d): return d.weekday() < 5 and d not in HOLIDAYS


def close_time(d):
    return dt.datetime.combine(d, dt.time(13, 0) if d in EARLY_CLOSE else dt.time(16, 0), ET)


def open_time(d): return dt.datetime.combine(d, dt.time(9, 30), ET)


def next_close(now):
    d = now.date()
    while not (is_trading_day(d) and close_time(d) > now):
        d += dt.timedelta(days=1)
    return close_time(d)


def session(now):
    d = now.date()
    if not is_trading_day(d):
        return "closed"
    if now < open_time(d):
        return "pre"
    return "open" if now < close_time(d) else "after"


def back_to_trading(d):
    while not is_trading_day(d):
        d -= dt.timedelta(days=1)
    return d


def nth_friday(y, m, n):
    first = dt.date(y, m, 1)
    return first + dt.timedelta(days=(4 - first.weekday()) % 7 + 7 * (n - 1))


def month_end(y, m):
    return back_to_trading(dt.date(y + (m == 12), m % 12 + 1, 1) - dt.timedelta(days=1))


def upcoming_events(today, days=3):
    ev = []
    for k in range(2):
        y, m = today.year + (today.month + k - 1) // 12, (today.month + k - 1) % 12 + 1
        q = m in (3, 6, 9, 12)
        ev.append((back_to_trading(nth_friday(y, m, 3)), "Triple witching (quarterly OPEX)" if q else "Monthly options expiration"))
        if q:
            ev.append((back_to_trading(nth_friday(y, m, 3)), "S&P 500 rebalance at the close"))
        if m == 12:
            ev.append((back_to_trading(nth_friday(y, m, 3)), "Nasdaq-100 reconstitution at the close"))
        if m in (6, 12):
            ev.append((back_to_trading(nth_friday(y, m, 4 if m == 6 else 2)), "Russell reconstitution at the close"))
        ev.append((month_end(y, m), "Quarter-end" if q else "Month-end"))
    out = []
    for d, name in sorted(set(ev)):
        gap = (d - today).days
        if 0 <= gap <= days:
            out.append(f"{name} {'TODAY' if gap == 0 else 'tomorrow' if gap == 1 else d.strftime('%a %b %d')}")
    return out


LOCAL_TZ = ZoneInfo(os.environ["FLOW_LOCAL_TZ"]) if os.environ.get("FLOW_LOCAL_TZ") else None


def local(t): return t.astimezone(LOCAL_TZ).strftime("%I:%M %p").lstrip("0")


# =============================================================================
# Tickers file
# =============================================================================
def load_tickers():
    if not os.path.exists(TICKERS_FILE):
        with open(TICKERS_FILE, "w") as f:
            f.write("# One or more tickers per line; '#' starts a comment.\n"
                    "# [index] = index/sector/commodity ETFs, [stocks] = single stocks\n\n[index]\n"
                    + "\n".join(SETTINGS["index"]) + "\n\n[stocks]\n" + "\n".join(SETTINGS["stocks"]) + "\n")
        print(f"Created {TICKERS_FILE} - edit it to change your list.")
    secs, cur = {"index": [], "stocks": []}, None
    alias = {"index": "index", "indexes": "index", "etf": "index", "etfs": "index",
             "stock": "stocks", "stocks": "stocks"}
    with open(TICKERS_FILE) as f:
        for raw in f:
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            if line.startswith("[") and line.endswith("]"):
                cur = alias.get(line[1:-1].strip().lower())
                continue
            for tok in re.split(r"[\s,]+", line.lstrip("- ")):
                tok = tok.upper()
                if tok and cur in secs and tok not in secs["index"] + secs["stocks"]:
                    secs[cur].append(tok)
    return secs["index"], secs["stocks"]


# =============================================================================
# Saved open interest
# Yahoo reports open interest as 0 outside market hours. Whenever it has real numbers we save
# them (one small file per ticker, overwritten each time) and reuse them when it doesn't.
# Open interest only updates once a day, so the saved copy is just as current.
# =============================================================================
def with_saved_oi(t: str, ch: pd.DataFrame, spot: float, save: bool = True):
    """Returns (chain, saved_date or None). Raises if no usable open interest exists."""
    path = os.path.join(CACHE_DIR, re.sub(r"[^A-Za-z0-9_-]", "_", t) + ".csv")
    oi = pd.to_numeric(ch["oi"], errors="coerce").fillna(0)
    if len(ch) and (oi > 0).mean() >= 0.10:  # live data looks real -> save it
        if not save:  # demo mode never touches saved real data
            return ch, None
        keep = ch[(ch.strike > spot * 0.7) & (ch.strike < spot * 1.3)]
        try:
            os.makedirs(CACHE_DIR, exist_ok=True)
            keep = keep[["expiry", "strike", "call", "oi", "iv"]].assign(saved_on=dt.date.today().isoformat())
            keep.to_csv(path, index=False)
        except OSError:
            pass
        return ch, None
    if not os.path.exists(path):
        raise RuntimeError("Yahoo shows no open interest right now (outside market hours) and there is "
                           "no saved copy yet - run once while the market is open")
    saved = pd.read_csv(path)
    saved["expiry"] = pd.to_datetime(saved["expiry"]).dt.date
    saved["call"] = saved["call"].astype(str).str.lower().isin(["true", "1"])
    if "saved_on" in saved.columns:
        saved_on = dt.date.fromisoformat(str(saved["saved_on"].iloc[0]))
    else:
        saved_on = dt.date.fromtimestamp(os.path.getmtime(path))
    return saved, saved_on


# =============================================================================
# Data
# =============================================================================
class Yahoo:
    name = "Yahoo (prices ~15 min delayed)"

    def __init__(self):
        try:
            import yfinance as yf
        except ImportError:
            sys.exit("Missing library: yfinance\nInstall with:  pip install yfinance")
        import logging
        logging.getLogger("yfinance").setLevel(logging.CRITICAL)
        self.yf, self._h, self._aum = yf, {}, {}

    def history(self, t):
        if t not in self._h:
            try:
                df = self.yf.Ticker(t).history(period="6mo", auto_adjust=False, actions=False)
            except Exception:
                df = None
            if df is None or df.empty:
                df = self.yf.download(t, period="6mo", auto_adjust=False, actions=False,
                                      progress=False, threads=False)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
            if df is None or df.empty:
                raise RuntimeError("no price history")
            df = df[["Close", "Volume"]].copy()
            df.index = [pd.Timestamp(x).date() for x in df.index]
            self._h[t] = df
        return self._h[t]

    def spot(self, t):
        try:
            p = self.yf.Ticker(t).fast_info["last_price"]
            if p and p > 0:
                return float(p)
        except Exception:
            pass
        return float(self.history(t)["Close"].iloc[-1])

    def chain(self, t, today, max_days):
        tk, frames = self.yf.Ticker(t), []
        for e in tk.options:
            ed = dt.date.fromisoformat(e)
            if ed < today or (ed - today).days > max_days:
                continue
            oc = tk.option_chain(e)
            for typ, df in (("C", oc.calls), ("P", oc.puts)):
                if df is not None and not df.empty:
                    d = df[["strike", "openInterest", "impliedVolatility"]].copy()
                    d.columns = ["strike", "oi", "iv"]
                    d["call"], d["expiry"] = typ == "C", ed
                    frames.append(d)
        if not frames:
            raise RuntimeError("no listed options")
        return pd.concat(frames, ignore_index=True)

    def aum(self, t):
        if t not in self._aum:
            try:
                v = self.yf.Ticker(t).info.get("totalAssets")
                self._aum[t] = float(v) if v else None
            except Exception:
                self._aum[t] = None
        return self._aum[t]


class Demo:
    """Fake data covering every case, for testing the report offline:  python flow_scan.py --demo
    Pretends it is 3:10 PM ET on the next trading day so the leveraged-ETF lines show."""
    name = "DEMO - fake data, one ticker per case"
    # ticker: (kind, price, normal daily $ volume, today's move, options flow target as % of daily $ volume)
    CASES = {
        "SPY":    ("index", 680.0, 40e9, -0.015, -6.0),   # STRONG SELL with leveraged ETFs piling on
        "QQQ":    ("index", 600.0, 22e9, +0.018, +5.0),   # STRONG BUY with leveraged ETFs piling on
        "IWM":    ("index", 245.0,  7e9, -0.004, -2.5),   # SELL
        "XLK":    ("index", 260.0,  2e9, +0.004, +1.8),   # LEAN BUY, dealers DAMPEN moves
        "XLE":    ("index",  90.0,  1.5e9, -0.003, -1.5), # LEAN SELL
        "GLD":    ("index", 380.0,  3e9, +0.002, +0.3),   # NEUTRAL
        "PLTR":   ("stock", 180.0,  8e9, +0.030, +11.0),  # STRONG BUY
        "NVDA":   ("stock", 190.0, 35e9, +0.006, +4.0),   # BUY (options + leveraged ETF)
        "TSLA":   ("stock", 430.0, 30e9, -0.025, -9.0),   # STRONG SELL
        "AMD":    ("stock", 160.0,  9e9, -0.006, -4.5),   # SELL
        "MSTR":   ("stock", 330.0,  4e9, +0.020, +1.2),   # LEAN BUY
        "AAPL":   ("stock", 255.0, 12e9, +0.001, -0.4),   # NEUTRAL
        "NEWCO":  ("stock",  25.0,  2e8, +0.010, None),   # NO DATA YET (zero open interest, nothing saved)
        "BADTKR": ("stock",  10.0,  1e8, +0.000, "err"),  # SKIPPED (no listed options)
    }
    AUM = {"SPXL": 5.5e9, "UPRO": 4.2e9, "SPXS": .6e9, "SPXU": .7e9, "SSO": 6.5e9, "SDS": .5e9,
           "SH": 1.1e9, "TQQQ": 28e9, "SQQQ": 3.2e9, "QLD": 9e9, "QID": .3e9, "PSQ": .6e9,
           "TNA": 2.2e9, "TZA": .4e9, "UWM": .4e9, "TWM": .05e9, "NVDL": 4.5e9, "TSLL": 6e9, "MSTU": 1.5e9}
    now: dt.datetime  # set by main()

    def tickers(self):
        return ([t for t, v in self.CASES.items() if v[0] == "index"],
                [t for t, v in self.CASES.items() if v[0] == "stock"])

    def history(self, t):
        kind, price, adv, move, _ = self.CASES[t]
        r = np.random.default_rng(zlib.crc32(t.encode()))
        rets = r.normal(0, 0.01 if kind == "index" else 0.025, 130)
        prev = price / (1 + move)
        px = prev * np.exp(np.cumsum(rets) - rets.sum())
        days, d = [], back_to_trading(self.now.date() - dt.timedelta(days=1))
        while len(days) < 130:
            days.append(d)
            d = back_to_trading(d - dt.timedelta(days=1))
        return pd.DataFrame({"Close": px, "Volume": adv / px}, index=days[::-1])

    def spot(self, t):
        return self.CASES[t][1]

    def _raw_chain(self, s, today, max_days, seed, call_skew, put_mu=(-.06, .01), call_mu=(-.02, .05),
                   floor=200, width=.03):
        r = np.random.default_rng(seed)
        step = 1.0 if s < 250 else 5.0
        K = np.arange(round(s * .85 / step) * step, s * 1.15, step)
        rows, d = [], today
        while (d - today).days <= max_days:
            if is_trading_day(d) and ((d - today).days <= 7 or d.weekday() == 4):
                m = K / s - 1
                w = 3.0 if d == back_to_trading(nth_friday(d.year, d.month, 3)) else 1.0
                for call, mu, wt in ((True, r.uniform(*call_mu), call_skew), (False, r.uniform(*put_mu), 1.0)):
                    oi = wt * w * (3e3 * np.exp(-((m - mu) / width) ** 2) + floor) * r.uniform(.5, 1.5, len(K))
                    rows.append(pd.DataFrame({"strike": K, "oi": oi, "iv": .25 + .6 * m ** 2,
                                              "call": call, "expiry": d}))
            d += dt.timedelta(days=1)
        return pd.concat(rows, ignore_index=True)

    WANT_MOVES = {"XLK": "DAMPEN", "SPY": "MIXED"}

    def chain(self, t, today, max_days):
        kind, price, adv, move, target = self.CASES[t]
        if target == "err":
            raise RuntimeError("no listed options")
        if target is None:  # simulate Yahoo's overnight zeros
            ch = self._raw_chain(price, today, max_days, 1, 1.0)
            ch["oi"] = 0
            return ch
        # Build chains until the options flow points the target way, then scale it to the target size.
        for seed in range(60):
            # Alternate between options mostly out of the money and mostly in the money
            # (e.g. after a big move through popular strikes); the two produce opposite flows.
            itm = seed % 2 == 1
            ch = self._raw_chain(price, today, max_days, zlib.crc32(t.encode()) + seed,
                                 [0.4, 1.0, 2.5][seed % 3],
                                 put_mu=(.02, .05) if itm else (-.06, .01),
                                 call_mu=(-.05, -.02) if itm else (-.02, .05),
                                 floor=0 if itm else 200, width=.01 if itm else .03)
            o = options_flow(t, ch, price, kind, self.now)
            d0 = o["drift"]
            want = self.WANT_MOVES.get(t)
            if d0 and (d0 > 0) == (target > 0) and (want is None or o["moves"] == want):
                ch["oi"] *= abs(target / 100 * adv / d0)
                return ch
        return ch

    def aum(self, t): return self.AUM.get(t)


# =============================================================================
# Math
# =============================================================================
def bs_delta(S, K, T, v, r, call):
    T = np.maximum(T, 1e-7)
    d1 = (np.log(S / K) + (r + .5 * v * v) * T) / (v * np.sqrt(T))
    return np.where(call, norm.cdf(d1), norm.cdf(d1) - 1.0)


def options_flow(t, chain, spot, model, now):
    """Dealer hedge flows (USD, + = dealers BUY) implied by open interest."""
    r, w = SETTINGS["risk_free"], SETTINGS["strike_window"]
    df = chain[(chain.strike > spot * (1 - w)) & (chain.strike < spot * (1 + w))].copy()
    df["iv"] = pd.to_numeric(df["iv"], errors="coerce")
    df.loc[(df.iv < .02) | (df.iv > 3), "iv"] = np.nan
    df["iv"] = df["iv"].fillna(df["iv"].median() if df["iv"].notna().any() else .3)
    df["oi"] = pd.to_numeric(df["oi"], errors="coerce").fillna(0)
    df = df[df.oi > 0]
    df["expiry"] = pd.to_datetime(df["expiry"]).dt.date
    secs = {d: (close_time(d) - now).total_seconds() for d in set(df["expiry"])}
    df["T"] = df["expiry"].map(secs).astype(float) / YEAR
    df = df[df["T"] > 0].reset_index(drop=True)
    if df.empty:
        raise RuntimeError("no live options")

    K, T, iv, call = df.strike.values, df["T"].values, df.iv.values, df.call.values.astype(bool)
    sign = np.where(call, 1.0, -1.0) if model == "index" else -np.ones(len(df))
    n = sign * df.oi.values * 100  # dealer position in share-equivalents

    horizon = next_close(now)
    h = (horizon - now).total_seconds() / YEAR
    expiring = (df["expiry"] <= horizon.date()).values
    d_now = bs_delta(spot, K, T, iv, r, call)
    intrinsic = np.where(call, (spot > K).astype(float), -(spot < K).astype(float))
    d_later = np.where(expiring | (T - h <= 1e-6), intrinsic,
                       bs_delta(spot, K, np.maximum(T - h, 1e-6), iv, r, call))
    drift = float((-(n * (d_later - d_now)) * spot).sum())  # time decay + expiring options

    def hedge_at(s1):
        return float((-(n * (bs_delta(s1, K, T, iv, r, call) - d_now))) .sum() * s1)
    up, dn = hedge_at(spot * 1.01), hedge_at(spot * 0.99)

    # Walls: strike with the most open interest above the price (calls) / below it (puts)
    calls = df[df.call & (df.strike > spot)].groupby("strike")["oi"].sum()
    puts = df[~df.call & (df.strike < spot)].groupby("strike")["oi"].sum()
    return {
        "drift": drift, "horizon": horizon,
        "moves": "AMPLIFY" if up > 0 and dn < 0 else "DAMPEN" if up < 0 and dn > 0 else "MIXED",
        "call_wall": float(calls.idxmax()) if len(calls) else None,
        "put_wall": float(puts.idxmax()) if len(puts) else None,
    }


def lev_etf_flow(prov, t, ret):
    funds = SETTINGS["leveraged_etfs"].get(t)
    if not funds:
        return 0.0
    return sum((prov.aum(e) or 0.0) * (L * L - L) * ret for e, L in funds.items())


def rating(pct):
    a, s = abs(pct), SETTINGS
    side = "BUY" if pct > 0 else "SELL"
    mark = "+" if pct > 0 else "-"
    if a >= s["strong_pct"]:
        return f"STRONG {side}", mark * 3, 3
    if a >= s["rating_pct"]:
        return side, mark * 2, 2
    if a >= s["lean_pct"]:
        return f"LEAN {side}", mark, 1
    return "NEUTRAL", "", 0


# =============================================================================
# Output helpers
# =============================================================================
COLOR = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None
if COLOR and sys.platform.startswith("win"):
    os.system("")


def c(txt, code): return f"\033[{code}m{txt}\033[0m" if COLOR else txt


def usd(x, sign=False):
    a = abs(x)
    for u, d in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if a >= d:
            s = f"${a / d:.1f}{u}" if a / d < 100 else f"${a / d:.0f}{u}"
            break
    else:
        s = f"${a:.0f}"
    return (("+" if x > 0 else "-" if x < 0 else "") + s) if sign else s


def lvl(p, spot): return "-" if p is None else f"{p:g} ({(p / spot - 1) * 100:+.1f}%)"


def et_clock(t): return t.strftime("%I:%M %p").lstrip("0")


# =============================================================================
# Scan (returns data; the report functions below turn it into text or HTML)
# =============================================================================
def demo_now(now):
    """Demo pretends it's 3:10 PM ET on the next trading day, so every section shows."""
    d = now.date() if now < close_time(now.date()) - dt.timedelta(minutes=50) else now.date() + dt.timedelta(days=1)
    while not is_trading_day(d):
        d += dt.timedelta(days=1)
    return min(dt.datetime.combine(d, dt.time(15, 10), ET), close_time(d) - dt.timedelta(minutes=50))


def scan(watchlist=None, demo=False, now=None, debug=False, progress=True):
    """watchlist: plain list of tickers (ETFs detected via KNOWN_ETFS), or None to use tickers.txt."""
    prov = Demo() if demo else Yahoo()
    now = now or dt.datetime.now(ET)
    if demo:
        now = demo_now(now)
        prov.now = now
        index, stocks = prov.tickers()
    elif watchlist is not None:
        seen = []
        for t in watchlist:
            t = str(t).strip().upper()
            if t and t not in seen:
                seen.append(t)
        index = [t for t in seen if t in KNOWN_ETFS]
        stocks = [t for t in seen if t not in KNOWN_ETFS]
    else:
        index, stocks = load_tickers()

    today, st = now.date(), session(now)
    close = next_close(now)
    is_open = st == "open"
    tag = ("INTO THE CLOSE" if is_open and close - now <= dt.timedelta(hours=2)
           else "INTRADAY" if is_open else "BEFORE THE OPEN")

    rows, skipped, used_saved = [], [], {}
    tickers = [(t, "index") for t in index] + [(t, "stock") for t in stocks]
    for i, (t, kind) in enumerate(tickers, 1):
        if progress:
            print(f"\r  scanning {i}/{len(tickers)} {t:<8}", end="", file=sys.stderr, flush=True)
        if i > 1 and not demo:
            time.sleep(SETTINGS["request_pause"])
        step = "price history"
        try:
            h = prov.history(t)
            step = "price"
            spot = prov.spot(t)
            step = "options"
            ch = prov.chain(t, today, SETTINGS["max_expiry_days"])
            ch, saved_on = with_saved_oi(t, ch, spot, save=not demo)
            if saved_on:
                used_saved[t] = saved_on
            step = "math"
            o = options_flow(t, ch, spot, kind, now)
        except Exception as e:
            skipped.append((t, step, str(e)))
            if debug:
                print(file=sys.stderr)
                traceback.print_exc()
            continue
        closes = h["Close"]
        prev = float(closes.iloc[-2] if closes.index[-1] == today and len(closes) > 1 else closes.iloc[-1])
        ret = spot / prev - 1 if is_open else 0.0
        lev = lev_etf_flow(prov, t, ret) if is_open else 0.0
        tail = h.tail(20)
        adv = float((tail["Close"] * tail["Volume"]).mean()) or float("nan")
        sigma = float(np.log(closes).diff().tail(20).std())  # daily volatility
        net = o["drift"] + lev
        pct = net / adv * 100
        push = math.copysign(sigma * math.sqrt(abs(net) / adv), net) * 100  # square-root impact
        label, marks, strength = rating(pct)
        rows.append(dict(t=t, kind=kind, spot=spot, ret=ret, net=net, opt=o["drift"], lev=lev,
                         pct=pct, push=push, label=label, marks=marks, strength=strength, **o))
    if progress:
        print("\r" + " " * 40 + "\r", end="", file=sys.stderr)

    spy = next((r for r in rows if r["t"] == "SPY"), None)
    notes = []
    if spy:
        notes.append(f"SPY dealers {spy['moves']} moves (put wall {lvl(spy['put_wall'], spy['spot'])}, "
                     f"call wall {lvl(spy['call_wall'], spy['spot'])})")
    notes += upcoming_events(today)

    no_oi = [t for t, _, e in skipped if "no saved copy yet" in e]
    return {
        "now": now, "close": close, "session": st, "is_open": is_open, "tag": tag, "source": prov.name,
        "notes": notes, "used_saved": used_saved,
        "active": sorted([r for r in rows if r["strength"] > 0], key=lambda r: (-r["strength"], -abs(r["pct"]))),
        "neutral": [r["t"] for r in rows if r["strength"] == 0],
        "no_oi": no_oi,
        "skipped": [f"{t} ({step}: {e})" for t, step, e in skipped if t not in no_oi],
    }


def timing_lines(res):
    now, close = res["now"], res["close"]
    lines = []
    if res["is_open"]:
        lines.append(f"Flow measured through today's close ({local(close)} your time, "
                     f"in {int((close - now).total_seconds() // 60)} min). Data: {res['source']}")
    else:
        lines.append(f"Flow measured through the next close: {close:%a} {local(close)} your time. "
                     f"Data: {res['source']}")
        if res["session"] in ("after", "closed"):
            lines.append("Market is closed: leveraged-ETF flow is added during market hours.")
    if res["used_saved"]:
        oldest = min(res["used_saved"].values())
        lines.append(f"Yahoo has no open interest right now, so {len(res['used_saved'])} tickers use the copy "
                     f"saved {oldest:%a %b %d} (open interest only changes once a day).")
    return lines


HOW_TO_READ = [
    ("RATING", "Direction of forced flow if price holds: +++ strong, ++ clear, + lean."),
    ("NET FORCED FLOW", "Dollars dealers (and, during market hours, leveraged ETFs) must trade by the close."),
    ("% OF DAY", "That flow vs. the ticker's normal daily $ volume. Above ~3% is meaningful."),
    ("EST. PUSH", "Rough price impact if nobody absorbs it (square-root rule). Order of magnitude, not a target."),
    ("MOVES", "AMPLIFY = dealers chase moves (breaks can run). DAMPEN = dealers fight moves "
              "(ranges tend to hold between the walls)."),
]
CAVEAT = ("Assumes dealers sold most stock options and index customers mostly buy puts / sell calls. "
          "News, earnings and real buyers/sellers outweigh all of this.")


# =============================================================================
# Text report (terminal)
# =============================================================================
def text_report(res, debug=False) -> str:
    out, now = [], res["now"]
    out.append("")
    out.append(c(f"FORCED FLOW SCAN - {res['tag']}", "1") +
               f"   {now:%a %b %d} {et_clock(now)} ET ({local(now)} your time)   {VERSION}")
    for line in timing_lines(res):
        out.append(c(line, "2") if line.startswith(("Market is closed", "Yahoo has")) else line)
    out += [f"  * {n}" for n in res["notes"]]
    out.append("")
    out.append(c(f"{'RATING':<16}{'TICKER':<7}{'PRICE':>9}{'TODAY':>8}  {'NET FORCED FLOW':<17}{'% OF DAY':>9}"
                 f"{'EST. PUSH':>11}  {'MOVES':<9}{'PUT WALL':<17}{'CALL WALL':<17}", "1"))
    for r in res["active"]:
        col = "92" if r["net"] > 0 else "91"
        today_s = f"{r['ret'] * 100:+.1f}%" if res["is_open"] else "-"
        flow = f"{'BUY' if r['net'] > 0 else 'SELL'} {usd(r['net'])}"
        mcol = {"AMPLIFY": "93", "DAMPEN": "96"}.get(r["moves"], "2")
        out.append(c(f"{r['label'] + ' ' + r['marks']:<16}", col + ";1") + f"{r['t']:<7}{r['spot']:>9.2f}{today_s:>8}  "
                   + c(f"{flow:<17}", col) + f"{abs(r['pct']):>8.1f}%{r['push']:>+10.2f}%  "
                   + c(f"{r['moves']:<9}", mcol) + f"{lvl(r['put_wall'], r['spot']):<17}{lvl(r['call_wall'], r['spot']):<17}")
        if r["lev"]:
            out.append(c(f"{'':<16}  from options {usd(r['opt'], True)}, from leveraged ETFs {usd(r['lev'], True)}", "2"))
    if not res["active"]:
        out.append("  No ticker has meaningful forced flow right now.")
    if res["neutral"]:
        out.append(c(f"\nNEUTRAL (forced flow under {SETTINGS['lean_pct']:g}% of a normal day): ", "2")
                   + ", ".join(res["neutral"]))
    if res["no_oi"]:
        out.append(c(f"\nNO DATA YET ({len(res['no_oi'])}): Yahoo shows no open interest outside market hours and "
                     f"these have no saved copy. Run once while the market is open; after that, early-morning "
                     f"runs work.\n  " + ", ".join(res["no_oi"]), "93"))
    if res["skipped"]:
        out.append(c("\nSKIPPED: " + "; ".join(res["skipped"]), "2")
                   + ("" if debug else c("  (run with --debug for details)", "2")))
    out.append(c("\nHOW TO READ IT", "2"))
    out += [c(f"  {k:<16}{v}", "2") for k, v in HOW_TO_READ]
    out.append(c(f"  {CAVEAT}", "2"))
    return "\n".join(out)


# =============================================================================
# HTML report (email) - narrow enough for a phone
# =============================================================================
def html_report(res) -> str:
    from html import escape as e
    green, red, grey = "#137333", "#c5221f", "#666"
    font = "font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif"
    now = res["now"]
    h = [f"<div style='{font};font-size:14px;color:#202124'>",
         f"<h3 style='margin:18px 0 2px'>Forced Flow - {e(res['tag'].title())}</h3>",
         f"<p style='color:{grey};margin:0 0 6px;font-size:13px'>{now:%a %b %d} {et_clock(now)} ET "
         f"({local(now)} your time)<br>" + "<br>".join(e(x) for x in timing_lines(res)) + "</p>"]
    if res["notes"]:
        h.append("<ul style='margin:4px 0 10px;padding-left:18px'>"
                 + "".join(f"<li>{e(n)}</li>" for n in res["notes"]) + "</ul>")

    if res["active"]:
        th = "style='text-align:left;padding:6px 8px;border-bottom:1px solid #ddd;font-size:12px;color:#555'"
        h.append("<table cellpadding='0' cellspacing='0' style='border-collapse:collapse;font-size:14px'>"
                 f"<tr style='background:#f4f4f4'><th {th}>Rating</th><th {th}>Ticker</th>"
                 f"<th {th}>Forced flow</th><th {th}>Moves</th><th {th}>Put / call wall</th></tr>")
        td = "style='padding:6px 8px;border-bottom:1px solid #eee;vertical-align:top'"
        for r in res["active"]:
            col = green if r["net"] > 0 else red
            word = "BUY" if r["net"] > 0 else "SELL"
            sub = f"{abs(r['pct']):.1f}% of a day<br>push {r['push']:+.2f}%"
            if r["lev"]:
                sub += f"<br>options {usd(r['opt'], True)}<br>lev. ETFs {usd(r['lev'], True)}"
            today_s = f"<br><span style='color:{grey};font-size:12px'>{r['ret'] * 100:+.1f}% today</span>" \
                if res["is_open"] else ""
            h.append(
                f"<tr><td {td}><b style='color:{col}'>{e(r['label'])}</b><br>"
                f"<span style='color:{col};letter-spacing:1px'>{e(r['marks'])}</span></td>"
                f"<td {td}><b>{e(r['t'])}</b><br><span style='color:{grey};font-size:12px'>{r['spot']:.2f}</span>{today_s}</td>"
                f"<td {td}><span style='color:{col}'>{word} {usd(r['net'])}</span><br>"
                f"<span style='color:{grey};font-size:12px'>{sub}</span></td>"
                f"<td {td}>{e(r['moves'].title())}</td>"
                f"<td {td}><span style='font-size:12px;white-space:nowrap'>{e(lvl(r['put_wall'], r['spot']))}<br>"
                f"{e(lvl(r['call_wall'], r['spot']))}</span></td></tr>")
        h.append("</table>")
    else:
        h.append("<p>No ticker has meaningful forced flow right now.</p>")

    if res["neutral"]:
        h.append(f"<p style='color:{grey};font-size:13px'><b>Neutral</b> (under {SETTINGS['lean_pct']:g}% of a "
                 f"normal day): {e(', '.join(res['neutral']))}</p>")
    if res["no_oi"]:
        h.append(f"<p style='color:#b06000;font-size:13px'><b>No data yet ({len(res['no_oi'])}):</b> Yahoo shows "
                 f"no open interest outside market hours and these have no saved copy yet. They fill in after "
                 f"a run during market hours.<br>{e(', '.join(res['no_oi']))}</p>")
    if res["skipped"]:
        h.append(f"<p style='color:{grey};font-size:12px'><b>Skipped:</b> {e('; '.join(res['skipped']))}</p>")
    h.append(f"<p style='color:{grey};font-size:12px;margin-top:12px'><b>How to read it</b><br>"
             + "<br>".join(f"<b>{e(k.title())}</b>: {e(v)}" for k, v in HOW_TO_READ)
             + f"<br>{e(CAVEAT)}</p></div>")
    return "".join(h)


# =============================================================================
# Command line
# =============================================================================
def main():
    ap = argparse.ArgumentParser(description="Forced order flow scan: BUY/SELL rating per ticker.")
    ap.add_argument("--demo", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--debug", action="store_true", help="show full error details for skipped tickers")
    ap.add_argument("--html", metavar="PATH", help="also write the report as an HTML file")
    args = ap.parse_args()
    res = scan(demo=args.demo, debug=args.debug)
    print(text_report(res, debug=args.debug))
    if args.html:
        with open(args.html, "w", encoding="utf-8") as f:
            f.write(html_report(res))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")
