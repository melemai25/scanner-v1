"""Setup Scanner — checks a watchlist for the setups in config.yaml and emails hits,
followed by the forced order flow scan (flow_scan.py) for the same watchlist.

Run locally:   python scanner.py --dry-run      (prints results, no email)
Env vars:      GMAIL_ADDRESS, GMAIL_APP_PASSWORD (sender account)
               FLOW_ONLY=1   skip the setups and send only the forced flow scan
"""
import argparse
import os
import re
import smtplib
import sys
from email.mime.text import MIMEText
from pathlib import Path

import pandas as pd
import yaml
import yfinance as yf

OPS = {"<", ">", "<=", ">=", "crosses_above", "crosses_below"}
INTRADAY = {"1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h"}


# ── Indicators ──────────────────────────────────────────────
def rsi(close, n):
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + gain / loss)


def macd(close, fast=12, slow=26, signal=9):
    line = close.ewm(span=fast, adjust=False).mean() - close.ewm(span=slow, adjust=False).mean()
    sig = line.ewm(span=signal, adjust=False).mean()
    return line, sig, line - sig


def series_for(token, df, cache):
    """Turn a config token ('rsi', 'sma_50', '2*avg_volume_20', '30') into a Series."""
    token = token.strip().lower()
    if token in cache:
        return cache[token]

    if "*" in token:  # multiple, e.g. 2*avg_volume_20
        factor, name = token.split("*", 1)
        out = float(factor) * series_for(name, df, cache)
    elif re.fullmatch(r"-?\d+(\.\d+)?", token):
        out = pd.Series(float(token), index=df.index)
    elif token in ("close", "open", "high", "low", "volume"):
        out = df[token.capitalize()]
    elif token == "ibs":  # internal bar strength: where the close sits in today's range, 0-100
        rng = (df["High"] - df["Low"]).replace(0, float("nan"))
        out = 100 * (df["Close"] - df["Low"]) / rng
    elif m := re.fullmatch(r"(lowest|highest)_close_(\d+)", token):
        roll = df["Close"].rolling(int(m.group(2)))
        out = roll.min() if m.group(1) == "lowest" else roll.max()
    elif m := re.fullmatch(r"prior_(high|low)_(\d+)", token):  # N-day high/low close, NOT counting today
        roll = df["Close"].shift(1).rolling(int(m.group(2)))
        out = roll.max() if m.group(1) == "high" else roll.min()
    elif m := re.fullmatch(r"return_(\d+)(?:_(\d+))?", token):  # % change from N bars ago (to M bars ago)
        n, skip = int(m.group(1)), int(m.group(2) or 0)
        out = 100 * (df["Close"].shift(skip) / df["Close"].shift(n) - 1)
    elif token == "rs_rank":
        raise ValueError("rs_rank needs the whole watchlist; it is filled in by main()")
    elif token in ("macd", "macd_signal", "macd_hist"):
        line, sig, hist = macd(df["Close"])
        cache.update(macd=line, macd_signal=sig, macd_hist=hist)
        out = cache[token]
    elif m := re.fullmatch(r"(rsi|sma|ema|avg_volume)(?:_(\d+))?", token):
        kind, n = m.group(1), int(m.group(2) or 14)
        out = {
            "rsi": lambda: rsi(df["Close"], n),
            "sma": lambda: df["Close"].rolling(n).mean(),
            "ema": lambda: df["Close"].ewm(span=n, adjust=False).mean(),
            "avg_volume": lambda: df["Volume"].rolling(n).mean(),
        }[kind]()
    else:
        raise ValueError(f"Unknown value in config: '{token}'")

    cache[token] = out
    return out


# ── Rules ───────────────────────────────────────────────────
def parse(condition):
    parts = condition.split()
    if len(parts) != 3 or parts[1] not in OPS:
        raise ValueError(f"Can't read condition '{condition}'. Use: <left> <operator> <right>")
    return parts


def check(condition, df, cache):
    left, op, right = parse(condition)
    a, b = series_for(left, df, cache), series_for(right, df, cache)
    now_a, now_b, prev_a, prev_b = a.iloc[-1], b.iloc[-1], a.iloc[-2], b.iloc[-2]
    if pd.isna([now_a, now_b, prev_a, prev_b]).any():
        return False
    return {
        "<": now_a < now_b,
        ">": now_a > now_b,
        "<=": now_a <= now_b,
        ">=": now_a >= now_b,
        "crosses_above": prev_a <= prev_b and now_a > now_b,
        "crosses_below": prev_a >= prev_b and now_a < now_b,
    }[op]


RS_LOOKBACK = "return_252_21"  # 12-month return skipping the latest month (classic momentum)


def rs_ranks(data):
    """Percentile (0-100) of each ticker's 12-1 month return vs the rest of the watchlist, per day."""
    rets = pd.DataFrame({t: series_for(RS_LOOKBACK, df, {}) for t, df in data.items()})
    return rets.rank(axis=1, pct=True) * 100


def scan_ticker(df, sections, extras=None):
    """sections: {"core": [setups], "speculative": [setups]} -> {"core": [names], ...}"""
    cache = dict(extras or {})
    hits = {sec: [s["name"] for s in setups if all(check(c, df, cache) for c in s["when"])]
            for sec, setups in sections.items()}
    last = lambda tok: series_for(tok, df, cache).iloc[-1]
    snapshot = {
        "close": last("close"),
        "rsi": last("rsi"),
        "macd_hist": last("macd_hist"),
        "ret_12_1": last(RS_LOOKBACK),
        "rs_rank": cache["rs_rank"].iloc[-1] if "rs_rank" in cache else float("nan"),
        "off_high": 100 * (last("close") / last("highest_close_252") - 1),
    }
    return hits, snapshot


# ── Data ────────────────────────────────────────────────────
def fetch(tickers, interval):
    period = "60d" if interval in INTRADAY else "2y"  # enough history for sma_200
    raw = yf.download(tickers, period=period, interval=interval, group_by="ticker",
                      auto_adjust=True, progress=False, threads=True)
    data = {}
    for t in tickers:
        df = (raw[t] if len(tickers) > 1 else raw).dropna(how="all")
        if interval in INTRADAY and len(df) > 1:
            df = df.iloc[:-1]  # drop the still-forming candle
        if len(df) >= 30:
            data[t] = df
        else:
            print(f"  ! {t}: not enough data, skipped", file=sys.stderr)
    return data


# ── Email ───────────────────────────────────────────────────
TABLE = '<table cellpadding="6" style="border-collapse:collapse;font-family:sans-serif;font-size:14px">'
HEAD = '<tr style="background:#f0f0f0;text-align:left">'

SECTIONS = {  # section -> (title, note, [(column, format)])
    "core": ("Core Setups", "High win-rate pullbacks in uptrends.",
             [("Close", "{close:.2f}"), ("RSI", "{rsi:.1f}"), ("MACD hist", "{macd_hist:+.3f}")]),
    "speculative": ("Speculative Setups",
                    "Momentum/breakout trades: lower win rate, bigger winners. Size small, use a stop.",
                    [("Close", "{close:.2f}"), ("12-1m return", "{ret_12_1:+.0f}%"),
                     ("RS rank", "{rs_rank:.0f}"), ("Off 52w high", "{off_high:+.1f}%")]),
}


def build_email(results, interval, flow=None, setups_ran=True):
    """results: [(ticker, {"core": [...], "speculative": [...]}, snapshot)]
    flow: (html, subject_bit) from run_flow(), shown below the setups, or None."""
    tag = f"Scanner ({os.environ['SCAN_LABEL']})" if os.environ.get("SCAN_LABEL") else "Scanner"
    flow_html, flow_bit = flow or ("", "")
    if not setups_ran:
        return f"{tag}: {flow_bit or 'forced flow'}", flow_html
    if not results:
        subject = f"{tag}: no setups triggered" + (f" | {flow_bit}" if flow_bit else "")
        return subject, "<p>No setups triggered this run.</p>" + flow_html

    total = sum(len(names) for _, hits, _ in results for names in hits.values())
    html = f"<p>{total} setup(s) on {len(results)} ticker(s) — {interval} candles</p>"
    for sec, (title, note, cols) in SECTIONS.items():
        rows = [(t, hits[sec], s) for t, hits, s in results if hits.get(sec)]
        if not rows:
            continue
        head = "".join(f"<th>{c}</th>" for c, _ in cols)
        body = "".join(
            f"<tr><td><b>{t}</b></td><td>{', '.join(names)}</td>"
            + "".join(f"<td>{fmt.format(**s)}</td>" for _, fmt in cols) + "</tr>"
            for t, names, s in rows
        )
        html += (f"<h3 style='font-family:sans-serif;margin:18px 0 2px'>{title}</h3>"
                 f"<p style='font-family:sans-serif;color:#666;margin:0 0 6px'>{note}</p>"
                 f"{TABLE}{HEAD}<th>Ticker</th><th>Setups</th>{head}</tr>{body}</table>")

    subject = f"{tag}: " + ", ".join(t for t, _, _ in results[:6]) + ("…" if len(results) > 6 else "")
    if flow_bit:
        subject += f" | {flow_bit}"
    return subject, html + flow_html


def send_email(to, subject, html):
    sender, password = os.environ["GMAIL_ADDRESS"], os.environ["GMAIL_APP_PASSWORD"]
    msg = MIMEText(html, "html")
    msg["Subject"], msg["From"], msg["To"] = subject, sender, to
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
        s.login(sender, password)
        s.send_message(msg)


# ── Forced flow ─────────────────────────────────────────────
def run_flow(watchlist):
    """Runs flow_scan.py on the watchlist. Returns (html, subject_bit, text).
    Never raises: if the flow scan breaks, the setup email still goes out with a note."""
    try:
        import flow_scan
        res = flow_scan.scan(watchlist=watchlist, progress=False)
        top = ", ".join(f"{r['t']} {r['marks']}" for r in res["active"][:3])
        bit = f"Flow: {top}" if top else "Flow: nothing notable"
        return flow_scan.html_report(res), bit, flow_scan.text_report(res)
    except Exception as e:  # keep the setups email alive no matter what
        import traceback
        traceback.print_exc()
        msg = f"Forced flow scan failed this run: {type(e).__name__}: {e}"
        return f"<p style='font-family:sans-serif;color:#b06000'>{msg}</p>", "", msg


# ── Main ────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=Path(__file__).with_name("config.yaml"))
    ap.add_argument("--dry-run", action="store_true", help="print results, don't email")
    args = ap.parse_args()

    try:
        cfg = yaml.safe_load(Path(args.config).read_text())
        sections = {"core": cfg["setups"] or [], "speculative": cfg.get("speculative_setups") or []}
        for setups in sections.values():  # fail fast on typos
            for s in setups:
                for c in s["when"]:
                    parse(c)
    except (yaml.YAMLError, KeyError, TypeError, ValueError) as e:
        sys.exit(f"CONFIG PROBLEM in config.yaml (check spacing/indentation): {e}")

    if not args.dry_run and not (os.environ.get("GMAIL_ADDRESS") and os.environ.get("GMAIL_APP_PASSWORD")):
        sys.exit("MISSING SECRETS: add GMAIL_ADDRESS and GMAIL_APP_PASSWORD under "
                 "Settings → Secrets and variables → Actions (names must match exactly).")

    interval = cfg.get("interval", "1d")
    flow_only = os.environ.get("FLOW_ONLY") == "1"
    flow_on = flow_only or cfg.get("flow_scan", True)
    results = []
    if not flow_only:
        data = fetch(cfg["watchlist"], interval)
        if not data:
            sys.exit("NO DATA: Yahoo returned nothing for any ticker. Usually a temporary Yahoo "
                     "outage or rate limit; re-run in a few minutes.")
        ranks = rs_ranks(data)
        for ticker, df in data.items():
            hits, snap = scan_ticker(df, sections, {"rs_rank": ranks[ticker].reindex(df.index)})
            if any(hits.values()):
                results.append((ticker, hits, snap))
                print(f"  ✓ {ticker}: {', '.join(n for names in hits.values() for n in names)}")
        print(f"{len(results)} ticker(s) with setups")

    flow = None
    if flow_on:
        print("Running forced flow scan...")
        html, bit, text = run_flow(cfg["watchlist"])
        print(text)
        flow = (html, bit)

    if args.dry_run or (not results and not flow_on and not cfg.get("email_when_empty")):
        return
    try:
        send_email(cfg.get("email_to") or os.environ["GMAIL_ADDRESS"],
                   *build_email(results, interval, flow, setups_ran=not flow_only))
    except smtplib.SMTPAuthenticationError:
        sys.exit("GMAIL LOGIN FAILED: GMAIL_APP_PASSWORD must be the 16-character app password "
                 "(not your normal Gmail password), and GMAIL_ADDRESS must be the same account.")
    print("Email sent.")


if __name__ == "__main__":
    main()
