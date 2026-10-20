"""Setup Scanner — checks a watchlist for the setups in config.yaml and emails hits.

Run locally:   python scanner.py --dry-run      (prints results, no email)
Env vars:      GMAIL_ADDRESS, GMAIL_APP_PASSWORD (sender account)
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


def scan_ticker(df, setups):
    cache, hits = {}, []
    for setup in setups:
        if all(check(c, df, cache) for c in setup["when"]):
            hits.append(setup["name"])
    snapshot = {
        "close": df["Close"].iloc[-1],
        "rsi": series_for("rsi", df, cache).iloc[-1],
        "macd_hist": series_for("macd_hist", df, cache).iloc[-1],
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
def build_email(results, interval):
    if not results:
        return "Scanner: no setups triggered", "<p>No setups triggered this run.</p>"
    rows = "".join(
        f"<tr><td><b>{t}</b></td><td>{', '.join(h)}</td><td>{s['close']:.2f}</td>"
        f"<td>{s['rsi']:.1f}</td><td>{s['macd_hist']:+.3f}</td></tr>"
        for t, h, s in results
    )
    html = f"""
    <p>{sum(len(h) for _, h, _ in results)} setup(s) on {len(results)} ticker(s) — {interval} candles</p>
    <table cellpadding="6" style="border-collapse:collapse;font-family:sans-serif;font-size:14px">
      <tr style="background:#f0f0f0;text-align:left">
        <th>Ticker</th><th>Setups</th><th>Close</th><th>RSI</th><th>MACD hist</th></tr>
      {rows}
    </table>"""
    subject = "Scanner: " + ", ".join(t for t, _, _ in results[:6]) + ("…" if len(results) > 6 else "")
    return subject, html


def send_email(to, subject, html):
    sender, password = os.environ["GMAIL_ADDRESS"], os.environ["GMAIL_APP_PASSWORD"]
    msg = MIMEText(html, "html")
    msg["Subject"], msg["From"], msg["To"] = subject, sender, to
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
        s.login(sender, password)
        s.send_message(msg)


# ── Main ────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=Path(__file__).with_name("config.yaml"))
    ap.add_argument("--dry-run", action="store_true", help="print results, don't email")
    args = ap.parse_args()

    try:
        cfg = yaml.safe_load(Path(args.config).read_text())
        for s in cfg["setups"]:  # fail fast on typos
            for c in s["when"]:
                parse(c)
    except (yaml.YAMLError, KeyError, TypeError, ValueError) as e:
        sys.exit(f"CONFIG PROBLEM in config.yaml (check spacing/indentation): {e}")

    if not args.dry_run and not (os.environ.get("GMAIL_ADDRESS") and os.environ.get("GMAIL_APP_PASSWORD")):
        sys.exit("MISSING SECRETS: add GMAIL_ADDRESS and GMAIL_APP_PASSWORD under "
                 "Settings → Secrets and variables → Actions (names must match exactly).")

    interval = cfg.get("interval", "1d")
    data = fetch(cfg["watchlist"], interval)
    if not data:
        sys.exit("NO DATA: Yahoo returned nothing for any ticker. Usually a temporary Yahoo "
                 "outage or rate limit; re-run in a few minutes.")
    results = []
    for ticker, df in data.items():
        hits, snap = scan_ticker(df, cfg["setups"])
        if hits:
            results.append((ticker, hits, snap))
            print(f"  ✓ {ticker}: {', '.join(hits)}")

    print(f"{len(results)} ticker(s) with setups")
    if args.dry_run or (not results and not cfg.get("email_when_empty")):
        return
    try:
        send_email(cfg.get("email_to") or os.environ["GMAIL_ADDRESS"], *build_email(results, interval))
    except smtplib.SMTPAuthenticationError:
        sys.exit("GMAIL LOGIN FAILED: GMAIL_APP_PASSWORD must be the 16-character app password "
                 "(not your normal Gmail password), and GMAIL_ADDRESS must be the same account.")
    print("Email sent.")


if __name__ == "__main__":
    main()
