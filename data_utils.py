import time
import threading
from datetime import datetime, timezone
import numpy as np
import requests
import yfinance as yf

from config import FAPI, EXCLUDED_CRYPTO_SYMS, CRYPTO_MAX_COINS, CRYPTO_MIN_VOL_24H

VISION_API = "https://data-api.binance.vision"
_USE_VISION = False

_WEIGHT_LOCK = threading.Lock()
_LAST_USED_WEIGHT = 0
_PAUSE_UNTIL = 0.0

YAHOO_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124.0.0.0 Safari/537.36"
}


def calc_sma(arr, period):
    arr = np.asarray(arr, dtype=np.float64)
    n = len(arr)
    out = np.full(n, np.nan, dtype=np.float64)
    valid = np.where(~np.isnan(arr))[0]
    if len(valid) < period:
        return out
    first = int(valid[0])
    sub = arr[first:]
    if len(sub) < period:
        return out
    cs = np.cumsum(np.insert(sub, 0, 0.0))
    out[first + period - 1 :] = (cs[period:] - cs[:-period]) / float(period)
    return out


def calc_ema(arr, period):
    arr = np.asarray(arr, dtype=np.float64)
    n = len(arr)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < period:
        return out
    out[period - 1] = float(np.mean(arr[:period]))
    k = 2.0 / (period + 1.0)
    for i in range(period, n):
        out[i] = arr[i] * k + out[i - 1] * (1.0 - k)
    return out


def calc_atr(h, l, c, period=14, fill_initial=False):
    h = np.asarray(h, dtype=np.float64)
    l = np.asarray(l, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64)
    n = len(c)
    tr = np.zeros(n, dtype=np.float64)
    if n == 0:
        return tr
    tr[0] = h[0] - l[0]
    if n > 1:
        prev_c = c[:-1]
        tr[1:] = np.maximum(
            h[1:] - l[1:],
            np.maximum(np.abs(h[1:] - prev_c), np.abs(l[1:] - prev_c)),
        )
    atr = np.full(n, np.nan, dtype=np.float64)
    if n >= period:
        atr[period - 1] = float(np.mean(tr[:period]))
        for i in range(period, n):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / float(period)
    if fill_initial:
        first_valid = period - 1
        if first_valid < n and not np.isnan(atr[first_valid]):
            atr[:first_valid] = atr[first_valid]
        else:
            atr[:] = np.maximum(h - l, 1e-12)
        atr = np.maximum(atr, 1e-12)
    return atr


def calc_rsi(c, period=14, fill_val=np.nan):
    c = np.asarray(c, dtype=np.float64)
    n = len(c)
    rsi = np.full(n, fill_val, dtype=np.float64)
    if n <= period:
        return rsi
    diff = np.diff(c)
    gains = np.where(diff > 0, diff, 0.0)
    losses = np.where(diff < 0, -diff, 0.0)
    avg_g = float(np.mean(gains[:period]))
    avg_l = float(np.mean(losses[:period]))
    rsi[period] = 100.0 if avg_l < 1e-12 else 100.0 - (100.0 / (1.0 + avg_g / avg_l))
    for i in range(period + 1, n):
        avg_g = (avg_g * (period - 1) + float(gains[i - 1])) / float(period)
        avg_l = (avg_l * (period - 1) + float(losses[i - 1])) / float(period)
        rsi[i] = 100.0 if avg_l < 1e-12 else 100.0 - (100.0 / (1.0 + avg_g / avg_l))
    return rsi


def make_sparkline(closes, points=20):
    arr = np.asarray(closes, dtype=np.float64)
    if len(arr) == 0:
        return []
    sub = arr[-points:]
    return [round(float(x), 6) for x in sub if not np.isnan(x)]


def _respect_rate_limit():
    while True:
        with _WEIGHT_LOCK:
            wait = _PAUSE_UNTIL - time.time()
        if wait > 0:
            time.sleep(min(wait, 2.0))
        else:
            break


def _update_weight(resp):
    global _LAST_USED_WEIGHT, _PAUSE_UNTIL
    w_hdr = resp.headers.get("X-MBX-USED-WEIGHT-1M")
    if not w_hdr:
        return
    try:
        w = int(w_hdr)
        limit_hi = 4800 if _USE_VISION else 1900
        limit_mid = 4200 if _USE_VISION else 1650
        with _WEIGHT_LOCK:
            _LAST_USED_WEIGHT = w
            if w >= limit_hi:
                _PAUSE_UNTIL = max(_PAUSE_UNTIL, time.time() + 12.0)
            elif w >= limit_mid:
                _PAUSE_UNTIL = max(_PAUSE_UNTIL, time.time() + 3.0)
    except Exception:
        pass


def fetch_binance_klines(session, symbol, interval, limit=400, end_time_ms=None, min_len=20):
    url = f"{VISION_API}/api/v3/klines" if _USE_VISION else f"{FAPI}/fapi/v1/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    if end_time_ms is not None:
        params["endTime"] = int(end_time_ms)

    for attempt in range(4):
        _respect_rate_limit()
        try:
            r = session.get(url, params=params, timeout=14)
            _update_weight(r)
            if r.status_code == 200:
                raw = r.json()
                if not isinstance(raw, list) or len(raw) < min_len:
                    return None
                v_arr = np.array([float(x[5]) for x in raw], dtype=np.float64)
                tbv_arr = np.array([float(x[9]) for x in raw], dtype=np.float64)
                return {
                    "t": np.array([int(x[0]) for x in raw], dtype=np.int64),
                    "o": np.array([float(x[1]) for x in raw], dtype=np.float64),
                    "h": np.array([float(x[2]) for x in raw], dtype=np.float64),
                    "l": np.array([float(x[3]) for x in raw], dtype=np.float64),
                    "c": np.array([float(x[4]) for x in raw], dtype=np.float64),
                    "v": v_arr,
                    "delta": (2.0 * tbv_arr) - v_arr,
                }
            if r.status_code in (418, 429):
                retry_after = float(r.headers.get("Retry-After", 10))
                with _WEIGHT_LOCK:
                    global _PAUSE_UNTIL
                    _PAUSE_UNTIL = max(_PAUSE_UNTIL, time.time() + max(retry_after, 8.0))
                time.sleep(min(retry_after, 12.0))
            else:
                time.sleep(0.4 * (attempt + 1))
        except Exception:
            time.sleep(0.5 * (attempt + 1))
    return None


def get_binance_universe_and_movers(session, max_coins=CRYPTO_MAX_COINS, min_vol=CRYPTO_MIN_VOL_24H):
    global _USE_VISION
    _USE_VISION = False

    r_info = session.get(f"{FAPI}/fapi/v1/exchangeInfo", timeout=15)
    r_tks = session.get(f"{FAPI}/fapi/v1/ticker/24hr", timeout=15)

    if r_info.status_code == 200 and r_tks.status_code == 200 and isinstance(r_tks.json(), list):
        ex_info = r_info.json()
        tks = r_tks.json()
        active = {
            s["symbol"]
            for s in ex_info.get("symbols", [])
            if s.get("status") == "TRADING"
            and s.get("quoteAsset") == "USDT"
            and s.get("contractType") == "PERPETUAL"
            and s["symbol"] not in EXCLUDED_CRYPTO_SYMS
        }
    else:
        _USE_VISION = True
        print(f"  [KRIPTO] Binance Futures bolge kisiti ({r_info.status_code}/{r_tks.status_code}) -> data-api.binance.vision yedege gecildi.")
        ex_info = session.get(f"{VISION_API}/api/v3/exchangeInfo", timeout=20).json()
        tks = session.get(f"{VISION_API}/api/v3/ticker/24hr", timeout=20).json()
        active = {
            s["symbol"]
            for s in ex_info.get("symbols", [])
            if s.get("status") == "TRADING"
            and s.get("quoteAsset") == "USDT"
            and s.get("isSpotTradingAllowed", True)
            and s["symbol"] not in EXCLUDED_CRYPTO_SYMS
        }

    liquid = [
        x for x in tks
        if isinstance(x, dict) and x.get("symbol") in active and float(x.get("quoteVolume", 0)) >= min_vol
    ]
    liquid.sort(key=lambda x: float(x["quoteVolume"]), reverse=True)
    syms = [x["symbol"] for x in liquid[:max_coins]]

    for must in ("BTCUSDT", "ETHUSDT", "SOLUSDT"):
        if must in active and must not in syms:
            syms.insert(0, must)

    ticker_map = {}
    for x in tks:
        if not isinstance(x, dict):
            continue
        sym = x.get("symbol")
        if sym in active:
            ticker_map[sym] = {
                "price": float(x.get("lastPrice", 0)),
                "change_24h": float(x.get("priceChangePercent", 0)),
                "quote_vol": float(x.get("quoteVolume", 0)),
                "high_24h": float(x.get("highPrice", 0)),
                "low_24h": float(x.get("lowPrice", 0)),
            }

    movers_pool = [
        {
            "symbol": s,
            "price": ticker_map[s]["price"],
            "change_pct": round(ticker_map[s]["change_24h"], 2),
            "quote_vol": round(ticker_map[s]["quote_vol"], 0),
        }
        for s in syms
        if s in ticker_map and ticker_map[s]["price"] > 0
    ]
    movers_pool.sort(key=lambda x: x["change_pct"], reverse=True)
    return syms, ticker_map, {"gainers": movers_pool[:5], "losers": list(reversed(movers_pool[-5:]))}


def fetch_yahoo_daily(session, ticker_code, range_str="2y", min_len=240):
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker_code}"
    params = {"interval": "1d", "range": range_str}

    for _ in range(2):
        try:
            r = session.get(url, params=params, headers=YAHOO_HEADERS, timeout=14)
            if r.status_code == 200:
                res = r.json().get("chart", {}).get("result")
                if res:
                    item = res[0]
                    ts_list = item.get("timestamp") or []
                    q = (item.get("indicators", {}).get("quote") or [{}])[0]
                    os_, hs_, ls_, cs_, vs_ = q.get("open"), q.get("high"), q.get("low"), q.get("close"), q.get("volume")
                    rows = []
                    for idx, ts in enumerate(ts_list):
                        o_v, h_v, l_v, c_v = os_[idx], hs_[idx], ls_[idx], cs_[idx]
                        v_v = vs_[idx] if vs_ and vs_[idx] is not None else 0.0
                        if None in (o_v, h_v, l_v, c_v):
                            continue
                        dt = datetime.fromtimestamp(int(ts), tz=timezone.utc)
                        rows.append((
                            int(ts) * 1000, dt.strftime("%d.%m.%Y"), int(dt.strftime("%Y%m%d")),
                            float(o_v), float(h_v), float(l_v), float(c_v), float(v_v),
                        ))
                    if len(rows) >= min_len:
                        return {
                            "t": np.array([x[0] for x in rows], dtype=np.int64),
                            "d": [x[1] for x in rows],
                            "di": np.array([x[2] for x in rows], dtype=np.int64),
                            "o": np.array([x[3] for x in rows], dtype=np.float64),
                            "h": np.array([x[4] for x in rows], dtype=np.float64),
                            "l": np.array([x[5] for x in rows], dtype=np.float64),
                            "c": np.array([x[6] for x in rows], dtype=np.float64),
                            "v": np.array([x[7] for x in rows], dtype=np.float64),
                        }
        except Exception:
            pass
        time.sleep(0.4)

    try:
        df = yf.Ticker(ticker_code).history(period=range_str, interval="1d", auto_adjust=True)
        if df is not None and len(df) >= min_len:
            df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
            if len(df) >= min_len:
                return {
                    "t": np.array([int(d.timestamp() * 1000) for d in df.index], dtype=np.int64),
                    "d": [d.strftime("%d.%m.%Y") for d in df.index],
                    "di": np.array([int(d.strftime("%Y%m%d")) for d in df.index], dtype=np.int64),
                    "o": df["Open"].to_numpy(dtype=np.float64),
                    "h": df["High"].to_numpy(dtype=np.float64),
                    "l": df["Low"].to_numpy(dtype=np.float64),
                    "c": df["Close"].to_numpy(dtype=np.float64),
                    "v": df["Volume"].to_numpy(dtype=np.float64),
                }
    except Exception:
        pass
    return None
