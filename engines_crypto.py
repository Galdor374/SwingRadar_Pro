import numpy as np

from config import MIN_DISPLAY_SCORE, SCALPER_MIN_VOL_USD
from data_utils import calc_sma, calc_atr, make_sparkline


def _build_major_resistances(h_d, c_d, d_idx, d_atr, include_d60=False):
    end_j = d_idx - 2
    if d_idx < 126 or end_j <= 30:
        return []
    scan_120, scan_90 = max(0, d_idx - 120), max(0, d_idx - 90)
    tol = max(d_atr * 0.55, float(c_d[d_idx]) * 0.011)
    abs_hi_90_i = scan_90 + int(np.argmax(h_d[scan_90:end_j]))
    abs_hi_90_full = float(np.max(h_d[scan_90 : d_idx + 1]))
    raw_highs = [(float(h_d[abs_hi_90_i]), abs_hi_90_i)]
    for w in ((120, 90, 60) if include_d60 else (120, 90)):
        s_j = max(0, d_idx - w)
        if end_j - s_j < 10:
            continue
        hi_i = s_j + int(np.argmax(h_d[s_j:end_j]))
        raw_highs.append((float(h_d[hi_i]), hi_i))
        pwin = 5 if w >= 90 else 4
        for j in range(s_j + pwin, min(end_j, d_idx - pwin + 1)):
            if h_d[j] >= np.max(h_d[j - pwin : j + pwin + 1]):
                raw_highs.append((float(h_d[j]), j))
    raw_highs.sort(key=lambda x: x[1])
    clusters = []
    for val, idx in raw_highs:
        placed = False
        for cl in clusters:
            if abs(val - cl["lvl"]) <= tol:
                cl["lvl"] = max(cl["lvl"], val)
                cl["oldest"] = min(cl["oldest"], idx)
                placed = True
                break
        if not placed:
            clusters.append({"lvl": val, "oldest": idx})
    min_age = 30 if include_d60 else 60
    out = []
    for cl in clusters:
        lvl, touches, last_t, first_hit = cl["lvl"], 0, -15, cl["oldest"]
        for j in range(scan_120, end_j):
            if (h_d[j] >= lvl - tol * 0.70) and (h_d[j] <= lvl + tol * 0.50):
                if j < first_hit:
                    first_hit = j
                if j - last_t >= 5:
                    touches += 1
                    last_t = j
        age_days = max(2, d_idx - first_hit)
        is_abs = lvl >= abs_hi_90_full - tol * 0.35
        if age_days <= min_age and not is_abs:
            continue
        tier, pool = (95, "ABS90") if is_abs else ((120, "D120") if age_days > 90 else ((90, "D90") if age_days > 60 else (60, "D60")))
        out.append({"lvl": lvl, "tier": tier, "pool": pool, "touches": max(1, touches), "is_abs_90_hi": is_abs})
    out.sort(key=lambda x: (1 if x["is_abs_90_hi"] else 0, 1 if x["pool"] == "D90" else 0, x["tier"], x["touches"]), reverse=True)
    return out[: 8 if include_d60 else 6]


def _build_major_supports(l_d, c_d, d_idx, d_atr, include_d60=False):
    end_j = d_idx - 2
    if d_idx < 126 or end_j <= 30:
        return []
    scan_120, scan_90 = max(0, d_idx - 120), max(0, d_idx - 90)
    tol = max(d_atr * 0.55, float(c_d[d_idx]) * 0.011)
    abs_lo_90_i = scan_90 + int(np.argmin(l_d[scan_90:end_j]))
    abs_lo_90_full = float(np.min(l_d[scan_90 : d_idx + 1]))
    raw_lows = [(float(l_d[abs_lo_90_i]), abs_lo_90_i)]
    for w in ((120, 90, 60) if include_d60 else (120, 90)):
        s_j = max(0, d_idx - w)
        if end_j - s_j < 10:
            continue
        lo_i = s_j + int(np.argmin(l_d[s_j:end_j]))
        raw_lows.append((float(l_d[lo_i]), lo_i))
        pwin = 5 if w >= 90 else 4
        for j in range(s_j + pwin, min(end_j, d_idx - pwin + 1)):
            if l_d[j] <= np.min(l_d[j - pwin : j + pwin + 1]):
                raw_lows.append((float(l_d[j]), j))
    raw_lows.sort(key=lambda x: x[1])
    clusters = []
    for val, idx in raw_lows:
        placed = False
        for cl in clusters:
            if abs(val - cl["lvl"]) <= tol:
                cl["lvl"] = min(cl["lvl"], val)
                cl["oldest"] = min(cl["oldest"], idx)
                placed = True
                break
        if not placed:
            clusters.append({"lvl": val, "oldest": idx})
    min_age = 30 if include_d60 else 60
    out = []
    for cl in clusters:
        lvl, touches, last_t, first_hit = cl["lvl"], 0, -15, cl["oldest"]
        for j in range(scan_120, end_j):
            if (l_d[j] <= lvl + tol * 0.70) and (l_d[j] >= lvl - tol * 0.50):
                if j < first_hit:
                    first_hit = j
                if j - last_t >= 5:
                    touches += 1
                    last_t = j
        age_days = max(2, d_idx - first_hit)
        is_abs = lvl <= abs_lo_90_full + tol * 0.35
        if age_days <= min_age and not is_abs:
            continue
        tier, pool = (95, "ABS90") if is_abs else ((120, "D120") if age_days > 90 else ((90, "D90") if age_days > 60 else (60, "D60")))
        out.append({"lvl": lvl, "tier": tier, "pool": pool, "touches": max(1, touches), "is_abs_90_lo": is_abs})
    out.sort(key=lambda x: (1 if x["is_abs_90_lo"] else 0, 1 if x["pool"] == "D90" else 0, x["tier"], x["touches"]), reverse=True)
    return out[: 8 if include_d60 else 6]


def _build_fresh_h4_levels(c_d, d_idx, d_atr, arr_4h, h4_idx, is_support=True):
    if d_idx < 35 or h4_idx < 190:
        return []
    tol = max(d_atr * 0.42, float(c_d[d_idx]) * 0.008)
    raw = []
    h4_start, h4_end = max(0, h4_idx - 180), h4_idx - 6
    for j in range(h4_start + 4, min(h4_end, h4_idx - 3)):
        if is_support and arr_4h[j] <= np.min(arr_4h[j - 4 : j + 5]):
            raw.append((float(arr_4h[j]), max(1, int((h4_idx - j) / 6))))
        elif (not is_support) and arr_4h[j] >= np.max(arr_4h[j - 4 : j + 5]):
            raw.append((float(arr_4h[j]), max(1, int((h4_idx - j) / 6))))
    if not raw:
        return []
    raw.sort(key=lambda x: x[1])
    clusters = []
    for val, age_d in raw:
        placed = False
        for cl in clusters:
            if abs(val - cl["lvl"]) <= tol:
                cl["lvl"] = min(cl["lvl"], val) if is_support else max(cl["lvl"], val)
                cl["age_days"] = max(cl["age_days"], age_d)
                placed = True
                break
        if not placed:
            clusters.append({"lvl": val, "age_days": age_d})
    scan_s = max(0, h4_idx - 210)
    out = []
    for cl in clusters:
        lvl, touches, last_t = cl["lvl"], 0, -20
        for j in range(scan_s, h4_idx - 4):
            hit = (arr_4h[j] <= lvl + tol * 0.70 and arr_4h[j] >= lvl - tol * 0.50) if is_support else (arr_4h[j] >= lvl - tol * 0.70 and arr_4h[j] <= lvl + tol * 0.50)
            if hit and (j - last_t >= 8):
                touches += 1
                last_t = j
        out.append({"lvl": lvl, "pool": "H4_DIP" if is_support else "H4_TEPE", "touches": max(1, touches), "age_days": cl["age_days"], "is_abs_90_hi": False})
    out.sort(key=lambda x: (x["touches"], -x["age_days"]), reverse=True)
    return out[: 8 if is_support else 6]


def _compute_sfp_stop(direction, price, lvl, ext_val, body_edge, wick_len, wick_ratio, atr_cur, atr_buf, min_atr, min_pct):
    min_dist = max(atr_cur * min_atr, price * min_pct)
    if direction == "LONG":
        raw_stop = min(ext_val - atr_cur * atr_buf, price - min_dist)
        raw_pct = ((price - raw_stop) / price) * 100.0
        if raw_pct <= 5.2:
            return raw_stop, raw_pct
        if wick_ratio >= 0.55:
            mid_stop = min(body_edge - wick_len * 0.55, price - min_dist)
            mid_pct = ((price - mid_stop) / price) * 100.0
            if mid_stop < lvl and mid_pct <= 5.2:
                return mid_stop, mid_pct
        return None, raw_pct
    raw_stop = max(ext_val + atr_cur * atr_buf, price + min_dist)
    raw_pct = ((raw_stop - price) / price) * 100.0
    if raw_pct <= 5.2:
        return raw_stop, raw_pct
    if wick_ratio >= 0.55:
        mid_stop = max(body_edge + wick_len * 0.55, price + min_dist)
        mid_pct = ((mid_stop - price) / price) * 100.0
        if mid_stop > lvl and mid_pct <= 5.2:
            return mid_stop, mid_pct
    return None, raw_pct


def _compute_bo_stop(direction, price, lvl, bar_ext, atr_cur, atr_buf, min_atr, min_pct):
    min_dist = max(atr_cur * min_atr, price * min_pct)
    if direction == "LONG":
        raw_stop = min(lvl - atr_cur * atr_buf, bar_ext - atr_cur * 0.35, price - min_dist)
        raw_pct = ((price - raw_stop) / price) * 100.0
        if raw_pct <= 5.2:
            return raw_stop, raw_pct
        alt_stop = min(lvl - atr_cur * 0.35, price - min_dist)
        alt_pct = ((price - alt_stop) / price) * 100.0
        return (alt_stop, alt_pct) if (alt_stop < lvl and alt_pct <= 5.2) else (None, raw_pct)
    raw_stop = max(lvl + atr_cur * atr_buf, bar_ext + atr_cur * 0.35, price + min_dist)
    raw_pct = ((raw_stop - price) / price) * 100.0
    if raw_pct <= 5.2:
        return raw_stop, raw_pct
    alt_stop = max(lvl + atr_cur * 0.35, price + min_dist)
    alt_pct = ((alt_stop - price) / price) * 100.0
    return (alt_stop, alt_pct) if (alt_stop > lvl and alt_pct <= 5.2) else (None, raw_pct)


# =============================================================================
# MOTOR 7: KRIPTO 1H SCALPER (KILITLI FINAL — >= 25M$ HACIM • DELTA • MA100 VETO)
# =============================================================================
def scan_crypto_scalper_symbol(symbol, pack, btc_h4_ctx, ticker_info):
    quote_vol = float(ticker_info.get("quote_vol", 0.0)) if ticker_info else 0.0
    if quote_vol < SCALPER_MIN_VOL_USD:
        return []

    d1, h4, h1 = pack.get("d1"), pack.get("h4"), pack.get("h1")
    if not d1 or not h4 or not h1:
        return []
    if len(d1["c"]) < 140 or len(h4["c"]) < 200 or len(h1["c"]) < 45:
        return []

    dt_d, h_d, l_d, c_d = d1["t"], d1["h"], d1["l"], d1["c"]
    dt_4h, h_4h, l_4h, c_4h = h4["t"], h4["h"], h4["l"], h4["c"]
    t, o, h, l, c, v, delta = h1["t"], h1["o"], h1["h"], h1["l"], h1["c"], h1["v"], h1["delta"]

    i = len(c) - 2
    s = i - 1
    ti = int(t[i])

    d_idx = int(np.searchsorted(dt_d, ti - 86_400_000, side="right")) - 1
    h4_idx = int(np.searchsorted(dt_4h, ti - 10_800_000, side="right")) - 1
    if d_idx < 128 or h4_idx < 192:
        return []

    d_atr = calc_atr(h_d, l_d, c_d, 14, fill_initial=True)
    ma50_d = calc_sma(c_d, 50)
    ma100_4h = calc_sma(c_4h, 100)
    atr_1h = calc_atr(h, l, c, 14, fill_initial=True)
    vol_ma = calc_sma(v, 24)
    if np.isnan(ma50_d[d_idx]) or np.isnan(ma100_4h[h4_idx]) or np.isnan(atr_1h[i]):
        return []

    btc_bull = True
    dt_btc, c_btc, ma100_btc = btc_h4_ctx
    if dt_btc is not None:
        b_idx = int(np.searchsorted(dt_btc, ti - 10_800_000, side="right")) - 1
        if b_idx >= 100 and not np.isnan(ma100_btc[b_idx]):
            btc_bull = float(c_btc[b_idx]) > float(ma100_btc[b_idx])

    price = float(c[i])
    atr_cur, atr_s = max(float(atr_1h[i]), 1e-12), max(float(atr_1h[s]), 1e-12)
    m100_4h_val = float(ma100_4h[h4_idx])

    t1d_up = float(c_d[d_idx]) > float(ma50_d[d_idx])
    t4h_up = float(c_4h[h4_idx]) > m100_4h_val
    trend_up_both = t1d_up and t4h_up
    trend_dn_both = (not t1d_up) and (not t4h_up)
    is_transition = (not trend_up_both) and (not trend_dn_both)

    s_rng = max(float(h[s] - l[s]), 1e-12)
    s_body = abs(float(c[s] - o[s]))
    s_up_wick, s_dn_wick = float(h[s] - max(o[s], c[s])), float(min(o[s], c[s]) - l[s])
    s_up_ratio, s_dn_ratio = s_up_wick / s_rng, s_dn_wick / s_rng

    i_rng = max(float(h[i] - l[i]), 1e-12)
    i_body = abs(float(c[i] - o[i]))
    i_up_ratio = float(h[i] - max(o[i], c[i])) / i_rng
    i_dn_ratio = float(min(o[i], c[i]) - l[i]) / i_rng
    i_clv = float(c[i] - l[i]) / i_rng

    vol_s = float(v[s] / vol_ma[s]) if (not np.isnan(vol_ma[s]) and vol_ma[s] > 0) else 1.0
    vol_i = float(v[i] / vol_ma[i]) if (not np.isnan(vol_ma[i]) and vol_ma[i] > 0) else 1.0
    vol_r = max(vol_s, vol_i)
    pre_span_atr = (float(np.max(h[i - 6 : i])) - float(np.min(l[i - 6 : i]))) / atr_cur
    pre_mean_c = float(np.mean(c[i - 6 : i]))
    delta_2b = float(delta[i] + delta[s]) / max(float(v[i] + v[s]), 1e-12)

    res_no60 = _build_major_resistances(h_d, c_d, d_idx, float(d_atr[d_idx]), include_d60=False)
    res_60 = _build_major_resistances(h_d, c_d, d_idx, float(d_atr[d_idx]), include_d60=True)
    sup_no60 = _build_major_supports(l_d, c_d, d_idx, float(d_atr[d_idx]), include_d60=False)
    sup_60 = _build_major_supports(l_d, c_d, d_idx, float(d_atr[d_idx]), include_d60=True)
    h4_sup = _build_fresh_h4_levels(c_d, d_idx, float(d_atr[d_idx]), l_4h, h4_idx, is_support=True)
    h4_res = _build_fresh_h4_levels(c_d, d_idx, float(d_atr[d_idx]), h_4h, h4_idx, is_support=False)

    signals = []
    chg_24h = float(ticker_info.get("change_24h", 0.0)) if ticker_info else 0.0

    def try_emit(sub_eng, direction, sc, stop, spct, is_3b=False, pool_lbl=""):
        if sc < MIN_DISPLAY_SCORE:
            return
        sgn = 1.0 if direction == "LONG" else -1.0
        if (delta_2b * sgn) <= 0.0:
            return

        risk = max((price - stop) if direction == "LONG" else (stop - price), 1e-12)
        if is_3b:
            tp1_p = max(price * 0.05, price - min(risk * 1.15, price * 0.040))
            tp2_p = max(price * 0.02, price - min(risk * 2.60, price * 0.100))
            tp1_r, tp2_r, tp1_w = (price - tp1_p) / risk, (price - tp2_p) / risk, 0.40
        else:
            tp1_r, tp2_r, tp1_w = 1.50, 3.20, 0.35
            tp1_p = (price + risk * tp1_r) if direction == "LONG" else max(price * 0.05, price - risk * tp1_r)
            tp2_p = (price + risk * tp2_r) if direction == "LONG" else max(price * 0.02, price - risk * tp2_r)

        if direction == "LONG" and price < m100_4h_val < tp1_p:
            return
        if direction == "SHORT" and tp1_p < m100_4h_val < price:
            return

        badge = "🔥 80+ Elit" if sc >= 80 else ("⚡ 70+ Kuvvetli" if sc >= 70 else "Vitrin")
        signals.append({
            "portfolio": "SCALPER", "market": "CRYPTO", "module": "CRYPTO_SCALPER",
            "engine_name": f"1H • {sub_eng}", "symbol": symbol, "direction": direction,
            "score": sc, "price": price, "entry_price": price, "stop_loss": stop,
            "tp1_price": tp1_p, "tp2_price": tp2_p, "rr_ratio": round(tp2_r, 2),
            "bar_ts": ti, "change_24h": chg_24h, "sparkline": make_sparkline(c[:-1]),
            "setup_tag": f"{pool_lbl} • Δ {'+' if delta_2b > 0 else ''}{delta_2b*100:.1f}%",
            "meta": {
                "badge": badge, "stop_pct": round(spct, 2), "vol_24h_m": round(quote_vol / 1e6, 1),
                "exit_desc": "Paket D: 48S • Saatlik Kapanış Stopu & -2.0R Şok • TP1+BE",
            },
            "exit_rules": {
                "mode": "SCALPER_PAKET_D", "init_risk": risk, "tp1_r": tp1_r, "tp2_r": tp2_r,
                "tp1_weight": tp1_w, "cost_r": (price * 0.0015) / risk, "max_bars": 48,
                "cooldown_hours": 48, "close_stop": True,
            },
        })

    if trend_up_both and (pre_span_atr <= 3.5) and (c[i] > o[i]) and (i_clv >= 0.60) and (i_up_ratio <= 0.32) and (atr_cur * 0.35 <= i_body <= atr_cur * 2.80):
        best = None
        for lv in res_no60:
            if lv["touches"] < 3:
                continue
            lvl = lv["lvl"]
            min_brk = max(atr_cur * 0.12, lvl * 0.0015)
            if not ((float(c[s]) <= lvl + atr_cur * 0.25) and (float(np.min(c[i - 6 : i])) < lvl) and (price > lvl + min_brk) and ((price - lvl) <= atr_cur * 2.40)):
                continue
            stop, spct = _compute_bo_stop("LONG", price, lvl, float(l[i]), atr_cur, 0.75, 1.80, 0.026)
            if stop is None:
                continue
            p_mum = 30.0 if (0.60 <= i_clv <= 0.88 and i_body <= atr_cur * 2.00) else 10.0
            p_pl = 24.0 if lv["pool"] == "ABS90" else (18.0 if lv["pool"] == "D90" else 10.0)
            p_vol = 20.0 if vol_i >= 1.80 else (16.0 if vol_i >= 1.40 else (8.0 if vol_i >= 1.15 else 0.0))
            sc = int(round(p_mum + 26.0 + p_pl + p_vol))
            if best is None or sc > best[0]:
                best = (sc, stop, spct, lv["pool"])
        if best:
            try_emit("Long Zirve Kırılımı", "LONG", best[0], best[1], best[2], False, best[3])

    if (not trend_dn_both) and (s_body <= atr_s * 2.8) and (i_body <= atr_cur * 3.2) and (s_up_ratio >= 0.35) and (s_up_wick >= atr_s * 0.45) and (h[i] < h[s]) and (c[i] < o[i]) and (i_clv <= 0.50):
        best = None
        for lv in res_60:
            if lv["touches"] < 2 or (trend_up_both and (not lv["is_abs_90_hi"])):
                continue
            lvl = lv["lvl"]
            if not (float(np.min(c[s - 6 : s])) < lvl and float(h[s]) > lvl + max(atr_s * 0.08, lvl * 0.0010) and price < lvl and (lvl - price) <= atr_cur * 2.40):
                continue
            stop, spct = _compute_sfp_stop("SHORT", price, lvl, float(h[s]), max(float(o[s]), float(c[s])), s_up_wick, s_up_ratio, atr_cur, 0.45, 1.15, 0.015)
            if stop is None:
                continue
            bear_engulf = (c[i] < l[s]) or ((o[i] >= max(o[s], c[s]) - atr_cur * 0.15) and (c[i] < min(o[s], c[s])))
            sc = int(round((0.0 if bear_engulf else 38.0) + (32.0 if vol_r < 1.40 else (10.0 if vol_r < 1.70 else 0.0)) + (20.0 if is_transition else 6.0) + (10.0 if 2.2 <= spct <= 5.5 else 2.0)))
            if best is None or sc > best[0]:
                best = (sc, stop, spct, lv["pool"])
        if best:
            try_emit("Short Zirve Dönüşü", "SHORT", best[0], best[1], best[2], False, best[3])

    if trend_up_both and (s_body <= atr_s * 2.8) and (i_body <= atr_cur * 3.2) and (s_dn_ratio >= 0.45) and (s_dn_wick >= atr_s * 0.50) and (c[s] >= o[s] - s_rng * 0.10) and (l[i] > l[s]) and (c[i] >= o[i] - i_rng * 0.18):
        best = None
        for lv in h4_sup:
            if lv["touches"] < 2:
                continue
            lvl = lv["lvl"]
            if not (float(np.max(c[s - 6 : s])) > lvl and float(l[s]) < lvl - max(atr_s * 0.08, lvl * 0.0010) and float(c[s]) >= lvl - atr_s * 0.15 and price > lvl and (price - lvl) <= atr_cur * 2.20):
                continue
            stop, spct = _compute_sfp_stop("LONG", price, lvl, float(l[s]), min(float(o[s]), float(c[s])), s_dn_wick, s_dn_ratio, atr_cur, 0.55, 1.35, 0.022)
            if stop is None:
                continue
            p_w = 32.0 if (0.45 <= s_dn_ratio < 0.55) else (12.0 if s_dn_ratio < 0.62 else 2.0)
            p_v = 30.0 if vol_r >= 1.80 else (24.0 if vol_r >= 1.40 else (10.0 if vol_r >= 1.15 else 0.0))
            dist_a = (price - lvl) / atr_cur
            p_e = 22.0 if (dist_a <= 1.35 and i_body <= atr_cur * 1.25) else (10.0 if dist_a <= 1.85 else 2.0)
            p_m = 16.0 if (c[i] > o[i] and i_clv >= 0.65 and c[i] > max(o[s], c[s])) else (8.0 if c[i] > o[i] else 2.0)
            sc = int(round(p_w + p_v + p_e + p_m))
            if best is None or sc > best[0]:
                best = (sc, stop, spct, lv["pool"])
        if best:
            try_emit("Long Trend Takip", "LONG", best[0], best[1], best[2], False, best[3])

    if trend_dn_both and (s_body <= atr_s * 2.8) and (i_body <= atr_cur * 3.2) and (s_up_ratio >= 0.35) and (s_up_wick >= atr_s * 0.45) and (h[i] < h[s]) and (c[i] < o[i]) and (i_clv <= 0.50):
        best = None
        for lv in res_no60 + h4_res:
            if lv["touches"] < 2 or (lv["pool"] == "H4_TEPE" and s_up_ratio >= 0.65):
                continue
            lvl = lv["lvl"]
            if not (float(np.min(c[s - 6 : s])) < lvl and float(h[s]) > lvl + max(atr_s * 0.08, lvl * 0.0010) and price < lvl and (lvl - price) <= atr_cur * 2.40):
                continue
            stop, spct = _compute_sfp_stop("SHORT", price, lvl, float(h[s]), max(float(o[s]), float(c[s])), s_up_wick, s_up_ratio, atr_cur, 0.50, 1.30, 0.020)
            if stop is None:
                continue
            p_c = 32.0 if (i_clv <= 0.35 and c[i] < min(o[s], c[s])) else 8.0
            p_p = 32.0 if lv["pool"] in ("D90", "ABS90") else (26.0 if lv["pool"] == "D120" else 8.0)
            p_e = 16.0 if ((lvl - price) / atr_cur <= 1.40) else 4.0
            p_b = 10.0 if (not btc_bull) else 2.0
            p_w = 10.0 if (0.35 <= s_up_ratio < 0.58) else 3.0
            sc = int(round(p_c + p_p + p_e + p_b + p_w))
            if best is None or sc > best[0]:
                best = (sc, stop, spct, lv["pool"])
        if best:
            try_emit("Short Trend Takip", "SHORT", best[0], best[1], best[2], False, best[3])

    if (not trend_up_both) and (s_body <= atr_s * 2.8) and (i_body <= atr_cur * 3.2) and (s_dn_ratio >= 0.35) and (s_dn_wick >= atr_s * 0.45) and (l[i] > l[s]) and (c[i] > o[i]) and (i_clv >= 0.50):
        best = None
        for lv in sup_60:
            if lv["touches"] < 2 or (trend_dn_both and (not lv["is_abs_90_lo"])):
                continue
            lvl = lv["lvl"]
            if not (float(np.max(c[s - 6 : s])) > lvl and float(l[s]) < lvl - max(atr_s * 0.08, lvl * 0.0010) and price > lvl and (price - lvl) <= atr_cur * 2.40):
                continue
            stop, spct = _compute_sfp_stop("LONG", price, lvl, float(l[s]), min(float(o[s]), float(c[s])), s_dn_wick, s_dn_ratio, atr_cur, 0.45, 1.15, 0.015)
            if stop is None:
                continue
            bull_engulf = (c[i] > h[s]) or ((o[i] <= min(o[s], c[s]) + atr_cur * 0.15) and (c[i] > max(o[s], c[s])))
            sc = int(round((0.0 if bull_engulf else 38.0) + (32.0 if vol_r < 1.40 else (10.0 if vol_r < 1.70 else 0.0)) + (20.0 if is_transition else 6.0) + (10.0 if 2.2 <= spct <= 5.5 else 2.0)))
            if best is None or sc > best[0]:
                best = (sc, stop, spct, lv["pool"])
        if best:
            try_emit("Long Dipten Dönüş", "LONG", best[0], best[1], best[2], False, f"R8 {best[3]}")

    if trend_dn_both and (pre_span_atr <= 3.5) and (c[i] < o[i]) and (i_clv <= 0.38) and (i_dn_ratio <= 0.30) and (atr_cur * 0.40 <= i_body <= atr_cur * 2.80):
        best = None
        for lv in sup_no60:
            if lv["touches"] < 2:
                continue
            lvl = lv["lvl"]
            min_brk = max(atr_cur * 0.12, lvl * 0.0015)
            if not (float(c[s]) >= lvl - atr_cur * 0.20 and float(np.max(c[i - 6 : i])) > lvl and price < lvl - min_brk and (lvl - price) <= atr_cur * 2.20):
                continue
            stop, spct = _compute_bo_stop("SHORT", price, lvl, float(h[i]), atr_cur, 0.65, 1.60, 0.025)
            if stop is None:
                continue
            p_m = 36.0 if (i_clv <= 0.28 and i_dn_ratio <= 0.22 and vol_i < 1.40) else (18.0 if vol_i < 1.40 else (6.0 if i_clv <= 0.28 else 0.0))
            p_b = 20.0 if (pre_span_atr <= 3.80 and abs(pre_mean_c - lvl) / atr_cur <= 2.20) else (8.0 if pre_span_atr <= 4.50 else 0.0)
            p_st = 20.0 if (2.5 <= spct <= 5.5) else 6.0
            p_c = 16.0 if (((lvl - price) / atr_cur) <= 1.25 and i_body <= atr_cur * 1.90) else (8.0 if ((lvl - price) / atr_cur) <= 1.65 else 2.0)
            p_p = 8.0 if lv["pool"] == "ABS90" else (6.0 if lv["pool"] == "D90" else 4.0)
            sc = int(round(p_m + p_b + p_st + p_c + p_p))
            if best is None or sc > best[0]:
                best = (sc, stop, spct, lv["pool"])
        if best:
            try_emit("Short Dip Kırılımı", "SHORT", best[0], best[1], best[2], True, best[3])

    return signals


# =============================================================================
# MOTOR 8: KRIPTO SWING (v2.5 + 45G VADE + BIRLESIK C: 1D + 4H TUTUNMA TETIK)
# =============================================================================
def _calc_w1_pts(w1, direction, price, atr_cur):
    if not w1 or len(w1["c"]) < 10:
        return 0.0
    wc, wh, wl = w1["c"][:-1], w1["h"][:-1], w1["l"][:-1]
    w_trend_ok = (wc[-1] > wc[-2]) if direction == "LONG" else (wc[-1] < wc[-2])
    pts_lvl = 0.0
    if direction == "LONG":
        lows = [float(wl[k]) for k in range(1, len(wl) - 1) if wl[k] <= min(wl[k - 1], wl[k + 1]) and wl[k] <= price]
        if lows and ((price - max(lows)) / atr_cur) <= 3.5:
            pts_lvl = 11.0
    else:
        highs = [float(wh[k]) for k in range(1, len(wh) - 1) if wh[k] >= max(wh[k - 1], wh[k + 1]) and wh[k] >= price]
        if highs and ((min(highs) - price) / atr_cur) <= 3.5:
            pts_lvl = 11.0
    return min(18.0, pts_lvl + (7.0 if w_trend_ok else 0.0))


def build_crypto_swing_ctx(crypto_packs):
    btc_up, eth_btc_up, breadth = True, True, 0.50
    btc_d1 = crypto_packs.get("BTCUSDT", {}).get("d1")
    if btc_d1 and len(btc_d1["c"]) >= 55:
        m50 = calc_sma(btc_d1["c"][:-1], 50)
        if not np.isnan(m50[-1]):
            btc_up = float(btc_d1["c"][-2]) > float(m50[-1])

    eth_d1 = crypto_packs.get("ETHUSDT", {}).get("d1")
    if btc_d1 and eth_d1:
        n_c = min(len(btc_d1["c"]), len(eth_d1["c"])) - 1
        if n_c >= 25:
            ratios = eth_d1["c"][-n_c - 1 : -1] / np.maximum(btc_d1["c"][-n_c - 1 : -1], 1e-12)
            r_ma20 = calc_sma(ratios, 20)
            if not np.isnan(r_ma20[-1]):
                eth_btc_up = float(ratios[-1]) > float(r_ma20[-1])

    above, tot = 0, 0
    for pack in crypto_packs.values():
        d1 = pack.get("d1")
        if d1 and len(d1["c"]) >= 55:
            m50 = calc_sma(d1["c"][:-1], 50)
            if not np.isnan(m50[-1]):
                tot += 1
                if float(d1["c"][-2]) > float(m50[-1]):
                    above += 1
    if tot >= 20:
        breadth = above / float(tot)
    return {"btc_up": btc_up, "eth_btc_up": eth_btc_up, "breadth": breadth}


def scan_crypto_swing_symbol(symbol, pack, swing_ctx, ticker_info):
    d1, h4, w1 = pack.get("d1"), pack.get("h4"), pack.get("w1")
    if not d1 or not h4 or len(d1["c"]) < 140 or len(h4["c"]) < 25:
        return None

    o_d, h_d, l_d, c_d = d1["o"], d1["h"], d1["l"], d1["c"]
    o_4h, h_4h, l_4h, c_4h = h4["o"], h4["h"], h4["l"], h4["c"]

    i = len(c_d) - 2
    prev_i = i - 1
    atr_d = calc_atr(h_d, l_d, c_d, 14, fill_initial=True)
    ma50, ma100, ma200 = calc_sma(c_d, 50), calc_sma(c_d, 100), calc_sma(c_d, 200)
    atr_4h = calc_atr(h_4h, l_4h, c_4h, 14, fill_initial=True)

    if np.isnan(ma50[i]) or np.isnan(ma100[i]):
        return None

    ma_spread = np.full(i + 1, np.nan, dtype=np.float64)
    for k in range(50, i + 1):
        vs_k = [x for x in (ma50[k], ma100[k], ma200[k]) if not np.isnan(x)]
        if len(vs_k) >= 2 and atr_d[k] > 0:
            ma_spread[k] = (max(vs_k) - min(vs_k)) / atr_d[k]

    recent_squeezed = False
    for k in range(max(50, i - 3), i + 1):
        sp_win = ma_spread[max(0, k - 119) : k + 1]
        sp_valid = sp_win[~np.isnan(sp_win)]
        if len(sp_valid) >= 15 and not np.isnan(ma_spread[k]):
            if ma_spread[k] <= max(float(np.percentile(sp_valid, 25)), 1.45):
                recent_squeezed = True
                break
    if not recent_squeezed:
        return None

    br = swing_ctx["breadth"]
    btc_up = swing_ctx["btc_up"]
    eth_btc_up = swing_ctx["eth_btc_up"]
    atr_cur = max(float(atr_d[i]), 1e-12)
    vs = [x for x in (ma50[i], ma100[i], ma200[i]) if not np.isnan(x)]
    vs_prev = [x for x in (ma50[prev_i], ma100[prev_i], ma200[prev_i]) if not np.isnan(x)]
    if len(vs) < 2 or len(vs_prev) < 2:
        return None

    chg_24h = float(ticker_info.get("change_24h", 0.0)) if ticker_info else 0.0
    k_last4 = len(c_4h) - 2

    def build_sig(layer_tag, direction, price, bar_ts, body_ok, wick_ok, trigger_tf):
        pts_rot = 11.0 if symbol == "BTCUSDT" else (22.0 if (eth_btc_up if direction == "LONG" else (not eth_btc_up)) else -15.0)
        pts_btc = 15.0 if (btc_up if direction == "LONG" else (not btc_up)) else -15.0
        pts_candle = 10.0 if (body_ok and wick_ok) else 0.0
        pts_w1 = _calc_w1_pts(w1, direction, price, atr_cur)
        score = int(round(min(100.0, max(0.0, 35.0 + pts_rot + pts_w1 + pts_btc + pts_candle))))
        if score < MIN_DISPLAY_SCORE:
            return None

        min_dist = max(atr_cur * 1.25, price * 0.035)
        if (min_dist / price) * 100.0 > 11.5:
            return None
        max_dist = min(atr_cur * 3.20, price * 0.115)
        dist = min(max_dist, max(min_dist, atr_cur * 1.85))
        stop = (price - dist) if direction == "LONG" else (price + dist)
        stop_pct = (dist / price) * 100.0
        if stop_pct > 11.5:
            return None

        tp1 = (price + dist * 1.80) if direction == "LONG" else max(price * 0.05, price - dist * 1.80)
        tp2 = (price + dist * 3.50) if direction == "LONG" else max(price * 0.02, price - dist * 3.50)

        diamond = " 💎 Dar Stop" if stop_pct <= 8.0 else ""
        badge = ("🔥 80+ Elit" if score >= 80 else ("⚡ 70+ Kuvvetli" if score >= 70 else "Vitrin")) + diamond
        return {
            "portfolio": "SWING", "market": "CRYPTO", "module": "CRYPTO_SWING",
            "engine_name": f"Kripto Swing ({layer_tag})", "symbol": symbol, "direction": direction,
            "score": score, "price": price, "entry_price": price, "stop_loss": stop,
            "tp1_price": tp1, "tp2_price": tp2, "rr_ratio": 3.5,
            "bar_ts": bar_ts, "change_24h": chg_24h, "sparkline": make_sparkline(c_d[:-1]),
            "setup_tag": f"{layer_tag} • MA Sıkışma Patlaması",
            "meta": {
                "badge": badge, "stop_pct": round(stop_pct, 2), "trigger_tf": trigger_tf,
                "exit_desc": "TP1 1.8R (%35)+Soft -0.35R / TP2 3.5R • 12G Ölü • 45G",
            },
            "exit_rules": {
                "mode": "SWING_V25_45G", "init_risk": dist, "tp1_r": 1.80, "tp2_r": 3.50,
                "tp1_weight": 0.35, "cost_r": (price * 0.0015) / dist, "max_bars": 45 * 6,
                "dead_bars": 12 * 6, "cooldown_hours": 120, "close_stop": False,
            },
        }

    p_d = float(c_d[i])
    l_hit = (p_d > max(vs)) and (c_d[i] > o_d[i]) and (c_d[prev_i] <= max(vs_prev) or (p_d - max(vs)) <= atr_cur * 1.50)
    s_hit = (p_d < min(vs)) and (c_d[i] < o_d[i]) and (c_d[prev_i] >= min(vs_prev) or (min(vs) - p_d) <= atr_cur * 1.50)
    if l_hit or s_hit:
        direction = "LONG" if l_hit else "SHORT"
        rng = max(float(h_d[i] - l_d[i]), 1e-12)
        body = abs(float(c_d[i] - o_d[i]))
        adv = float(h_d[i] - max(o_d[i], c_d[i])) if direction == "LONG" else float(min(o_d[i], c_d[i]) - l_d[i])
        day_chg = ((c_d[i] - c_d[prev_i]) / max(c_d[prev_i], 1e-12)) * 100.0
        st4 = ((float(c_4h[k_last4]) - float(c_4h[k_last4 - 4])) / max(float(atr_4h[k_last4]), 1e-12)) * (1.0 if direction == "LONG" else -1.0)
        ok_l = (direction == "LONG") and not (day_chg >= 12.0 or (p_d - o_d[i]) > atr_cur * 2.5 or adv >= rng * 0.48 or br > 0.85 or st4 >= 2.8)
        ok_s = (direction == "SHORT") and not (day_chg <= -12.0 or (o_d[i] - p_d) > atr_cur * 2.5 or adv >= rng * 0.48 or br < 0.15 or st4 >= 2.8)
        if ok_l or ok_s:
            sig = build_sig("✅ 1D Kapanış", direction, p_d, int(d1["t"][i]), body >= rng * 0.45, adv <= rng * 0.28, "1D")
            if sig is not None:
                return sig

    k_hold = k_last4
    k_brk = k_last4 - 1
    if k_brk >= 5:
        band_hi, band_lo = max(vs), min(vs)
        p_brk, p_hold = float(c_4h[k_brk]), float(c_4h[k_hold])
        l2_l = (p_d <= band_hi) and (float(c_4h[k_brk - 1]) <= band_hi) and (p_brk > band_hi) and (c_4h[k_brk] > o_4h[k_brk]) and ((p_brk - band_hi) <= atr_cur * 1.50) and (p_hold > band_hi)
        l2_s = (p_d >= band_lo) and (float(c_4h[k_brk - 1]) >= band_lo) and (p_brk < band_lo) and (c_4h[k_brk] < o_4h[k_brk]) and ((band_lo - p_brk) <= atr_cur * 1.50) and (p_hold < band_lo)
        if l2_l or l2_s:
            direction = "LONG" if l2_l else "SHORT"
            rng = max(float(h_4h[k_brk] - l_4h[k_brk]), 1e-12)
            body = abs(float(c_4h[k_brk] - o_4h[k_brk]))
            clv = float(c_4h[k_brk] - l_4h[k_brk]) / rng
            adv = float(h_4h[k_brk] - max(o_4h[k_brk], c_4h[k_brk])) if direction == "LONG" else float(min(o_4h[k_brk], c_4h[k_brk]) - l_4h[k_brk])
            mv_pct = ((p_brk - p_d) / max(p_d, 1e-12)) * 100.0
            st4 = ((p_brk - float(c_4h[k_brk - 4])) / max(float(atr_4h[k_brk]), 1e-12)) * (1.0 if direction == "LONG" else -1.0)
            ok_l = (direction == "LONG") and not (clv < 0.60 or mv_pct >= 12.0 or adv >= rng * 0.48 or br > 0.85 or st4 >= 2.8)
            ok_s = (direction == "SHORT") and not (clv > 0.40 or mv_pct <= -12.0 or adv >= rng * 0.48 or br < 0.15 or st4 >= 2.8)
            if ok_l or ok_s:
                return build_sig("⚡ 4H Tutunma", direction, p_hold, int(h4["t"][k_hold]), body >= rng * 0.45, adv <= rng * 0.28, "4H")

    return None