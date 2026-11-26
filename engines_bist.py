import numpy as np

from config import BIST_100, BIST_30, BIST_31_50, VIOP_ALL, MIN_DISPLAY_SCORE
from data_utils import calc_sma, calc_ema, calc_atr, calc_rsi, make_sparkline


def _pct_change_1d(c):
    if len(c) < 2 or c[-2] <= 0:
        return 0.0
    return ((float(c[-1]) - float(c[-2])) / float(c[-2])) * 100.0


def _index_tier_long_dip(symbol):
    if symbol in BIST_30:
        return "BIST30", 9
    if symbol in BIST_31_50:
        return "BIST50", 5
    return "BIST100", 0


def _build_bist_market_stats(bist_data):
    items = []
    above, tot = 0, 0
    for sym in BIST_100:
        d1 = bist_data.get(sym)
        if d1 is None or len(d1["c"]) < 55:
            continue
        c_arr, v_arr = d1["c"], d1["v"]
        val = float(np.mean(c_arr[-20:] * v_arr[-20:]))
        if val > 0:
            items.append((sym, val))
        ma50 = calc_sma(c_arr, 50)
        if not np.isnan(ma50[-1]):
            tot += 1
            if float(c_arr[-1]) > float(ma50[-1]):
                above += 1

    items.sort(key=lambda x: x[1], reverse=True)
    n_liq = float(len(items)) if items else 1.0
    vol_ranks = {sym: (idx + 1) / n_liq for idx, (sym, _) in enumerate(items)}
    breadth = (above / float(tot)) if tot >= 30 else 0.50
    return vol_ranks, breadth


def _eval_xu100_context(xu_data, breadth):
    ctx = {
        "ok": False, "bull_rally": False, "bear_crash": False,
        "panic_recent": False, "wind_label": "Veri Yok",
        "breadth": round(breadth * 100.0, 1),
    }
    if xu_data is None or len(xu_data["c"]) < 60:
        return ctx

    h, l, c = xu_data["h"], xu_data["l"], xu_data["c"]
    e8, e21, ma20 = calc_ema(c, 8), calc_ema(c, 21), calc_sma(c, 20)
    atr = calc_atr(h, l, c, 14)
    rsi7 = calc_rsi(c, 7, fill_val=50.0)

    i = len(c) - 1
    if any(np.isnan(x) for x in (e8[i], e21[i], ma20[i], atr[i])):
        return ctx

    c_now, e8_now, e21_now = float(c[i]), float(e8[i]), float(e21[i])
    bull_rally = (c_now >= e8_now) and (c_now >= e21_now)
    bear_crash = (c_now <= e8_now) and (c_now <= e21_now)

    st_vals = [
        (float(c[k]) - float(ma20[k])) / float(atr[k])
        for k in range(max(20, i - 2), i + 1)
        if not np.isnan(ma20[k]) and not np.isnan(atr[k]) and atr[k] > 0
    ]
    min_r3 = float(np.min(rsi7[max(0, i - 2) : i + 1]))
    panic_recent = (min_r3 <= 34.0) or ((min(st_vals) if st_vals else 0.0) <= -1.2)
    wind = "Boğa Rallisi (Long Açık)" if bull_rally else ("Satıcılı / Düşüş (Short Açık)" if bear_crash else "Kararsız / Geçiş")

    ctx.update({
        "ok": True, "bull_rally": bull_rally, "bear_crash": bear_crash,
        "panic_recent": panic_recent, "wind_label": wind,
    })
    return ctx


# =============================================================================
# 1. SPOT LONG DIPTEN TOPLA (PAKET C + BABADAGI HACIMLI DOJI +8)
# =============================================================================
def _scan_spot_long_dip(sym, d1, xu_ctx, vol_ranks, breadth):
    if breadth > 0.85:
        return None
    vol_pct = vol_ranks.get(sym, 1.0)
    if vol_pct > 0.50:
        return None

    o, h, l, c, v = d1["o"], d1["h"], d1["l"], d1["c"], d1["v"]
    n = len(c)
    if n < 220:
        return None

    atr = calc_atr(h, l, c, 14)
    vol_ma = calc_sma(v, 20)
    ma20, ma200 = calc_sma(c, 20), calc_sma(c, 200)
    rsi7, rsi14 = calc_rsi(c, 7, fill_val=50.0), calc_rsi(c, 14, fill_val=50.0)

    i = n - 1
    if any(np.isnan(x) for x in (ma20[i], ma200[i], atr[i])):
        return None

    disaster_ok = any(
        (k >= 20 and not np.isnan(ma20[k]) and rsi7[k] <= 22.0 and c[k] < ma20[k])
        for k in range(i - 3, i)
    )
    prev_body_hi = float(max(o[i - 1], c[i - 1]))
    if not (disaster_ok and (h[i] > h[i - 1]) and (c[i] > prev_body_hi) and (c[i] > o[i])):
        return None

    price, atr_cur = float(c[i]), max(float(atr[i]), 1e-12)
    m20, m200 = float(ma20[i]), float(ma200[i])
    if price >= m20 + atr_cur * 0.35 or price >= m200:
        return None

    for k in (i - 1, i):
        vma_k = float(vol_ma[k]) if (not np.isnan(vol_ma[k]) and vol_ma[k] > 0) else 0.0
        if c[k] < o[k] and vma_k > 0 and (float(v[k]) / vma_k) >= 1.35:
            return None

    day_pct = (price - float(c[i - 1])) / max(float(c[i - 1]), 1e-12)
    if day_pct > 0.090 or (float(h[i] - l[i]) / atr_cur) > 3.00:
        return None

    min_r7_3 = float(np.min(rsi7[i - 3 : i]))
    min_r14_3 = float(np.min(rsi14[i - 3 : i]))
    rsi_pts = 20 if (min_r7_3 <= 18.0 or min_r14_3 <= 28.0) else (10 if (min_r7_3 <= 21.0 or min_r14_3 <= 32.0) else 0)

    w0 = max(0, i - 119)
    rng_hi, rng_lo = float(np.max(h[w0 : i + 1])), float(np.min(l[w0 : i + 1]))
    rng_120 = max(rng_hi - rng_lo, 1e-12)
    cheap_pts = 18 if price <= rng_lo + rng_120 * 0.25 else (9 if price <= rng_lo + rng_120 * 0.40 else 0)

    ma200_dist = ((price - m200) / max(m200, 1e-12)) * 100.0
    m200_bonus = 14 if ma200_dist < -25.0 else 0
    idx_tier_name, idx_tier_pts = _index_tier_long_dip(sym)
    vol_pts = 7 if vol_pct <= 0.30 else 4
    xu_pts = 4 if xu_ctx["panic_recent"] else 0

    # Babadağı Destekçi: Önceki gün Yüksek Hacimli Doji / Dragonfly + Yeşil Teyit (+8)
    doji_bonus, doji_tag = 0, ""
    if not np.isnan(vol_ma[i - 1]) and vol_ma[i - 1] > 0:
        r_prev = max(float(h[i - 1] - l[i - 1]), 1e-12)
        b_prev = abs(float(c[i - 1] - o[i - 1]))
        dn_wick_prev = (min(float(o[i - 1]), float(c[i - 1])) - float(l[i - 1])) / r_prev
        if (b_prev / r_prev <= 0.25) and (float(v[i - 1]) / float(vol_ma[i - 1]) >= 1.50) and (dn_wick_prev >= 0.40):
            doji_bonus = 8
            doji_tag = " + 🕯️ Hacimli Doji"

    score = min(100, 28 + rsi_pts + cheap_pts + m200_bonus + idx_tier_pts + vol_pts + xu_pts + doji_bonus)
    if score < MIN_DISPLAY_SCORE:
        return None

    panic_lo = float(np.min(l[i - 3 : i + 1]))
    stop = max(price - atr_cur * 3.0, min(panic_lo - atr_cur * 0.55, price - atr_cur * 1.40))
    risk = max(price - stop, 1e-12)
    stop_pct = ((price - stop) / price) * 100.0
    if stop_pct > 11.0:
        return None

    tp1 = price + risk * 2.50
    badge = "🎯 80+ Nişancı" if score >= 80 else ("⚡ 75+ A+ Kalite" if score >= 75 else "Vitrin")

    return {
        "portfolio": "BIST", "market": "BIST", "module": "BIST_LONG",
        "engine_name": "Spot Long Dipten Topla", "symbol": sym, "direction": "LONG",
        "score": score, "price": price, "entry_price": price, "stop_loss": stop,
        "tp1_price": tp1, "tp2_price": None, "rr_ratio": 2.5,
        "bar_ts": int(d1["t"][i]), "change_24h": _pct_change_1d(c),
        "sparkline": make_sparkline(c), "setup_tag": f"Paket C Dip{doji_tag}",
        "meta": {
            "badge": badge, "stop_pct": round(stop_pct, 2), "ma200_dist": round(ma200_dist, 1),
            "idx_tier": idx_tier_name, "exit_desc": "Saf +2.5R • Günlük Kapanış Stopu & -2.0R Şok (40G)",
        },
        "exit_rules": {
            "mode": "SAF_25R_CLOSE_STOP", "init_risk": risk, "tp1_r": 2.50,
            "tp1_weight": 1.0, "cost_r": (price * 0.0020) / risk, "max_bars": 40,
            "cooldown_days": 5, "close_stop": True,
        },
    }


# =============================================================================
# 2. SPOT LONG TREND (v1.3 FINAL — TP1+BE + EMA21 TRAIL)
# =============================================================================
def _scan_spot_long_trend(sym, d1, xu_ctx):
    o, h, l, c, v = d1["o"], d1["h"], d1["l"], d1["c"], d1["v"]
    n = len(c)
    if n < 220 or xu_ctx["bear_crash"]:
        return None

    ema8, ema21 = calc_ema(c, 8), calc_ema(c, 21)
    sma50, sma200 = calc_sma(c, 50), calc_sma(c, 200)
    atr14 = calc_atr(h, l, c, 14)
    rsi14 = calc_rsi(c, 14)
    rsi_ma, vol20 = calc_sma(rsi14, 14), calc_sma(v, 20)

    i = n - 1
    if any(np.isnan(x) for x in (sma50[i], sma200[i], ema21[i - 5], atr14[i], rsi_ma[i], rsi_ma[i - 3], vol20[i])):
        return None

    price = float(c[i])
    if price <= float(sma200[i]):
        return None

    atr_cur = max(float(atr14[i]), 1e-9)
    rng = max(float(h[i] - l[i]), 1e-9)
    body = abs(float(c[i] - o[i]))
    body_atr, body_ratio, clv = body / atr_cur, body / rng, float(c[i] - l[i]) / rng
    vol_r = float(v[i] / vol20[i]) if vol20[i] > 0 else 1.0
    r14, rma = float(rsi14[i]), float(rsi_ma[i])

    if not (c[i] > o[i] and body_ratio >= 0.35 and clv >= 0.58):
        return None
    if body_atr < 0.25 or body_atr > 1.00 or (rng / atr_cur) > 1.85:
        return None
    if not (50.0 <= r14 <= 65.0) or (r14 <= rma):
        return None

    hi_90, lo_90 = float(np.max(h[i - 90 : i])), float(np.min(l[i - 90 : i]))
    pos_90 = (price - lo_90) / max(hi_90 - lo_90, 1e-9)
    if pos_90 < 0.15 or pos_90 > 0.50:
        return None

    e8, e21 = float(ema8[i]), float(ema21[i])
    e8_up = e8 > float(ema8[i - 3])
    ema821_full = (price > e8 > e21) and e8_up
    brk_e21_fresh = (float(c[i - 1]) <= float(ema21[i - 1])) and (price > e21 * 1.004) and e8_up
    cross_821_fresh = ema821_full and ((float(ema8[i - 1]) <= float(ema21[i - 1])) or (float(ema8[i - 2]) <= float(ema21[i - 2])))
    if not (brk_e21_fresh or cross_821_fresh):
        return None

    stop = min(min(e21, float(l[i])) - atr_cur * 0.20, price * 0.982)
    stop_pct = ((price - stop) / price) * 100.0
    if stop_pct <= 0.0 or stop_pct > 6.5:
        return None
    risk = max(price - stop, 1e-9)

    p_sma50 = 28.0 if price <= float(sma50[i]) * 1.012 else 10.0
    rsi_50_62, rsi_ma_up = (50.0 <= r14 <= 62.0), (rma > float(rsi_ma[i - 3]))
    p_rsi = 26.0 if (rsi_50_62 and rsi_ma_up) else (16.0 if (rsi_50_62 or rsi_ma_up) else 8.0)
    p_stop = 20.0 if stop_pct <= 3.8 else (10.0 if stop_pct <= 4.8 else 0.0)
    p_candle = 16.0 if (body_atr >= 0.55 and clv >= 0.80) else (8.0 if (body_atr >= 0.55 or clv >= 0.80) else 0.0)
    p_vol = 6.0 if vol_r >= 1.30 else (3.0 if vol_r >= 1.10 else 0.0)
    p_fib = 4.0 if pos_90 < 0.25 else 2.0

    score = int(round(min(100.0, p_sma50 + p_rsi + p_stop + p_candle + p_vol + p_fib)))
    if score < MIN_DISPLAY_SCORE:
        return None

    tp1 = price + risk * 1.00
    setup_lbl = "EMA 8-21 Kesişim" if cross_821_fresh else "EMA 21 Taze Kırılım"
    badge = "⚡ 70+ Elit" if score >= 70 else "Vitrin"

    return {
        "portfolio": "BIST", "market": "BIST", "module": "BIST_LONG",
        "engine_name": "Spot Long Trend", "symbol": sym, "direction": "LONG",
        "score": score, "price": price, "entry_price": price, "stop_loss": stop,
        "tp1_price": tp1, "tp2_price": None, "rr_ratio": 2.0,
        "bar_ts": int(d1["t"][i]), "change_24h": _pct_change_1d(c),
        "sparkline": make_sparkline(c), "setup_tag": setup_lbl,
        "meta": {
            "badge": badge, "stop_pct": round(stop_pct, 2), "rsi14": round(r14, 1),
            "vol_r": round(vol_r, 2), "exit_desc": "TP1 1.0R (%30)+BE / %70 EMA21 Trail (45G)",
        },
        "exit_rules": {
            "mode": "TP1_BE_EMA21", "init_risk": risk, "tp1_r": 1.00,
            "tp1_weight": 0.30, "cost_r": (price * 0.0015) / risk, "max_bars": 45,
            "cooldown_days": 5, "close_stop": False,
        },
    }


# =============================================================================
# 3. SPOT LONG ZIRVE KIRILIMI (v1.2 + KAPANIS STOPU & SOK)
# =============================================================================
def _scan_spot_long_breakout(sym, d1, xu_ctx):
    o, h, l, c, v = d1["o"], d1["h"], d1["l"], d1["c"], d1["v"]
    n = len(c)
    if n < 310 or xu_ctx["bear_crash"]:
        return None

    sma200, ema21 = calc_sma(c, 200), calc_ema(c, 21)
    atr5, atr14 = calc_atr(h, l, c, 5), calc_atr(h, l, c, 14)
    rsi14 = calc_rsi(c, 14)
    rsi_ma, vol20 = calc_sma(rsi14, 14), calc_sma(v, 20)

    i = n - 1
    if any(np.isnan(x) for x in (sma200[i], sma200[i - 10], ema21[i], atr14[i], atr5[i - 1], rsi_ma[i], vol20[i])):
        return None

    price, m200 = float(c[i]), float(sma200[i])
    if price <= m200:
        return None

    atr_cur = max(float(atr14[i]), 1e-9)
    rng = max(float(h[i] - l[i]), 1e-9)
    body = abs(float(c[i] - o[i]))
    body_atr, body_ratio, clv = body / atr_cur, body / rng, float(c[i] - l[i]) / rng
    up_wick_ratio = float(h[i] - max(o[i], c[i])) / rng
    vol_r = float(v[i] / vol20[i]) if vol20[i] > 0 else 1.0
    r14, rma = float(rsi14[i]), float(rsi_ma[i])

    if not (c[i] > o[i] and body_ratio >= 0.45 and clv >= 0.65 and up_wick_ratio <= 0.30):
        return None
    if body_atr < 0.25 or (rng / atr_cur) > 2.10 or r14 > 72.0:
        return None

    brk = None
    for W, w_key, w_pts in ((240, "240G", 18.0), (180, "180G", 16.0), (300, "300G", 12.0)):
        if i - W < 0:
            continue
        res_lvl = float(np.max(h[i - W : i - 2]))
        if np.any(c[i - 2 : i] > res_lvl * 1.006) or price <= res_lvl * 1.008 or (price - res_lvl) > atr_cur * 1.50:
            continue
        atr_prev = max(float(atr14[i - 1]), 1e-9)
        atr_ratio_prev = float(atr5[i - 1]) / atr_prev if not np.isnan(atr5[i - 1]) else 1.0
        pre_h, pre_l, pre_vol = h[i - 6 : i], l[i - 6 : i], v[i - 6 : i]
        pre_body = np.abs(c[i - 6 : i] - o[i - 6 : i])
        near_res = float(np.max(pre_h)) <= res_lvl * 1.055
        band_ok = (float(np.max(pre_h)) - float(np.min(pre_l))) <= atr_cur * 2.50
        atr_ok = (atr_ratio_prev <= 0.90) or (float(np.mean(pre_body)) <= atr_cur * 0.65)
        vol_calm = float(np.mean(pre_vol)) <= (float(vol20[i]) * 1.05 if vol20[i] > 0 else 1.0)
        comp_score = int(near_res) + int(band_ok) + int(atr_ok) + int(vol_calm)
        comp_lbl, comp_grade = ("Zirve Altı Sıkışma", 2) if (comp_score >= 3 and near_res) else (("Kısmi Sıkışma", 1) if (comp_score >= 2 and near_res) else ("Direkt Zirve Kırılımı", 0))
        brk = {"w_key": w_key, "w_pts": w_pts, "res": res_lvl, "comp_lbl": comp_lbl, "comp_grade": comp_grade, "pre_vol_avg": float(np.mean(pre_vol))}
        break

    if brk is None:
        return None

    stop = min(min(brk["res"] * 0.992, float(l[i])) - atr_cur * 0.15, price * 0.984)
    stop_pct = ((price - stop) / price) * 100.0
    if stop_pct <= 0.0 or stop_pct > 6.0:
        return None
    risk = max(price - stop, 1e-9)

    p_win_comp = brk["w_pts"] + (16.0 if brk["comp_grade"] == 2 else (8.0 if brk["comp_grade"] == 1 else 0.0))
    p_candle = 24.0 if ((0.55 <= body_atr <= 1.15) and (clv >= 0.72) and (up_wick_ratio <= 0.20)) else 12.0
    comp_surge = (brk["pre_vol_avg"] > 0) and (float(v[i]) >= brk["pre_vol_avg"] * 1.25) and (vol_r >= 1.15)
    p_vol = 18.0 if ((vol_r >= 1.40) or comp_surge) else (9.0 if vol_r >= 1.15 else 0.0)
    p_stop = 16.0 if (3.5 <= stop_pct <= 5.8) else (6.0 if 2.8 <= stop_pct < 3.5 else 0.0)
    p_rsi = 6.0 if ((52.0 <= r14 <= 68.0) and (r14 > rma)) else (3.0 if ((45.0 <= r14 <= 72.0) and (r14 > rma)) else 0.0)
    p_sma200 = 4.0 if (m200 > float(sma200[i - 10]) * 1.001) else 2.0

    score = int(round(min(100.0, p_win_comp + p_candle + p_vol + p_stop + p_rsi + p_sma200)))
    if score < MIN_DISPLAY_SCORE:
        return None

    tp1 = price + risk * 1.05
    badge = "⚡ 70+ Elit" if score >= 70 else "Vitrin"

    return {
        "portfolio": "BIST", "market": "BIST", "module": "BIST_LONG",
        "engine_name": "Spot Long Zirve Kırılımı", "symbol": sym, "direction": "LONG",
        "score": score, "price": price, "entry_price": price, "stop_loss": stop,
        "tp1_price": tp1, "tp2_price": None, "rr_ratio": 2.1,
        "bar_ts": int(d1["t"][i]), "change_24h": _pct_change_1d(c),
        "sparkline": make_sparkline(c), "setup_tag": f"{brk['w_key']} • {brk['comp_lbl']}",
        "meta": {
            "badge": badge, "stop_pct": round(stop_pct, 2), "rsi14": round(r14, 1),
            "vol_r": round(vol_r, 2), "exit_desc": "TP1 1.05R (%35)+BE / EMA21 • Kapanış Stopu & Şok (45G)",
        },
        "exit_rules": {
            "mode": "TP1_BE_EMA21", "init_risk": risk, "tp1_r": 1.05,
            "tp1_weight": 0.35, "cost_r": (price * 0.0015) / risk, "max_bars": 45,
            "cooldown_days": 5, "close_stop": True,
        },
    }


# =============================================================================
# 4. SHORT VIOP DIP KIRILIMI (v1.6 FINAL)
# =============================================================================
def _scan_short_viop_dip(sym, d1, xu_ctx):
    o, h, l, c, v = d1["o"], d1["h"], d1["l"], d1["c"], d1["v"]
    n = len(c)
    if n < 220 or xu_ctx["bull_rally"]:
        return None

    sma200, atr14 = calc_sma(c, 200), calc_atr(h, l, c, 14)
    rsi14 = calc_rsi(c, 14)
    rsi_ma, vol20 = calc_sma(rsi14, 14), calc_sma(v, 20)

    i = n - 1
    if any(np.isnan(x) for x in (sma200[i], atr14[i], rsi_ma[i], vol20[i])):
        return None

    price = float(c[i])
    if price >= float(sma200[i]):
        return None

    atr_cur = max(float(atr14[i]), 1e-9)
    rng = max(float(h[i] - l[i]), 1e-9)
    body = abs(float(c[i] - o[i]))
    body_atr, body_ratio, clv = body / atr_cur, body / rng, float(c[i] - l[i]) / rng
    low_wick_ratio = float(min(o[i], c[i]) - l[i]) / rng
    has_vol = (sym != "XU030") and (vol20[i] > 0) and (v[i] > 0)
    vol_r = float(v[i] / vol20[i]) if has_vol else 1.0
    r14, rma = float(rsi14[i]), float(rsi_ma[i])

    if c[i] >= o[i] or body_atr < 0.25 or body_atr > 1.05 or (rng / atr_cur) > 1.85:
        return None
    if body_ratio < 0.35 or clv > 0.42 or low_wick_ratio > 0.35 or not (32.0 <= r14 <= 56.0):
        return None

    hi_90, lo_90 = float(np.max(h[i - 90 : i])), float(np.min(l[i - 90 : i]))
    span_90 = max(hi_90 - lo_90, 1e-9)
    pos_cur = (price - lo_90) / span_90
    pos_prev = (float(c[i - 1]) - lo_90) / span_90
    if not ((0.25 < pos_cur <= 0.50) and (pos_prev > 0.50)):
        return None
    eq_lvl = lo_90 + 0.50 * span_90

    if has_vol:
        for k in (i - 1, i - 2):
            if k >= 20 and not np.isnan(vol20[k]) and vol20[k] > 0 and float(v[k] / vol20[k]) >= 1.30:
                rk = max(float(h[k] - l[k]), 1e-9)
                bk = abs(float(c[k] - o[k]))
                lwk = float(min(o[k], c[k]) - l[k])
                if float(l[k]) <= eq_lvl + atr_cur * 0.55 and ((bk / rk) <= 0.24 or (lwk / rk) >= 0.48):
                    return None

    vol_pre5 = float(np.mean(v[i - 5 : i])) if has_vol else 0.0
    vol_dry = has_vol and (vol_pre5 <= float(vol20[i]) * 0.95)
    hugging = float(np.max(h[i - 5 : i])) <= eq_lvl + atr_cur * 2.25
    tam_buildup = vol_dry and hugging

    buf = 0.15 if tam_buildup else (0.22 if (vol_dry or hugging) else 0.35)
    stop = max(max(eq_lvl, float(h[i])) + atr_cur * buf, price * (1.018 if tam_buildup else 1.022))
    stop_pct = ((stop - price) / price) * 100.0
    if stop_pct > 7.5:
        return None
    risk = max(stop - price, 1e-9)
    corridor_r = (price - lo_90) / risk

    p_buildup = 26.0 if tam_buildup else (16.0 if vol_dry else (7.0 if hugging else 0.0))
    purposeful, non_panic = (0.45 <= body_atr <= 1.05), (clv > 0.12)
    p_mum = 22.0 if (purposeful and non_panic) else (11.0 if (purposeful or non_panic) else 0.0)
    rsi_below_ma, rsi_40_52 = (r14 < rma), (40.0 <= r14 <= 52.0)
    p_rsi = 22.0 if (rsi_40_52 and rsi_below_ma) else (12.0 if (rsi_40_52 or (((36.0 <= r14 < 40.0) or (52.0 < r14 <= 56.0)) and rsi_below_ma)) else (5.0 if rsi_below_ma else 0.0))
    if has_vol:
        vol_strong = (vol_r >= 1.25) or (vol_pre5 > 0 and float(v[i]) >= vol_pre5 * 1.35 and vol_r >= 0.95)
        vol_mid = (vol_r >= 1.05) or (vol_pre5 > 0 and float(v[i]) >= vol_pre5 * 1.15)
        p_vol = 18.0 if vol_strong else (9.0 if vol_mid else 0.0)
    else:
        p_vol = 9.0
    p_corr = 12.0 if corridor_r >= 2.20 else (6.0 if corridor_r >= 1.50 else 0.0)

    score = int(round(min(100.0, p_buildup + p_mum + p_rsi + p_vol + p_corr)))
    if score < MIN_DISPLAY_SCORE:
        return None

    tp1, tp2 = price - risk * 1.50, price - risk * 3.20
    tag_lbl = "90G 0.50 • Tam Build-Up" if tam_buildup else ("90G 0.50 • Kuru Hacim" if vol_dry else "90G 0.50 • Denge Kırılımı")
    badge = "⚡ 70+ Elit" if score >= 70 else "Vitrin"

    return {
        "portfolio": "BIST", "market": "BIST", "module": "BIST_VIOP_SHORT",
        "engine_name": "Short VİOP Dip Kırılımı", "symbol": sym, "direction": "SHORT",
        "score": score, "price": price, "entry_price": price, "stop_loss": stop,
        "tp1_price": tp1, "tp2_price": tp2, "rr_ratio": 2.6,
        "bar_ts": int(d1["t"][i]), "change_24h": _pct_change_1d(c),
        "sparkline": make_sparkline(c), "setup_tag": tag_lbl,
        "meta": {
            "badge": badge, "stop_pct": round(stop_pct, 2), "rsi14": round(r14, 1),
            "vol_r": round(vol_r, 2), "exit_desc": "TP1 1.5R (%35)+BE / TP2 3.2R (%65) • 30G",
        },
        "exit_rules": {
            "mode": "TIERED_BE", "init_risk": risk, "tp1_r": 1.50, "tp2_r": 3.20,
            "tp1_weight": 0.35, "cost_r": (price * 0.0015) / risk, "max_bars": 30,
            "cooldown_days": 5, "close_stop": False,
        },
    }


# =============================================================================
# 5. SHORT VIOP TREND (v1.7 FINAL)
# =============================================================================
def _scan_short_viop_trend(sym, d1, xu_ctx):
    o, h, l, c, v = d1["o"], d1["h"], d1["l"], d1["c"], d1["v"]
    n = len(c)
    if n < 200 or xu_ctx["bull_rally"]:
        return None

    ema8, ema21, atr14 = calc_ema(c, 8), calc_ema(c, 21), calc_atr(h, l, c, 14)
    rsi14 = calc_rsi(c, 14)
    rsi_ma, vol20 = calc_sma(rsi14, 14), calc_sma(v, 20)

    i = n - 1
    if any(np.isnan(x) for x in (ema21[i], ema21[i - 5], atr14[i], rsi_ma[i], rsi_ma[i - 3], vol20[i])):
        return None

    price = float(c[i])
    atr_cur = max(float(atr14[i]), 1e-9)
    rng = max(float(h[i] - l[i]), 1e-9)
    body = abs(float(c[i] - o[i]))
    body_atr, body_ratio, clv = body / atr_cur, body / rng, float(c[i] - l[i]) / rng
    has_vol = (sym != "XU030") and (vol20[i] > 0) and (v[i] > 0)
    vol_r = float(v[i] / vol20[i]) if has_vol else 1.0
    r14, rma = float(rsi14[i]), float(rsi_ma[i])

    if not (c[i] < o[i] and body_ratio >= 0.35 and clv <= 0.42):
        return None
    if body_atr < 0.25 or body_atr > 1.00 or (rng / atr_cur) > 1.85 or r14 >= rma or r14 < 32.0:
        return None

    hi_180, lo_180 = float(np.max(h[i - 180 : i])), float(np.min(l[i - 180 : i]))
    pos_180 = (price - lo_180) / max(hi_180 - lo_180, 1e-9)
    if not (0.48 <= pos_180 <= 0.82):
        return None

    min_12 = float(np.min(l[i - 12 : i]))
    if ((float(h[i]) - min_12) / max(min_12, 1e-9) >= 0.18) and (price > float(c[i - 5]) * 1.06):
        return None

    e8, e21 = float(ema8[i]), float(ema21[i])
    e8_dn, e21_dn = (e8 < float(ema8[i - 3])), (e21 < float(ema21[i - 5]))
    ema821_full = (price < e8 < e21) and e8_dn
    ema821_strict = ema821_full and e21_dn
    brk_e21_fresh = (float(c[i - 1]) >= float(ema21[i - 1])) and (price < e21 * 0.996) and e8_dn
    cross_821_fresh = ema821_full and ((float(ema8[i - 1]) >= float(ema21[i - 1])) or (float(ema8[i - 2]) >= float(ema21[i - 2])))
    if not (brk_e21_fresh or cross_821_fresh):
        return None

    stop = max(max(e21, float(h[i])) + atr_cur * 0.30, price * 1.020)
    stop_pct = ((stop - price) / price) * 100.0
    if stop_pct > 7.5:
        return None
    risk = max(stop - price, 1e-9)
    if ((price - lo_180) / risk) < 1.20:
        return None

    p_ema = 28.0 if ema821_strict else (18.0 if ema821_full else 10.0)
    p_zone = 24.0 if (0.50 <= pos_180 <= 0.75) else 10.0
    p_candle = 20.0 if (body_atr >= 0.55 and clv > 0.15) else (12.0 if (body_atr >= 0.55 or clv > 0.15) else 4.0)
    breathable = (4.2 <= stop_pct <= 6.8) or ((risk / atr_cur) >= 1.25 and stop_pct <= 6.8)
    p_stop = 16.0 if breathable else (8.0 if 3.2 <= stop_pct < 4.2 else 0.0)
    p_rsi = 6.0 if ((42.0 <= r14 <= 62.0) and ((float(rsi14[i - 3]) - r14) < 4.5)) else 0.0
    p_vol = (6.0 if vol_r >= 1.30 else (3.0 if vol_r >= 1.10 else 0.0)) if has_vol else 3.0

    score = int(round(min(100.0, p_ema + p_zone + p_candle + p_stop + p_rsi + p_vol)))
    if score < MIN_DISPLAY_SCORE:
        return None

    tp1, tp2 = price - risk * 1.20, price - risk * 2.40
    setup_lbl = "EMA 8-21 Çift Eğim" if ema821_strict else ("EMA 8-21 Kesişim" if cross_821_fresh else "EMA 21 Taze Kırılım")
    badge = "⚡ 70+ Seçkin" if score >= 70 else "Vitrin"

    return {
        "portfolio": "BIST", "market": "BIST", "module": "BIST_VIOP_SHORT",
        "engine_name": "Short VİOP Trend", "symbol": sym, "direction": "SHORT",
        "score": score, "price": price, "entry_price": price, "stop_loss": stop,
        "tp1_price": tp1, "tp2_price": tp2, "rr_ratio": 1.8,
        "bar_ts": int(d1["t"][i]), "change_24h": _pct_change_1d(c),
        "sparkline": make_sparkline(c), "setup_tag": setup_lbl,
        "meta": {
            "badge": badge, "stop_pct": round(stop_pct, 2), "rsi14": round(r14, 1),
            "vol_r": round(vol_r, 2), "exit_desc": "TP1 1.2R (%50)+BE / TP2 2.4R (%50) • 20G",
        },
        "exit_rules": {
            "mode": "TIERED_BE", "init_risk": risk, "tp1_r": 1.20, "tp2_r": 2.40,
            "tp1_weight": 0.50, "cost_r": (price * 0.0015) / risk, "max_bars": 20,
            "cooldown_days": 5, "close_stop": False,
        },
    }


# =============================================================================
# 6. SHORT VIOP ZIRVE DONUSU (v3.6 MOD A + BABADAGI MEZAR TASI DOJI +8)
# =============================================================================
def _scan_short_viop_peak(sym, d1):
    o, h, l, c, v = d1["o"], d1["h"], d1["l"], d1["c"], d1["v"]
    n = len(c)
    if n < 120:
        return None

    atr, vol_ma = calc_atr(h, l, c, 14), calc_sma(v, 20)
    rsi14 = calc_rsi(c, 14, fill_val=50.0)
    is_index = (sym == "XU030")

    i = n - 1
    if np.isnan(atr[i]):
        return None

    price, prev_c = float(c[i]), float(c[i - 1])
    atr_cur = max(float(atr[i]), 1e-12)
    day_chg = ((price - prev_c) / max(prev_c, 1e-12)) * 100.0
    if day_chg <= (-6.5 if is_index else -8.5):
        return None

    hi_90, lo_90 = float(np.max(h[i - 90 : i - 10])), float(np.min(l[i - 90 : i - 10]))
    rng_90 = hi_90 - lo_90
    if rng_90 < atr_cur * 4.8:
        return None
    q50 = lo_90 + rng_90 * 0.50

    bar_rng = max(float(h[i] - l[i]), 1e-12)
    body = abs(float(o[i] - c[i]))
    up_wick = float(h[i] - max(o[i], c[i]))
    clv = float(c[i] - l[i]) / bar_rng

    is_red = c[i] < o[i]
    bear_engulf = (c[i - 1] > o[i - 1] and is_red and c[i] <= o[i - 1] and o[i] >= c[i - 1] * 0.995 and body >= abs(float(c[i - 1] - o[i - 1])) * 0.95)
    bear_pin = (up_wick >= bar_rng * 0.42) and (clv <= 0.40) and (bar_rng >= atr_cur * 0.70)
    bear_reject = bear_engulf or bear_pin or (is_red and clv <= 0.45)

    rec_hi3 = float(np.max(h[i - 2 : i + 1]))
    if not ((rec_hi3 > hi_90) and (price < hi_90) and ((hi_90 - price) <= atr_cur * 2.00) and bear_reject):
        return None

    raw_stop = rec_hi3 + atr_cur * 0.45
    if (raw_stop - price) > atr_cur * 3.00:
        return None
    stop = max(raw_stop, price + atr_cur * 1.10)
    risk = max(stop - price, 1e-12)
    stop_pct = ((stop - price) / price) * 100.0

    target = max(price * 0.15, max(price - risk * 3.20, q50)) if q50 <= price - risk * 1.60 else max(price * 0.15, price - risk * 2.00)

    cur_rsi, prev_rsi = float(rsi14[i]), float(rsi14[i - 1])
    max_rsi3 = float(np.max(rsi14[i - 3 : i]))
    rsi_75_down = (max_rsi3 >= 75.0) and (cur_rsi < 75.0)
    rsi_55_cross = (prev_rsi >= 55.0) and (cur_rsi < 55.0)
    p_rsi = 30.0 if rsi_75_down else (18.0 if rsi_55_cross else (12.0 if 35.0 <= cur_rsi < 55.0 else 0.0))

    has_vol = (not is_index) and (not np.isnan(vol_ma[i])) and (vol_ma[i] > 0) and (v[i] > 0)
    r2, b2 = max(float(h[i - 2] - l[i - 2]), 1e-12), abs(float(c[i - 2] - o[i - 2]))
    r1, b1 = max(float(h[i - 1] - l[i - 1]), 1e-12), abs(float(c[i - 1] - o[i - 1]))
    peak_exh = ((h[i - 1] > hi_90) or (h[i - 2] > hi_90)) and ((r1 < r2 * 0.85) or (b1 < b2 * 0.72)) and ((v[i - 1] < v[i - 2]) if has_vol else True) and is_red
    p_exh = 24.0 if peak_exh else 0.0

    if has_vol:
        vol_r = float(v[i] / vol_ma[i])
        vol_strong = (vol_r >= 1.50) or ((v[i] > v[i - 1]) and (vol_r >= 1.35))
        p_vol = 16.0 if vol_strong else 0.0
    else:
        vol_r, vol_strong, p_vol = 1.0, False, 8.0

    # Babadağı Destekçi: Zirvede Yüksek Hacimli Mezar Taşı Doji + Kırmızı Teyit (+8)
    doji_bonus, has_gravestone = 0.0, False
    if has_vol and not np.isnan(vol_ma[i - 1]) and vol_ma[i - 1] > 0 and is_red:
        up_wick_prev = (float(h[i - 1]) - max(float(o[i - 1]), float(c[i - 1]))) / r1
        if (b1 / r1 <= 0.25) and (float(v[i - 1]) / float(vol_ma[i - 1]) >= 1.50) and (up_wick_prev >= 0.40):
            doji_bonus = 8.0
            has_gravestone = True

    score = int(round(min(100.0, 30.0 + p_rsi + p_exh + p_vol + doji_bonus)))
    if score < MIN_DISPLAY_SCORE:
        return None

    badge = "🎯 75+ Sniper" if score >= 75 else ("⚡ 70+ Kuvvetli" if score >= 70 else "Vitrin")
    tags = ["90G Zirve SFP"]
    if rsi_75_down:
        tags.append("RSI75↓")
    elif rsi_55_cross:
        tags.append("RSI55↓")
    if peak_exh:
        tags.append("Sönüm")
    if vol_strong:
        tags.append("Hacim")
    if has_gravestone:
        tags.append("🕯️ Hacimli Doji")

    return {
        "portfolio": "BIST", "market": "BIST", "module": "BIST_VIOP_SHORT",
        "engine_name": "Short VİOP Zirve Dönüşü", "symbol": sym, "direction": "SHORT",
        "score": score, "price": price, "entry_price": price, "stop_loss": stop,
        "tp1_price": target, "tp2_price": None, "rr_ratio": round((price - target) / risk, 2),
        "bar_ts": int(d1["t"][i]), "change_24h": _pct_change_1d(c),
        "sparkline": make_sparkline(c), "setup_tag": " + ".join(tags),
        "meta": {
            "badge": badge, "stop_pct": round(stop_pct, 2), "rsi14": round(cur_rsi, 1),
            "vol_r": round(vol_r, 2), "exit_desc": "MOD A • Kapanış Stopu & Şok • 5G Kesici • 22G",
        },
        "exit_rules": {
            "mode": "MOD_A_5G", "init_risk": risk, "tp1_r": (price - target) / risk,
            "tp1_weight": 1.0, "cost_r": (price * (0.0012 if is_index else 0.0020)) / risk,
            "max_bars": 22, "cooldown_days": 5, "close_stop": True,
        },
    }


def scan_all_bist(bist_data_map, xu100_data):
    vol_ranks, breadth = _build_bist_market_stats(bist_data_map)
    xu_ctx = _eval_xu100_context(xu100_data, breadth)

    long_signals, short_signals = [], []
    for sym in BIST_100:
        d1 = bist_data_map.get(sym)
        if d1 is None:
            continue
        for fn in (
            lambda: _scan_spot_long_dip(sym, d1, xu_ctx, vol_ranks, breadth),
            lambda: _scan_spot_long_trend(sym, d1, xu_ctx),
            lambda: _scan_spot_long_breakout(sym, d1, xu_ctx),
        ):
            sig = fn()
            if sig is not None:
                long_signals.append(sig)

    for sym in VIOP_ALL:
        d1 = bist_data_map.get(sym)
        if d1 is None:
            continue
        for fn in (
            lambda: _scan_short_viop_dip(sym, d1, xu_ctx),
            lambda: _scan_short_viop_trend(sym, d1, xu_ctx),
            lambda: _scan_short_viop_peak(sym, d1),
        ):
            sig = fn()
            if sig is not None:
                short_signals.append(sig)

    long_signals.sort(key=lambda x: x["score"], reverse=True)
    short_signals.sort(key=lambda x: x["score"], reverse=True)
    return long_signals, short_signals, xu_ctx