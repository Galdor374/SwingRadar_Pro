import csv
import json
import os
from datetime import datetime
import numpy as np

from config import STATE_FILE, TRADES_CSV, PORTFOLIOS
from data_utils import calc_ema


def _now_str():
    return datetime.now().strftime("%d.%m.%Y %H:%M")


def load_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    state = {"accounts": {}, "open_positions": [], "closed_trades": [], "cooldowns": {}, "last_scan": ""}
    now = _now_str()
    for key, cfg in PORTFOLIOS.items():
        state["accounts"][key] = {
            "name": cfg["name"],
            "currency": cfg["currency"],
            "initial": cfg["initial"],
            "alloc_pct": cfg["alloc_pct"],
            "equity": cfg["initial"],
            "peak_equity": cfg["initial"],
            "max_dd_pct": 0.0,
            "cum_r": 0.0,
            "curve": [{"date": now, "equity": cfg["initial"], "ret_pct": 0.0, "cum_r": 0.0, "dd_pct": 0.0}],
        }
    return state


def save_state(state):
    state["last_scan"] = _now_str()
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def _append_trade_csv(tr):
    exists = os.path.exists(TRADES_CSV)
    fields = [
        "closed_at", "opened_at", "portfolio", "engine", "symbol", "direction", "score",
        "entry_price", "exit_price", "stop_loss", "stop_pct", "pos_size", "r_val",
        "pnl_money", "exit_reason", "equity_after",
    ]
    with open(TRADES_CSV, "a", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            w.writeheader()
        w.writerow({k: tr.get(k, "") for k in fields})


def _eval_position_bars(pos, series, live_price=None):
    """Acik islemi giris barindan sonraki kapanmis mumlarda kilitli cikis kuraliyla denetler."""
    t, o, h, l, c = series["t"], series["o"], series["h"], series["l"], series["c"]
    # Kripto'da son bar acik (forming), BIST'te son bar gun sonu/guncel
    end_idx = len(c) - 1 if pos["market"] == "CRYPTO" else len(c)
    idxs = [k for k in range(end_idx) if int(t[k]) > int(pos["entry_bar_ts"])]

    rules = pos["exit_rules"]
    mode = rules["mode"]
    price = float(pos["entry_price"])
    stop = float(pos["stop_loss"])
    risk = float(rules["init_risk"])
    cost_r = float(rules["cost_r"])
    direction = pos["direction"]
    tp1_r = float(rules.get("tp1_r", 1.5))
    tp2_r = float(rules.get("tp2_r", 3.2))
    tp1_w = float(rules.get("tp1_weight", 0.35))
    rem_w = 1.0 - tp1_w
    max_bars = int(rules.get("max_bars", 45))
    close_stop = bool(rules.get("close_stop", False))

    tp1_p = float(pos["tp1_price"]) if pos.get("tp1_price") else (price + risk * tp1_r if direction == "LONG" else price - risk * tp1_r)
    tp2_p = float(pos["tp2_price"]) if pos.get("tp2_price") else (price + risk * tp2_r if direction == "LONG" else price - risk * tp2_r)
    shock_stop = (price - risk * 2.00) if direction == "LONG" else (price + risk * 2.00)
    soft_stop = (price - risk * 0.35) if direction == "LONG" else (price + risk * 0.35)
    ema21 = calc_ema(c, 21) if mode == "TP1_BE_EMA21" else None

    tp1_hit = False
    for step, j in enumerate(idxs, 1):
        # 1) SAF +2.5R & Kapanis Stopu (BIST Spot Dip)
        if mode == "SAF_25R_CLOSE_STOP":
            if l[j] <= shock_stop:
                fp = min(float(o[j]), shock_stop)
                return True, "ŞOK STOP (-2R)", ((fp - price) / risk) - cost_r, fp
            if h[j] >= tp1_p:
                return True, "HEDEF +2.5R", tp1_r - cost_r, tp1_p
            if c[j] < stop:
                return True, "KAPANIŞ STOPU", ((float(c[j]) - price) / risk) - cost_r, float(c[j])

        # 2) MOD A + 5G Kesici (BIST Short VIOP Zirve)
        elif mode == "MOD_A_5G":
            if h[j] >= shock_stop:
                fp = max(float(o[j]), shock_stop)
                return True, "ŞOK STOP (-2R)", ((price - fp) / risk) - cost_r, fp
            if l[j] <= tp1_p:
                return True, "HEDEF MOD A", tp1_r - cost_r, tp1_p
            if c[j] > stop:
                return True, "KAPANIŞ STOPU", ((price - float(c[j])) / risk) - cost_r, float(c[j])
            if step == 5 and ((price - float(c[j])) / risk) < 0.25:
                return True, "5G ÖLÜ KESİCİ", ((price - float(c[j])) / risk) - cost_r, float(c[j])

        # 3) TP1+BE + EMA21 Trail (BIST Spot Trend & Zirve Kirilimi)
        elif mode == "TP1_BE_EMA21":
            if not tp1_hit:
                if close_stop:
                    if l[j] <= shock_stop:
                        fp = min(float(o[j]), shock_stop)
                        return True, "ŞOK STOP (-2R)", ((fp - price) / risk) - cost_r, fp
                    if h[j] >= tp1_p:
                        tp1_hit = True
                    elif c[j] < stop:
                        return True, "KAPANIŞ STOPU", ((float(c[j]) - price) / risk) - cost_r, float(c[j])
                else:
                    if l[j] <= stop:
                        return True, "STOP", -1.0 - cost_r, stop
                    if h[j] >= tp1_p:
                        tp1_hit = True
            else:
                if l[j] <= price:
                    return True, "TP1+BE", (tp1_w * tp1_r) - cost_r, price
                if ema21 is not None and not np.isnan(ema21[j]) and c[j] < ema21[j]:
                    rem_r = (float(c[j]) - price) / risk
                    return True, "TP1+EMA21", (tp1_w * tp1_r + rem_w * rem_r) - cost_r, float(c[j])

        # 4) Kademeli TP1/TP2 (Scalper Paket D, Swing v2.5, VIOP Dip/Trend)
        else:
            is_soft = (mode == "SWING_V25_45G")
            dead_bars = int(rules.get("dead_bars", 0))
            if direction == "LONG":
                if not tp1_hit:
                    if close_stop:
                        if l[j] <= shock_stop:
                            fp = min(float(o[j]), shock_stop)
                            return True, "ŞOK STOP (-2R)", ((fp - price) / risk) - cost_r, fp
                        if h[j] >= tp2_p:
                            return True, "TP2", (tp1_w * tp1_r + rem_w * tp2_r) - cost_r, tp2_p
                        if h[j] >= tp1_p:
                            tp1_hit = True
                        elif c[j] < stop:
                            return True, "KAPANIŞ STOPU", ((float(c[j]) - price) / risk) - cost_r, float(c[j])
                    else:
                        if l[j] <= stop:
                            return True, "STOP", -1.0 - cost_r, stop
                        if h[j] >= tp2_p:
                            return True, "TP2", (tp1_w * tp1_r + rem_w * tp2_r) - cost_r, tp2_p
                        if h[j] >= tp1_p:
                            tp1_hit = True
                        elif dead_bars > 0 and step == dead_bars and ((float(c[j]) - price) / risk) < 0.25:
                            return True, "12G ÖLÜ KESİCİ", ((float(c[j]) - price) / risk) - cost_r, float(c[j])
                else:
                    st_lvl = soft_stop if is_soft else price
                    if l[j] <= st_lvl:
                        rem_r = -0.35 if is_soft else 0.0
                        return True, ("TP1+SOFT" if is_soft else "TP1+BE"), (tp1_w * tp1_r + rem_w * rem_r) - cost_r, st_lvl
                    if h[j] >= tp2_p:
                        return True, "TP2", (tp1_w * tp1_r + rem_w * tp2_r) - cost_r, tp2_p
            else:
                if not tp1_hit:
                    if close_stop:
                        if h[j] >= shock_stop:
                            fp = max(float(o[j]), shock_stop)
                            return True, "ŞOK STOP (-2R)", ((price - fp) / risk) - cost_r, fp
                        if l[j] <= tp2_p:
                            return True, "TP2", (tp1_w * tp1_r + rem_w * tp2_r) - cost_r, tp2_p
                        if l[j] <= tp1_p:
                            tp1_hit = True
                        elif c[j] > stop:
                            return True, "KAPANIŞ STOPU", ((price - float(c[j])) / risk) - cost_r, float(c[j])
                    else:
                        if h[j] >= stop:
                            return True, "STOP", -1.0 - cost_r, stop
                        if l[j] <= tp2_p:
                            return True, "TP2", (tp1_w * tp1_r + rem_w * tp2_r) - cost_r, tp2_p
                        if l[j] <= tp1_p:
                            tp1_hit = True
                        elif dead_bars > 0 and step == dead_bars and ((price - float(c[j])) / risk) < 0.25:
                            return True, "12G ÖLÜ KESİCİ", ((price - float(c[j])) / risk) - cost_r, float(c[j])
                else:
                    st_lvl = soft_stop if is_soft else price
                    if h[j] >= st_lvl:
                        rem_r = -0.35 if is_soft else 0.0
                        return True, ("TP1+SOFT" if is_soft else "TP1+BE"), (tp1_w * tp1_r + rem_w * rem_r) - cost_r, st_lvl
                    if l[j] <= tp2_p:
                        return True, "TP2", (tp1_w * tp1_r + rem_w * tp2_r) - cost_r, tp2_p

        if step >= max_bars:
            rem_r = ((float(c[j]) - price) / risk) if direction == "LONG" else ((price - float(c[j])) / risk)
            final_r = (tp1_w * tp1_r + rem_w * rem_r - cost_r) if tp1_hit else (rem_r - cost_r)
            return True, f"VADE SONU", final_r, float(c[j])

    # Hala acik: anlik mark-to-market degeri hesapla
    cur_p = float(live_price) if (live_price and live_price > 0) else float(c[-1])
    rem_r = ((cur_p - price) / risk) if direction == "LONG" else ((price - cur_p) / risk)
    open_r = (tp1_w * tp1_r + rem_w * rem_r) if tp1_hit else rem_r
    return False, ("TP1 ALINDI • AÇIK" if tp1_hit else "AÇIK"), open_r, cur_p


def update_open_positions(state, bist_data, crypto_packs, ticker_map):
    still_open = []
    newly_closed = []

    for pos in state["open_positions"]:
        sym = pos["symbol"]
        series, live_p = None, None
        if pos["portfolio"] == "BIST":
            series = bist_data.get(sym)
            if series is not None:
                live_p = float(series["c"][-1])
        elif pos["portfolio"] == "SCALPER":
            pack = crypto_packs.get(sym)
            if pack and pack.get("h1"):
                series = pack["h1"]
            if sym in ticker_map:
                live_p = float(ticker_map[sym]["price"])
        else:
            pack = crypto_packs.get(sym)
            if pack and pack.get("h4"):
                series = pack["h4"]
            if sym in ticker_map:
                live_p = float(ticker_map[sym]["price"])

        if series is None:
            still_open.append(pos)
            continue

        closed, reason, r_val, ex_price = _eval_position_bars(pos, series, live_p)
        stop_pct_dec = abs(float(pos["entry_price"]) - float(pos["stop_loss"])) / max(float(pos["entry_price"]), 1e-12)
        pnl_money = float(pos["pos_size"]) * (r_val * stop_pct_dec)

        if not closed:
            pos["current_price"] = round(ex_price, 6)
            pos["status_note"] = reason
            pos["unrealized_r"] = round(r_val, 2)
            pos["unrealized_pnl"] = round(pnl_money, 2)
            still_open.append(pos)
            continue

        # Islem kapandi -> Bilesik kasaya isle
        acc = state["accounts"][pos["portfolio"]]
        acc["equity"] = round(float(acc["equity"]) + pnl_money, 2)
        acc["cum_r"] = round(float(acc["cum_r"]) + r_val, 2)
        if acc["equity"] > acc["peak_equity"]:
            acc["peak_equity"] = acc["equity"]
        dd_pct = ((acc["peak_equity"] - acc["equity"]) / max(acc["peak_equity"], 1e-9)) * 100.0
        acc["max_dd_pct"] = round(max(float(acc["max_dd_pct"]), dd_pct), 2)
        ret_pct = ((acc["equity"] - acc["initial"]) / acc["initial"]) * 100.0

        now_s = _now_str()
        acc["curve"].append({
            "date": now_s,
            "equity": acc["equity"],
            "ret_pct": round(ret_pct, 2),
            "cum_r": acc["cum_r"],
            "dd_pct": round(dd_pct, 2),
        })

        closed_rec = {
            "closed_at": now_s,
            "opened_at": pos["opened_at"],
            "portfolio": pos["portfolio"],
            "engine": pos["engine_name"],
            "symbol": sym,
            "direction": pos["direction"],
            "score": pos["score"],
            "entry_price": pos["entry_price"],
            "exit_price": round(ex_price, 6),
            "stop_loss": pos["stop_loss"],
            "stop_pct": round(stop_pct_dec * 100.0, 2),
            "pos_size": pos["pos_size"],
            "r_val": round(r_val, 2),
            "pnl_money": round(pnl_money, 2),
            "exit_reason": reason,
            "equity_after": acc["equity"],
        }
        state["closed_trades"].append(closed_rec)
        _append_trade_csv(closed_rec)
        newly_closed.append(closed_rec)

    state["open_positions"] = still_open
    return newly_closed


def register_new_signals(state, signals):
    """Yeni sinyalleri bilesik kasanin %10 / %1 / %5 pozisyon buyukluguyle portfoye acar."""
    opened = []
    open_keys = {(p["portfolio"], p["engine_name"], p["symbol"]) for p in state["open_positions"]}

    for sig in signals:
        pf_key = sig["portfolio"]
        cd_key = f"{pf_key}|{sig['engine_name']}|{sig['symbol']}"
        if (pf_key, sig["engine_name"], sig["symbol"]) in open_keys:
            continue

        bar_ts = int(sig["bar_ts"])
        rules = sig["exit_rules"]
        cd_ms = int(rules.get("cooldown_hours", rules.get("cooldown_days", 5) * 24)) * 3_600_000
        last_ts = int(state["cooldowns"].get(cd_key, 0))
        if bar_ts <= last_ts or (bar_ts - last_ts) < cd_ms:
            continue

        acc = state["accounts"][pf_key]
        pos_size = round(float(acc["equity"]) * float(acc["alloc_pct"]), 2)
        qty = pos_size / max(float(sig["entry_price"]), 1e-12)

        pos = {
            "opened_at": _now_str(),
            "portfolio": pf_key,
            "market": sig["market"],
            "engine_name": sig["engine_name"],
            "symbol": sig["symbol"],
            "direction": sig["direction"],
            "score": sig["score"],
            "entry_price": round(float(sig["entry_price"]), 6),
            "current_price": round(float(sig["entry_price"]), 6),
            "stop_loss": round(float(sig["stop_loss"]), 6),
            "tp1_price": round(float(sig["tp1_price"]), 6) if sig.get("tp1_price") else None,
            "tp2_price": round(float(sig["tp2_price"]), 6) if sig.get("tp2_price") else None,
            "pos_size": pos_size,
            "qty": round(qty, 6),
            "entry_bar_ts": bar_ts,
            "setup_tag": sig["setup_tag"],
            "exit_rules": rules,
            "status_note": "YENİ AÇILDI",
            "unrealized_r": 0.0,
            "unrealized_pnl": 0.0,
        }
        state["open_positions"].append(pos)
        state["cooldowns"][cd_key] = bar_ts
        open_keys.add((pf_key, sig["engine_name"], sig["symbol"]))
        opened.append(pos)
    return opened