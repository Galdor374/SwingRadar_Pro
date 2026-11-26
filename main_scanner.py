import argparse
import os
import sys
import time
import webbrowser
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import requests

from config import BIST_100, VIOP_ALL, CRYPTO_MAX_COINS, SCALPER_MIN_VOL_USD
from data_utils import (
    calc_sma,
    fetch_binance_klines,
    fetch_yahoo_daily,
    get_binance_universe_and_movers,
)
from engines_bist import scan_all_bist
from engines_crypto import (
    build_crypto_swing_ctx,
    scan_crypto_scalper_symbol,
    scan_crypto_swing_symbol,
)
from portfolio_tracker import (
    load_state,
    register_new_signals,
    save_state,
    update_open_positions,
)
from dashboard_builder import build_terminal_html

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def fetch_all_bist():
    print("  [BIST] XU100 + BIST100 & VIOP verileri indiriliyor...")
    all_syms = list(dict.fromkeys(BIST_100 + VIOP_ALL))
    sess = requests.Session()
    xu100 = fetch_yahoo_daily(sess, "XU100.IS", range_str="2y", min_len=180)

    def _dl(sym):
        code = "XU030.IS" if sym == "XU030" else f"{sym}.IS"
        return sym, fetch_yahoo_daily(requests.Session(), code, range_str="2y", min_len=180)

    bist_map = {}
    with ThreadPoolExecutor(max_workers=10) as ex:
        for fut in as_completed([ex.submit(_dl, s) for s in all_syms]):
            sym, d1 = fut.result()
            if d1 is not None:
                bist_map[sym] = d1
    print(f"  [BIST] Hazir: {len(bist_map)}/{len(all_syms)} enstruman.")
    return bist_map, xu100


def fetch_all_crypto():
    sess = requests.Session()
    syms, ticker_map, _ = get_binance_universe_and_movers(sess, max_coins=CRYPTO_MAX_COINS, min_vol=0.0)
    scalp_eligible = {
        s for s in syms
        if float(ticker_map.get(s, {}).get("quote_vol", 0.0)) >= SCALPER_MIN_VOL_USD
    }
    scalp_eligible.add("BTCUSDT")
    print(f"  [KRIPTO] 200 Swing coini (ici >=25M$ Scalper: {len(scalp_eligible)} coin) indiriliyor...")

    def _dl(sym):
        s_local = requests.Session()
        d1 = fetch_binance_klines(s_local, sym, "1d", limit=260, min_len=140)
        if d1 is None:
            return sym, None
        h4 = fetch_binance_klines(s_local, sym, "4h", limit=260, min_len=60)
        w1 = fetch_binance_klines(s_local, sym, "1w", limit=60, min_len=12)
        h1 = fetch_binance_klines(s_local, sym, "1h", limit=120, min_len=45) if sym in scalp_eligible else None
        return sym, {"d1": d1, "h4": h4, "w1": w1, "h1": h1}

    packs = {}
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(_dl, s) for s in syms]
        for idx, fut in enumerate(as_completed(futs), 1):
            sym, pack = fut.result()
            if pack is not None:
                packs[sym] = pack
            if idx % 50 == 0 or idx == len(syms):
                print(f"           -> {idx}/{len(syms)} (Hazir: {len(packs)})")
    return packs, ticker_map


def run_once(open_browser=True):
    t0 = time.time()
    print(f"\n=== SWINGRADAR PRO TARAMA BASLADI ({datetime.now().strftime('%d.%m.%Y %H:%M')}) ===")

    bist_map, xu100 = fetch_all_bist()
    crypto_packs, ticker_map = fetch_all_crypto()

    # 1) BIST 6 Motor
    bist_long, bist_short, xu_ctx = scan_all_bist(bist_map, xu100)

    # 2) Kripto Motor 7 (1H Scalper) & Motor 8 (Swing)
    btc_h4 = crypto_packs.get("BTCUSDT", {}).get("h4")
    btc_h4_ctx = (btc_h4["t"], btc_h4["c"], calc_sma(btc_h4["c"], 100)) if btc_h4 else (None, None, None)
    swing_ctx = build_crypto_swing_ctx(crypto_packs)

    scalper_sigs, swing_sigs = [], []
    for sym, pack in crypto_packs.items():
        tk = ticker_map.get(sym, {})
        scalper_sigs.extend(scan_crypto_scalper_symbol(sym, pack, btc_h4_ctx, tk))
        sw_sig = scan_crypto_swing_symbol(sym, pack, swing_ctx, tk)
        if sw_sig is not None:
            swing_sigs.append(sw_sig)

    scalper_sigs.sort(key=lambda x: x["score"], reverse=True)
    swing_sigs.sort(key=lambda x: x["score"], reverse=True)

    # 3) Portfoy Guncelleme & Bilesik Takip
    state = load_state()
    closed_now = update_open_positions(state, bist_map, crypto_packs, ticker_map)
    all_new_sigs = bist_long + bist_short + scalper_sigs + swing_sigs
    opened_now = register_new_signals(state, all_new_sigs)
    save_state(state)

    scan_results = {
        "bist_long": bist_long,
        "bist_short": bist_short,
        "scalper": scalper_sigs,
        "swing": swing_sigs,
    }
    html_path = build_terminal_html(state, scan_results, {"xu_ctx": xu_ctx, "swing_ctx": swing_ctx})

    print(
        f"  [OZET] Sinyaller -> BIST Long: {len(bist_long)} | BIST Short: {len(bist_short)} | "
        f"Scalper: {len(scalper_sigs)} | Swing: {len(swing_sigs)}"
    )
    print(f"  [PORTFOY] Yeni Acilan: {len(opened_now)} | Kapanan: {len(closed_now)} | Toplam Acik: {len(state['open_positions'])}")
    print(f"  Tamamlandi ({time.time() - t0:.1f} sn) -> {html_path}")

    if open_browser:
        webbrowser.open("file://" + os.path.abspath(html_path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--loop", action="store_true", help="Her saat basi (:01) otomatik tarama yapar")
    parser.add_argument("--no-browser", action="store_true", help="Tarama sonrasi tarayiciyi acmaz")
    args = parser.parse_args()

    if not args.loop:
        run_once(open_browser=not args.no_browser)
        return

    print("Otomatik saatlik tarama modu aktif (durdurmak icin CTRL+C)...")
    run_once(open_browser=not args.no_browser)
    while True:
        now = datetime.now()
        wait_sec = (60 - now.minute) * 60 - now.second + 65
        print(f"  Sonraki taramaya {wait_sec // 60} dk {wait_sec % 60} sn bekleniyor...")
        time.sleep(max(30, wait_sec))
        try:
            run_once(open_browser=False)
        except Exception as exc:
            print(f"  [HATA] Tarama sirasinda sorun olustu: {exc}")


if __name__ == "__main__":
    main()