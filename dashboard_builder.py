from datetime import datetime
from config import REPORT_HTML, SCALPER_MIN_VOL_USD


def _fmt_price(p):
    if p is None:
        return "—"
    p = float(p)
    if p >= 100:
        return f"{p:,.2f}"
    if p >= 1:
        return f"{p:.4f}"
    return f"{p:.6f}"


def _build_spark_svg(vals, direction):
    if not vals or len(vals) < 2:
        return ""
    lo, hi = min(vals), max(vals)
    span = max(hi - lo, 1e-12)
    w, h = 76, 22
    pts = [
        f"{(idx / (len(vals) - 1)) * w:.1f},{h - ((v - lo) / span) * (h - 4) - 2:.1f}"
        for idx, v in enumerate(vals)
    ]
    col = "#38c996" if direction == "LONG" else "#f06560"
    return f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}"><polyline fill="none" stroke="{col}" stroke-width="1.8" points="{" ".join(pts)}"/></svg>'


def _build_curve_svg(points, value_key, suffix, accent_color, baseline=None):
    w, h = 520, 165
    pad_l, pad_r, pad_t, pad_b = 64, 18, 18, 24
    plot_w, plot_h = w - pad_l - pad_r, h - pad_t - pad_b

    vals = [float(pt[value_key]) for pt in points]
    if len(vals) == 1:
        vals = [vals[0], vals[0]]
    lo, hi = min(vals), max(vals)
    if baseline is not None:
        lo, hi = min(lo, baseline), max(hi, baseline)
    if abs(hi - lo) < 1e-6:
        pad_v = max(abs(hi) * 0.02, 1.0)
        lo, hi = lo - pad_v, hi + pad_v
    else:
        margin = (hi - lo) * 0.12
        lo, hi = lo - margin, hi + margin

    def to_xy(idx, val):
        x = pad_l + (idx / max(len(vals) - 1, 1)) * plot_w
        y = pad_t + plot_h - ((val - lo) / (hi - lo)) * plot_h
        return x, y

    coords = [to_xy(i, v) for i, v in enumerate(vals)]
    line_pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in coords)
    area_pts = f"{coords[0][0]:.1f},{pad_t + plot_h:.1f} {line_pts} {coords[-1][0]:.1f},{pad_t + plot_h:.1f}"
    gid = f"g_{value_key}_{abs(hash(accent_color)) % 9999}"

    grid_svg = ""
    for frac in (0.0, 0.5, 1.0):
        gv = lo + (hi - lo) * frac
        gy = pad_t + plot_h - frac * plot_h
        if abs(gv) >= 100:
            lbl = f"{gv:,.0f}{suffix}"
        elif suffix == "%":
            lbl = f"{gv:+.2f}{suffix}"
        else:
            lbl = f"{gv:,.2f}{suffix}"
        grid_svg += f'<line x1="{pad_l}" y1="{gy:.1f}" x2="{w - pad_r}" y2="{gy:.1f}" stroke="rgba(233,196,106,0.12)" stroke-dasharray="3,3"/>'
        grid_svg += f'<text x="{pad_l - 6}" y="{gy + 3:.1f}" text-anchor="end" fill="#a89b8c" font-size="9.5">{lbl}</text>'

    base_svg = ""
    if baseline is not None and lo <= baseline <= hi:
        _, by = to_xy(0, baseline)
        base_svg = f'<line x1="{pad_l}" y1="{by:.1f}" x2="{w - pad_r}" y2="{by:.1f}" stroke="rgba(246,239,232,0.28)" stroke-width="1"/>'

    lx, ly = coords[-1]
    last_val = vals[-1]
    last_lbl = f"{last_val:,.0f}{suffix}" if abs(last_val) >= 100 else f"{last_val:+.2f}{suffix}"
    first_dt = points[0]["date"][:10]
    last_dt = points[-1]["date"][:10]

    return f"""<svg viewBox="0 0 {w} {h}" style="width:100%;height:auto;display:block">
      <defs>
        <linearGradient id="{gid}" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stop-color="{accent_color}" stop-opacity="0.32"/>
          <stop offset="100%" stop-color="{accent_color}" stop-opacity="0.02"/>
        </linearGradient>
      </defs>
      {grid_svg}{base_svg}
      <polygon points="{area_pts}" fill="url(#{gid})"/>
      <polyline fill="none" stroke="{accent_color}" stroke-width="2.2" stroke-linejoin="round" points="{line_pts}"/>
      <circle cx="{lx:.1f}" cy="{ly:.1f}" r="3.8" fill="{accent_color}" stroke="#151312" stroke-width="1.5"/>
      <text x="{pad_l}" y="{h - 6}" fill="#a89b8c" font-size="9.5">{first_dt}</text>
      <text x="{w - pad_r}" y="{h - 6}" text-anchor="end" fill="#e9c46a" font-size="9.5" font-weight="700">{last_dt} • {last_lbl}</text>
    </svg>"""


def _portfolio_stats(state, pf_key):
    acc = state["accounts"][pf_key]
    closed = [t for t in state["closed_trades"] if t["portfolio"] == pf_key]
    opens = [p for p in state["open_positions"] if p["portfolio"] == pf_key]
    open_pnl = sum(float(p.get("unrealized_pnl", 0.0)) for p in opens)
    live_eq = float(acc["equity"]) + open_pnl
    live_ret = ((live_eq - float(acc["initial"])) / float(acc["initial"])) * 100.0
    wins = [t for t in closed if float(t["r_val"]) > 0]
    loss_r = sum(abs(float(t["r_val"])) for t in closed if float(t["r_val"]) < 0)
    win_r = sum(float(t["r_val"]) for t in wins)
    pf = (win_r / loss_r) if loss_r > 1e-9 else (9.9 if win_r > 0 else 0.0)
    wr = (len(wins) / len(closed) * 100.0) if closed else 0.0

    curve = list(acc["curve"])
    if opens:
        curve.append({
            "date": "CANLI",
            "equity": round(live_eq, 2),
            "ret_pct": round(live_ret, 2),
            "cum_r": round(float(acc["cum_r"]) + sum(float(p.get("unrealized_r", 0.0)) for p in opens), 2),
        })
    return {
        "acc": acc, "closed": closed, "opens": opens,
        "closed_n": len(closed), "open_n": len(opens),
        "open_pnl": open_pnl, "live_eq": live_eq, "live_ret": live_ret,
        "wr": wr, "pf": min(9.9, pf), "curve": curve,
    }


def _base_css():
    return """<style>
      :root{
        --bg:#151312; --card:#1f1c1a; --card2:#272220;
        --border:rgba(224,122,95,0.22); --border2:rgba(233,196,106,0.16);
        --text:#f6efe8; --muted:#a89b8c;
        --terracotta:#e07a5f; --champagne:#e9c46a; --copper:#d98a5b;
        --green:#38c996; --red:#f06560;
      }
      *{box-sizing:border-box;margin:0;padding:0}
      body{font-family:Inter,-apple-system,Segoe UI,sans-serif;background:radial-gradient(ellipse at 12% 0%, #2c1e1a 0%, var(--bg) 52%);color:var(--text);padding:22px 28px 52px;max-width:1460px;margin:auto}
      a{color:inherit;text-decoration:none}
      .top-bar{background:linear-gradient(135deg,#251d1a 0%,#1c1816 100%);border:1px solid var(--border);border-radius:16px;padding:18px 22px;margin-bottom:20px;display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:12px}
      h1{font-size:19px;font-weight:800;color:var(--champagne)}
      .sub{font-size:11.5px;color:var(--muted);margin-top:3px}
      .nav-pills{display:flex;gap:8px;flex-wrap:wrap;align-items:center}
      .nav-btn{padding:7px 14px;border-radius:10px;font-size:12px;font-weight:700;background:var(--card2);border:1px solid var(--border2);color:var(--text);transition:.15s}
      .nav-btn:hover,.nav-btn.active{background:rgba(224,122,95,0.22);border-color:var(--terracotta);color:var(--champagne)}
      .pill{display:inline-block;padding:4px 10px;border-radius:999px;font-size:11px;font-weight:600;background:rgba(224,122,95,0.13);border:1px solid rgba(224,122,95,0.28);color:#f4d8cd}
      .home-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:18px}
      @media(max-width:1050px){.home-grid{grid-template-columns:1fr}}
      .hub-card{display:block;background:var(--card);border:1px solid var(--border);border-radius:16px;padding:20px;transition:.18s}
      .hub-card:hover{transform:translateY(-3px);border-color:var(--champagne);box-shadow:0 10px 28px rgba(0,0,0,0.35)}
      .hub-top{display:flex;justify-content:space-between;align-items:flex-start;gap:10px;margin-bottom:14px}
      .hub-title{font-size:16px;font-weight:800;color:var(--terracotta)}
      .hub-eq{font-size:21px;font-weight:800;font-variant-numeric:tabular-nums;text-align:right}
      .hub-stats{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-bottom:14px}
      .mini-stat{background:var(--card2);border-radius:10px;padding:8px 10px}
      .mini-stat span{display:block;font-size:10px;color:var(--muted);text-transform:uppercase}
      .mini-stat b{font-size:13px;font-variant-numeric:tabular-nums}
      .hub-cta{display:flex;justify-content:space-between;align-items:center;padding-top:12px;border-top:1px solid rgba(255,244,230,0.08);font-size:12px;font-weight:700;color:var(--champagne)}
      .panel{background:var(--card);border:1px solid var(--border2);border-radius:15px;padding:16px 20px;margin-bottom:16px;overflow-x:auto}
      .kpi-row{display:grid;grid-template-columns:repeat(5,1fr);gap:10px;margin-bottom:14px}
      @media(max-width:800px){.kpi-row{grid-template-columns:repeat(2,1fr)}}
      .kpi{background:var(--card2);border-radius:10px;padding:9px 12px}
      .kpi span{display:block;font-size:10px;color:var(--muted);text-transform:uppercase;margin-bottom:2px}
      .kpi b{font-size:14px;font-variant-numeric:tabular-nums}
      .charts-2{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:16px}
      @media(max-width:950px){.charts-2{grid-template-columns:1fr}}
      .chart-box{background:var(--card);border:1px solid var(--border2);border-radius:15px;padding:14px 16px}
      .ctitle{font-size:11px;font-weight:800;color:var(--champagne);letter-spacing:.4px;margin-bottom:8px}
      .ptitle{font-size:13.5px;font-weight:800;color:var(--champagne);margin-bottom:10px;display:flex;align-items:center;gap:8px}
      .cnt{background:rgba(224,122,95,0.20);color:#ffd5c7;padding:2px 8px;border-radius:999px;font-size:11px}
      table{width:100%;border-collapse:collapse;font-size:12px}
      th{text-align:left;font-size:10px;color:var(--muted);text-transform:uppercase;padding:7px 6px;border-bottom:1px solid var(--border2)}
      td{padding:9px 6px;border-top:1px solid rgba(255,244,230,0.06);vertical-align:middle}
      tr:hover td{background:rgba(224,122,95,0.05)}
      .num{font-variant-numeric:tabular-nums;white-space:nowrap}
      .pos{color:var(--green)} .neg{color:var(--red)}
      .tag{padding:3px 8px;border-radius:6px;font-size:10.5px;font-weight:800}
      .tag-long{background:rgba(56,201,150,0.16);color:var(--green);border:1px solid rgba(56,201,150,0.32)}
      .tag-short{background:rgba(240,101,96,0.16);color:var(--red);border:1px solid rgba(240,101,96,0.32)}
      .sc{display:inline-block;padding:2px 8px;border-radius:6px;font-weight:800;background:rgba(233,196,106,0.18);color:var(--champagne)}
      .empty{color:var(--muted);font-size:12px;padding:10px 4px}
    </style>"""


def _nav_html(active_page, now_str, ctx_pills=""):
    links = [
        ("index.html", "Ana Ekran", "home"),
        ("bist.html", "BIST", "bist"),
        ("scalper.html", "Kripto 1H Scalper", "scalper"),
        ("swing.html", "Kripto Swing", "swing"),
    ]
    nav_btns = "".join(
        f'<a class="nav-btn {"active" if key == active_page else ""}" href="{href}">{label}</a>'
        for href, label, key in links
    )
    return f"""
    <div class="top-bar">
      <div>
        <h1>SWINGRADAR PRO</h1>
        <div class="sub">Son Tarama: <b>{now_str}</b> {("• " + ctx_pills) if ctx_pills else ""}</div>
      </div>
      <div class="nav-pills">{nav_btns}</div>
    </div>"""


def _signals_table_html(title, sigs, acc, empty_msg):
    if not sigs:
        return f'<div class="panel"><div class="ptitle">{title} <span class="cnt">0</span></div><div class="empty">{empty_msg}</div></div>'
    alloc_amt = float(acc["equity"]) * float(acc["alloc_pct"])
    cur = acc["currency"]
    rows = ""
    for s in sigs:
        dir_cls = "tag-long" if s["direction"] == "LONG" else "tag-short"
        rows += f"""
        <tr>
          <td><b>{s['symbol']}</b><div class="sub">{s['engine_name']}</div></td>
          <td><span class="tag {dir_cls}">{s['direction']}</span></td>
          <td class="num"><span class="sc">{s['score']}</span><div class="sub">{s['meta'].get('badge','')}</div></td>
          <td class="num"><b>{_fmt_price(s['entry_price'])}</b><div class="sub {'pos' if s['change_24h']>=0 else 'neg'}">{s['change_24h']:+.2f}%</div></td>
          <td class="num neg">{_fmt_price(s['stop_loss'])}<div class="sub">%{s['meta'].get('stop_pct',0):.2f}</div></td>
          <td class="num pos">{_fmt_price(s['tp1_price'])} / {_fmt_price(s['tp2_price'])}</td>
          <td class="num"><b>{alloc_amt:,.0f} {cur}</b><div class="sub">%{acc['alloc_pct']*100:.0f} Kasa</div></td>
          <td><b>{s['setup_tag']}</b><div class="sub">{s['meta'].get('exit_desc','')}</div></td>
          <td>{_build_spark_svg(s.get('sparkline',[]), s['direction'])}</td>
        </tr>"""
    return f"""
    <div class="panel">
      <div class="ptitle">{title} <span class="cnt">{len(sigs)}</span></div>
      <table>
        <thead><tr><th>Sembol / Motor</th><th>Yön</th><th>Puan</th><th>Giriş</th><th>Stop</th><th>TP1 / TP2</th><th>Pozisyon</th><th>Kurulum & Çıkış</th><th>Trend</th></tr></thead>
        <tbody>{rows}</tbody>
      </table>
    </div>"""


def _build_detail_page(page_key, page_title, subtitle_info, pf_key, signals, state, now_str, ctx_pills, empty_msg):
    st = _portfolio_stats(state, pf_key)
    acc = st["acc"]
    cur = acc["currency"]
    pos_size_now = float(acc["equity"]) * float(acc["alloc_pct"])
    ret_cls = "pos" if st["live_ret"] >= 0 else "neg"
    pnl_cls = "pos" if st["open_pnl"] >= 0 else "neg"

    eq_svg = _build_curve_svg(st["curve"], "equity", f" {cur}", "#e07a5f", baseline=float(acc["initial"]))
    ret_svg = _build_curve_svg(st["curve"], "ret_pct", "%", "#e9c46a", baseline=0.0)

    open_rows = ""
    for p in reversed(st["opens"]):
        u_r, u_p = float(p.get("unrealized_r", 0.0)), float(p.get("unrealized_pnl", 0.0))
        cls = "pos" if u_p >= 0 else "neg"
        open_rows += f"""
        <tr>
          <td><b>{p['symbol']}</b><div class="sub">{p['engine_name']} • {p['opened_at']}</div></td>
          <td><span class="tag {'tag-long' if p['direction']=='LONG' else 'tag-short'}">{p['direction']}</span></td>
          <td class="num">{p['score']}</td>
          <td class="num">{_fmt_price(p['entry_price'])} → <b>{_fmt_price(p.get('current_price'))}</b></td>
          <td class="num neg">{_fmt_price(p['stop_loss'])}</td>
          <td class="num pos">{_fmt_price(p.get('tp1_price'))} / {_fmt_price(p.get('tp2_price'))}</td>
          <td class="num">{p['pos_size']:,.0f} {cur}</td>
          <td class="num {cls}"><b>{"+" if u_r>=0 else ""}{u_r:.2f}R</b><div class="sub">{"+" if u_p>=0 else ""}{u_p:,.2f} {cur}</div></td>
          <td><span class="pill">{p.get('status_note','AÇIK')}</span></td>
        </tr>"""

    closed_rows = ""
    for t in reversed(st["closed"][-30:]):
        cls = "pos" if float(t["pnl_money"]) >= 0 else "neg"
        closed_rows += f"""
        <tr>
          <td><b>{t['symbol']}</b><div class="sub">{t['engine']}</div></td>
          <td>{t['closed_at']}</td>
          <td><span class="tag {'tag-long' if t['direction']=='LONG' else 'tag-short'}">{t['direction']}</span></td>
          <td class="num">{_fmt_price(t['entry_price'])} → {_fmt_price(t['exit_price'])}</td>
          <td class="num">{t['pos_size']:,.0f} {cur}</td>
          <td class="num {cls}"><b>{"+" if t['r_val']>=0 else ""}{t['r_val']:.2f}R</b></td>
          <td class="num {cls}"><b>{"+" if t['pnl_money']>=0 else ""}{t['pnl_money']:,.2f} {cur}</b></td>
          <td><span class="pill">{t['exit_reason']}</span></td>
          <td class="num"><b>{t['equity_after']:,.2f} {cur}</b></td>
        </tr>"""

    body = f"""
    {_nav_html(page_key, now_str, ctx_pills)}
    <div class="panel">
      <div class="hub-top">
        <div>
          <div class="hub-title">{page_title}</div>
          <div class="sub">{subtitle_info} • Başlangıç: <b>{acc['initial']:,.0f} {cur}</b> • Pozisyon: <b>%{acc['alloc_pct']*100:.0f}</b> (~{pos_size_now:,.0f} {cur}) • Bileşik</div>
        </div>
        <div style="text-align:right">
          <div class="hub-eq">{st['live_eq']:,.2f} <span style="color:var(--champagne);font-size:14px">{cur}</span></div>
          <div class="sub {ret_cls}"><b>{"+" if st['live_ret']>=0 else ""}{st['live_ret']:.2f}%</b> • Açık K/Z: <span class="{pnl_cls}">{"+" if st['open_pnl']>=0 else ""}{st['open_pnl']:,.2f} {cur}</span></div>
        </div>
      </div>
      <div class="kpi-row">
        <div class="kpi"><span>Kapalı / Açık</span><b>{st['closed_n']} / {st['open_n']}</b></div>
        <div class="kpi"><span>Win Rate</span><b>%{st['wr']:.1f}</b></div>
        <div class="kpi"><span>Profit Factor</span><b>{st['pf']:.2f}</b></div>
        <div class="kpi"><span>Toplam R</span><b class="{'pos' if acc['cum_r']>=0 else 'neg'}">{"+" if acc['cum_r']>=0 else ""}{acc['cum_r']:.2f}R</b></div>
        <div class="kpi"><span>Max DD</span><b class="neg">-%{acc['max_dd_pct']:.2f}</b></div>
      </div>
    </div>

    <div class="charts-2">
      <div class="chart-box"><div class="ctitle">PORTFÖY EĞRİSİ ({cur})</div>{eq_svg}</div>
      <div class="chart-box"><div class="ctitle">GETİRİ EĞRİSİ (%)</div>{ret_svg}</div>
    </div>

    {_signals_table_html("AKTİF SİNYALLER", signals, acc, empty_msg)}

    <div class="panel">
      <div class="ptitle">AÇIK POZİSYONLAR <span class="cnt">{st['open_n']}</span></div>
      {f'<table><thead><tr><th>Sembol / Motor</th><th>Yön</th><th>Puan</th><th>Giriş → Güncel</th><th>Stop</th><th>TP1 / TP2</th><th>Pozisyon</th><th>Anlık K/Z</th><th>Durum</th></tr></thead><tbody>{open_rows}</tbody></table>' if open_rows else '<div class="empty">Bu kasada açık pozisyon yok.</div>'}
    </div>

    <div class="panel">
      <div class="ptitle">KAPANAN İŞLEMLER (SON 30) <span class="cnt">{st['closed_n']}</span></div>
      {f'<table><thead><tr><th>Sembol / Motor</th><th>Kapanış</th><th>Yön</th><th>Giriş → Çıkış</th><th>Pozisyon</th><th>Sonuç R</th><th>Net K/Z</th><th>Çıkış Sebebi</th><th>Yeni Kasa</th></tr></thead><tbody>{closed_rows}</tbody></table>' if closed_rows else '<div class="empty">Bu kasada henüz kapanan işlem yok.</div>'}
    </div>"""

    return f"""<!DOCTYPE html><html lang="tr"><head><meta charset="UTF-8">
    <title>{page_title} — SwingRadar Pro</title>{_base_css()}</head><body>{body}</body></html>"""


def build_terminal_html(state, scan_results, market_ctx):
    now_str = datetime.now().strftime("%d.%m.%Y %H:%M")
    xu = market_ctx.get("xu_ctx", {})
    sw = market_ctx.get("swing_ctx", {})
    ctx_pills = f"XU100: {xu.get('wind_label','—')} (%{xu.get('breadth',50):.1f}) • BTC: {'BOĞA' if sw.get('btc_up',True) else 'AYI'} (Genişlik %{sw.get('breadth',0.5)*100:.0f})"

    cards_cfg = [
        ("bist.html", "BIST 100 + VİOP", "BIST", len(scan_results["bist_long"]) + len(scan_results["bist_short"]), "100 Spot + 51 VİOP • %10 Pozisyon"),
        ("scalper.html", "KRİPTO 1H SCALPER", "SCALPER", len(scan_results["scalper"]), f"≥ ${SCALPER_MIN_VOL_USD/1e6:.0f}M Hacim • %1 Pozisyon"),
        ("swing.html", "KRİPTO SWİNG", "SWING", len(scan_results["swing"]), "200 Coin • 1D + 4H Tutunma • %5 Pozisyon"),
    ]

    cards_html = ""
    for href, title, pf_key, sig_cnt, note in cards_cfg:
        st = _portfolio_stats(state, pf_key)
        acc = st["acc"]
        cur = acc["currency"]
        ret_cls = "pos" if st["live_ret"] >= 0 else "neg"
        cards_html += f"""
        <a class="hub-card" href="{href}">
          <div class="hub-top">
            <div>
              <div class="hub-title">{title}</div>
              <div class="sub">{note}</div>
            </div>
            <div>
              <div class="hub-eq">{st['live_eq']:,.2f} <span style="color:var(--champagne);font-size:13px">{cur}</span></div>
              <div class="sub {ret_cls}" style="text-align:right"><b>{"+" if st['live_ret']>=0 else ""}{st['live_ret']:.2f}%</b></div>
            </div>
          </div>
          <div class="hub-stats">
            <div class="mini-stat"><span>Sinyal</span><b>{sig_cnt}</b></div>
            <div class="mini-stat"><span>Açık / Kapalı</span><b>{st['open_n']} / {st['closed_n']}</b></div>
            <div class="mini-stat"><span>PF / Top. R</span><b>{st['pf']:.2f} / {acc['cum_r']:+.1f}R</b></div>
          </div>
          <div class="hub-cta"><span>Ekrana Gir</span><span>→</span></div>
        </a>"""

    home_html = f"""<!DOCTYPE html><html lang="tr"><head><meta charset="UTF-8">
    <title>SwingRadar Pro — Ana Ekran</title>{_base_css()}</head><body>
    {_nav_html("home", now_str, ctx_pills)}
    <div class="home-grid">{cards_html}</div>
    </body></html>"""

    with open(REPORT_HTML, "w", encoding="utf-8") as f:
        f.write(home_html)

    bist_sigs = scan_results["bist_long"] + scan_results["bist_short"]
    pages = {
        "bist.html": _build_detail_page(
            "bist", "BIST 100 + VİOP EKRANI", "6 BIST & VİOP Motoru",
            "BIST", bist_sigs, state, now_str, ctx_pills,
            "Şu an BIST 100 / VİOP tarafında aktif sinyal yok."
        ),
        "scalper.html": _build_detail_page(
            "scalper", "KRİPTO 1H SCALPER EKRANI", "Motor 7 • 2 Bar Delta & 4H MA100 Veto",
            "SCALPER", scan_results["scalper"], state, now_str, ctx_pills,
            "Şu an ≥25M$ hacimli coinlerde 1H Scalper sinyali yok."
        ),
        "swing.html": _build_detail_page(
            "swing", "KRİPTO SWİNG EKRANI", "Motor 8 • v2.5 + 45G + Birleşik C",
            "SWING", scan_results["swing"], state, now_str, ctx_pills,
            "Şu an 200 coin evreninde aktif Swing sinyali yok."
        ),
    }

    for fname, content in pages.items():
        with open(fname, "w", encoding="utf-8") as f:
            f.write(content)

    return REPORT_HTML