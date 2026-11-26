# =============================================================================
# SWINGRADAR PRO — MERKEZİ AYARLAR & PORTFÖY PARAMETRELERİ
# =============================================================================

FAPI = "https://fapi.binance.com"

# Kripto Evren & Hacim Ayarları
CRYPTO_MAX_COINS = 200
CRYPTO_MIN_VOL_24H = 0.0
SCALPER_MIN_VOL_USD = 3_000_000.0  # 1H Scalper günlük min hacim: 25M$

MIN_DISPLAY_SCORE = 60

# Çıktı & Takip Dosyaları
STATE_FILE = "portfolio_state.json"
TRADES_CSV = "trades_history.csv"
REPORT_HTML = "index.html"

# 3 Bağımsız Bileşik Kasa (Pozisyon Büyüklüğü Bazlı)
PORTFOLIOS = {
    "BIST": {
        "name": "BIST 100 + VİOP Kasası",
        "currency": "TL",
        "initial": 100_000.0,
        "alloc_pct": 0.10,  # Her işleme güncel kasanın %10'u
    },
    "SCALPER": {
        "name": "Kripto 1H Scalper Kasası",
        "currency": "$",
        "initial": 10_000.0,
        "alloc_pct": 0.01,  # Her işleme güncel kasanın %1'i
    },
    "SWING": {
        "name": "Kripto Swing Kasası",
        "currency": "$",
        "initial": 10_000.0,
        "alloc_pct": 0.05,  # Her işleme güncel kasanın %5'i
    },
}

EXCLUDED_CRYPTO_SYMS = {
    "USDCUSDT", "FDUSDUSDT", "USDEUSDT", "TUSDUSDT", "BUSDUSDT",
    "DAIUSDT", "USDPUSDT", "EURUSDT", "AEURUSDT", "USD1USDT",
}

BIST_100 = [
    "AKBNK", "GARAN", "ISCTR", "YKBNK", "HALKB", "VAKBN", "SKBNK", "TSKB", "KCHOL", "SAHOL",
    "AGHOL", "ALARK", "DOHOL", "ECZYT", "ENKAI", "GSDHO", "SISE", "TKFEN", "TUPRS", "PETKM",
    "THYAO", "PGSUS", "TAVHL", "CLEBI", "FROTO", "TOASO", "DOAS", "OTKAR", "TTRAK", "ASELS",
    "EREGL", "KRDMD", "TRALT", "TRMET", "ARCLK", "VESBE", "VESTL", "TCELL", "TTKOM", "BRSAN",
    "BIMAS", "MGROS", "SOKM", "AEFES", "CCOLA", "ULKER", "TABGD", "OBAMS", "HEKTS", "SASA",
    "ASTOR", "EUPWR", "GESAN", "KONTR", "SMRTG", "CWENE", "ALFAS", "YEOTK", "ENERY", "ENJSA",
    "ODAS", "ZOREN", "AKSEN", "AKFYE", "GWIND", "CANTE", "EKGYO", "ISGYO", "TRGYO", "EGEEN",
    "OYAKC", "CIMSA", "AKCNS", "BTCIM", "BSOKE", "GUBRF", "BAGFS", "AKSA", "BRYAT", "KCAER",
    "MIATK", "REEDR", "ARDYZ", "LOGO", "SDTTR", "MPARK", "SELEC", "ECILC", "RGYAS", "PASEU",
    "TURSG", "ANSGR", "ISMEN", "KLSER", "QUAGR", "MAVI", "BFREN", "KARSN", "KTLEV", "TUKAS",
]

BIST_30 = {
    "AKBNK", "AEFES", "ASELS", "ASTOR", "BIMAS", "CIMSA", "EKGYO", "ENKAI", "EREGL", "FROTO",
    "GARAN", "GUBRF", "ISCTR", "KCHOL", "KRDMD", "MGROS", "PETKM", "PGSUS", "SAHOL", "SASA",
    "SISE", "TAVHL", "TCELL", "THYAO", "TOASO", "TRALT", "TTKOM", "TUPRS", "ULKER", "YKBNK",
}

BIST_31_50 = {
    "AGHOL", "AKSA", "AKSEN", "ALARK", "ARCLK", "BRSAN", "CCOLA", "DOAS", "DOHOL", "ENJSA",
    "HALKB", "HEKTS", "ISMEN", "KONTR", "MPARK", "ODAS", "OTKAR", "OYAKC", "SOKM", "VAKBN",
}

VIOP_ALL = [
    "XU030",
    "AEFES", "AKBNK", "AKSEN", "ALARK", "ARCLK", "ASELS", "ASTOR", "BIMAS", "BRSAN", "CIMSA",
    "DOAS", "DOHOL", "EKGYO", "ENJSA", "ENKAI", "EREGL", "FROTO", "GARAN", "GUBRF", "HALKB",
    "HEKTS", "ISCTR", "ISGYO", "ISMEN", "KCHOL", "KONTR", "KRDMD", "MGROS", "ODAS", "OTKAR",
    "OYAKC", "PETKM", "PGSUS", "SAHOL", "SASA", "SISE", "SOKM", "TAVHL", "TCELL", "THYAO",
    "TKFEN", "TOASO", "TRALT", "TSKB", "TTKOM", "TUPRS", "ULKER", "VAKBN", "VESTL", "YKBNK",
]
