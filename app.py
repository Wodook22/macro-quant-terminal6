# ============================================================
# Macro Quant Terminal v15
# fredapi 방식으로 FRED 데이터 로드 (504 타임아웃 해결)
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, date
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
import plotly.express as px
import requests
from dotenv import load_dotenv

load_dotenv()

# ── 선택 패키지 ──────────────────────────────────────────────
try:
    from fredapi import Fred
    HAS_FREDAPI = True
except Exception:
    HAS_FREDAPI = False

try:
    import google.generativeai as genai
    HAS_GENAI = True
except Exception:
    genai = None
    HAS_GENAI = False

try:
    from anthropic import Anthropic as AnthropicClient
    HAS_ANTHROPIC = True
except Exception:
    AnthropicClient = None
    HAS_ANTHROPIC = False

try:
    from pykrx import stock as krx_stock
    HAS_PYKRX = True
except Exception:
    HAS_PYKRX = False

try:
    from pycoingecko import CoinGeckoAPI
    cg = CoinGeckoAPI()
    HAS_COINGECKO = True
except Exception:
    HAS_COINGECKO = False

try:
    from streamlit_autorefresh import st_autorefresh
    HAS_AUTOREFRESH = True
except Exception:
    HAS_AUTOREFRESH = False

# ── Streamlit Secrets / 환경변수에서 키 로드 ──────────────────
def read_secret(key: str, default: str = "") -> str:
    """Streamlit Cloud Secrets → .env 순서로 읽기."""
    try:
        return st.secrets.get(key, "") or os.getenv(key, default)
    except Exception:
        return os.getenv(key, default)

ENV_GEMINI_KEY    = read_secret("GEMINI_API_KEY")
ENV_ANTHROPIC_KEY = read_secret("ANTHROPIC_API_KEY")
ENV_FRED_KEY      = read_secret("FRED_API_KEY")

# ============================================================
# 0. 페이지 설정 & CSS
# ============================================================

st.set_page_config(
    page_title="MACRO QUANT TERMINAL v15",
    layout="wide",
    page_icon="🏛️",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Noto+Sans+KR:wght@400;700&display=swap');
html, body, [data-testid="stAppViewContainer"] {
    background-color: #0b0e14;
    font-family: 'Noto Sans KR', sans-serif;
    color: #adbac7;
}
[data-testid="stSidebar"] { background-color: #15181f; border-right: 1px solid #2d3139; }
h1 { color: #4da6ff !important; font-weight: 700; font-family: 'JetBrains Mono', sans-serif; }
h2, h3 { color: #f0f6fc !important; border-left: 4px solid #4da6ff; padding-left: 12px; }
.stMetric { background-color: #15181f; border: 1px solid #2d3139; padding: 15px; border-radius: 6px; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# 1. 상수
# ============================================================

SECTOR_ETFS: Dict[str, str] = {
    "Communication Services": "XLC",
    "Consumer Discretionary": "XLY",
    "Consumer Staples":       "XLP",
    "Energy":                 "XLE",
    "Financials":             "XLF",
    "Health Care":            "XLV",
    "Industrials":            "XLI",
    "Materials":              "XLB",
    "Real Estate":            "XLRE",
    "Technology":             "XLK",
    "Utilities":              "XLU",
}

SECTOR_STOCKS: Dict[str, List[str]] = {
    "Communication Services": ["GOOGL","META","NFLX","DIS","TMUS","CMCSA"],
    "Consumer Discretionary": ["AMZN","TSLA","HD","MCD","NKE","BKNG"],
    "Consumer Staples":       ["PG","COST","WMT","KO","PEP","PM"],
    "Energy":                 ["XOM","CVX","COP","SLB","EOG","OXY"],
    "Financials":             ["JPM","BAC","WFC","GS","MS","BLK"],
    "Health Care":            ["LLY","UNH","JNJ","ABBV","MRK","TMO"],
    "Industrials":            ["GE","CAT","RTX","HON","UNP","DE"],
    "Materials":              ["LIN","SHW","FCX","NEM","APD","ECL"],
    "Real Estate":            ["PLD","AMT","EQIX","WELL","SPG","DLR"],
    "Technology":             ["NVDA","MSFT","AAPL","AVGO","AMD","ORCL"],
    "Utilities":              ["NEE","SO","DUK","AEP","SRE","EXC"],
}

KR_STOCKS: Dict[str, List[str]] = {
    "반도체": ["005930.KS","000660.KS"],
    "2차전지": ["006400.KS","051910.KS"],
    "바이오":  ["207940.KS","068270.KS"],
    "금융":   ["055550.KS","105560.KS"],
    "자동차":  ["005380.KS","000270.KS"],
}
KR_ETFS: Dict[str, str] = {
    "KOSPI200":  "069500.KS",
    "KOSDAQ150": "229200.KS",
    "KR반도체":  "091160.KS",
    "KR인버스":  "114800.KS",
}
COMMODITIES: Dict[str, str] = {
    "WTI 원유": "CL=F", "브렌트 원유": "BZ=F",
    "금": "GC=F", "은": "SI=F", "구리": "HG=F", "천연가스": "NG=F",
}
CRYPTO_TICKERS: Dict[str, str] = {
    "Bitcoin": "BTC-USD", "Ethereum": "ETH-USD", "BNB": "BNB-USD",
}
GLOBAL_INDICES: Dict[str, str] = {
    "S&P500":"^GSPC","나스닥":"^IXIC","DOW":"^DJI",
    "KOSPI":"^KS11","KOSDAQ":"^KQ11","닛케이":"^N225","항셍":"^HSI",
    "상해종합":"000001.SS","DAX":"^GDAXI","VIX":"^VIX",
    "달러인덱스":"DX-Y.NYB","원달러":"KRW=X","TLT(장기채)":"TLT","GLD(금)":"GLD",
}

ETF_TO_SECTOR = {v: k for k, v in SECTOR_ETFS.items()}
ALL_SECTOR_ETFS = list(SECTOR_ETFS.values())
ALL_STOCKS = sorted(set(t for tl in SECTOR_STOCKS.values() for t in tl))
STOCK_SECTOR_MAP = {t: s for s, tl in SECTOR_STOCKS.items() for t in tl}

# FRED 시리즈 메타 (fredapi 사용 — CSV URL 불필요)
FRED_META: Dict[str, Dict] = {
    "WALCL":         {"label": "Fed Total Assets (B USD)",       "transform": "/1000", "freq": "weekly"},
    "WTREGEN":       {"label": "TGA Balance (B USD)",            "transform": "/1000", "freq": "weekly"},
    "RRPONTSYD":     {"label": "Reverse Repo RRP (B USD)",       "transform": "none",  "freq": "daily"},
    "M2SL":          {"label": "M2 Money Supply (B USD)",        "transform": "none",  "freq": "monthly"},
    "TB3MS":         {"label": "3M T-Bill Rate",                 "transform": "/100",  "freq": "monthly"},
    "DGS10":         {"label": "10Y Treasury Yield (%)",         "transform": "none",  "freq": "daily"},
    "DGS2":          {"label": "2Y Treasury Yield (%)",          "transform": "none",  "freq": "daily"},
    "T10Y2Y":        {"label": "10Y-2Y Yield Spread (%)",        "transform": "none",  "freq": "daily"},
    "BAMLH0A0HYM2":  {"label": "HY Credit Spread OAS (%)",       "transform": "none",  "freq": "daily"},
    "NFCI":          {"label": "Financial Conditions Index",     "transform": "none",  "freq": "weekly"},
    "GDPC1":         {"label": "Real GDP (B USD)",               "transform": "none",  "freq": "quarterly"},
    "GDP":           {"label": "Nominal GDP (B USD)",            "transform": "none",  "freq": "quarterly"},
    "NCBEILQ027S":   {"label": "US Equity Market Cap (B USD)",   "transform": "none",  "freq": "quarterly"},
    "WILL5000IND": {"label": "Wilshire 5000 Full Cap (B USD)", "transform": "none",  "freq": "daily"},
    "CPIAUCSL":      {"label": "CPI (Index)",                    "transform": "none",  "freq": "monthly"},
    "UNRATE":        {"label": "Unemployment Rate (%)",          "transform": "none",  "freq": "monthly"},
}

STALE_WARN_DAYS = 7

# ============================================================
# 2. 유틸리티
# ============================================================

def safe_zscore(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    std = s.std(ddof=0)
    if pd.isna(std) or std == 0:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - s.mean()) / std

def pct_n(s: pd.Series, n: int) -> float:
    s = pd.to_numeric(s, errors="coerce").dropna()
    if len(s) <= n or s.iloc[-n] == 0:
        return np.nan
    return float(s.iloc[-1] / s.iloc[-n] - 1.0)

def ann_vol(r: pd.Series, window: int = None) -> float:
    r = pd.to_numeric(r, errors="coerce").dropna()
    if window:
        r = r.tail(window)
    return float(r.std() * np.sqrt(252)) if len(r) >= 5 else np.nan

def max_dd(price: pd.Series, w: int = 126) -> float:
    s = pd.to_numeric(price, errors="coerce").dropna().tail(w)
    return float((s / s.cummax() - 1).min()) if len(s) >= 2 else np.nan

def get_val(df: pd.DataFrame, col: str) -> float:
    if col not in df.columns:
        return np.nan
    s = df[col].dropna()
    return float(s.iloc[-1]) if not s.empty else np.nan

def get_delta(df: pd.DataFrame, col: str, n: int = 21) -> float:
    if col not in df.columns:
        return np.nan
    s = df[col].dropna()
    return float(s.iloc[-1] - s.iloc[-n]) if len(s) >= n else np.nan

# ============================================================
# 3. 데이터 로더 — fredapi 방식 (핵심 변경)
# ============================================================

def get_fred_client() -> Optional[object]:
    """fredapi Fred 클라이언트 반환. 키 없으면 None."""
    if not HAS_FREDAPI:
        return None
    key = ENV_FRED_KEY
    if not key:
        return None
    try:
        return Fred(api_key=key)
    except Exception:
        return None


def _apply_transform(s: pd.Series, transform: str) -> pd.Series:
    if transform == "/1000": return s / 1000
    if transform == "/100":  return s / 100
    return s


def _load_fred_via_api(series_id: str, fred_key: str, transform: str) -> pd.Series:
    """fredapi 방식 — 요청 간 0.5초 딜레이로 Rate Limit 방지."""
    fred = Fred(api_key=fred_key)
    time.sleep(0.5)                  # Rate Limit 방지
    s = fred.get_series(series_id)
    s = pd.to_numeric(s, errors="coerce").dropna()
    s.index = pd.to_datetime(s.index)
    s.index.name = "DATE"
    return _apply_transform(s, transform).rename(series_id)


def _load_fred_via_csv(series_id: str, transform: str) -> pd.Series:
    """CSV fallback — pandas timeout 파라미터 없이 requests로 직접."""
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    resp = requests.get(url, timeout=20)
    resp.raise_for_status()
    from io import StringIO
    df = pd.read_csv(StringIO(resp.text))
    df.columns = [c.strip() for c in df.columns]
    date_col = next((c for c in df.columns if "date" in c.lower()), df.columns[0])
    val_cols  = [c for c in df.columns if c != date_col]
    if not val_cols:
        return pd.Series(dtype=float, name=series_id)
    val_col = val_cols[0]
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[val_col]  = pd.to_numeric(df[val_col].replace(".", np.nan), errors="coerce")
    s = df.dropna(subset=[date_col, val_col]).set_index(date_col)[val_col].sort_index()
    s.index.name = "DATE"
    return _apply_transform(s, transform).rename(series_id)


@st.cache_data(ttl=3600, show_spinner=False)
def load_fred_series(series_id: str, fred_key: str) -> pd.Series:
    """
    fredapi → CSV fallback 순서로 FRED 시리즈 로드.
    Rate Limit 방지를 위해 요청 간 딜레이 적용.
    """
    meta      = FRED_META.get(series_id, {})
    transform = meta.get("transform", "none")

    # ── 방법 1: fredapi ───────────────────────────────────────
    if HAS_FREDAPI and fred_key:
        try:
            return _load_fred_via_api(series_id, fred_key, transform)
        except Exception as e:
            st.warning(f"fredapi {series_id} 실패, CSV fallback 시도: {e}")

    # ── 방법 2: CSV fallback (requests 기반, timeout 인수 없음) ─
    try:
        return _load_fred_via_csv(series_id, transform)
    except Exception as e:
        st.warning(f"FRED {series_id} 로드 실패: {e}")
        return pd.Series(dtype=float, name=series_id)


@st.cache_data(ttl=3600, show_spinner=False)
def build_macro(fred_key: str) -> pd.DataFrame:
    series_dict = {}
    # 순차 로드 — 동시 요청 방지 (Rate Limit 해결)
    for sid in FRED_META:
        s = load_fred_series(sid, fred_key)
        if not s.empty:
            series_dict[sid] = s

    if not series_dict:
        st.error("FRED 데이터를 하나도 불러오지 못했습니다.\n\nStreamlit Cloud → Settings → Secrets에 FRED_API_KEY를 입력해주세요.\n무료 발급: https://fred.stlouisfed.org/docs/api/api_key.html")
        st.stop()

    macro = pd.concat(series_dict.values(), axis=1).sort_index().ffill().dropna(how="all").tail(1500)

    if all(c in macro.columns for c in ["WALCL","WTREGEN","RRPONTSYD"]):
        macro["Net_Liq"]      = macro["WALCL"] - macro["WTREGEN"] - macro["RRPONTSYD"]
        # WALCL/WTREGEN 주간 데이터 기준 diff
        macro["Net_Liq_1W"]   = macro["Net_Liq"].diff(1)   # 1주 변화 (주간 데이터)
        macro["Net_Liq_1M"]   = macro["Net_Liq"].diff(4)   # 4주 ≈ 1개월
        macro["Net_Liq_MA20"] = macro["Net_Liq"].rolling(20).mean()
        macro["Net_Liq_MA60"] = macro["Net_Liq"].rolling(60).mean()

    if "M2SL" in macro.columns:
        # M2SL 월간 데이터 → pct_change(12) = 전년동월비 YoY
        macro["M2_YoY"] = macro["M2SL"].pct_change(12) * 100
    if "CPIAUCSL" in macro.columns:
        macro["CPI_YoY"] = macro["CPIAUCSL"].pct_change(12) * 100
    if "BAMLH0A0HYM2" in macro.columns:
        macro["HY_1M_Chg"] = macro["BAMLH0A0HYM2"].diff(21)
    if "DGS10" in macro.columns:
        macro["DGS10_1M_Chg"] = macro["DGS10"].diff(21)
    if "NFCI" in macro.columns:
        macro["NFCI_1M_Chg"] = macro["NFCI"].diff(4)

    return macro


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_prices(tickers: List[str], period: str = "1y") -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not tickers:
        return pd.DataFrame(), pd.DataFrame()
    try:
        raw = yf.download(tickers, period=period, auto_adjust=True, progress=False, threads=True)
        if raw.empty:
            return pd.DataFrame(), pd.DataFrame()
        def extract(field):
            if isinstance(raw.columns, pd.MultiIndex):
                if field in raw.columns.get_level_values(0):
                    return raw[field].copy().ffill()
            else:
                if field in raw.columns:
                    return raw[[field]].copy().ffill()
            return pd.DataFrame()
        return extract("Close"), extract("Volume")
    except Exception:
        return pd.DataFrame(), pd.DataFrame()


@st.cache_data(ttl=7200, show_spinner=False)
def fetch_fear_greed() -> dict:
    try:
        r = requests.get(
            "https://production.dataviz.cnn.io/index/fearandgreed/graphdata",
            timeout=10, headers={"User-Agent": "Mozilla/5.0"},
        )
        cur = r.json()["fear_and_greed"]
        return {"score": float(cur["score"]), "rating": cur["rating"]}
    except Exception:
        return {"score": 50, "rating": "N/A"}


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_buffett(fred_key: str) -> dict:
    """
    버핏지표 = 미국 주식시장 시총 / 미국 명목 GDP

    시총 계산 방식:
      FRED NCBEILQ027S = 비금융 기업 주식 시가총액 (분기, 십억달러)
      → 가장 정확한 방식이지만 분기 지연
      fallback: SPY 시총 / SPY S&P500 비중 (0.80) 으로 전체 시장 추정

    GDP:
      FRED GDP (명목GDP, 십억달러) — GDPC1(실질) 아닌 명목 사용이 정확
      fallback: GDPC1 사용
    """
    try:
        # ── GDP: 명목 GDP 우선 (FRED 'GDP' 시리즈) ──────────────
        gdp_series = load_fred_series("GDP", fred_key)
        if gdp_series.empty:
            gdp_series = load_fred_series("GDPC1", fred_key)
        if gdp_series.empty:
            return {}
        latest_gdp_bn = float(gdp_series.dropna().iloc[-1])  # 십억달러

        # ── 시총: NCBEILQ027S (비금융기업 주식 시총, 십억달러) ──
        mktcap_bn = None
        try:
            ncb = load_fred_series("NCBEILQ027S", fred_key)
            if not ncb.empty:
                mktcap_bn = float(ncb.dropna().iloc[-1])  # 이미 십억달러
        except Exception:
            pass

        # ── fallback: Wilshire 5000 Full Cap Index ──────────────
        if mktcap_bn is None or mktcap_bn <= 0:
            try:
                # WILL5000IND: 윌셔5000 풀캡 (달러 기준 시총 근사값, 십억달러)
                will = load_fred_series("WILL5000IND", fred_key)
                if not will.empty:
                    mktcap_bn = float(will.dropna().iloc[-1])
            except Exception:
                pass

        # ── fallback: yfinance SPY 시총으로 전체 시장 추정 ───────
        if mktcap_bn is None or mktcap_bn <= 0:
            try:
                spy_info = yf.Ticker("SPY").fast_info
                spy_mktcap = getattr(spy_info, "market_cap", None)
                if spy_mktcap and spy_mktcap > 0:
                    # SPY는 S&P500 추종 → S&P500이 전체 시장의 약 80%
                    mktcap_bn = (spy_mktcap / 1e9) / 0.80
            except Exception:
                pass

        if mktcap_bn is None or mktcap_bn <= 0 or latest_gdp_bn <= 0:
            return {}

        ratio = mktcap_bn / latest_gdp_bn

        if ratio > 1.8:    val = "극단적 고평가 (역사적 최고 수준)"
        elif ratio > 1.4:  val = "크게 고평가 (과열 주의)"
        elif ratio > 1.1:  val = "다소 고평가"
        elif ratio > 0.85: val = "적정 수준"
        elif ratio > 0.65: val = "다소 저평가"
        else:              val = "크게 저평가"

        return {
            "ratio": ratio,
            "valuation": val,
            "mktcap_bn": mktcap_bn,
            "gdp_bn": latest_gdp_bn,
        }
    except Exception:
        return {}


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_sector_pe() -> Dict[str, float]:
    pe_map = {}
    for sector, etf in SECTOR_ETFS.items():
        try:
            info = yf.Ticker(etf).info
            pe = info.get("trailingPE") or info.get("forwardPE")
            if pe and isinstance(pe, (int, float)) and 0 < pe < 500:
                pe_map[sector] = float(pe)
        except Exception:
            pass
    return pe_map


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_crypto_dom() -> dict:
    if not HAS_COINGECKO:
        return {}
    try:
        data = cg.get_global()
        return {
            "btc": data["market_cap_percentage"].get("btc", 0),
            "eth": data["market_cap_percentage"].get("eth", 0),
            "total_mc": data["total_market_cap"].get("usd", 0),
            "mc_chg_24h": data.get("market_cap_change_percentage_24h_usd", 0),
        }
    except Exception:
        return {}


# ============================================================
# 4. 데이터 신선도 체크
# ============================================================

def check_data_freshness(macro: pd.DataFrame) -> List[Dict]:
    today = datetime.now().date()
    results = []
    for sid, meta in FRED_META.items():
        if sid not in macro.columns:
            continue
        s = macro[sid].dropna()
        if s.empty:
            continue
        last_date = s.index[-1].date()
        days_old = (today - last_date).days
        freq = meta.get("freq", "")
        # 주기별 허용 지연
        ok_days = {"daily": 3, "weekly": 10, "monthly": 40, "quarterly": 100}.get(freq, STALE_WARN_DAYS)
        results.append({
            "지표": meta.get("label", sid),
            "ID": sid,
            "최신일": str(last_date),
            "지연(일)": days_old,
            "발표주기": freq,
            "상태": "✅ 정상" if days_old <= ok_days else "⚠️ 지연",
        })
    return results


# ============================================================
# 5. 레짐 분류
# ============================================================

def classify_regime(macro: pd.DataFrame, vix: float) -> Dict:
    if macro.empty:
        return {"master": "Unknown", "score": 0}

    def g(col):
        return get_val(macro, col)

    net_liq    = g("Net_Liq")
    net_liq_ma = g("Net_Liq_MA20")
    net_liq_1m = g("Net_Liq_1M")
    hy         = g("BAMLH0A0HYM2")
    hy_1m      = g("HY_1M_Chg")
    nfci       = g("NFCI")
    curve      = g("T10Y2Y")
    dgs10_1m   = g("DGS10_1M_Chg")
    cpi_yoy    = g("CPI_YoY")
    m2_yoy     = g("M2_YoY")

    liq_ok   = pd.notna(net_liq) and pd.notna(net_liq_ma)
    liquidity = "Expanding" if (liq_ok and net_liq > net_liq_ma and net_liq_1m >= 0) else "Contracting"
    vol_reg  = "High Vol" if vix >= 30 else ("Elevated" if vix >= 20 else "Calm")
    credit   = "Stress Rising"  if (pd.notna(hy_1m) and hy_1m > 0) else "Stress Easing"
    fin      = "Tight"  if (pd.notna(nfci) and nfci > 0) else "Loose"
    curve_r  = "Inverted" if (pd.notna(curve) and curve < 0) else "Normal"
    rates    = "Rising" if (pd.notna(dgs10_1m) and dgs10_1m > 0.1) else "Falling/Stable"
    infl     = "High" if pd.notna(cpi_yoy) and cpi_yoy > 4 else ("Moderate" if pd.notna(cpi_yoy) and cpi_yoy > 2 else "Low")
    m2_g     = "Expanding" if pd.notna(m2_yoy) and m2_yoy > 5 else ("Contracting" if pd.notna(m2_yoy) and m2_yoy < 0 else "Stable")

    score = 0.0
    score += 1.5  if liquidity == "Expanding"      else -1.5
    score += 1.0  if vol_reg == "Calm"              else (-1.5 if vol_reg == "High Vol" else -0.5)
    score += 1.0  if credit == "Stress Easing"      else -1.0
    score += 0.5  if fin == "Loose"                 else -0.5
    score += 0.5  if curve_r == "Normal"            else -0.5
    score += 0.5  if m2_g == "Expanding"            else (-0.3 if m2_g == "Contracting" else 0)
    if infl == "High": score -= 0.5

    if score >= 2.5:   master = "강한 Risk-On"
    elif score >= 1.0: master = "Risk-On"
    elif score <= -2.5:master = "강한 Risk-Off"
    elif score <= -1.0:master = "Risk-Off"
    else:              master = "Mixed/Transition"

    if "Risk-On" in master:
        preferred = "성장주·기술주·경기민감섹터·크립토"
        avoid     = "장기채·달러·방어주"
    elif "Risk-Off" in master:
        preferred = "금·달러·단기채·방어주(필수소비재·헬스케어·유틸리티)"
        avoid     = "성장주·크립토·고위험채권"
    elif infl == "High":
        preferred = "실물자산·에너지·금·원자재"
        avoid     = "장기채·고밸류에이션 성장주"
    else:
        preferred = "분산 포트폴리오"
        avoid     = "과도한 단일 자산 집중"

    return {
        "master": master, "score": round(score, 2),
        "liquidity": liquidity, "vol": vol_reg, "credit": credit,
        "fin": fin, "curve": curve_r, "rates": rates,
        "inflation": infl, "m2": m2_g,
        "preferred": preferred, "avoid": avoid,
        "net_liq": net_liq, "net_liq_1m": net_liq_1m,
        "hy": hy, "hy_1m": hy_1m, "nfci": nfci,
        "curve_val": curve, "dgs10": g("DGS10"), "dgs2": g("DGS2"),
        "cpi_yoy": cpi_yoy, "m2_yoy": m2_yoy, "unrate": g("UNRATE"),
    }


# ============================================================
# 6. 섹터 스코어링
# ============================================================

@dataclass
class ScoreConfig:
    rs_1m: float = 0.30
    rs_3m: float = 0.20
    volume: float = 0.12
    trend: float = 0.16
    low_vol: float = 0.10
    drawdown: float = 0.06
    macro_fit: float = 0.06


def calc_macro_fit(sector: str, regime: Dict) -> float:
    score = 0.0
    master = regime.get("master","")
    liq    = regime.get("liquidity","")
    vol    = regime.get("vol","")
    credit = regime.get("credit","")
    rates  = regime.get("rates","")
    infl   = regime.get("inflation","")

    offensive  = {"Technology","Communication Services","Consumer Discretionary","Industrials","Materials","Financials"}
    defensive  = {"Utilities","Consumer Staples","Health Care"}
    rate_sens  = {"Real Estate","Utilities","Technology"}
    infl_ben   = {"Energy","Materials"}

    if "Risk-On"  in master and sector in offensive:  score += 0.70
    if "Risk-Off" in master and sector in defensive:  score += 0.90
    if "Risk-Off" in master and sector in {"Technology","Consumer Discretionary"}: score -= 0.50
    if liq == "Expanding"     and sector in {"Technology","Communication Services"}: score += 0.35
    if liq == "Contracting"   and sector in defensive:    score += 0.35
    if vol == "High Vol"      and sector in defensive:    score += 0.45
    if vol == "High Vol"      and sector in {"Technology","Consumer Discretionary"}: score -= 0.30
    if credit == "Stress Rising" and sector in {"Consumer Staples","Health Care","Utilities"}: score += 0.35
    if credit == "Stress Rising" and sector in {"Consumer Discretionary","Financials","Real Estate"}: score -= 0.35
    if rates == "Rising"      and sector in {"Financials","Energy"}: score += 0.25
    if rates == "Rising"      and sector in rate_sens:  score -= 0.30
    if infl == "High"         and sector in infl_ben:   score += 0.40
    if infl == "High"         and sector in {"Technology","Real Estate"}: score -= 0.30
    return float(score)


def pe_penalty(pe: float) -> float:
    if pd.isna(pe) or pe <= 0: return 0.0
    if pe < 18:  return 0.0
    if pe < 25:  return -0.1
    if pe < 35:  return -0.25
    return -0.5

def pe_text(pe: float) -> str:
    if pd.isna(pe) or pe <= 0: return "N/A"
    if pe < 12:  return "저평가"
    if pe < 18:  return "적정"
    if pe < 25:  return "다소 고평가"
    if pe < 35:  return "고평가"
    return "극단적 고평가"


def score_sectors(
    close: pd.DataFrame, volume: pd.DataFrame,
    regime: Dict, cfg: ScoreConfig,
    sector_pe: Dict[str, float] = None,
) -> pd.DataFrame:
    spy = close.get("SPY") if "SPY" in close.columns else None
    if spy is None or spy.dropna().empty:
        return pd.DataFrame()

    rows = []
    for sector, etf in SECTOR_ETFS.items():
        if etf not in close.columns: continue
        price = close[etf].dropna()
        if len(price) < 80: continue
        ret  = price.pct_change().dropna()
        spy_a = spy.reindex(price.index).ffill().dropna()
        common = price.index.intersection(spy_a.index)
        price, spy_p = price.reindex(common).dropna(), spy_a.reindex(common).dropna()
        if len(price) < 80: continue

        rs1m = pct_n(price,21); spy1m = pct_n(spy_p,21)
        rs3m = pct_n(price,63); spy3m = pct_n(spy_p,63)
        rs1m_v = (rs1m - spy1m) if (pd.notna(rs1m) and pd.notna(spy1m)) else np.nan
        rs3m_v = (rs3m - spy3m) if (pd.notna(rs3m) and pd.notna(spy3m)) else np.nan

        vol_surge = np.nan
        if etf in volume.columns:
            v = volume[etf].dropna()
            if len(v) >= 21 and v.iloc[-21:-1].mean() > 0:
                vol_surge = float(v.iloc[-1] / v.iloc[-21:-1].mean())

        ma20 = price.rolling(20).mean().iloc[-1]
        ma50 = price.rolling(50).mean().iloc[-1]
        trend = 0.0
        if pd.notna(ma20) and pd.notna(ma50) and ma50 > 0:
            trend = float((price.iloc[-1]/ma50 - 1) + (0.05 if ma20 > ma50 else -0.05))

        pe = (sector_pe or {}).get(sector, np.nan)
        rows.append({
            "Sector": sector, "ETF": etf,
            "Last": float(price.iloc[-1]),
            "1W": pct_n(price,5), "1M": pct_n(price,21),
            "3M": pct_n(price,63), "6M": pct_n(price,126),
            "RS_1M": rs1m_v, "RS_3M": rs3m_v,
            "VolSurge": vol_surge, "Trend": trend,
            "Vol20": ann_vol(ret,20), "MDD126": max_dd(price,126),
            "MacroFit": calc_macro_fit(sector, regime),
            "PER": pe, "PER_Penalty": pe_penalty(pe),
        })

    df = pd.DataFrame(rows)
    if df.empty: return df

    df["Z_RS1M"]     = safe_zscore(df["RS_1M"])
    df["Z_RS3M"]     = safe_zscore(df["RS_3M"])
    df["Z_Vol"]      = safe_zscore(df["VolSurge"])
    df["Z_Trend"]    = safe_zscore(df["Trend"])
    df["Z_LowVol"]   = safe_zscore(-df["Vol20"])
    df["Z_DD"]       = safe_zscore(df["MDD126"])
    df["Z_MacroFit"] = safe_zscore(df["MacroFit"])

    df["Score"] = (
        cfg.rs_1m    * df["Z_RS1M"]  + cfg.rs_3m    * df["Z_RS3M"]  +
        cfg.volume   * df["Z_Vol"]   + cfg.trend     * df["Z_Trend"] +
        cfg.low_vol  * df["Z_LowVol"]+ cfg.drawdown  * df["Z_DD"]   +
        cfg.macro_fit* df["Z_MacroFit"] + df["PER_Penalty"]
    )
    return df.sort_values("Score", ascending=False).reset_index(drop=True)


def score_stocks(close: pd.DataFrame, volume: pd.DataFrame, selected_sectors: List[str]) -> pd.DataFrame:
    universe = sorted(set(t for s in selected_sectors for t in SECTOR_STOCKS.get(s,[])))
    rows = []
    for t in universe:
        if t not in close.columns: continue
        price = close[t].dropna()
        if len(price) < 80: continue
        ret = price.pct_change().dropna()
        vol_surge = np.nan
        if t in volume.columns:
            v = volume[t].dropna()
            if len(v) >= 21 and v.iloc[-21:-1].mean() > 0:
                vol_surge = float(v.iloc[-1] / v.iloc[-21:-1].mean())
        ma20 = price.rolling(20).mean().iloc[-1]
        ma50 = price.rolling(50).mean().iloc[-1]
        trend = 0.0
        if pd.notna(ma20) and pd.notna(ma50) and ma50 > 0:
            trend = float((price.iloc[-1]/ma50 - 1) + (0.05 if ma20 > ma50 else -0.05))
        rows.append({
            "Ticker": t, "Sector": STOCK_SECTOR_MAP.get(t,"Unknown"),
            "Last": float(price.iloc[-1]),
            "1W": pct_n(price,5), "1M": pct_n(price,21), "3M": pct_n(price,63),
            "VolSurge": vol_surge, "Trend": trend,
            "Vol20": ann_vol(ret,20), "MDD126": max_dd(price,126),
        })
    df = pd.DataFrame(rows)
    if df.empty: return df
    for col in ["1M","3M","VolSurge","Trend"]:
        df[f"Z_{col}"] = safe_zscore(df[col])
    df["Z_LowVol"] = safe_zscore(-df["Vol20"])
    df["Z_MDD"]    = safe_zscore(df["MDD126"])
    df["Score"] = (
        0.30*df["Z_1M"] + 0.20*df["Z_3M"] + 0.15*df["Z_VolSurge"] +
        0.18*df["Z_Trend"] + 0.10*df["Z_LowVol"] + 0.07*df["Z_MDD"]
    )
    return df.sort_values("Score", ascending=False).reset_index(drop=True)


# ============================================================
# 7. MPT 최적화 (벡터화)
# ============================================================

def optimize_mpt(
    price_df: pd.DataFrame, assets: List[str],
    rf: float, cash_w: float,
    max_asset: float, max_sector: float,
    a2s: Dict[str, str], n_sim: int = 8000,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    assets = [a for a in assets if a in price_df.columns]
    if not assets:
        return {"CASH": 1.0}, pd.DataFrame()
    prices = price_df[assets].dropna(how="all").ffill().dropna()
    rets = prices.pct_change().dropna()
    if len(rets) < 30:
        w = (1 - cash_w) / len(assets)
        return {**{a: w for a in assets}, "CASH": cash_w}, pd.DataFrame()

    mu  = rets.mean().values * 252
    cov = rets.cov().values  * 252
    n   = len(assets)
    invest = 1 - cash_w

    rng   = np.random.default_rng(42)
    raw_w = rng.random((n_sim, n))
    raw_w = np.clip(raw_w, 0, max_asset)
    raw_w = raw_w / raw_w.sum(axis=1, keepdims=True)

    p_ret = raw_w @ mu
    p_var = np.einsum("ij,jk,ik->i", raw_w, cov, raw_w)
    p_vol = np.sqrt(np.maximum(p_var, 0))
    sharpe = np.where(p_vol > 0, (p_ret - rf) / p_vol, -np.inf)

    best_i = int(np.argmax(sharpe))
    best_w = raw_w[best_i]

    result = {a: float(best_w[i] * invest) for i, a in enumerate(assets)}
    result["CASH"] = float(cash_w)
    cloud = pd.DataFrame({"Volatility": p_vol[:500], "Return": p_ret[:500], "Sharpe": sharpe[:500]})
    return result, cloud


def calc_cash(vix: float, regime: Dict, geo_risk: str) -> float:
    c = 0.10
    if vix >= 30:   c += 0.20
    elif vix >= 25: c += 0.12
    elif vix >= 20: c += 0.06
    if regime.get("liquidity") == "Contracting": c += 0.08
    if regime.get("credit") == "Stress Rising":  c += 0.08
    if regime.get("fin") == "Tight":             c += 0.05
    if "Risk-Off" in regime.get("master",""):    c += 0.07
    if geo_risk == "Medium": c += 0.05
    elif geo_risk == "High": c += 0.12
    return float(min(max(c, 0.05), 0.50))


def perf_stats(rets: pd.Series, rf: float = 0.0) -> Dict:
    rets = pd.to_numeric(rets, errors="coerce").dropna()
    if len(rets) < 2:
        return {k: np.nan for k in ["CAGR","Vol","Sharpe","Sortino","MDD","WinRate"]}
    eq   = (1 + rets).cumprod()
    yrs  = max((rets.index[-1] - rets.index[0]).days / 365.25, 1/365.25)
    cagr = eq.iloc[-1] ** (1/yrs) - 1
    vol  = rets.std() * np.sqrt(252)
    exc  = rets - rf / 252
    dv   = rets[rets < 0].std() * np.sqrt(252)
    return {
        "CAGR": cagr, "Vol": vol,
        "Sharpe":  exc.mean() * 252 / vol if vol > 0 else np.nan,
        "Sortino": exc.mean() * 252 / dv  if dv  > 0 else np.nan,
        "MDD": (eq / eq.cummax() - 1).min(),
        "WinRate": (rets > 0).mean(),
    }


# ============================================================
# 8. 내 포트폴리오 파서
# ============================================================

def parse_my_portfolio(text: str) -> Dict[str, float]:
    result = {}
    for line in str(text or "").strip().splitlines():
        line = line.strip()
        if not line or line.startswith("#"): continue
        parts = line.replace(",",":").replace("=",":").split(":")
        if len(parts) < 2: continue
        ticker = parts[0].strip().upper()
        try:
            w_raw = float(parts[1].strip().replace("%",""))
            w = w_raw / 100 if w_raw > 1.5 else w_raw
            if ticker and 0 < w <= 1.0:
                result[ticker] = w
        except Exception:
            pass
    total = sum(result.values())
    if total > 0:
        result = {k: v/total for k, v in result.items()}
    return result


def compare_portfolios(my_p: Dict[str,float], rec_p: Dict[str,float], a2s: Dict[str,str]) -> pd.DataFrame:
    all_assets = sorted(set(list(my_p.keys()) + list(rec_p.keys())))
    rows = []
    for asset in all_assets:
        my_w  = my_p.get(asset, 0.0)
        rec_w = rec_p.get(asset, 0.0)
        diff  = rec_w - my_w
        sector = a2s.get(asset, ETF_TO_SECTOR.get(asset,"기타"))
        if diff > 0.02:   action = f"▲ {diff:+.1%} 늘리기"
        elif diff < -0.02:action = f"▼ {diff:+.1%} 줄이기"
        else:             action = "≈ 유지"
        rows.append({
            "자산": asset, "섹터": sector if asset != "CASH" else "현금",
            "내 비중": my_w, "추천 비중": rec_w, "차이": diff, "조정 방향": action,
        })
    return pd.DataFrame(rows).sort_values("차이", key=abs, ascending=False)


# ============================================================
# 9. 거래비용 반영 백테스트
# ============================================================

COST_BPS = {"US_ETF": 2, "US_STOCK": 2, "KR_STOCK": 33, "CRYPTO": 10}

def run_backtest(
    sector_close: pd.DataFrame, start: str,
    top_n: int = 3, benchmark: str = "SPY", cost_bps: int = 0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    needed = ALL_SECTOR_ETFS + [benchmark]
    close = sector_close[[c for c in needed if c in sector_close.columns]].dropna(how="all").ffill().dropna()
    if close.empty or benchmark not in close.columns or len(close) < 150:
        return pd.DataFrame(), pd.DataFrame()
    close = close.loc[pd.to_datetime(start):]
    daily = close.pct_change().fillna(0)
    rebal = close.resample("ME").last().index
    actual = []
    for d in rebal:
        idx = close.index[close.index <= d]
        if len(idx) > 0: actual.append(idx[-1])
    actual = sorted(set(actual))

    strat = pd.Series(index=close.index, dtype=float)
    logs  = []
    wts   = {e: 0.0 for e in ALL_SECTOR_ETFS}
    prev_wts = {e: 0.0 for e in ALL_SECTOR_ETFS}

    for i, dt in enumerate(actual):
        loc = close.index.get_loc(dt)
        if loc < 90: continue
        hist = close.iloc[:loc+1]
        spy  = hist[benchmark]
        rows = []
        for etf in ALL_SECTOR_ETFS:
            if etf not in hist.columns: continue
            p = hist[etf].dropna()
            if len(p) < 90: continue
            rs1m = (pct_n(p,21) or 0) - (pct_n(spy,21) or 0)
            rs3m = (pct_n(p,63) or 0) - (pct_n(spy,63) or 0)
            tr   = 1.0 if p.iloc[-1] > p.rolling(50).mean().iloc[-1] else -1.0
            rows.append({"ETF": etf, "Score": 0.5*rs1m + 0.4*rs3m + 0.1*tr})

        chosen = pd.DataFrame(rows).sort_values("Score", ascending=False).head(top_n)["ETF"].tolist()
        new_wts = {e: (1/top_n if e in chosen else 0.0) for e in ALL_SECTOR_ETFS}
        turnover = sum(abs(new_wts[e] - prev_wts.get(e,0.0)) for e in ALL_SECTOR_ETFS) / 2
        period_cost = turnover * cost_bps / 10000
        prev_wts = new_wts.copy()
        wts = new_wts
        logs.append({"날짜": str(dt.date()), "보유섹터": ", ".join(chosen),
                     "회전율": f"{turnover:.1%}", "비용(bps)": round(turnover*cost_bps,2)})

        next_dt = actual[i+1] if i+1 < len(actual) else close.index[-1]
        period  = close.index[(close.index > dt) & (close.index <= next_dt)]
        for j, day in enumerate(period):
            day_ret = sum(daily.loc[day,e]*w for e,w in wts.items() if e in daily.columns)
            if j == 0: day_ret -= period_cost
            strat.loc[day] = day_ret

    strat = strat.dropna()
    equity = pd.DataFrame({
        "섹터로테이션": (1 + strat).cumprod(),
        benchmark: (1 + daily.loc[strat.index, benchmark]).cumprod(),
    })
    return equity, pd.DataFrame(logs)


# ============================================================
# 10. LLM
# ============================================================

SYSTEM_RISK = """당신은 퀀트 모델 결과를 설명하는 리스크 분석 보조자입니다.
매수/매도 단정 표현을 절대 금지합니다. 데이터에 근거해 한국어로 설명하세요.
초보자도 이해할 수 있도록 어려운 용어는 괄호로 풀어 쓰세요.
불확실성, 데이터 지연 가능성, 모델 한계를 반드시 포함하세요."""

SYSTEM_ANALYST = """당신은 퀀트 모델 결과를 설명하는 애널리스트입니다.
매수/매도 지시, 목표가, 수익 보장 표현을 절대 금지합니다.
제공된 데이터에만 근거해 설명하고, 없는 정보를 지어내지 마세요.
초보자도 이해하도록 쉽게 설명하되, 근거 지표와 리스크를 반드시 제시하세요."""

LLM_LIMIT = 5  # 세션당 최대 호출 횟수

def call_llm(system_prompt: str, user_prompt: str, gemini_key: str, anthropic_key: str) -> str:
    if "llm_count" not in st.session_state:
        st.session_state.llm_count = 0
    if st.session_state.llm_count >= LLM_LIMIT:
        return f"⚠️ 이 세션에서 LLM 호출 횟수({LLM_LIMIT}회)를 초과했습니다. 페이지를 새로고침하면 초기화됩니다."

    if anthropic_key and HAS_ANTHROPIC:
        try:
            client = AnthropicClient(api_key=anthropic_key)
            resp = client.messages.create(
                model="claude-opus-4-5", max_tokens=2048,
                system=system_prompt,
                messages=[{"role":"user","content":user_prompt}],
            )
            st.session_state.llm_count += 1
            return resp.content[0].text
        except Exception as e:
            return f"Anthropic 오류: {e}"

    if gemini_key and HAS_GENAI:
        try:
            genai.configure(api_key=gemini_key)
            preferred = ["gemini-2.5-flash","gemini-2.0-flash","gemini-1.5-flash"]
            available = []
            try:
                for m in genai.list_models():
                    name = m.name.replace("models/","")
                    if "generateContent" in (getattr(m,"supported_generation_methods",[]) or []):
                        available.append(name)
            except Exception:
                pass
            model_name = next((p for p in preferred if p in available), available[0] if available else "gemini-2.5-flash")
            model = genai.GenerativeModel(model_name=model_name, system_instruction=system_prompt)
            resp = model.generate_content(user_prompt)
            st.session_state.llm_count += 1
            return resp.text
        except Exception as e:
            return f"Gemini 오류: {e}"

    return "사이드바에서 Gemini 또는 Anthropic API Key를 입력해주세요."

# ============================================================
# 11. 사이드바
# ============================================================

st.sidebar.markdown("### 🛠️ CONTROL PANEL")

if HAS_AUTOREFRESH:
    auto_ref = st.sidebar.selectbox("자동 새로고침", ["끄기","15분","30분","60분"], index=2)
    if auto_ref != "끄기":
        ms = {"15분":15*60000,"30분":30*60000,"60분":60*60000}[auto_ref]
        st_autorefresh(interval=ms, key="auto_ref")

view_mode = st.sidebar.radio("보기 모드 ⓘ", ["Beginner","Advanced"], index=0,
    help="Beginner: 핵심 정보만 · Advanced: 전체 수치·차트")
is_advanced = (view_mode == "Advanced")

if st.sidebar.button("📡 데이터 새로고침"):
    st.cache_data.clear()
    st.rerun()

geo_risk        = st.sidebar.selectbox("지정학 위험", ["Low","Medium","High"], index=1)
analysis_period = st.sidebar.selectbox("분석 기간", ["6mo","1y","2y","5y"], index=1)
top_sector_n    = st.sidebar.slider("상위 섹터 수", 2, 6, 3)
top_stock_n     = st.sidebar.slider("후보 종목 수", 4, 16, 8)
portfolio_mode  = st.sidebar.selectbox("포트폴리오 방식",
    ["Sector ETF Only","Stocks Only","Hybrid ETF + Stocks"], index=2)
max_single  = st.sidebar.slider("단일 자산 최대 비중", 0.10, 0.50, 0.25, 0.05)
max_sect_w  = st.sidebar.slider("단일 섹터 최대 비중", 0.25, 0.80, 0.45, 0.05)
n_sim       = st.sidebar.slider("MPT 시뮬레이션 수", 2000, 20000, 8000, 1000)
bt_start    = st.sidebar.selectbox("백테스트 시작",
    ["2019-01-01","2020-01-01","2021-01-01","2022-01-01","2023-01-01"], index=1)
bt_cost_bps = st.sidebar.slider("백테스트 거래비용 (bps)", 0, 30, 2, 1,
    help="0=비용없음, 2=미국ETF현실적, 5=보수적")

show_kr     = st.sidebar.checkbox("한국 시장 포함", value=False)
show_crypto = st.sidebar.checkbox("크립토 포함",   value=True)
show_comm   = st.sidebar.checkbox("원자재 포함",   value=True)

st.sidebar.markdown("### ⚖️ 섹터 스코어 가중치")
cfg = ScoreConfig(
    rs_1m   = st.sidebar.slider("1M 상대강도",  0.0, 0.6, 0.30, 0.05),
    rs_3m   = st.sidebar.slider("3M 상대강도",  0.0, 0.6, 0.20, 0.05),
    volume  = st.sidebar.slider("거래량 급증",  0.0, 0.4, 0.12, 0.02),
    trend   = st.sidebar.slider("추세",         0.0, 0.4, 0.16, 0.02),
    low_vol = st.sidebar.slider("저변동성",     0.0, 0.3, 0.10, 0.02),
    drawdown= st.sidebar.slider("낙폭 방어",    0.0, 0.3, 0.06, 0.02),
    macro_fit=st.sidebar.slider("매크로 적합도",0.0, 0.3, 0.06, 0.02),
)

st.sidebar.markdown("### 🤖 LLM API Key")
gemini_key    = st.sidebar.text_input("Gemini API Key",   type="password", value=ENV_GEMINI_KEY,    placeholder="AIzaSy...")
anthropic_key = st.sidebar.text_input("Anthropic API Key (선택)", type="password", value=ENV_ANTHROPIC_KEY, placeholder="sk-ant-...")
llm_used = st.session_state.get("llm_count", 0)
st.sidebar.caption(f"LLM 사용: {llm_used}/{LLM_LIMIT}회 (세션 기준)")

st.sidebar.markdown("### 📋 내 포트폴리오 입력")
my_portfolio_text = st.sidebar.text_area(
    "종목:비중% 형식",
    value="",
    placeholder="QQQ:30\nNVDA:15\nGLD:10\nTLT:15\nCASH:30",
    height=120,
    help="티커:비중%. 합계 자동 정규화.",
)


# ============================================================
# 12. 데이터 로딩
# ============================================================

st.title("🏛️ MACRO QUANT TERMINAL v15")
st.caption("fredapi 기반 · Beginner/Advanced 모드 · 추천 이유표 · LLM 사용량 제한 · 섹터 PER · 내 포트폴리오 비교 · 거래비용 백테스트 · 데이터 기준일 · 자동 새로고침")

# FRED 키 안내
if not ENV_FRED_KEY:
    st.warning("""
    ⚠️ **FRED API 키가 없습니다.**

    데이터를 불러오려면 Streamlit Cloud → Settings → Secrets에 아래를 입력하세요:
    ```toml
    FRED_API_KEY = "발급받은키"
    ```
    무료 발급: https://fred.stlouisfed.org/docs/api/api_key.html
    """)

all_us = ALL_SECTOR_ETFS + ["SPY","QQQ","RSP","TLT","HYG","GLD"] + ALL_STOCKS
comm_tickers   = list(COMMODITIES.values())
crypto_tickers = list(CRYPTO_TICKERS.values())
global_tickers = list(GLOBAL_INDICES.values())
kr_tickers     = list(KR_ETFS.values()) + [t for tl in KR_STOCKS.values() for t in tl]

with st.spinner("📡 데이터 수집 중... (FRED API Rate Limit 방지로 순차 로드, 최초 1~2분 소요)"):
    macro_df               = build_macro(ENV_FRED_KEY)
    sector_close, sect_vol = fetch_prices(all_us,         period=analysis_period)
    stock_close,  stk_vol  = fetch_prices(ALL_STOCKS,     period=analysis_period)
    global_close, _        = fetch_prices(global_tickers, period="1y")
    fear_greed             = fetch_fear_greed()
    buffett                = fetch_buffett(ENV_FRED_KEY)
    sector_pe              = fetch_sector_pe()

    comm_close   = fetch_prices(comm_tickers,   period="1y")[0] if show_comm   else pd.DataFrame()
    crypto_close = fetch_prices(crypto_tickers, period="1y")[0] if show_crypto else pd.DataFrame()
    crypto_dom   = fetch_crypto_dom()                            if show_crypto else {}
    kr_close     = fetch_prices(kr_tickers,     period="1y")[0] if show_kr     else pd.DataFrame()

vix = 20.0
if "^VIX" in global_close.columns:
    v = global_close["^VIX"].dropna()
    if not v.empty: vix = float(v.iloc[-1])

regime  = classify_regime(macro_df, vix)
rf_rate = float(macro_df["TB3MS"].dropna().iloc[-1]) if "TB3MS" in macro_df.columns and not macro_df["TB3MS"].dropna().empty else 0.04
cash_w  = calc_cash(vix, regime, geo_risk)

sector_rank = score_sectors(sector_close, sect_vol, regime, cfg, sector_pe)
sel_sectors = sector_rank.head(top_sector_n)["Sector"].tolist() if not sector_rank.empty else []
stock_rank  = score_stocks(stock_close, stk_vol, sel_sectors)
sel_etfs    = [SECTOR_ETFS[s] for s in sel_sectors if s in SECTOR_ETFS]
sel_stocks  = stock_rank.head(top_stock_n)["Ticker"].tolist() if not stock_rank.empty else []

if portfolio_mode == "Sector ETF Only":
    p_assets, p_price = sel_etfs, sector_close
    a2s = {e: ETF_TO_SECTOR.get(e,e) for e in p_assets}
elif portfolio_mode == "Stocks Only":
    p_assets, p_price = sel_stocks, stock_close
    a2s = STOCK_SECTOR_MAP
else:
    p_assets = sel_etfs + sel_stocks
    frames = []
    if sel_etfs:   frames.append(sector_close[[e for e in sel_etfs if e in sector_close.columns]])
    if sel_stocks: frames.append(stock_close[[t for t in sel_stocks if t in stock_close.columns]])
    p_price = pd.concat(frames, axis=1) if frames else pd.DataFrame()
    a2s = {**{e: ETF_TO_SECTOR.get(e,e) for e in sel_etfs}, **STOCK_SECTOR_MAP}

p_weights, mpt_cloud = optimize_mpt(p_price, p_assets, rf_rate, cash_w, max_single, max_sect_w, a2s, n_sim)
my_portfolio = parse_my_portfolio(my_portfolio_text) if my_portfolio_text.strip() else {}

sec_exp: Dict[str,float] = {}
for a, w in p_weights.items():
    if a == "CASH": continue
    s = a2s.get(a, ETF_TO_SECTOR.get(a,"Unknown"))
    sec_exp[s] = sec_exp.get(s,0) + w

risk_flags = []
if vix >= 30:   risk_flags.append(("🚨", f"VIX {vix:.1f} — 공포 수준 매우 높음"))
elif vix >= 25: risk_flags.append(("⚠️", f"VIX {vix:.1f} — 변동성 확대 구간"))
if regime.get("liquidity") == "Contracting": risk_flags.append(("⚠️","순유동성 수축 — 위험자산 부담"))
if regime.get("credit") == "Stress Rising":  risk_flags.append(("⚠️","하이일드 스프레드 확대 — 신용 리스크 상승"))
if regime.get("curve") == "Inverted":        risk_flags.append(("⚠️","10Y-2Y 금리 역전 — 경기 침체 선행 신호"))
if regime.get("inflation") == "High":        risk_flags.append(("⚠️","고인플레이션 — 성장주 밸류에이션 부담"))
if cash_w >= 0.30:                           risk_flags.append(("🛡️",f"동적 현금 {cash_w:.0%} — 방어 포지션 권고"))
if not risk_flags:                           risk_flags.append(("✅","주요 리스크 플래그 없음"))


# ============================================================
# 13. 탭 UI
# ============================================================

tabs = st.tabs([
    "① 초보자 가이드","② 글로벌 시장","③ 유동성 모니터","④ 매크로 레짐",
    "⑤ 섹터 로테이션","⑥ 한국 시장","⑦ 크립토","⑧ 원자재",
    "⑨ 포트폴리오","⑩ 내 포트폴리오","⑪ 백테스트","⑫ AI 분석","⑬ 데이터 품질",
])

# ───────── ① 초보자 가이드 ─────────
with tabs[0]:
    st.markdown("### 🧭 이 앱을 읽는 순서")
    st.info("**분석 흐름**: 글로벌 유동성 → 매크로 레짐 판단 → 강한 섹터 선택 → 종목 스캔 → 포트폴리오 구성")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        #### 1단계: 유동성 확인
        - **Fed 자산**: 연준 보유 자산 (클수록 우호)
        - **TGA**: 재무부 현금통장 (클수록 시장 부담)
        - **역레포(RRP)**: 연준에 맡긴 돈 (클수록 시장 부담)
        - **순유동성** = Fed자산 - TGA - RRP
        - **M2**: 시중 통화량
        """)
    with c2:
        st.markdown("""
        #### 2단계: 레짐 판단
        - **Risk-On**: 성장주·크립토 유리
        - **Risk-Off**: 금·달러·채권·방어주 유리
        - **VIX**: 20↓=안정, 25↑=주의, 30↑=위험
        - **버핏지표**: 1.5↑ = 역사적 고평가
        - **공포탐욕지수**: 25↓=공포, 75↑=탐욕
        """)
    with c3:
        st.markdown("""
        #### 3단계: 자산 선택
        - **섹터 ETF**: 11개 중 강한 섹터
        - **섹터 PER**: 밸류에이션 고평가 여부
        - **한국·크립토·원자재**: 자산군 다변화
        - **내 포트폴리오**: 현재 vs 추천 비교
        - **백테스트**: 거래비용 포함 성과 검증
        """)
    glossary = pd.DataFrame([
        {"용어": "순유동성",      "설명": "Fed자산 - TGA - RRP. 시장 유동성 간이 측정값"},
        {"용어": "VIX",          "설명": "공포지수. 높을수록 시장 불안"},
        {"용어": "하이일드 스프레드","설명": "위험 회사채 추가 금리. 오르면 신용 위험 증가"},
        {"용어": "수익률 커브 역전","설명": "단기>장기 금리. 경기침체 선행 신호"},
        {"용어": "NFCI",         "설명": "금융환경 지수. 양수=긴축, 음수=완화"},
        {"용어": "버핏지표",      "설명": "시총/GDP. 1.5↑=역사적 고평가"},
        {"용어": "M2",           "설명": "시중 통화량. 빠르게 늘면 유동성 확대"},
        {"용어": "PER",          "설명": "주가수익비율. 낮을수록 상대적 저평가"},
        {"용어": "BTC 도미넌스",  "설명": "크립토 중 BTC 비중. 낮으면 알트시즌"},
        {"용어": "공포탐욕지수",  "설명": "0=극도공포, 100=극도탐욕"},
        {"용어": "상대강도(RS)",  "설명": "SPY 대비 초과 수익률"},
        {"용어": "MPT",          "설명": "수익률-위험 균형 최적 비중 계산 이론"},
    ])
    st.dataframe(glossary, hide_index=True, use_container_width=True)
    st.warning("⚠️ 연구·학습용 도구입니다. 실제 투자 전 수수료·세금·환율·슬리피지·개별 기업 분석을 별도 확인하세요.")

# ───────── ② 글로벌 시장 ─────────
with tabs[1]:
    st.markdown("### 🌍 글로벌 시장 현황")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("공포탐욕지수", f"{fear_greed.get('score',50):.0f}", fear_greed.get("rating","N/A"))
    c2.metric("버핏지표", f"{buffett.get('ratio',0):.2f}x" if buffett else "N/A", buffett.get("valuation","") if buffett else "")
    c3.metric("VIX", f"{vix:.2f}", regime.get("vol",""))
    c4.metric("레짐", regime.get("master","Unknown"), f"점수 {regime.get('score',0):+.1f}")

    idx_rows = []
    for name, ticker in GLOBAL_INDICES.items():
        if ticker not in global_close.columns: continue
        s = global_close[ticker].dropna()
        if len(s) < 2: continue
        idx_rows.append({
            "자산": name, "현재가": float(s.iloc[-1]),
            "1일(%)":   (float(s.iloc[-1]/s.iloc[-2]-1)*100)  if len(s)>=2  else np.nan,
            "1개월(%)": (float(s.iloc[-1]/s.iloc[-21]-1)*100) if len(s)>=21 else np.nan,
            "3개월(%)": (float(s.iloc[-1]/s.iloc[-63]-1)*100) if len(s)>=63 else np.nan,
        })
    if idx_rows:
        st.dataframe(pd.DataFrame(idx_rows), hide_index=True, use_container_width=True,
            column_config={
                "현재가":   st.column_config.NumberColumn("현재가",   format="%.2f"),
                "1일(%)":   st.column_config.NumberColumn("1일(%)",   format="%+.2f%%"),
                "1개월(%)": st.column_config.NumberColumn("1개월(%)", format="%+.2f%%"),
                "3개월(%)": st.column_config.NumberColumn("3개월(%)", format="%+.2f%%"),
            })

    chart_data = {}
    for name, t in {"S&P500":"^GSPC","KOSPI":"^KS11","나스닥":"^IXIC","금":"GLD","TLT":"TLT"}.items():
        if t in global_close.columns:
            s = global_close[t].dropna()
            if not s.empty: chart_data[name] = s / s.iloc[0] * 100
    if chart_data:
        st.markdown("### 주요 지수 추이 (정규화 100)")
        st.line_chart(pd.DataFrame(chart_data).dropna(how="all"), height=360)

# ───────── ③ 유동성 모니터 ─────────
with tabs[2]:
    st.markdown("### 💧 글로벌 유동성 모니터")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Fed 자산",  f"${get_val(macro_df,'WALCL'):,.1f}B",     f"1M {get_delta(macro_df,'WALCL'):+.1f}B")
    c2.metric("TGA 잔고",  f"${get_val(macro_df,'WTREGEN'):,.1f}B",   f"1M {get_delta(macro_df,'WTREGEN'):+.1f}B")
    c3.metric("역레포",    f"${get_val(macro_df,'RRPONTSYD'):,.1f}B", f"1M {get_delta(macro_df,'RRPONTSYD'):+.1f}B")
    c4.metric("순유동성",  f"${get_val(macro_df,'Net_Liq'):,.1f}B",   f"1M {get_delta(macro_df,'Net_Liq'):+.1f}B")
    c5,c6,c7,c8 = st.columns(4)
    c5.metric("M2",      f"${get_val(macro_df,'M2SL'):,.0f}B",  f"YoY {get_val(macro_df,'M2_YoY'):+.1f}%")
    c6.metric("CPI YoY", f"{get_val(macro_df,'CPI_YoY'):.1f}%", regime.get("inflation",""))
    c7.metric("실업률",  f"{get_val(macro_df,'UNRATE'):.1f}%",  "")
    c8.metric("10Y 금리",f"{get_val(macro_df,'DGS10'):.2f}%",  f"{get_delta(macro_df,'DGS10'):+.2f}%")

    nl = get_val(macro_df,"Net_Liq"); nl_ma = get_val(macro_df,"Net_Liq_MA20")
    if pd.notna(nl) and pd.notna(nl_ma):
        if nl > nl_ma: st.success(f"순유동성 ${nl:,.1f}B → 20일 평균 ${nl_ma:,.1f}B 상회 (우호적)")
        else:          st.warning(f"순유동성 ${nl:,.1f}B → 20일 평균 ${nl_ma:,.1f}B 하회 (부담)")

    ca, cb = st.columns(2)
    with ca:
        cols = [c for c in ["Net_Liq","Net_Liq_MA20","Net_Liq_MA60"] if c in macro_df.columns]
        if cols:
            st.markdown("#### 순유동성 추이")
            st.line_chart(macro_df[cols].tail(730), height=280)
    with cb:
        if "M2SL" in macro_df.columns:
            st.markdown("#### M2 통화량")
            st.line_chart(macro_df[["M2SL"]].tail(730), height=280)

    cc, cd = st.columns(2)
    with cc:
        lc = [c for c in ["WALCL","WTREGEN","RRPONTSYD"] if c in macro_df.columns]
        if lc:
            st.markdown("#### Fed자산 vs TGA vs RRP")
            st.line_chart(macro_df[lc].tail(730), height=280)
    with cd:
        rc = [c for c in ["DGS10","DGS2","T10Y2Y"] if c in macro_df.columns]
        if rc:
            st.markdown("#### 금리 커브")
            st.line_chart(macro_df[rc].tail(365), height=280)

    if "Net_Liq" in macro_df.columns:
        st.markdown("### 날짜별 유동성 기록")
        col_map = {"WALCL":"Fed자산(B)","WTREGEN":"TGA(B)","RRPONTSYD":"RRP(B)",
                   "Net_Liq":"순유동성(B)","Net_Liq_1W":"1W변화","Net_Liq_1M":"1M변화",
                   "Net_Liq_MA20":"20일MA","Net_Liq_MA60":"60일MA"}
        avail = [c for c in col_map if c in macro_df.columns]
        hist = macro_df[avail].rename(columns={c: col_map[c] for c in avail})
        hist = hist.dropna(subset=["순유동성(B)"]).tail(90).sort_index(ascending=False).reset_index()
        first_col = hist.columns[0]
        hist = hist.rename(columns={first_col: "날짜"})
        hist["날짜"] = pd.to_datetime(hist["날짜"]).dt.date.astype(str)
        st.dataframe(hist, hide_index=True, use_container_width=True, height=320)
        st.download_button("⬇️ CSV 다운로드", data=hist.to_csv(index=False).encode("utf-8-sig"),
                           file_name="liquidity.csv", mime="text/csv")

# ───────── ④ 매크로 레짐 ─────────
with tabs[3]:
    st.markdown("### 🌐 매크로 레짐 대시보드")
    master = regime.get("master","Unknown")
    score  = regime.get("score",0)
    if "강한 Risk-On" in master: st.success(f"## {master}  (점수: {score:+.2f})")
    elif "Risk-On"    in master: st.success(f"## {master}  (점수: {score:+.2f})")
    elif "강한 Risk-Off" in master: st.error(f"## {master}  (점수: {score:+.2f})")
    elif "Risk-Off"   in master: st.warning(f"## {master}  (점수: {score:+.2f})")
    else:                        st.info(f"## {master}  (점수: {score:+.2f})")

    st.markdown(f"**선호 자산**: {regime.get('preferred','-')}  |  **주의 자산**: {regime.get('avoid','-')}")

    r1,r2,r3 = st.columns(3)
    r1.metric("유동성",  regime.get("liquidity","-"))
    r2.metric("변동성",  regime.get("vol","-"), f"VIX {vix:.1f}")
    r3.metric("신용",    regime.get("credit","-"), f"HY {regime.get('hy',0):.2f}%")
    r4,r5,r6 = st.columns(3)
    r4.metric("금융환경",regime.get("fin","-"),    f"NFCI {regime.get('nfci',0):.2f}")
    r5.metric("금리커브",regime.get("curve","-"),  f"10Y-2Y {regime.get('curve_val',0):.2f}%")
    r6.metric("인플레",  regime.get("inflation","-"), f"CPI {regime.get('cpi_yoy',0):.1f}%")
    r7,r8,_ = st.columns(3)
    r7.metric("금리방향",regime.get("rates","-"), f"10Y {regime.get('dgs10',0):.2f}%")
    r8.metric("M2 성장", regime.get("m2","-"),    f"{regime.get('m2_yoy',0):.1f}% YoY")

    if buffett:
        bf_r  = buffett.get("ratio", 0)
        bf_mc = buffett.get("mktcap_bn", 0)
        bf_gd = buffett.get("gdp_bn", 0)
        st.info(
            f"버핏지표: **{bf_r:.2f}x** → {buffett.get('valuation','')}"
            + (f"  |  시총 ${bf_mc:,.0f}B / GDP ${bf_gd:,.0f}B" if bf_mc and bf_gd else "")
        )
        if is_advanced:
            st.caption("""
            **버핏지표 계산 방식**: 미국 주식시장 시총 ÷ 명목 GDP
            - 시총: FRED NCBEILQ027S (비금융기업 주식 시가총액, 분기) → fallback: WILL5000IND → SPY 추정
            - GDP: FRED GDP (명목 GDP, 십억달러, 분기)
            - 0.85 이하=저평가, 0.85~1.1=적정, 1.1~1.4=고평가, 1.4 이상=크게 고평가
            """)

    st.markdown("### 리스크 플래그")
    for emoji, msg in risk_flags:
        if emoji == "🚨":  st.error(f"{emoji} {msg}")
        elif emoji == "✅": st.success(f"{emoji} {msg}")
        else:               st.warning(f"{emoji} {msg}")

    if is_advanced:
        ca, cb = st.columns(2)
        with ca:
            if "BAMLH0A0HYM2" in macro_df.columns:
                st.markdown("#### 하이일드 스프레드")
                st.line_chart(macro_df[["BAMLH0A0HYM2"]].tail(365), height=250)
        with cb:
            if "NFCI" in macro_df.columns:
                st.markdown("#### 금융환경지수 (NFCI)")
                st.line_chart(macro_df[["NFCI"]].tail(365), height=250)

# ───────── ⑤ 섹터 로테이션 ─────────
with tabs[4]:
    st.markdown("### 🔁 섹터 로테이션 분석")
    if sector_rank.empty:
        st.error("섹터 데이터를 불러오지 못했습니다.")
    else:
        st.success(f"선택 섹터: **{', '.join(sel_sectors)}**")
        disp = sector_rank.copy()
        disp["PER 평가"] = disp["PER"].apply(pe_text)
        show_cols = ["Sector","ETF","Score","1W","1M","3M","RS_1M","RS_3M","VolSurge","Vol20","MacroFit","PER","PER 평가"]
        st.dataframe(disp[[c for c in show_cols if c in disp.columns]], hide_index=True, use_container_width=True,
            column_config={
                "Score":    st.column_config.NumberColumn("Score",   format="%.3f"),
                "1W":       st.column_config.NumberColumn("1W",      format="%+.2%"),
                "1M":       st.column_config.NumberColumn("1M",      format="%+.2%"),
                "3M":       st.column_config.NumberColumn("3M",      format="%+.2%"),
                "RS_1M":    st.column_config.NumberColumn("RS 1M",   format="%+.2%"),
                "RS_3M":    st.column_config.NumberColumn("RS 3M",   format="%+.2%"),
                "VolSurge": st.column_config.NumberColumn("VolSurge",format="%.2fx"),
                "Vol20":    st.column_config.NumberColumn("Vol20",   format="%.2%"),
                "MacroFit": st.column_config.NumberColumn("MacroFit",format="%+.2f"),
                "PER":      st.column_config.NumberColumn("PER",     format="%.1f"),
            })
        if sector_pe:
            overvalued = [(s, pe) for s,pe in sector_pe.items() if pe > 30]
            for s, pe in overvalued[:3]:
                st.warning(f"⚠️ {s}: PER {pe:.1f} — 역사적 고평가 구간")
        st.bar_chart(sector_rank.set_index("Sector")["Score"], height=300)

        st.markdown("### 상위 섹터 내 종목")
        if not stock_rank.empty:
            st.dataframe(stock_rank[["Ticker","Sector","Score","Last","1W","1M","3M","VolSurge","Vol20","MDD126"]],
                hide_index=True, use_container_width=True,
                column_config={
                    "Score":    st.column_config.NumberColumn("Score",   format="%.3f"),
                    "Last":     st.column_config.NumberColumn("Last",    format="$%.2f"),
                    "1W":       st.column_config.NumberColumn("1W",      format="%+.2%"),
                    "1M":       st.column_config.NumberColumn("1M",      format="%+.2%"),
                    "3M":       st.column_config.NumberColumn("3M",      format="%+.2%"),
                    "VolSurge": st.column_config.NumberColumn("VolSurge",format="%.2fx"),
                    "Vol20":    st.column_config.NumberColumn("Vol20",   format="%.2%"),
                    "MDD126":   st.column_config.NumberColumn("MDD126",  format="%.2%"),
                })

# ───────── ⑥ 한국 시장 ─────────
with tabs[5]:
    st.markdown("### 🇰🇷 한국 시장 분석")
    if not show_kr:
        st.info("사이드바에서 '한국 시장 포함'을 체크해주세요.")
    else:
        if "KRW=X" in global_close.columns:
            s = global_close["KRW=X"].dropna()
            if not s.empty:
                krw = float(s.iloc[-1])
                krw_1m = float(s.iloc[-21]) if len(s)>=21 else krw
                st.metric("원/달러", f"{krw:,.1f}", f"1M {krw-krw_1m:+.1f}")
        kr_chart = {}
        for name, ticker in KR_ETFS.items():
            if ticker in kr_close.columns:
                s = kr_close[ticker].dropna()
                if not s.empty: kr_chart[name] = s / s.iloc[0] * 100
        if kr_chart:
            st.markdown("### 한국 ETF 추이")
            st.line_chart(pd.DataFrame(kr_chart).dropna(how="all"), height=300)
        kr_rows = []
        for sector, tickers in KR_STOCKS.items():
            for t in tickers:
                if t not in kr_close.columns: continue
                s = kr_close[t].dropna()
                if len(s) < 2: continue
                kr_rows.append({"섹터":sector,"종목":t,"현재가":float(s.iloc[-1]),
                    "1M(%)": pct_n(s,21)*100 if pd.notna(pct_n(s,21)) else np.nan,
                    "3M(%)": pct_n(s,63)*100 if pd.notna(pct_n(s,63)) else np.nan})
        if kr_rows:
            st.dataframe(pd.DataFrame(kr_rows), hide_index=True, use_container_width=True,
                column_config={"1M(%)":st.column_config.NumberColumn("1M(%)",format="%+.2f%%"),
                               "3M(%)":st.column_config.NumberColumn("3M(%)",format="%+.2f%%")})

# ───────── ⑦ 크립토 ─────────
with tabs[6]:
    st.markdown("### ₿ 크립토 시장")
    if not show_crypto:
        st.info("사이드바에서 '크립토 포함'을 체크해주세요.")
    else:
        if crypto_dom:
            c1,c2,c3,c4 = st.columns(4)
            btc_dom = crypto_dom.get("btc",0)
            c1.metric("BTC 도미넌스", f"{btc_dom:.1f}%",
                      "알트시즌 가능" if btc_dom<45 else ("BTC 강세" if btc_dom>55 else "중립"))
            c2.metric("ETH 도미넌스", f"{crypto_dom.get('eth',0):.1f}%","")
            total_mc = crypto_dom.get("total_mc",0)
            c3.metric("전체 시총", f"${total_mc/1e12:.2f}T" if total_mc>1e12 else f"${total_mc/1e9:.0f}B","")
            c4.metric("24h 변화", f"{crypto_dom.get('mc_chg_24h',0):+.2f}%","")
        crypto_rows = []
        for name, ticker in CRYPTO_TICKERS.items():
            if ticker not in crypto_close.columns: continue
            s = crypto_close[ticker].dropna()
            if len(s) < 2: continue
            crypto_rows.append({"코인":name,"현재가($)":float(s.iloc[-1]),
                "1일(%)": pct_n(s,1)*100 if len(s)>=2 else np.nan,
                "1M(%)":  pct_n(s,30)*100 if len(s)>=30 else np.nan,
                "3M(%)":  pct_n(s,90)*100 if len(s)>=90 else np.nan,
                "ATH 대비(%)": (float(s.iloc[-1])/float(s.max())-1)*100})
        if crypto_rows:
            st.dataframe(pd.DataFrame(crypto_rows), hide_index=True, use_container_width=True,
                column_config={
                    "현재가($)":   st.column_config.NumberColumn("현재가",   format="$%,.0f"),
                    "1일(%)":      st.column_config.NumberColumn("1일",      format="%+.2f%%"),
                    "1M(%)":       st.column_config.NumberColumn("1M",       format="%+.2f%%"),
                    "3M(%)":       st.column_config.NumberColumn("3M",       format="%+.2f%%"),
                    "ATH 대비(%)": st.column_config.NumberColumn("ATH 대비", format="%.1f%%"),
                })
        crypto_chart = {}
        for name, ticker in CRYPTO_TICKERS.items():
            if ticker in crypto_close.columns:
                s = crypto_close[ticker].dropna()
                if not s.empty: crypto_chart[name] = s / s.iloc[0] * 100
        if crypto_chart:
            st.line_chart(pd.DataFrame(crypto_chart).dropna(how="all"), height=300)
        m = regime.get("master","")
        if "Risk-On" in m:    st.success("Risk-On 환경: 크립토는 상대적으로 유리할 수 있습니다.")
        elif "Risk-Off" in m: st.warning("Risk-Off 환경: 크립토는 하락 압력을 받는 경향이 있습니다.")
        else:                 st.info("Mixed 환경: BTC 도미넌스와 유동성을 함께 확인하세요.")

# ───────── ⑧ 원자재 ─────────
with tabs[7]:
    st.markdown("### 🛢️ 원자재")
    if not show_comm:
        st.info("사이드바에서 '원자재 포함'을 체크해주세요.")
    else:
        comm_rows = []
        for name, ticker in COMMODITIES.items():
            if ticker not in comm_close.columns: continue
            s = comm_close[ticker].dropna()
            if len(s) < 2: continue
            comm_rows.append({"원자재":name,"현재가":float(s.iloc[-1]),
                "1M(%)": pct_n(s,21)*100 if pd.notna(pct_n(s,21)) else np.nan,
                "3M(%)": pct_n(s,63)*100 if pd.notna(pct_n(s,63)) else np.nan,
                "6M(%)": pct_n(s,126)*100 if pd.notna(pct_n(s,126)) else np.nan})
        if comm_rows:
            st.dataframe(pd.DataFrame(comm_rows), hide_index=True, use_container_width=True,
                column_config={
                    "현재가": st.column_config.NumberColumn("현재가", format="%.2f"),
                    "1M(%)": st.column_config.NumberColumn("1M(%)", format="%+.2f%%"),
                    "3M(%)": st.column_config.NumberColumn("3M(%)", format="%+.2f%%"),
                    "6M(%)": st.column_config.NumberColumn("6M(%)", format="%+.2f%%"),
                })
            norm = {}
            for name, ticker in COMMODITIES.items():
                if ticker in comm_close.columns:
                    s = comm_close[ticker].dropna()
                    if not s.empty: norm[name] = s / s.iloc[0] * 100
            if norm:
                st.line_chart(pd.DataFrame(norm).dropna(how="all"), height=300)
        infl = regime.get("inflation","")
        if infl == "High":          st.warning("고인플레이션: 금·원유·원자재가 헤지 수단으로 주목받을 수 있습니다.")
        elif regime.get("rates") == "Rising": st.info("금리 상승: 금은 기회비용 증가로 부담. 에너지·구리는 경기에 따라 다름.")
        else:                       st.info("원자재는 레짐 변화의 선행 신호로 관찰 가치가 있습니다.")

# ───────── ⑨ 포트폴리오 ─────────
with tabs[8]:
    st.markdown("### 🎯 포트폴리오 빌더")
    st.info(f"레짐: **{regime.get('master')}** | 동적 현금: **{cash_w:.0%}** | 방식: **{portfolio_mode}**")
    c1,c2,c3 = st.columns([4,3,3])
    with c1:
        w_df = pd.DataFrame([
            {"자산":a,"비중":w,"섹터":a2s.get(a,"Cash") if a!="CASH" else "Cash"}
            for a,w in sorted(p_weights.items(), key=lambda x:x[1], reverse=True)
        ])
        st.dataframe(w_df, hide_index=True, use_container_width=True,
            column_config={"비중":st.column_config.NumberColumn("비중",format="%.2%")})
    with c2:
        st.markdown("**자산 비중**")
        fig, ax = plt.subplots(figsize=(4,4)); fig.patch.set_alpha(0)
        ax.pie(p_weights.values(), labels=p_weights.keys(), autopct="%1.1f%%", textprops={"fontsize":8})
        ax.add_artist(plt.Circle((0,0),0.55,fc="#0b0e14"))
        st.pyplot(fig)
    with c3:
        st.markdown("**섹터 익스포저**")
        if sec_exp:
            fig2, ax2 = plt.subplots(figsize=(4,4)); fig2.patch.set_alpha(0)
            ax2.pie(sec_exp.values(), labels=sec_exp.keys(), autopct="%1.1f%%", textprops={"fontsize":8})
            ax2.add_artist(plt.Circle((0,0),0.55,fc="#0b0e14"))
            st.pyplot(fig2)
    if is_advanced and not mpt_cloud.empty:
        st.markdown("### MPT 시뮬레이션")
        fig_mpt = px.scatter(mpt_cloud, x="Volatility", y="Return", color="Sharpe",
            color_continuous_scale="RdYlGn",
            labels={"Volatility":"연율화 변동성","Return":"연율화 수익률"})
        fig_mpt.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=350)
        st.plotly_chart(fig_mpt, use_container_width=True)
    st.markdown("### 리스크 플래그")
    for emoji, msg in risk_flags:
        if emoji == "🚨":  st.error(f"{emoji} {msg}")
        elif emoji == "✅": st.success(f"{emoji} {msg}")
        else:               st.warning(f"{emoji} {msg}")

# ───────── ⑩ 내 포트폴리오 ─────────
with tabs[9]:
    st.markdown("### 📋 내 포트폴리오 vs 추천 포트폴리오")
    if not my_portfolio:
        st.info("사이드바 '내 포트폴리오 입력'에 보유 종목을 입력해주세요.\n\n예시:\n```\nQQQ:30\nNVDA:15\nGLD:10\nTLT:15\nCASH:30\n```")
    else:
        mc1, mc2 = st.columns(2)
        with mc1:
            st.markdown("#### 내 포트폴리오")
            my_df = pd.DataFrame([
                {"자산":k,"비중":v,"섹터":a2s.get(k,ETF_TO_SECTOR.get(k,"기타")) if k!="CASH" else "현금"}
                for k,v in sorted(my_portfolio.items(), key=lambda x:x[1], reverse=True)
            ])
            st.dataframe(my_df, hide_index=True, use_container_width=True,
                column_config={"비중":st.column_config.NumberColumn("비중",format="%.1%")})
        with mc2:
            fig_my, ax_my = plt.subplots(figsize=(4,4)); fig_my.patch.set_alpha(0)
            ax_my.pie(my_portfolio.values(), labels=my_portfolio.keys(), autopct="%1.1f%%", textprops={"fontsize":8})
            ax_my.add_artist(plt.Circle((0,0),0.55,fc="#0b0e14"))
            st.pyplot(fig_my)

        st.markdown("#### 비교표 — 조정 방향")
        cmp_df = compare_portfolios(my_portfolio, p_weights, a2s)
        st.dataframe(cmp_df, hide_index=True, use_container_width=True,
            column_config={
                "내 비중":   st.column_config.NumberColumn("내 비중",   format="%.1%"),
                "추천 비중": st.column_config.NumberColumn("추천 비중", format="%.1%"),
                "차이":      st.column_config.NumberColumn("차이",      format="%+.1%"),
            })

        st.markdown("#### 리밸런싱 금액 계산")
        total_amt = st.number_input("총 포트폴리오 금액 (원 또는 달러)", min_value=100, value=10_000_000, step=1_000_000)
        rebal_rows = []
        for _, row in cmp_df.iterrows():
            if abs(row["차이"]) < 0.01: continue
            rebal_rows.append({"자산":row["자산"],"방향":"매수" if row["차이"]>0 else "매도",
                               "현재":f"{row['내 비중']:.1%}","목표":f"{row['추천 비중']:.1%}",
                               "금액":row["차이"]*total_amt})
        if rebal_rows:
            st.dataframe(pd.DataFrame(rebal_rows), hide_index=True, use_container_width=True,
                column_config={"금액":st.column_config.NumberColumn("금액",format="%+,.0f")})
        st.warning("⚠️ 매매 금액은 참고용입니다. 실제 거래 시 수수료·세금·환율·최소 거래 단위를 반드시 확인하세요.")

# ───────── ⑪ 백테스트 ─────────
with tabs[10]:
    st.markdown("### 📈 섹터 로테이션 백테스트")
    st.caption(f"매 월말 리밸런싱. 거래비용: {bt_cost_bps}bps 설정")

    bt_nc, _ = run_backtest(sector_close, start=bt_start, top_n=top_sector_n, cost_bps=0)
    bt_c, bt_log = run_backtest(sector_close, start=bt_start, top_n=top_sector_n, cost_bps=bt_cost_bps)

    if bt_c.empty:
        st.warning("데이터 부족. 분석 기간을 2y 또는 5y로 늘려보세요.")
    else:
        if not bt_nc.empty:
            compare_bt = pd.concat([
                bt_nc[["섹터로테이션"]].rename(columns={"섹터로테이션":"비용없음"}),
                bt_c[["섹터로테이션","SPY"]].rename(columns={"섹터로테이션":f"비용{bt_cost_bps}bps"}),
            ], axis=1).dropna(how="all")
            st.line_chart(compare_bt, height=360)
        else:
            st.line_chart(bt_c, height=360)

        strat_r = bt_c["섹터로테이션"].pct_change().dropna()
        spy_r   = bt_c["SPY"].pct_change().dropna()
        stats_list = [perf_stats(strat_r, rf_rate), perf_stats(spy_r, rf_rate)]
        idx_list   = [f"섹터로테이션({bt_cost_bps}bps)", "SPY(벤치마크)"]
        if not bt_nc.empty:
            nc_r = bt_nc["섹터로테이션"].pct_change().dropna()
            stats_list.insert(0, perf_stats(nc_r, rf_rate))
            idx_list.insert(0, "섹터로테이션(비용없음)")
        stats = pd.DataFrame(stats_list, index=idx_list)
        st.dataframe(stats, use_container_width=True,
            column_config={
                "CAGR":    st.column_config.NumberColumn("CAGR",    format="%.2%"),
                "Vol":     st.column_config.NumberColumn("변동성",  format="%.2%"),
                "Sharpe":  st.column_config.NumberColumn("Sharpe",  format="%.2f"),
                "Sortino": st.column_config.NumberColumn("Sortino", format="%.2f"),
                "MDD":     st.column_config.NumberColumn("MDD",     format="%.2%"),
                "WinRate": st.column_config.NumberColumn("승률",    format="%.2%"),
            })
        if not bt_nc.empty:
            nc_cagr = perf_stats(bt_nc["섹터로테이션"].pct_change().dropna(), rf_rate)["CAGR"]
            c_cagr  = perf_stats(strat_r, rf_rate)["CAGR"]
            if pd.notna(nc_cagr) and pd.notna(c_cagr):
                drag = nc_cagr - c_cagr
                st.info(f"💡 거래비용 {bt_cost_bps}bps 적용 시 연간 CAGR 약 **{drag:.2%}** 감소.")
        with st.expander("리밸런싱 로그"):
            st.dataframe(bt_log.tail(24), hide_index=True, use_container_width=True)

# ───────── ⑫ AI 분석 ─────────
with tabs[11]:
    st.markdown("### 🤖 AI 분석")
    st.caption(f"LLM 사용: {st.session_state.get('llm_count',0)}/{LLM_LIMIT}회. AI는 매수/매도를 단정하지 않습니다.")

    if not (gemini_key or anthropic_key):
        st.warning("사이드바에서 Gemini 또는 Anthropic API Key를 입력하세요.")

    analysis_type = st.radio("분석 유형", ["리스크 분석","섹터/종목 추천 이유","자유 질문"], horizontal=True)
    user_q = ""
    if analysis_type == "자유 질문":
        user_q = st.text_area("질문 입력", height=100,
            placeholder="예: 지금 KOSPI와 나스닥 중 어느 쪽이 더 유리한 환경인가요?")

    if st.button("🔎 AI 분석 실행", disabled=not(gemini_key or anthropic_key)):
        macro_sum = {"레짐":regime.get("master"),"점수":regime.get("score"),
                     "유동성":regime.get("liquidity"),"VIX":vix,
                     "인플레":regime.get("inflation"),"금리커브":regime.get("curve"),
                     "선호자산":regime.get("preferred"),"주의자산":regime.get("avoid")}
        top_s = sector_rank.head(5)[["Sector","Score","RS_1M","RS_3M","MacroFit","PER"]].to_string(index=False) if not sector_rank.empty else "N/A"
        top_t = stock_rank.head(8)[["Ticker","Sector","Score","1M","3M"]].to_string(index=False)               if not stock_rank.empty else "N/A"
        flags = "\n".join(f"{e} {m}" for e,m in risk_flags)

        if analysis_type == "리스크 분석":
            prompt = f"""
[매크로 레짐] {macro_sum}
[상위 섹터] {top_s}
[포트폴리오 비중] {p_weights}
[리스크 플래그] {flags}
[공포탐욕지수] {fear_greed}

출력 형식:
1. 현재 레짐 요약 (초보자용)
2. 가장 큰 리스크 요인 3가지
3. 이 환경에서 주의할 자산
4. 모델 한계와 추가 확인 사항
"""
            system = SYSTEM_RISK
        elif analysis_type == "섹터/종목 추천 이유":
            prompt = f"""
[매크로 레짐] {macro_sum}
[선택 섹터] {', '.join(sel_sectors)}
[상위 섹터 랭킹] {top_s}
[상위 종목 랭킹] {top_t}
[포트폴리오] {p_weights}

출력 형식:
1. 한 줄 결론
2. 섹터 선택 이유 (지표 근거)
3. 반대 시나리오
4. 초보자 체크리스트
5. 모델 한계
"""
            system = SYSTEM_ANALYST
        else:
            prompt = f"""
[현재 시장 환경]
레짐: {regime.get('master')} | VIX: {vix:.1f} | 인플레: {regime.get('inflation')} | 금리커브: {regime.get('curve')}

[사용자 질문] {user_q}
"""
            system = SYSTEM_ANALYST

        with st.spinner("AI 분석 중..."):
            result_text = call_llm(system, prompt, gemini_key, anthropic_key)
            st.info(result_text)

    st.warning("⚠️ AI 해석은 참고용입니다. 실제 투자 전 반드시 추가 확인이 필요합니다.")

# ───────── ⑬ 데이터 품질 ─────────
with tabs[12]:
    st.markdown("### 🔍 데이터 기준일 & 품질 점검")

    freshness = check_data_freshness(macro_df)
    if freshness:
        fresh_df = pd.DataFrame(freshness)
        stale = sum(1 for r in freshness if r["상태"] != "✅ 정상")
        if stale > 0:
            st.warning(f"⚠️ {stale}개 지표의 데이터가 발표 주기 기준보다 지연됐습니다.")
        else:
            st.success("✅ 모든 주요 지표가 발표 주기 기준 내에 있습니다.")
        st.dataframe(fresh_df, hide_index=True, use_container_width=True,
            column_config={"지연(일)":st.column_config.NumberColumn("지연(일)",format="%d")})
    else:
        st.warning("데이터 기준일을 확인할 수 없습니다.")

    st.markdown("### FRED 데이터 소스 공식 링크")
    link_rows = [{"지표":meta["label"],"ID":sid,"공식링크":f"https://fred.stlouisfed.org/series/{sid}"}
                 for sid, meta in FRED_META.items()]
    st.dataframe(pd.DataFrame(link_rows), hide_index=True, use_container_width=True,
        column_config={"공식링크":st.column_config.LinkColumn("공식링크")})

    if sector_pe:
        st.markdown("### 섹터 PER 수집 현황")
        pe_status = pd.DataFrame([
            {"섹터":s,"ETF":SECTOR_ETFS.get(s,""),"Trailing PER":pe,"평가":pe_text(pe)}
            for s,pe in sorted(sector_pe.items(), key=lambda x:x[1])
        ])
        st.dataframe(pe_status, hide_index=True, use_container_width=True,
            column_config={"Trailing PER":st.column_config.NumberColumn("Trailing PER",format="%.1f")})
        st.caption(f"PER 수집: {len(sector_pe)}/{len(SECTOR_ETFS)}개 섹터. yfinance 제공값, 지연 데이터입니다.")

    st.markdown("### fredapi 사용 상태")
    if HAS_FREDAPI and ENV_FRED_KEY:
        st.success("✅ fredapi + FRED_API_KEY 사용 중 (안정적)")
    elif HAS_FREDAPI and not ENV_FRED_KEY:
        st.warning("⚠️ fredapi 설치됨, 하지만 FRED_API_KEY 없음 → CSV fallback 사용 중 (불안정)")
    else:
        st.warning("⚠️ fredapi 미설치 → CSV fallback 사용 중 (불안정)\nrequirements.txt에 fredapi 추가 후 재배포 권장")

    if is_advanced:
        st.markdown("### 지표 계산 공식 & 검증")

        st.markdown("#### 순유동성")
        st.code("Net Liquidity (B USD) = WALCL/1000 - WTREGEN/1000 - RRPONTSYD", language="text")
        st.caption("WALCL·WTREGEN: 주간(수요일) 데이터, 단위 백만→십억 변환 | RRPONTSYD: 일간, 이미 십억달러 단위")
        st.warning("순유동성은 공식 통계 명칭이 아닙니다. 파생 계산값이며 방향성 참고 용도입니다.")

        st.markdown("#### 버핏지표")
        st.code("버핏지표 = 미국 주식시장 시총(B USD) / 명목 GDP(B USD)", language="text")
        st.caption("""
        - 시총 우선순위: NCBEILQ027S(분기) → WILL5000IND(일간) → SPY 시총 추정
        - GDP: FRED GDP (명목 GDP, 분기)
        - 이전 버전 오류: ^W5000 포인트값 * 1e6 방식은 단위가 맞지 않아 수정됨
        """)

        st.markdown("#### M2 YoY")
        st.code("M2_YoY = M2SL.pct_change(12) * 100  # 월간 데이터 → 12개월 = 전년동월비", language="text")
        st.caption("이전 버전 오류: pct_change(252)는 일간 데이터 기준 — 월간인 M2SL에는 pct_change(12)가 정확")

        st.markdown("#### 순유동성 1W/1M 변화")
        st.code("Net_Liq_1W = diff(1)  # 주간 1행=1주\nNet_Liq_1M = diff(4)  # 주간 4행=1개월", language="text")
        st.caption("이전 버전 오류: diff(5)/diff(21)은 일간 데이터 기준 — 주간 데이터에는 diff(1)/diff(4)가 정확")

st.caption("Disclaimer: 연구·학습용 프로토타입입니다. 투자 조언이 아닙니다.")
