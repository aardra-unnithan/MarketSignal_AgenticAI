
import dash
from dash import dcc, html, Input, Output, State, callback_context
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go
import pandas as pd
import json
import time
import anthropic
from fredapi import Fred
from datetime import datetime, timedelta
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from bs4 import BeautifulSoup
import numpy as np

# ── API KEYS ──────────────────────────────────────────────────────────────────
import os
from dotenv import load_dotenv
load_dotenv()

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
FRED_API_KEY      = os.environ.get("FRED_API_KEY")
GOOGLE_API_KEY    = os.environ.get("GOOGLE_API_KEY")
CENSUS_API_KEY    = os.environ.get("CENSUS_API_KEY")

missing = [name for name, val in [
    ("ANTHROPIC_API_KEY", ANTHROPIC_API_KEY),
    ("FRED_API_KEY", FRED_API_KEY),
    ("GOOGLE_API_KEY", GOOGLE_API_KEY),
    ("CENSUS_API_KEY", CENSUS_API_KEY),
] if not val]
if missing:
    raise RuntimeError(
        f"Missing required environment variable(s): {', '.join(missing)}. "
        "Set them in a local .env file (see .env.example) — never hardcode them in source."
    )

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
CLAUDE_MODEL = "claude-sonnet-5"
fred   = Fred(api_key=FRED_API_KEY)

# ── DATA FUNCTIONS (MUSCLE) ───────────────────────────────────────────────────
def fetch_macro_data(overrides={}):
    end_date   = datetime.today()
    start_date = end_date - timedelta(days=365*3)
    series = {
        "fed_funds_rate":     "FEDFUNDS",
        "retail_sales":       "RETAILSMNSA",
        "cpi":                "CPIAUCSL",
        "gdp":                "GDP",
        "pce":                "DPCERE1Q156NBEA",
        "prime_rate":         "DPRIME",
        "disposable_income":  "DSPIC96",
    }
    raw = {}
    for name, code in series.items():
        try:
            raw[name] = fred.get_series(code, start_date, end_date)
        except:
            pass

    # Safe defaults if FRED API unavailable
    macro = {}
    try:
        macro["fed_funds_rate"] = round(raw["fed_funds_rate"].dropna().iloc[-1], 2)
    except: macro["fed_funds_rate"] = 3.64
    try:
        macro["prime_rate"] = round(raw["prime_rate"].dropna().iloc[-1], 2)
    except: macro["prime_rate"] = 6.75
    try:
        cpi = raw["cpi"].dropna()
        macro["inflation_rate"] = round(((cpi.iloc[-1] - cpi.iloc[-13]) / cpi.iloc[-13]) * 100, 2)
    except: macro["inflation_rate"] = 2.83
    try:
        gdp = raw["gdp"].dropna()
        macro["gdp_growth"] = round(((gdp.iloc[-1] - gdp.iloc[-5]) / gdp.iloc[-5]) * 100, 2)
    except: macro["gdp_growth"] = 5.58
    try:
        retail = raw["retail_sales"].dropna()
        macro["retail_sales_growth"] = round(((retail.iloc[-1] - retail.iloc[-13]) / retail.iloc[-13]) * 100, 2)
    except: macro["retail_sales_growth"] = 3.72
    try:
        pce = raw["pce"].dropna()
        macro["pce_growth"] = round(((pce.iloc[-1] - pce.iloc[-5]) / pce.iloc[-5]) * 100, 2)
    except: macro["pce_growth"] = -0.44
    try:
        income = raw["disposable_income"].dropna()
        macro["income_growth"] = round(((income.iloc[-1] - income.iloc[-5]) / income.iloc[-5]) * 100, 2)
    except: macro["income_growth"] = 0.03
    # Apply scenario overrides
    for k, v in overrides.items():
        if k in macro:
            macro[k] = v
    return macro

def fetch_fomc_statement():
    url     = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
    headers = {"User-Agent": "Mozilla/5.0"}
    resp    = requests.get(url, headers=headers)
    soup    = BeautifulSoup(resp.content, "html.parser")
    link    = None
    for a in soup.find_all("a", href=True):
        if "newsevents/pressreleases/monetary" in a["href"] and "a.htm" in a["href"]:
            link = "https://www.federalreserve.gov" + a["href"]
            break
    if not link:
        return "Fed statement unavailable.", "unknown"
    r    = requests.get(link, headers=headers)
    s    = BeautifulSoup(r.content, "html.parser")
    body = s.find("div", {"id": "article"}) or s.find("div", {"class": "col-xs-12 col-sm-8 col-md-8"})
    text = body.get_text(separator=" ", strip=True) if body else "Could not extract."
    return text, link

def analyze_fomc_tone(fomc_text):
    prompt = f"""
    You are an expert monetary economist. Analyze this FOMC statement and return JSON only.
    No markdown, no backticks, just raw JSON.
    FOMC Statement: {fomc_text}
    Return:
    {{
        "tone": "hawkish" or "neutral" or "dovish",
        "tone_score": number from -1.0 to 1.0,
        "key_signals": [3 phrases from the statement],
        "policy_direction": "tightening" or "holding" or "easing",
        "inflation_concern": "high" or "moderate" or "low",
        "growth_concern": "high" or "moderate" or "low",
        "one_line_summary": "one sentence plain English summary"
    }}
    """
    r = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=800,
        system="Monetary policy expert. JSON only.",
        messages=[{"role": "user", "content": prompt}],
    )
    raw = "".join(block.text for block in r.content if block.type == "text").strip().replace("```json","").replace("```","").strip()
    print(f"Claude raw response length: {len(raw)}")
    print(f"Claude first 300 chars: {raw[:300]}")
    print(f"Claude last 200 chars: {raw[-200:]}")
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        print(f"JSON parse error. Raw response length: {len(raw)}")
        print(f"Last 200 chars: {raw[-200:]}")
        # Claude returned malformed JSON — attempt repair by truncating at last valid closing brace
        try:
            last_brace = raw.rfind("}")
            if last_brace > 0:
                return json.loads(raw[:last_brace+1])
        except:
            pass
        # Final fallback — return safe defaults
        return {
            "expansion_score": 50, "local_market_score": 50, "pricing_power_score": 50,
            "expansion_recommendation": "Hold", "recommended_price_tier": "Mid-Range",
            "confidence_level": "Low", "confidence_reason": "Analysis temporarily unavailable.",
            "macro_risk": "Moderate", "reasoning_trace": ["Analysis unavailable — please retry."],
            "key_opportunities": ["Retry analysis for full results."],
            "key_risks": ["Retry analysis for full results."],
            "swot": {"strengths": ["Retry for full analysis."], "weaknesses": ["Retry for full analysis."],
                     "opportunities": ["Retry for full analysis."], "threats": ["Retry for full analysis."]},
            "market_entry_window": "Cautious", "market_entry_score": 50,
            "market_entry_reason": "Analysis temporarily unavailable — please retry.",
            "advisory_memo": "Analysis temporarily unavailable. Please click Run Analysis again.",
            "pricing_rationale": "Please retry the analysis."
        }


# ── ZIP TO FED DISTRICT MAPPING ───────────────────────────────────────────────
# Maps ZIP code prefix ranges to Federal Reserve Districts
# Source: Federal Reserve Bank district boundaries
FED_DISTRICT_MAP = {
    "boston":       {"states": ["ME","NH","VT","MA","RI","CT"], "beige_url": "https://www.federalreserve.gov/monetarypolicy/beigebook/beigebook_boston.htm"},
    "new_york":     {"states": ["NY","NJ","PR","VI"], "beige_url": "https://www.federalreserve.gov/monetarypolicy/beigebook/beigebook_newyork.htm"},
    "philadelphia": {"states": ["PA","DE","MD"], "beige_url": "https://www.federalreserve.gov/monetarypolicy/beigebook/beigebook_philadelphia.htm"},
    "cleveland":    {"states": ["OH","KY","WV","PA"], "beige_url": "https://www.federalreserve.gov/monetarypolicy/beigebook/beigebook_cleveland.htm"},
    "richmond":     {"states": ["VA","NC","SC","DC","MD"], "beige_url": "https://www.federalreserve.gov/monetarypolicy/beigebook/beigebook_richmond.htm"},
    "atlanta":      {"states": ["GA","FL","AL","TN","MS","LA"], "beige_url": "https://www.federalreserve.gov/monetarypolicy/beigebook/beigebook_atlanta.htm"},
    "chicago":      {"states": ["IL","IN","MI","WI","IA"], "beige_url": "https://www.federalreserve.gov/monetarypolicy/beigebook/beigebook_chicago.htm"},
    "st_louis":     {"states": ["MO","AR","IL","IN","KY","MS","TN"], "beige_url": "https://www.federalreserve.gov/monetarypolicy/beigebook/beigebook_stlouis.htm"},
    "minneapolis":  {"states": ["MN","MT","ND","SD","WI","MI"], "beige_url": "https://www.federalreserve.gov/monetarypolicy/beigebook/beigebook_minneapolis.htm"},
    "kansas_city":  {"states": ["KS","CO","NE","OK","WY","MO","NM"], "beige_url": "https://www.federalreserve.gov/monetarypolicy/beigebook/beigebook_kansascity.htm"},
    "dallas":       {"states": ["TX","LA","NM"], "beige_url": "https://www.federalreserve.gov/monetarypolicy/beigebook/beigebook_dallas.htm"},
    "san_francisco":{"states": ["CA","AK","HI","OR","WA","NV","ID","UT","AZ"], "beige_url": "https://www.federalreserve.gov/monetarypolicy/beigebook/beigebook_sanfrancisco.htm"},
}

def get_fed_district(state_abbr):
    """Map state abbreviation to Fed District."""
    for district, info in FED_DISTRICT_MAP.items():
        if state_abbr.upper() in info["states"]:
            return district, info["beige_url"]
    return "national", None

def get_state_from_zip(zip_code):
    """Get state abbreviation from ZIP code using Census geocoding."""
    try:
        url = "https://api.census.gov/data/2023/acs/acs5"
        params = {"get": "B01003_001E", "for": f"zip code tabulation area:{zip_code}", "key": CENSUS_API_KEY}
        r = requests.get(url, params=params)
        # Census doesn't return state directly but we can get it from zip lookup
        # Use a simple ZIP prefix approach as backup
        geo_r = requests.get(
            "https://maps.googleapis.com/maps/api/geocode/json",
            params={"address": zip_code, "key": GOOGLE_API_KEY}
        ).json()
        if geo_r.get("results"):
            for component in geo_r["results"][0]["address_components"]:
                if "administrative_area_level_1" in component["types"]:
                    return component["short_name"]
    except:
        pass
    return "OH"  # default fallback

# ── ZILLOW RENT DATA ──────────────────────────────────────────────────────────
import pandas as pd
_zillow_df = None

def load_zillow_data():
    """Load Zillow ZORI CSV once and cache it."""
    global _zillow_df
    if _zillow_df is None:
        try:
            _zillow_df = pd.read_csv(
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "zillow_rent.csv"),
                dtype={"RegionName": str}
            )
        except:
            _zillow_df = pd.DataFrame()
    return _zillow_df

def fetch_zillow_rent(zip_code):
    """
    Fetch latest median rent for a ZIP code from Zillow ZORI.
    Falls back to BLS metro-level or national median if ZIP not found.
    Source: Zillow Observed Rent Index (ZORI), updated monthly.
    """
    try:
        df = load_zillow_data()
        if df.empty:
            return None, None, "unavailable"
        
        row = df[df["RegionName"] == str(zip_code)]
        if row.empty:
            return None, None, "not_found"
        
        date_cols = [c for c in df.columns if c.startswith("20")]
        values = row[date_cols].iloc[0].dropna()
        
        if values.empty:
            return None, None, "no_data"
        
        latest_rent = round(values.iloc[-1], 2)
        latest_date = values.index[-1]
        
        # Calculate rent trend (6 month change)
        if len(values) >= 6:
            rent_6mo_ago = values.iloc[-6]
            rent_trend_pct = round(((latest_rent - rent_6mo_ago) / rent_6mo_ago) * 100, 1)
        else:
            rent_trend_pct = 0
            
        return latest_rent, rent_trend_pct, latest_date
        
    except Exception as e:
        print(f"Zillow rent error: {e}")
        return None, None, "error"

# ── BLS COUNTY UNEMPLOYMENT ───────────────────────────────────────────────────
def fetch_bls_unemployment(zip_code, state_abbr="OH"):
    """
    Fetch monthly county-level unemployment from BLS LAUS.
    Much more current than Census ACS (monthly vs 2-year lag).
    Source: BLS Local Area Unemployment Statistics (LAUS).
    """
    try:
        # Map ZIP to county FIPS using Google Geocoding
        geo_r = requests.get(
            "https://maps.googleapis.com/maps/api/geocode/json",
            params={"address": zip_code, "key": GOOGLE_API_KEY}
        ).json()
        
        county_fips = None
        state_fips = None
        county_name = None
        
        if geo_r.get("results"):
            for component in geo_r["results"][0]["address_components"]:
                if "administrative_area_level_2" in component["types"]:
                    county_name = component["long_name"]
        
        # Use FIPS lookup via Census Geocoder
        fips_r = requests.get(
            "https://geocoding.geo.census.gov/geocoder/geographies/address",
            params={
                "benchmark": "Public_AR_Current",
                "vintage": "Current_Current",
                "layers": "Counties",
                "format": "json",
                "address": f"{zip_code}",
                "city": "",
                "state": state_abbr,
            }
        ).json()
        
        matches = fips_r.get("result", {}).get("addressMatches", [])
        if matches:
            geo = matches[0].get("geographies", {}).get("Counties", [])
            if geo:
                state_fips = geo[0]["STATE"]
                county_fips = geo[0]["COUNTY"]
        
        if not state_fips or not county_fips:
            return None, None
        
        # Build BLS LAUS series ID: LAUCN{state_fips}{county_fips}0000000003
        series_id = f"LAUCN{state_fips}{county_fips}0000000003"
        
        from datetime import datetime
        current_year = str(datetime.now().year)
        prev_year = str(datetime.now().year - 1)
        
        bls_r = requests.post(
            "https://api.bls.gov/publicAPI/v2/timeseries/data/",
            json={"seriesid": [series_id], "startyear": prev_year, "endyear": current_year}
        ).json()
        
        series = bls_r.get("Results", {}).get("series", [])
        if series and series[0].get("data"):
            latest = series[0]["data"][0]
            unemp_rate = float(latest["value"])
            period = f"{latest['periodName']} {latest['year']}"
            return unemp_rate, period
            
    except Exception as e:
        print(f"BLS unemployment error: {e}")
    
    return None, None

# ── BEIGE BOOK NLP ────────────────────────────────────────────────────────────
_beige_cache = {}

def fetch_beige_book_sentiment(district, beige_url):
    """
    Scrape the Fed Beige Book for the relevant district and classify
    economic sentiment as Positive/Neutral/Negative using keyword NLP.
    Source: Federal Reserve Beige Book — published 8x per year.
    Cleveland Fed (2024): District-level Beige Book sentiment predicts regional recessions.
    """
    global _beige_cache
    if district in _beige_cache:
        return _beige_cache[district]
    
    try:
        # Scrape the main Beige Book page to find latest report
        headers = {"User-Agent": "Mozilla/5.0"}
        main_r = requests.get(
            "https://www.federalreserve.gov/monetarypolicy/beige-book-default.htm",
            headers=headers
        )
        soup = BeautifulSoup(main_r.content, "html.parser")
        
        # Find latest Beige Book link
        latest_link = None
        for a in soup.find_all("a", href=True):
            if "beigebook" in a["href"].lower() and ".htm" in a["href"]:
                latest_link = "https://www.federalreserve.gov" + a["href"] if a["href"].startswith("/") else a["href"]
                break
        
        if not latest_link:
            return {"tone": "neutral", "score": 0, "summary": "Beige Book unavailable"}
        
        # Fetch the full Beige Book
        book_r = requests.get(latest_link, headers=headers)
        book_soup = BeautifulSoup(book_r.content, "html.parser")
        full_text = book_soup.get_text(separator=" ", strip=True).lower()
        
        # Find district section
        district_keywords = {
            "boston": "first district",
            "new_york": "second district",
            "philadelphia": "third district", 
            "cleveland": "fourth district",
            "richmond": "fifth district",
            "atlanta": "sixth district",
            "chicago": "seventh district",
            "st_louis": "eighth district",
            "minneapolis": "ninth district",
            "kansas_city": "tenth district",
            "dallas": "eleventh district",
            "san_francisco": "twelfth district",
        }
        
        district_key = district_keywords.get(district, "")
        district_text = ""
        
        if district_key and district_key in full_text:
            start = full_text.find(district_key)
            # Get next ~3000 chars as district section
            district_text = full_text[start:start+3000]
        else:
            # Use national summary if district not found
            district_text = full_text[:3000]
        
        # Keyword NLP scoring — Cleveland Fed (2024) methodology
        positive_words = [
            "grew", "growth", "expanded", "expansion", "increased", "improvement",
            "optimistic", "strong", "robust", "gained", "rising", "positive",
            "higher", "accelerated", "solid", "healthy", "steady", "pickup"
        ]
        negative_words = [
            "declined", "decreased", "weakened", "slowed", "contraction", "fell",
            "pessimistic", "weak", "uncertain", "concern", "lower", "reduced",
            "softened", "deteriorated", "challenging", "difficult", "modest decline"
        ]
        
        pos_count = sum(district_text.count(w) for w in positive_words)
        neg_count = sum(district_text.count(w) for w in negative_words)
        total = pos_count + neg_count or 1
        
        score = round((pos_count - neg_count) / total, 2)
        
        if score > 0.1:
            tone = "positive"
        elif score < -0.1:
            tone = "negative"
        else:
            tone = "neutral"
        
        # Extract first meaningful sentence from district text as summary
        # This is actual Beige Book text — not generated
        sentences = [s.strip() for s in district_text.replace(".", ". ").split(". ") if len(s.strip()) > 40]
        real_summary = sentences[1] if len(sentences) > 1 else sentences[0] if sentences else "See source for details."
        real_summary = real_summary[:200].capitalize()

        result = {
            "tone": tone,
            "score": score,
            "district": district.replace("_", " ").title(),
            "summary": real_summary,
            "source_url": latest_link,
            "pos_count": pos_count,
            "neg_count": neg_count,
        }
        _beige_cache[district] = result
        return result
        
    except Exception as e:
        print(f"Beige Book error: {e}")
        return {"tone": "neutral", "score": 0, "district": district, "summary": "Regional data unavailable"}

# ── GOOGLE NEWS LOCAL SENTIMENT ───────────────────────────────────────────────
def fetch_local_news_sentiment(city, business_type):
    """
    Scrape Google News RSS for local economic headlines and score sentiment.
    Uses VADER-style keyword NLP for local business climate signal.
    Provides city-level real-time economic sentiment.
    """
    try:
        # Google News RSS — free, no API key needed
        query = f"{city} economy business {business_type}".replace(" ", "+")
        rss_url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
        
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(rss_url, headers=headers, timeout=10)
        soup = BeautifulSoup(r.content, "xml")
        
        titles = [item.find("title").text for item in soup.find_all("item")[:15]]
        
        if not titles:
            return {"tone": "neutral", "score": 0, "headlines": [], "city": city}
        
        # Keyword sentiment scoring
        positive_words = [
            "growth", "boom", "thriving", "opens", "expands", "hiring", "investment",
            "record", "surge", "strong", "gain", "up", "rise", "optimism", "recovery",
            "success", "new business", "development", "opportunity", "flourishing"
        ]
        negative_words = [
            "closes", "layoffs", "decline", "recession", "struggles", "bankruptcy",
            "loss", "down", "falling", "weak", "concern", "crisis", "shutdown",
            "unemployment", "slow", "cut", "reduced", "difficulty", "challenging"
        ]
        
        combined_text = " ".join(titles).lower()
        pos_count = sum(combined_text.count(w) for w in positive_words)
        neg_count = sum(combined_text.count(w) for w in negative_words)
        total = pos_count + neg_count or 1
        
        score = round((pos_count - neg_count) / total, 2)
        
        if score > 0.1:
            tone = "positive"
        elif score < -0.1:
            tone = "negative"
        else:
            tone = "neutral"
        
        return {
            "tone": tone,
            "score": score,
            "headlines": titles[:5],
            "city": city,
            "pos_count": pos_count,
            "neg_count": neg_count,
            "source_url": rss_url,
        }
        
    except Exception as e:
        print(f"Local news error: {e}")
        return {"tone": "neutral", "score": 0, "headlines": [], "city": city, "summary": "Local news unavailable"}


def fetch_local_data(zip_code):
    url = "https://api.census.gov/data/2023/acs/acs5"
    base = f"zip code tabulation area:{zip_code}"

    # ── Call 1: Base demographics + age 0-34 ─────────────────────────────
    params1 = {
        "get": ",".join([
            "B19013_001E","B01003_001E","B23025_005E","B23025_003E",
            "B01001_003E","B01001_027E",  # under 5
            "B01001_004E","B01001_028E",  # 5-9
            "B01001_005E","B01001_029E",  # 10-14
            "B01001_006E","B01001_030E",  # 15-17
            "B01001_007E","B01001_031E",  # 18-19
            "B01001_008E","B01001_032E",  # 20
            "B01001_009E","B01001_033E",  # 21
            "B01001_010E","B01001_034E",  # 22-24
            "B01001_011E","B01001_035E",  # 25-29
            "B01001_012E","B01001_036E",  # 30-34
        ]),
        "for": base,
        "key": CENSUS_API_KEY,
    }

    # ── Call 2: Age 35+ + Education ───────────────────────────────────────
    params2 = {
        "get": ",".join([
            "B01001_013E","B01001_037E",  # 35-39
            "B01001_014E","B01001_038E",  # 40-44
            "B01001_015E","B01001_039E",  # 45-49
            "B01001_016E","B01001_040E",  # 50-54
            "B01001_017E","B01001_041E",  # 55-59
            "B01001_018E","B01001_042E",  # 60-61
            "B01001_019E","B01001_043E",  # 62-64
            "B01001_020E","B01001_044E",  # 65-66
            "B01001_021E","B01001_045E",  # 67-69
            "B01001_022E","B01001_046E",  # 70-74
            "B01001_023E","B01001_047E",  # 75-79
            "B01001_024E","B01001_048E",  # 80-84
            "B01001_025E","B01001_049E",  # 85+
            "B15003_022E","B15003_023E",  # bachelor + master
            "B15003_024E","B15003_025E",  # professional + doctorate
            "B15003_001E",               # total 25+ for education base
        ]),
        "for": base,
        "key": CENSUS_API_KEY,
    }

    try:
        r1 = requests.get(url, params=params1).json()
        r2 = requests.get(url, params=params2).json()
        v1 = r1[1]
        v2 = r2[1]

        def iv(x): return max(0, int(x)) if x and x != "-1" else 0

        # ── Base variables ────────────────────────────────────────────────
        median_income     = iv(v1[0])
        population        = iv(v1[1])
        unemployed        = iv(v1[2])
        labor_force       = iv(v1[3]) or 1
        unemployment_rate = round((unemployed / labor_force) * 100, 2)

        # ── Age segments ─────────────────────────────────────────────────
        # v1 structure: [income, pop, unemployed, laborforce, then age pairs M+F]
        # v1 indices 4-23 = age data (20 values = 10 pairs), index 24 = ZIP
        # Children: under 18 (under5, 5-9, 10-14, 15-17) = indices 4-11
        children = sum(iv(v1[i]) for i in range(4, 12))

        # Young adults: 18-24 (18-19, 20, 21, 22-24) = indices 12-19
        young_adults = sum(iv(v1[i]) for i in range(12, 20))

        # Prime age 25-34 from v1 indices 20-23, 25-44 continues in v2 indices 0-7
        prime_age = sum(iv(v1[i]) for i in range(20, 24)) + sum(iv(v2[i]) for i in range(0, 8))

        # Middle age 45-64: v2 indices 8-23
        middle_age = sum(iv(v2[i]) for i in range(8, 24))

        # Seniors 65+: v2 indices 24-30 (65-66 to 85+), stop before education cols
        seniors = sum(iv(v2[i]) for i in range(24, 31))

        total_age = max(population, children + young_adults + prime_age + middle_age + seniors)

        pct_children = round((children     / total_age) * 100, 1) if total_age > 0 else 0
        pct_young    = round((young_adults / total_age) * 100, 1) if total_age > 0 else 0
        pct_prime    = round((prime_age    / total_age) * 100, 1) if total_age > 0 else 0
        pct_middle   = round((middle_age   / total_age) * 100, 1) if total_age > 0 else 0
        pct_seniors  = round((seniors      / total_age) * 100, 1) if total_age > 0 else 0

        # ── Education ─────────────────────────────────────────────────────
        # v2 structure: ...age pairs..., bach(26), master(27), prof(28), doc(29), total25+(30), ZIP(31)
        # Luttmer (2005) — education as premium pricing predictor
        edu_base     = iv(v2[30]) or 1
        college_plus = iv(v2[26]) + iv(v2[27]) + iv(v2[28]) + iv(v2[29])
        pct_college  = round((college_plus / edu_base) * 100, 1) if edu_base > 0 else 0

        # ── Dominant age segment ──────────────────────────────────────────
        segments = {
            "Children (0-17)":      pct_children,
            "Young Adults (18-24)": pct_young,
            "Prime Age (25-44)":    pct_prime,
            "Middle Age (45-64)":   pct_middle,
            "Seniors (65+)":        pct_seniors,
        }
        dominant_segment = max(segments, key=segments.get)

        # Get city name from Google Geocoding
        city_name = zip_code
        try:
            geo_r = requests.get(
                "https://maps.googleapis.com/maps/api/geocode/json",
                params={"address": zip_code, "key": GOOGLE_API_KEY}
            ).json()
            if geo_r.get("results"):
                for component in geo_r["results"][0]["address_components"]:
                    if "locality" in component["types"]:
                        city_name = component["long_name"]
                        break
                    elif "sublocality" in component["types"]:
                        city_name = component["long_name"]
        except:
            pass

        return {
            "zip_code":         zip_code,
            "city":             city_name,
            "median_income":    median_income,
            "population":       population,
            "unemployment_rate":unemployment_rate,
            "labor_force":      labor_force,
            "pct_children":     pct_children,
            "pct_young":        pct_young,
            "pct_prime":        pct_prime,
            "pct_middle":       pct_middle,
            "pct_seniors":      pct_seniors,
            "pct_college":      pct_college,
            "dominant_segment": dominant_segment,
        }

    except Exception as e:
        print(f"Census error: {e}")
        return {
            "zip_code":         zip_code,
            "median_income":    58000,
            "population":       45000,
            "unemployment_rate":4.2,
            "labor_force":      22000,
            "pct_children":     20.0,
            "pct_young":        10.0,
            "pct_prime":        30.0,
            "pct_middle":       25.0,
            "pct_seniors":      15.0,
            "pct_college":      30.0,
            "dominant_segment": "Prime Age (25-44)",
        }

def normalize_business_type(business_type):
    """
    BRAIN: Uses GPT to normalize business type into a standard
    Google Places search keyword to ensure consistent competitor search.
    """
    prompt = f"""
    Convert this business description into the best single Google Places search keyword.
    Return only the keyword, nothing else. No explanation, no punctuation.
    
    Examples:
    "cafe" -> "coffee shop"
    "coffee house" -> "coffee shop"
    "drugstore" -> "pharmacy"
    "clothes shop" -> "clothing store"
    "burger place" -> "restaurant"
    "gym" -> "fitness center"
    
    Business type: {business_type}
    """
    r = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=20,
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(block.text for block in r.content if block.type == "text")
    return text.strip().lower()

def fetch_competitor_data(zip_code, business_type):
    """
    MUSCLE: Fetches ALL competitor data from Google Places API using pagination.
    Retrieves up to 60 results (3 pages of 20) to avoid the 20-result cap.
    Calculates Proxy HHI following Anderson-Magruder (2012).
    """
    try:
        # Geocode ZIP code
        geo_resp = requests.get(
            "https://maps.googleapis.com/maps/api/geocode/json",
            params={"address": zip_code, "key": GOOGLE_API_KEY}
        ).json()

        if not geo_resp["results"]:
            return _empty_competitor_data()

        loc = geo_resp["results"][0]["geometry"]["location"]
        lat, lng = loc["lat"], loc["lng"]

        # Paginate through ALL results — fixes the 20-result cap
        all_competitors = []
        next_page_token = None

        for page in range(3):  # Max 3 pages = 60 results
            params = {
                "location": f"{lat},{lng}",
                "radius": 3219,  # 2 miles in meters
                "keyword": business_type,
                "key": GOOGLE_API_KEY,
            }
            if next_page_token:
                params = {"pagetoken": next_page_token, "key": GOOGLE_API_KEY}
                import time
                time.sleep(2)  # Google requires delay before using next_page_token

            resp = requests.get(
                "https://maps.googleapis.com/maps/api/place/nearbysearch/json",
                params=params
            ).json()

            results = resp.get("results", [])
            all_competitors.extend(results)
            next_page_token = resp.get("next_page_token")

            if not next_page_token:
                break  # No more pages

        count = len(all_competitors)
        population = 19326  # Will be overridden by actual local data in caller

        # Proxy HHI — Anderson-Magruder (2012)
        # review count as market share proxy
        reviews = [c.get("user_ratings_total", 0) for c in all_competitors]
        total_reviews = sum(reviews)
        if total_reviews > 0:
            shares = [r / total_reviews for r in reviews]
            proxy_hhi = round(sum(s ** 2 for s in shares) * 10000)
        else:
            proxy_hhi = 0

        # Competition level classified in fetch_competitor_data_with_density using density
        level = "Unknown"  # will be overridden by density-based classification






        # Top 5 for display
        # Top 5 for display — include rating for quality signal
        nearby = [
            {
                "name": c.get("name", "Unknown"),
                "rating": c.get("rating", 0) or 0,
                "vicinity": c.get("vicinity", ""),
                "reviews": c.get("user_ratings_total", 0),
            }
            for c in all_competitors[:5]
        ]

        # Competitor quality signal — average rating of all competitors
        # Higher avg rating = stronger competition = harder to enter
        ratings = [c.get("rating", 0) for c in all_competitors if c.get("rating")]
        avg_competitor_rating = round(sum(ratings) / len(ratings), 2) if ratings else 0

        return {
            "competitor_count": count,
            "competitor_level": level,
            "nearby_competitors": nearby,
            "proxy_hhi": proxy_hhi,
            "review_counts": reviews[:20],
            "avg_competitor_rating": avg_competitor_rating,
            "lat": lat,
            "lng": lng,
        }

    except Exception as e:
        print(f"Google Places error: {e}")
        return _empty_competitor_data()


def _empty_competitor_data():
    return {
        "competitor_count": 0,
        "competitor_level": "Unknown",
        "nearby_competitors": [],
        "proxy_hhi": 0,
        "review_counts": [],
        "avg_competitor_rating": 0,
    }


def fetch_competitor_data_with_density(zip_code, business_type, population):
    """Wrapper that adds density-based classification and quality signal once population is known."""
    data = fetch_competitor_data(zip_code, business_type)
    density = 0
    if population > 0:
        density = round(data["competitor_count"] / (population / 10000), 2)
        data["density_per_10k"] = density
    else:
        data["density_per_10k"] = 0


    # HHI classification — DOJ thresholds (Anderson & Magruder 2012)
    # HHI and density measure different things — interpret together
    proxy_hhi = data.get("proxy_hhi", 0)
    if proxy_hhi < 1500:
        data["hhi_level"] = "Unconcentrated"
        if density < 10:
            data["hhi_interpretation"] = "few competitors and no dominant player — excellent entry conditions"
        else:
            data["hhi_interpretation"] = "crowded market but no single dominant player — differentiation possible"
    elif proxy_hhi < 2500:
        data["hhi_level"] = "Moderately Concentrated"
        if density < 10:
            data["hhi_interpretation"] = "few competitors with some concentration — easy to enter, quality differentiation needed"
        else:
            data["hhi_interpretation"] = "moderate concentration with significant competition — differentiation essential"
    else:
        data["hhi_level"] = "Highly Concentrated"
        if density < 10:
            data["hhi_interpretation"] = "few competitors but one clearly dominates on reviews — market easy to enter physically, strong differentiation required to challenge incumbent"
        else:
            data["hhi_interpretation"] = "market saturated and dominated — worst case for independent entry"

    # Density-based competition level — Bresnahan & Reiss (1991)
    # Uses density not raw count — internally consistent with scoring engine
    # Thresholds are neutral defaults — overridden by firm type in calculate_scores_deterministic
    if density < 4:
        data["competitor_level"] = "Low"
    elif density < 10:
        data["competitor_level"] = "Moderate"
    else:
        data["competitor_level"] = "High"

    # Competitor quality signal — average Google Places rating
    # High avg rating = stronger incumbents = harder to displace
    # Porter (1980): incumbent strength is a barrier to entry
    avg_rating = data.get("avg_competitor_rating", 0)
    if avg_rating >= 4.5:
        data["quality_level"] = "High"
        data["quality_note"] = f"Strong incumbents (avg rating {avg_rating}) — differentiation essential"
    elif avg_rating >= 4.0:
        data["quality_level"] = "Moderate"
        data["quality_note"] = f"Moderate incumbent quality (avg rating {avg_rating}) — quality parity needed"
    elif avg_rating > 0:
        data["quality_level"] = "Low"
        data["quality_note"] = f"Weak incumbents (avg rating {avg_rating}) — quality advantage possible"
    else:
        data["quality_level"] = "Unknown"
        data["quality_note"] = "Competitor quality data unavailable"

    return data


def calculate_scores_deterministic(macro_data, local_data, competitor_data, firm_type, price_tier):
    """
    MUSCLE: All scores calculated deterministically from verified research literature.
    Weights reflect relative explanatory power in source papers — rank-ordered, not arbitrary.
    All geographic levels explicitly matched to appropriate data sources.

    EXPANSION SCORE (100 pts) — Bresnahan & Reiss (1991, JPE) primary:
      Competitor Density 35pts [Local-ZIP] | Income 15pts [Local-ZIP] |
      BLS Unemployment 12pts [Local-County] | Population 12pts [Local-ZIP] |
      Rent Burden 10pts [Local-ZIP] | Age Mix 8pts [Local-ZIP] | Prime Rate 8pts [National]

    LOCAL MARKET SCORE (100 pts) — Huff Gravity Model (1964):
      Population 25pts [Local-ZIP] | Income 25pts [Local-ZIP] |
      BLS Unemployment 15pts [Local-County] | Age Mix 12pts [Local-ZIP] |
      Education 8pts [Local-ZIP] | Rent-to-Income 8pts [Local-ZIP] |
      Income Growth 7pts [National — no free ZIP alternative]
      + Firm type trade area bonus: National +8pts | Regional +4pts
      Citation: Ailawadi & Keller (2004) — brands draw customers beyond ZIP boundary
      Citation: Ellickson et al. (2013, RAND JE) — chains viable in moderate-income markets

    PRICING POWER SCORE (100 pts) — measures RELATIVE pricing power vs competitors
      Porter (1980) five forces: market structure constrains pricing — firms are price takers
      Ailawadi et al. (2003, JM): brand equity enables premium OVER competitors not free pricing
      Brand Equity 28pts [Firm+Local] | Market Structure HHI+Competition 22pts [Local-ZIP] |
      Income Premium 18pts [Local-ZIP, varies by firm type] |
      Education Premium 10pts [Local-ZIP] | Local Rent Burden 8pts [Local-ZIP] |
      Beige Book Tone 8pts [Regional-District] | Retail Growth 4pts [National-minor] |
      Inflation Penalty up to -2pts [National]

    MARKET ENTRY WINDOW (100 pts) — Nakamura & Steinsson (2013, AER):
      BOTH national AND local must support entry — Huff (1964) gravity requires both.
      LOCAL VETO: critically weak local readiness downgrades MEW regardless of national macro.
      National Macro 30pts | Local Readiness 28pts | Competitive Landscape 22pts |
      Local News Sentiment 10pts [City-level NLP] | Beige Book 8pts [District NLP] |
      Firm Resilience bonus: National +4pts | Regional +2pts
    """

    # ── VARIABLE EXTRACTION ───────────────────────────────────────────────
    # Local ZIP-level variables (Census ACS 2023)
    income        = local_data.get("median_income", 58000)
    population    = local_data.get("population", 45000)
    pct_children  = local_data.get("pct_children", 20.0)
    pct_young     = local_data.get("pct_young", 10.0)
    pct_prime     = local_data.get("pct_prime", 30.0)
    pct_middle    = local_data.get("pct_middle", 25.0)
    pct_seniors   = local_data.get("pct_seniors", 15.0)
    pct_college   = local_data.get("pct_college", 30.0)

    # Local county unemployment (BLS LAUS — monthly, fresher than Census ACS)
    unemployment  = local_data.get("unemployment_rate", 4.2)

    # Local ZIP rent (Zillow ZORI — monthly updated)
    median_rent   = local_data.get("median_rent", None)

    # Regional NLP signals
    beige_tone    = local_data.get("beige_tone", "neutral")

    # City-level NLP signal
    news_tone     = local_data.get("news_tone", "neutral")

    # Competitor data (Google Places — live)
    density       = competitor_data.get("density_per_10k", 0)
    proxy_hhi     = competitor_data.get("proxy_hhi", 1000)
    comp_level    = competitor_data.get("competitor_level", "Moderate")

    # National macro variables (FRED)
    fed_rate      = macro_data.get("fed_funds_rate", 3.64)
    prime_rate    = macro_data.get("prime_rate", 6.75)
    gdp_growth    = macro_data.get("gdp_growth", 2.0)
    inflation     = macro_data.get("inflation_rate", 2.5)
    retail_growth = macro_data.get("retail_sales_growth", 2.0)
    income_growth = macro_data.get("income_growth", 0.5)  # national — no free ZIP alt

    national_median = 74580  # US median household income 2023

    # Rent burden — annual rent / annual income
    # Mian, Rao & Sufi (2013, QJE): housing cost burden constrains discretionary spending
    if median_rent and income > 0:
        rent_burden = (median_rent * 12) / income
    else:
        rent_burden = 0.30  # US national average fallback

    # ── EXPANSION SCORE (100 pts) ─────────────────────────────────────────
    # Bresnahan & Reiss (1991, JPE): competitor density is THE primary entry finding
    # Magnolfi et al. (2024): B&R remains the current standard framework
    # GRADUATED formula — B&R show diminishing returns not cliff effects
    # Ellickson, Houghton & Timmins (2013, RAND JE): firm-type specific thresholds
    if firm_type == "Independent":
        max_density = 20
    elif firm_type == "Regional Chain":
        max_density = 35
    else:
        max_density = 50
    density_pts = max(0, round(35 * (1 - density / max_density)))

    # ZIP median income — Dube et al. (2010, QJE): income sets revenue ceiling — 15pts
    # GRADUATED: linear scale from $20K (min viable) to $120K (max)
    income_exp_pts = round(max(1, min(15, 15 * (income - 20000) / 100000)))

    # BLS county unemployment — labor demand proxy — 12pts
    # GRADUATED: optimal at 3%, zero at 15%+
    unemp_exp_pts = round(max(0, min(12, 12 * (1 - (unemployment - 3) / 12))))

    # ZIP population — Berry & Waldfogel (1999, JPE) — 12pts
    # GRADUATED: linear from 5K to 80K
    pop_pts = round(max(1, min(12, 12 * (population - 5000) / 75000)))

    # Rent burden — Bank of America Institute (2024) — 10pts
    # GRADUATED: optimal at 0.15, zero at 0.60+
    rent_exp_pts = round(max(0, min(10, 10 * (1 - (rent_burden - 0.15) / 0.45))))

    # Age mix — Azoulay et al. (2020, AER:Insights) — 8pts
    dominant_age = max(pct_children, pct_young, pct_prime, pct_middle, pct_seniors)
    age_exp_pts = round(max(1, min(8, 8 * dominant_age / 50)))

    # Prime rate — Bernanke & Gertler (1995, AER) — 8pts
    # GRADUATED: optimal at 3%, zero at 12%+
    prime_pts = round(max(0, min(8, 8 * (1 - (prime_rate - 3) / 9))))

    expansion = min(100, density_pts + income_exp_pts + unemp_exp_pts + pop_pts +
                    rent_exp_pts + age_exp_pts + prime_pts)

    # Breakdown for visualization
    expansion_breakdown = {
        f"Competitor Density ({density:.1f}/10K)":      (density_pts, 35),
        f"Local Income (${income:,.0f})":                (income_exp_pts, 15),
        f"Unemployment ({unemployment:.1f}%)":           (unemp_exp_pts, 12),
        f"Population ({population:,})":                  (pop_pts, 12),
        f"Rent Burden ({rent_burden*100:.0f}%)":         (rent_exp_pts, 10),
        f"Age Mix ({dominant_age:.0f}% dominant)":       (age_exp_pts, 8),
        f"Prime Rate ({prime_rate:.2f}%)":               (prime_pts, 8),
    }

    # ── LOCAL MARKET SCORE (100 pts) ──────────────────────────────────────
    # Huff (1964) Gravity Model: population and income are co-equal primary factors

    # ZIP population — 25pts GRADUATED
    pop_huff = round(max(1, min(25, 25 * (population - 5000) / 95000)))

    # ZIP median income — 25pts GRADUATED
    inc_huff = round(max(1, min(25, 25 * (income - 20000) / 130000)))

    # BLS unemployment — 15pts GRADUATED
    unemp_huff = round(max(0, min(15, 15 * (1 - (unemployment - 2) / 13))))

    # Age mix — 12pts — priority: prime > young > seniors
    age_local_pts = 12 if pct_prime > 35 else (9 if pct_prime > 25 else (7 if pct_young > 30 else (6 if pct_seniors > 25 else 4)))

    # Education — Luttmer (2005, QJE); Datta et al. (2017, JM) — 8pts GRADUATED
    edu_pts = round(max(1, min(8, 8 * pct_college / 70)))

    # Rent-to-income — Mian, Rao & Sufi (2013, QJE) — 8pts GRADUATED
    rent_local_pts = round(max(0, min(8, 8 * (1 - (rent_burden - 0.15) / 0.45))))

    # National income growth — 7pts
    income_growth_pts = 7 if income_growth > 2 else (5 if income_growth >= 0 else 2)

    local = min(100, pop_huff + inc_huff + unemp_huff + age_local_pts +
                edu_pts + rent_local_pts + income_growth_pts)

    # Firm type trade area bonus — Ailawadi & Keller (2004)
    firm_bonus = 8 if firm_type == "National Brand" else (4 if firm_type == "Regional Chain" else 0)
    local = min(100, local + firm_bonus)

    # Breakdown for visualization
    local_breakdown = {
        f"Population ({population:,})":                  (pop_huff, 25),
        f"Median Income (${income:,.0f})":               (inc_huff, 25),
        f"Unemployment ({unemployment:.1f}%)":           (unemp_huff, 15),
        f"Age Mix ({pct_prime:.0f}% prime)":             (age_local_pts, 12),
        f"Education ({pct_college:.0f}% college)":       (edu_pts, 8),
        f"Rent Burden ({rent_burden*100:.0f}%)":         (rent_local_pts, 8),
        f"Income Growth ({income_growth:.1f}%)":         (income_growth_pts, 7),
    }

    # ── PRICING POWER SCORE (100 pts) ─────────────────────────────────────
    # Porter (1980): relative pricing power vs competitors

    # Brand equity — Ailawadi et al. (2003, JM) — 28pts
    if firm_type == "Independent":
        brand_base = 0
    elif firm_type == "Regional Chain":
        brand_base = 12
    else:
        brand_base = 22
    local_support = 8 if income > national_median else (6 if income > 60000 else (4 if income > 40000 else 4))
    brand_pts = min(28, brand_base + local_support)

    # Market structure — Porter (1980); Anderson & Magruder (2012) — 22pts
    # HHI weight reduced to 4pts — Google review proxy too weak to carry more
    # Density-based competition level carries primary weight (12pts)
    hhi_pricing_pts = 4 if proxy_hhi < 1500 else (2 if proxy_hhi < 2500 else 1)
    comp_pricing_pts = 12 if comp_level == "Low" else (7 if comp_level == "Moderate" else 2)
    # Competitor quality adjustment — Porter (1980): strong incumbents reduce pricing power
    quality_level = competitor_data.get("quality_level", "Unknown")
    quality_adj = -2 if quality_level == "High" else (0 if quality_level == "Moderate" else 1)
    market_struct_pts = hhi_pricing_pts + comp_pricing_pts + quality_adj

    # Income premium — Dube et al. (2010, QJE) — 18pts, varies by firm type
    income_ratio = income / national_median
    if firm_type == "Independent":
        income_premium_pts = min(18, max(2, round((income_ratio - 0.5) * 18)))
    elif firm_type == "Regional Chain":
        income_premium_pts = min(18, max(4, round((income_ratio - 0.3) * 18)))
    else:
        income_premium_pts = min(18, max(6, round((income_ratio - 0.1) * 18)))

    # Education premium — Luttmer (2005); Datta et al. (2017) — 10pts GRADUATED
    edu_premium_pts = round(max(1, min(10, 10 * pct_college / 70)))

    # Rent burden — Mian et al. (2013) — 8pts GRADUATED
    rent_pricing_pts = round(max(0, min(8, 8 * (1 - (rent_burden - 0.15) / 0.45))))

    # Beige Book tone — Cleveland Fed (2024) — 8pts
    beige_pricing_pts = 8 if beige_tone == "positive" else (5 if beige_tone == "neutral" else 2)

    # Retail sales growth — Census Bureau — 4pts
    retail_pts = 4 if retail_growth > 4 else (2 if retail_growth > 2 else (1 if retail_growth >= 0 else 0))

    # Inflation penalty — BCG (2024)
    inflation_penalty = 2 if inflation > 4 else 0









    pricing = min(100, brand_pts + market_struct_pts + income_premium_pts +
                  edu_premium_pts + rent_pricing_pts + beige_pricing_pts +
                  retail_pts - inflation_penalty)

    # Breakdown for visualization
    pricing_breakdown = {
        f"Brand Equity ({firm_type})":                        (brand_pts, 28),
        f"Market Structure (HHI={proxy_hhi:.0f}, Quality={quality_level})": (market_struct_pts, 17),
        f"Income Premium (${income:,.0f})":                   (income_premium_pts, 18),
        f"Education ({pct_college:.0f}% college)":            (edu_premium_pts, 10),
        f"Rent Burden ({rent_burden*100:.0f}%)":              (rent_pricing_pts, 8),
        f"Beige Book ({beige_tone.title()})":                 (beige_pricing_pts, 8),
        f"Retail Growth ({retail_growth:.1f}%)":              (retail_pts, 4),
        f"Price Tier Alignment":                             (0, 8),  # updated after tier calc
    }

    # ── MARKET ENTRY WINDOW (100 pts) ─────────────────────────────────────
    # Nakamura & Steinsson (2013, AER)

    # National macro — 30pts GRADUATED
    fed_pts  = round(max(0, min(10, 10 * (1 - (fed_rate - 2) / 8))))
    gdp_pts  = round(max(0, min(10, 10 * gdp_growth / 6)))
    inf_pts  = round(max(0, min(4, 4 * (1 - abs(inflation - 2) / 4))))
    national_pts = fed_pts + gdp_pts + inf_pts

    # Local readiness — 28pts
    inc_entry_pts   = 14 if income > national_median else (9 if income > 60000 else (5 if income > 40000 else 1))
    unemp_entry_pts = 11 if unemployment < 3.5 else (7 if unemployment < 5 else (4 if unemployment < 7 else 1))
    pop_entry_pts   = 5 if population > 40000 else (3 if population > 20000 else 1)
    age_entry_pts   = 2 if (pct_prime > 25 or pct_young > 25) else 1
    local_pts = inc_entry_pts + unemp_entry_pts + pop_entry_pts + age_entry_pts

    # Competitive landscape — B&R (1991) + Anderson & Magruder (2012) — 22pts
    if firm_type == "Independent":
        density_entry_pts = max(1, round(13 * (1 - density / max_density)))
    elif firm_type == "Regional Chain":
        density_entry_pts = max(1, round(13 * (1 - density / max_density)))
    else:
        density_entry_pts = max(1, round(13 * (1 - density / max_density)))
    hhi_entry_pts = 4 if proxy_hhi < 1500 else (2 if proxy_hhi < 2500 else 1)  # reduced — review proxy too weak
    comp_entry_pts = density_entry_pts + hhi_entry_pts

    # Local news sentiment — Google News NLP — 13pts
    news_entry_pts = 8 if news_tone == "positive" else (5 if news_tone == "neutral" else 2)  # reduced from 13 — RSS too volatile

    # Beige Book regional tone — Fed District NLP — 8pts
    # Cleveland Fed (2024): district-level sentiment predicts regional economic conditions
    beige_entry_pts = 11 if beige_tone == "positive" else (7 if beige_tone == "neutral" else 3)  # +3pts replacing national sentiment

    # Firm type resilience bonus — Decker & Haltiwanger (2024, Fed FEDS Notes) — max 4pts
    # Established brands show greater entry resilience during economic contractions
    if firm_type == "National Brand":
        firm_resilience = 4
    elif firm_type == "Regional Chain":
        firm_resilience = 2
    else:
        firm_resilience = 0

    market_entry_score = min(100, national_pts + local_pts + comp_entry_pts +
                             news_entry_pts + beige_entry_pts + firm_resilience)

    # Initial classification
    market_entry_window = "Open" if market_entry_score > 70 else ("Cautious" if market_entry_score >= 45 else "Closed")

    # LOCAL VETO — Huff (1964): gravity model requires BOTH size AND attractiveness
    # Critically weak local readiness overrides national macro — national conditions
    # cannot rescue a severely distressed local market
    if local_pts <= 8:  # out of 28 — critically distressed neighborhood
        market_entry_window = "Closed"
        market_entry_score  = min(market_entry_score, 44)
    elif local_pts <= 14 and market_entry_window == "Open":
        market_entry_window = "Cautious"
        market_entry_score  = min(market_entry_score, 69)

    # ── MARKET ENTRY WINDOW VETO IN GPT PROMPT ───────────────────────────
    # If MEW is Closed, GPT recommendation cannot be Expand (handled in prompt)

    # ── PRICE TIER ALIGNMENT ──────────────────────────────────────────────
    # Based on local ZIP median income only — Dube et al. (2010, QJE)
    # Income is the direct purchasing power measure; education is a proxy
    if firm_type == "Independent":
        if income < 45000:
            recommended_tier = "Budget"
        elif income < 75000:
            recommended_tier = "Mid-Range"
        else:
            recommended_tier = "Premium"
    elif firm_type == "Regional Chain":
        if income < 55000:
            recommended_tier = "Mid-Range"
        else:
            recommended_tier = "Premium"
    else:  # National Brand
        if income < 45000:
            recommended_tier = "Mid-Range"
        else:
            recommended_tier = "Premium"

    pricing_aligned = (price_tier == recommended_tier)

    # Price tier alignment penalty — Dube et al. (2010)
    # Misaligned price tier reduces effective pricing power
    tier_order = ["Budget", "Mid-Range", "Premium"]
    selected_idx = tier_order.index(price_tier) if price_tier in tier_order else 1
    recommended_idx = tier_order.index(recommended_tier) if recommended_tier in tier_order else 1
    tier_gap = abs(selected_idx - recommended_idx)
    tier_penalty = tier_gap * 8  # 8pts per tier misalignment
    pricing = min(100, max(0, pricing - tier_penalty))
    # Update pricing breakdown with tier alignment
    pricing_breakdown[f"Price Tier Alignment (gap={tier_gap})"] = (max(0, 8 - tier_penalty), 8)
    if "Price Tier Alignment" in pricing_breakdown:
        del pricing_breakdown["Price Tier Alignment"]

    # MEW breakdown for visualization
    # Check if local veto fired
    veto_fired = local_pts <= 8
    veto_cautious = local_pts <= 14 and not veto_fired

    mew_breakdown = {
        f"Fed Rate ({fed_rate:.2f}%)":              (fed_pts, 10),
        f"GDP Growth ({gdp_growth:.1f}%)":          (gdp_pts, 10),
        f"Inflation ({inflation:.1f}%)":            (inf_pts, 4),
        f"Income Readiness (${income:,.0f})":       (inc_entry_pts, 14),
        f"Unemployment ({unemployment:.1f}%)":      (unemp_entry_pts, 11),
        f"Population ({population:,})":             (pop_entry_pts, 5),
        f"Age Mix":                                 (age_entry_pts, 2),
        f"Competitor Density ({density:.1f}/10K)":  (density_entry_pts, 13),
        f"Market Concentration (HHI={proxy_hhi:.0f})": (hhi_entry_pts, 4),
        f"Local News ({news_tone.title()})":        (news_entry_pts, 8),
        f"Beige Book ({beige_tone.title()})":       (beige_entry_pts, 11),
        f"Local Readiness Total":                   (local_pts, 28),
    }
    if veto_fired:
        mew_breakdown[f"LOCAL VETO TRIGGERED (local={local_pts}/28)"] = (0, 28)
    elif veto_cautious:
        mew_breakdown[f"LOCAL CAUTION (local={local_pts}/28 — downgraded to Cautious)"] = (local_pts, 28)

    return {
        "expansion_score":      expansion,
        "local_market_score":   local,
        "pricing_power_score":  pricing,
        "market_entry_score":   market_entry_score,
        "market_entry_window":  market_entry_window,
        "recommended_tier":     recommended_tier,
        "pricing_aligned":      pricing_aligned,
        "pricing_direction":    "aligned" if pricing_aligned else (
            "overpriced" if ["Budget","Mid-Range","Premium"].index(price_tier) > ["Budget","Mid-Range","Premium"].index(recommended_tier)
            else "underpriced"
        ),
        "expansion_breakdown":  expansion_breakdown,
        "local_breakdown":      local_breakdown,
        "pricing_breakdown":    pricing_breakdown,
        "mew_breakdown":        mew_breakdown,
    }

def run_strategy_agent(macro_data, fomc_analysis, local_data, firm_inputs, competitor_data={}, scores={}):
    prompt = f"""
You are a senior economic strategy advisor at McKinsey. Be specific and cite exact numbers.
Never write generic advice. Every sentence must reference actual data points provided.
Return JSON only. No markdown. No preamble.

FIRM PROFILE:
- Business Type: {firm_inputs["business_type"]}
- Firm Type: {firm_inputs.get("firm_type", "Independent")}
- Price Tier Selected: {firm_inputs["price_tier"]}
- Target ZIP: {local_data["zip_code"]}

PRE-COMPUTED SCORES (from deterministic research-grounded scoring engine):
- Expansion Score: {firm_inputs.get("expansion_score", 50)}/100
- Local Market Score: {firm_inputs.get("local_market_score", 50)}/100
- Pricing Power Score: {firm_inputs.get("pricing_power_score", 50)}/100
- Market Entry Window: {firm_inputs.get("market_entry_window", "Cautious")} ({firm_inputs.get("market_entry_score", 50)}/100)
These scores are calculated from verified economic research. Use them directly in your decision rules below.

NATIONAL MACRO DATA (FRED):
- Fed Funds Rate: {macro_data["fed_funds_rate"]}%
- Prime Rate: {macro_data["prime_rate"]}%
- Inflation Rate: {macro_data["inflation_rate"]}%
- GDP Growth: {macro_data["gdp_growth"]}%
- Retail Sales Growth: {macro_data["retail_sales_growth"]}%
- PCE Growth: {macro_data.get("pce_growth", -0.29)}%
- Disposable Income Growth: {macro_data["income_growth"]}%

FED SIGNAL (FOMC NLP):
- Tone: {fomc_analysis["tone"]} (score: {fomc_analysis["tone_score"]})
- Direction: {fomc_analysis["policy_direction"]}
- Summary: {fomc_analysis["one_line_summary"]}

LOCAL MARKET DATA (ZIP {local_data["zip_code"]} — Census ACS 2023 | BLS LAUS | Zillow ZORI):
- Median Household Income: ${local_data["median_income"]:,} (National median: $74,580)
- Population: {local_data["population"]:,}
- Unemployment Rate: {local_data["unemployment_rate"]}% (BLS LAUS county level)
- Median Rent: ${local_data.get("median_rent", "N/A")} /month (Zillow ZORI)
- Rent Burden: {round((local_data.get("median_rent", 0) * 12 / max(local_data["median_income"], 1) * 100) if local_data.get("median_rent") else 30, 1)}% of annual income
- College Educated: {local_data.get("pct_college", 30)}%
- Prime Age 25-44: {local_data.get("pct_prime", 30)}%
- Young Adults 18-24: {local_data.get("pct_young", 10)}%
- Seniors 65+: {local_data.get("pct_seniors", 15)}%

NLP INTELLIGENCE SIGNALS:
- Beige Book ({local_data.get("beige_district", "Unknown")} District): {local_data.get("beige_tone", "neutral").upper()} (score: {local_data.get("beige_score", 0)})
- Beige Book Summary: {local_data.get("beige_summary", "Unavailable")[:150]}
- Local News ({local_data.get("city", local_data["zip_code"])}): {local_data.get("news_tone", "neutral").upper()} (score: {local_data.get("news_score", 0)})

COMPETITOR LANDSCAPE:
- Competitor count within 2 miles: {competitor_data.get("competitor_count", 0)}
- Competitor density: {competitor_data.get("density_per_10k", 0)} per 10,000 people
- Competition level: {competitor_data.get("competitor_level", "Unknown")}
- Market structure: {competitor_data.get("hhi_interpretation", "unknown")}
- Competitor quality: {competitor_data.get("quality_note", "unknown")}
- Top competitors: {", ".join([c["name"] for c in competitor_data.get("nearby_competitors", [])[:5]])}

ECONOMIC FRAMEWORK RULES:

1. FIRM TYPE DIFFERENTIATION (Ailawadi et al. 2003; Ellickson et al. 2013):
   - Independent: fully constrained by local income. No brand premium. Density >4/10K = saturated.
   - Regional Chain: 10-15% premium possible. Density >8/10K = saturated.
   - National Brand: 20-30% premium regardless of local income. Density >12/10K = saturated.
   Recommendation MUST differ between Independent and National Brand in same market.

2. EDUCATION PREMIUM (Luttmer 2005; Datta et al. 2017):
   High college education signals willingness to pay for quality independent of income.
   University areas with low income but high education support quality businesses better than income alone suggests.

3. RENT BURDEN (Mian et al. 2013):
   Rent burden above 35% = significant constraint on discretionary spending.

4. AGE MIX: Prime age = highest disposable income. Young adults = high frequency, price sensitive.

5. NLP SIGNALS: Incorporate all three — FOMC (macro timing), Beige Book (regional), News (local).


CRITICAL DECISION RULES — YOU MUST FOLLOW THESE EXACTLY:
- If market_entry_window = OPEN AND expansion_score >= 70: expansion_recommendation MUST be "Expand". No exceptions.
- If market_entry_window = OPEN AND expansion_score 50-69: "Hold" is acceptable.
- If market_entry_window = CAUTIOUS: "Hold" at best. Never "Expand".
- If market_entry_window = CLOSED: "Do Not Expand". Non-negotiable.
- FOMC neutral tone = no signal. Do NOT use neutral FOMC as reason for Hold.
- High HHI with low density means differentiation challenge NOT entry barrier. Do NOT use HHI alone to justify Hold.
- The scores are calculated from verified economic research. Trust the scores. Follow the rules above.

Return ONLY this JSON:
{{
    "expansion_recommendation": "Expand" or "Hold" or "Do Not Expand",
    "confidence_level": "High" or "Moderate" or "Low",
    "confidence_reason": "one sentence explaining WHY confidence is high/moderate/low. Must reference: (1) at least one score number, (2) at least one local variable with its value, (3) at least one competitive or NLP signal. Do not just repeat data points — explain what combination of signals drives the confidence level.",
    "macro_risk": "High" or "Moderate" or "Low",
    "reasoning_trace": [
        "Step 1 - Fed Signal: FOMC tone with specific numbers",
        "Step 2 - Macro Screen: GDP, inflation, credit with numbers",
        "Step 3 - Local Market: income, unemployment, education %, rent burden %, age mix % with numbers",
        "Step 4 - NLP Signals: Beige Book score and News sentiment score with district name",
        "Step 5 - Final Decision: how all signals combined into recommendation"
    ],
    "swot": {{
        "strengths": ["strength with specific number", "strength with specific number"],
        "weaknesses": ["weakness with specific number", "weakness with specific number"],
        "opportunities": ["opportunity with specific number", "opportunity with specific number"],
        "threats": ["threat with specific number", "threat with specific number"]
    }},
    "market_entry_window": "Open" or "Cautious" or "Closed",
    "market_entry_score": 0-100,
    "market_entry_reason": "one sentence with BOTH national macro AND local numbers",
        "advisory_memo": "Write a 4-sentence strategic analysis — do NOT just list numbers, INTERPRET them. Sentence 1: identify the single biggest barrier to entry and why it matters for this specific business type. Sentence 2: find the key tension or contradiction in the data (e.g. high education but low income, fragmented HHI but saturated density) and what it implies. Sentence 3: assess what the NLP signals (Beige Book tone and local news) add to the picture that the numbers alone do not show. Sentence 4: give a clear actionable strategic recommendation — not just expand or not, but HOW the firm should position itself or what conditions would change the recommendation. Every sentence must reference at least one specific number but the insight must go beyond the number.",
    "pricing_rationale": "EXACTLY 2 sentences with numbers. Sentence 1: local income $, education %, rent burden % and pricing constraint. Sentence 2: firm type premium possible given these conditions."
}}
    """
    r = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=3000,
        system="Senior McKinsey economic strategist. Always cite specific numbers. Return ONLY valid JSON. No markdown. No extra text.",
        messages=[{"role": "user", "content": prompt}],
    )
    raw = "".join(block.text for block in r.content if block.type == "text").strip().replace("```json","").replace("```","").strip()
    print(f"Claude raw response length: {len(raw)}")
    print(f"Claude first 300 chars: {raw[:300]}")
    print(f"Claude last 200 chars: {raw[-200:]}")
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        print(f"JSON parse error. Raw response length: {len(raw)}")
        print(f"Last 200 chars: {raw[-200:]}")
        # Claude returned malformed JSON — attempt repair by truncating at last valid closing brace
        try:
            last_brace = raw.rfind("}")
            if last_brace > 0:
                return json.loads(raw[:last_brace+1])
        except:
            pass
        # Final fallback — return safe defaults
        return {
            "expansion_score": 50, "local_market_score": 50, "pricing_power_score": 50,
            "expansion_recommendation": "Hold", "recommended_price_tier": "Mid-Range",
            "confidence_level": "Low", "confidence_reason": "Analysis temporarily unavailable.",
            "macro_risk": "Moderate", "reasoning_trace": ["Analysis unavailable — please retry."],
            "key_opportunities": ["Retry analysis for full results."],
            "key_risks": ["Retry analysis for full results."],
            "swot": {"strengths": ["Retry for full analysis."], "weaknesses": ["Retry for full analysis."],
                     "opportunities": ["Retry for full analysis."], "threats": ["Retry for full analysis."]},
            "market_entry_window": "Cautious", "market_entry_score": 50,
            "market_entry_reason": "Analysis temporarily unavailable — please retry.",
            "advisory_memo": "Analysis temporarily unavailable. Please click Run Analysis again.",
            "pricing_rationale": "Please retry the analysis."
        }

# ── COLOUR HELPERS ────────────────────────────────────────────────────────────
def score_color(score):
    if score >= 70: return "#00d4aa"
    if score >= 45: return "#f5a623"
    return "#ff4757"

def gauge_fig(value, title):
    color = score_color(value)
    fig = go.Figure(go.Indicator(
        mode  = "gauge+number",
        value = value,
        title = {"text": title, "font": {"size": 13, "color": "#8892a4"}},
        number= {"font": {"size": 36, "color": color, "family": "Inter"}},
        gauge = {
            "axis":  {"range": [0, 100], "tickcolor": "#2d3748", "tickwidth": 1, "tickvals": [0, 50, 100],
                      "ticktext": ["0", "50", "100"], "tickfont": {"size": 9, "color": "#4a5568"}},
            "bar":   {"color": color, "thickness": 0.15},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [0,  45], "color": "rgba(255,71,87,0.15)"},
                {"range": [45, 70], "color": "rgba(245,166,35,0.15)"},
                {"range": [70,100], "color": "rgba(0,212,170,0.15)"},
            ],
        },
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor ="rgba(0,0,0,0)",
        margin=dict(t=50, b=10, l=30, r=30),
        height=170,
    )
    return fig

def macro_bar_fig(macro_data, scenario_macro=None):
    """National macro indicators chart — all FRED data, national level."""
    labels = ["Fed Rate", "Prime Rate", "Inflation", "GDP Growth",
              "Retail Growth", "PCE Growth", "Income Growth"]
    values = [
        macro_data.get("fed_funds_rate", 3.64),
        macro_data.get("prime_rate", 6.75),
        macro_data.get("inflation_rate", 2.83),
        macro_data.get("gdp_growth", 5.58),
        macro_data.get("retail_sales_growth", 3.72),
        macro_data.get("pce_growth", -0.44),
        macro_data.get("income_growth", 0.03),
    ]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels, y=values,
        name="Current",
        marker_color=["#ff4757" if v < 0 else "#3d7aed" for v in values],
        text=[f"{v:.2f}" for v in values],
        textposition="outside",
        textfont={"size": 10, "color": "#8892a4"},
    ))
    if scenario_macro:
        scenario_values = [
            scenario_macro.get("fed_funds_rate", values[0]),
            scenario_macro.get("prime_rate", values[1]),
            scenario_macro.get("inflation_rate", values[2]),
            scenario_macro.get("gdp_growth", values[3]),
            scenario_macro.get("retail_sales_growth", values[4]),
            scenario_macro.get("pce_growth", values[5]),
            scenario_macro.get("income_growth", values[6]),
        ]
        fig.add_trace(go.Bar(
            x=labels, y=scenario_values,
            name="Scenario",
            marker_color="rgba(0,212,170,0.6)",
            text=[f"{v:.2f}" for v in scenario_values],
            textposition="outside",
            textfont={"size": 10, "color": "#00d4aa"},
        ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor ="rgba(0,0,0,0)",
        font_color="#8892a4",
        margin=dict(t=20, b=10, l=10, r=10),
        height=260,
        barmode="group",
        legend=dict(font=dict(size=10), bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(tickangle=-30, gridcolor="rgba(0,0,0,0)",
                   tickfont=dict(size=9)),
        yaxis=dict(gridcolor="rgba(45,55,72,0.5)", tickfont=dict(size=9)),
    )
    return fig

def local_bar_fig(local_data):
    """Local market indicators chart — ZIP/county level data."""
    income      = local_data.get("median_income", 58000)
    unemp       = local_data.get("unemployment_rate", 4.2)
    rent        = local_data.get("median_rent", None)
    college     = local_data.get("pct_college", 30.0)
    prime_age   = local_data.get("pct_prime", 30.0)
    young       = local_data.get("pct_young", 10.0)

    # Rent burden — annual rent / annual income
    rent_burden = round(((rent * 12) / income) * 100, 1) if rent and income > 0 else 30.0

    labels = ["Income ($k)", "Unemployment %", "Rent Burden %",
              "College Edu %", "Prime Age %", "Young Adults %"]
    values = [
        round(income / 1000, 1),
        unemp,
        rent_burden,
        college,
        prime_age,
        young,
    ]

    # Color coding: green = good, red = concerning
    # For income: higher is better
    # For unemployment and rent burden: lower is better
    # For education/age: higher is better
    colors = [
        "#3d7aed",                              # income — blue
        "#ff4757" if unemp > 6 else "#f5a623" if unemp > 4 else "#00d4aa",  # unemployment
        "#ff4757" if rent_burden > 40 else "#f5a623" if rent_burden > 30 else "#00d4aa",  # rent burden
        "#00d4aa" if college > 40 else "#f5a623" if college > 25 else "#ff4757",  # education
        "#00d4aa" if prime_age > 30 else "#f5a623" if prime_age > 20 else "#ff4757",  # prime age
        "#00d4aa" if young > 20 else "#f5a623",  # young adults
    ]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels, y=values,
        marker_color=colors,
        text=[f"{v:.1f}" for v in values],
        textposition="outside",
        textfont={"size": 10, "color": "#8892a4"},
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor ="rgba(0,0,0,0)",
        font_color="#8892a4",
        margin=dict(t=20, b=10, l=10, r=10),
        height=260,
        legend=dict(font=dict(size=10), bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(tickangle=-30, gridcolor="rgba(0,0,0,0)",
                   tickfont=dict(size=9)),
        yaxis=dict(gridcolor="rgba(45,55,72,0.5)", tickfont=dict(size=9)),
    )
    return fig

def swot_html(swot):
    CARD   = "#111f33"
    BORDER = "#1e3a5f"
    def quadrant(title, items, border_color, title_color):
        return html.Div(style={
            "backgroundColor": CARD,
            "border": f"1px solid {border_color}",
            "borderTop": f"3px solid {border_color}",
            "borderRadius": "6px",
            "padding": "14px",
        }, children=[
            html.P(title, style={
                "color": title_color, "fontWeight": "700",
                "fontSize": "11px", "letterSpacing": "1px",
                "margin": "0 0 10px",
            }),
            html.Div([
                html.P(f"• {item}", style={
                    "fontSize": "11px", "color": "#e2e8f0",
                    "margin": "0 0 6px", "lineHeight": "1.5",
                }) for item in items
            ]),
        ])
    return html.Div(style={
        "display": "grid",
        "gridTemplateColumns": "1fr 1fr",
        "gap": "10px",
        "height": "100%",
    }, children=[
        quadrant("STRENGTHS",     swot["strengths"],    "#00d4aa", "#00d4aa"),
        quadrant("WEAKNESSES",    swot["weaknesses"],   "#ff4757", "#ff4757"),
        quadrant("OPPORTUNITIES", swot["opportunities"],"#7bed9f", "#7bed9f"),
        quadrant("THREATS",       swot["threats"],      "#f5a623", "#f5a623"),
    ])

def tone_color(tone):
    return {"hawkish": "#ff4757", "neutral": "#f5a623", "dovish": "#00d4aa"}.get(tone, "#8892a4")

# ── APP ───────────────────────────────────────────────────────────────────────
app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server
app.title = "MarketSignal"

BG     = "#070d1a"
PANEL  = "#0d1829"
CARD   = "#111f33"
BORDER = "#1e3a5f"
ACCENT = "#3d7aed"
TEXT   = "#e2e8f0"
MUTED  = "#8892a4"
GREEN  = "#00d4aa"
RED    = "#ff4757"
YELLOW = "#f5a623"

app.layout = html.Div(style={
    "backgroundColor": BG, "minHeight": "100vh",
    "fontFamily": "'Inter', 'Segoe UI', sans-serif", "color": TEXT,
    "overflowX": "hidden",
}, children=[


    # Store for score breakdowns
    dcc.Store(id="breakdown-store", data={}),
    dcc.Store(id="page-store", data="landing"),
    dcc.Store(id="zip-history-store", data={}, storage_type="local"),
    dcc.Store(id="last-fetched-store", data={}),

    # ── LANDING PAGE ──────────────────────────────────────────────────────────
    html.Div(id="landing-page", style={
        "display": "flex", "flexDirection": "column",
        "alignItems": "center", "justifyContent": "center",
        "minHeight": "100vh", "backgroundColor": BG, "padding": "40px",
    }, children=[

        # Logo and title
        html.Div(style={"textAlign": "center", "marginBottom": "40px"}, children=[
            html.Div(style={
                "width": "80px", "height": "80px", "margin": "0 auto 20px",
                "backgroundColor": CARD, "borderRadius": "20px",
                "border": f"1px solid {ACCENT}",
                "display": "flex", "alignItems": "center", "justifyContent": "center",
            }, children=[
                html.Div("MS", style={
                    "fontSize": "28px", "fontWeight": "800",
                    "color": ACCENT, "letterSpacing": "-1px",
                }),
            ]),
            html.Div("MarketSignal", style={
                "fontSize": "48px", "fontWeight": "800",
                "color": TEXT, "letterSpacing": "-2px", "lineHeight": "1",
                "marginBottom": "10px",
            }),
            html.Div("Agentic Market Expansion & Pricing Advisor", style={
                "fontSize": "15px", "color": MUTED, "marginBottom": "6px",
            }),
            html.Div("Federal Reserve Data · Census ACS · BLS · Zillow · NLP Intelligence", style={
                "fontSize": "10px", "color": MUTED, "opacity": "0.5",
            }),
        ]),

        # Feature pills
        html.Div(style={
            "display": "flex", "gap": "10px", "marginBottom": "40px",
            "flexWrap": "wrap", "justifyContent": "center",
        }, children=[
            html.Div(pill, style={
                "backgroundColor": CARD, "border": f"1px solid {BORDER}",
                "borderRadius": "20px", "padding": "5px 14px",
                "fontSize": "10px", "color": MUTED,
            }) for pill in [
                "Real-Time FRED Data", "ZIP-Level Analysis",
                "NLP Intelligence", "Bresnahan-Reiss Entry Model",
                "Huff Gravity Model", "Beige Book Sentiment",
            ]
        ]),

        # Input card
        html.Div(style={
            "backgroundColor": PANEL, "borderRadius": "16px",
            "padding": "36px", "border": f"1px solid {BORDER}",
            "width": "100%", "maxWidth": "480px",
            "boxShadow": "0 20px 60px rgba(0,0,0,0.3)",
        }, children=[
            html.P("CONFIGURE YOUR ANALYSIS", style={
                "fontSize": "10px", "fontWeight": "700",
                "color": MUTED, "letterSpacing": "2px", "margin": "0 0 20px",
            }),

            html.Label("Business Type", style={"fontSize": "11px", "color": MUTED, "display": "block", "marginBottom": "5px"}),
            dcc.Input(id="landing-business", type="text",
                      placeholder="e.g. Coffee Shop, Gym, Restaurant",
                      style={"width": "100%", "backgroundColor": CARD,
                             "border": f"1px solid {BORDER}", "borderRadius": "8px",
                             "padding": "10px 14px", "color": TEXT, "fontSize": "13px",
                             "marginBottom": "14px", "boxSizing": "border-box"}),

            html.Label("Target ZIP Code", style={"fontSize": "11px", "color": MUTED, "display": "block", "marginBottom": "5px"}),
            dcc.Input(id="landing-zip", type="text", placeholder="e.g. 45219",
                      style={"width": "100%", "backgroundColor": CARD,
                             "border": f"1px solid {BORDER}", "borderRadius": "8px",
                             "padding": "10px 14px", "color": TEXT, "fontSize": "13px",
                             "marginBottom": "14px", "boxSizing": "border-box"}),

            html.Label("Firm Type", style={"fontSize": "11px", "color": MUTED, "display": "block", "marginBottom": "5px"}),
            dcc.Dropdown(id="landing-firm-type",
                options=[
                    {"label": "Independent", "value": "Independent"},
                    {"label": "Regional Chain", "value": "Regional Chain"},
                    {"label": "National Brand", "value": "National Brand"},
                ],
                value="Independent",
                style={"backgroundColor": CARD, "borderRadius": "8px",
                       "color": "#0f1923", "marginBottom": "14px"},
            ),

            html.Label("Price Tier", style={"fontSize": "11px", "color": MUTED, "display": "block", "marginBottom": "5px"}),
            dcc.Dropdown(id="landing-price-tier",
                options=[
                    {"label": "$ Budget", "value": "Budget"},
                    {"label": "$$ Mid-Range", "value": "Mid-Range"},
                    {"label": "$$$ Premium", "value": "Premium"},
                ],
                value="Mid-Range",
                style={"backgroundColor": CARD, "borderRadius": "8px",
                       "color": "#0f1923", "marginBottom": "24px"},
            ),

            html.Button("Run Analysis →", id="landing-run-btn", n_clicks=0,
                style={
                    "width": "100%", "backgroundColor": ACCENT,
                    "color": "#fff", "border": "none", "borderRadius": "10px",
                    "padding": "14px", "fontSize": "14px", "fontWeight": "700",
                    "cursor": "pointer", "letterSpacing": "0.5px",
                }),
        ]),

        html.Div("Built for AI Challenge 2026 · Anthropic Claude · Economic Research Framework", style={
            "fontSize": "10px", "color": MUTED, "opacity": "0.4",
            "marginTop": "28px", "textAlign": "center",
        }),
    ]),


    # ── HEADER ────────────────────────────────────────────────────────────────
    html.Div(style={
        "borderBottom": f"1px solid {BORDER}",
        "padding": "16px 40px",
        "display": "flex", "justifyContent": "space-between", "alignItems": "center",
        "backgroundColor": PANEL,
    }, children=[
        html.Div([
            html.Div([
                html.Div("MarketSignal", style={
                    "fontSize": "22px", "fontWeight": "700",
                    "color": TEXT, "letterSpacing": "-0.5px",
                }),
                html.Div("Agentic Market Expansion & Pricing Advisor", style={
                    "fontSize": "11px", "color": MUTED, "marginTop": "2px",
                }),
            ]),
        ]),
        html.Div(id="live-time", style={"fontSize": "11px", "color": MUTED}),
        dcc.Interval(id="clock", interval=1000, n_intervals=0),
    ]),

    html.Div(style={"maxWidth": "1400px", "margin": "0 auto", "padding": "24px 32px"}, children=[

        # ── TWO COLUMN LAYOUT ─────────────────────────────────────────────────
        html.Div(style={"display": "grid", "gridTemplateColumns": "340px 1fr",
                        "gap": "20px", "alignItems": "start"}, children=[

            # ── LEFT SIDEBAR ──────────────────────────────────────────────────
            html.Div([

                # Firm Inputs
                html.Div(style={
                    "backgroundColor": PANEL, "borderRadius": "10px",
                    "padding": "20px", "border": f"1px solid {BORDER}",
                    "marginBottom": "16px",
                }, children=[
                    html.P("FIRM INPUTS", style={
                        "fontSize": "10px", "fontWeight": "700",
                        "color": ACCENT, "letterSpacing": "1.5px", "margin": "0 0 16px",
                    }),
                    html.Div([
                        html.Label("Business Type", style={"fontSize": "11px", "color": MUTED, "marginBottom": "6px", "display": "block"}),
                        dcc.Input(id="business-type", value="Coffee Shop", type="text", style={
                            "width": "100%", "padding": "10px 12px",
                            "borderRadius": "6px", "backgroundColor": CARD,
                            "color": TEXT, "border": f"1px solid {BORDER}",
                            "fontSize": "13px", "boxSizing": "border-box",
                        }),
                    ], style={"marginBottom": "14px"}),
                    html.Div([
                        html.Label("Price Tier", style={"fontSize": "11px", "color": MUTED, "marginBottom": "6px", "display": "block"}),
                        dcc.Dropdown(id="price-tier",
                            options=[
                                {"label": "$ Budget",     "value": "Budget"},
                                {"label": "$$ Mid-Range", "value": "Mid-Range"},
                                {"label": "$$$ Premium",  "value": "Premium"},
                            ],
                            value="Mid-Range",
                            style={"backgroundColor": CARD, "borderRadius": "6px", "color": "#0f1923"},
                        ),
                    ], style={"marginBottom": "14px"}),
                    html.Div([
                        html.Label("Firm Type", style={"fontSize": "11px", "color": MUTED, "marginBottom": "6px", "display": "block"}),
                        dcc.Dropdown(id="firm-type",
                            options=[
                                {"label": "Independent",      "value": "Independent"},
                                {"label": "Regional Chain",   "value": "Regional Chain"},
                                {"label": "National Brand",   "value": "National Brand"},
                            ],
                            value="Independent",
                            style={"backgroundColor": CARD, "borderRadius": "6px", "color": "#0f1923"},
                        ),
                    ], style={"marginBottom": "14px"}),
                                                                                html.Div([
                        html.Label("Target ZIP Code", style={"fontSize": "11px", "color": MUTED, "marginBottom": "6px", "display": "block"}),
                        dcc.Input(id="zip-code", value="45219", type="text", style={
                            "width": "100%", "padding": "10px 12px",
                            "borderRadius": "6px", "backgroundColor": CARD,
                            "color": TEXT, "border": f"1px solid {BORDER}",
                            "fontSize": "13px", "boxSizing": "border-box",
                        }),
                    ], style={"marginBottom": "20px"}),
                    html.Button("▶  Run Analysis", id="run-btn", style={
                        "width": "100%", "backgroundColor": ACCENT,
                        "color": "white", "border": "none", "borderRadius": "6px",
                        "padding": "12px", "cursor": "pointer",
                        "fontSize": "13px", "fontWeight": "600",
                        "letterSpacing": "0.3px",
                    }),
                ]),

                # Scenario Panel
                html.Div(style={
                    "backgroundColor": PANEL, "borderRadius": "10px",
                    "padding": "20px", "border": f"1px solid {BORDER}",
                    "marginBottom": "16px",
                }, children=[
                    html.P("SCENARIO SIMULATOR", style={
                        "fontSize": "10px", "fontWeight": "700",
                        "color": YELLOW, "letterSpacing": "1.5px", "margin": "0 0 4px",
                    }),
                    html.P("Adjust variables to simulate alternative conditions",
                           style={"fontSize": "10px", "color": MUTED, "margin": "0 0 16px"}),

                    # National Variables
                    html.P("NATIONAL INDICATORS", style={
                        "fontSize": "9px", "color": ACCENT, "letterSpacing": "1px",
                        "fontWeight": "700", "margin": "0 0 8px"
                    }),

                    html.Label("Fed Funds Rate (%)", style={"fontSize": "11px", "color": MUTED}),
                    dcc.Slider(id="s-fed", min=0, max=8, step=0.25, value=3.64,
                               marks={0: {"label":"0%","style":{"color":"#8892a4"}}, 4: {"label":"4%","style":{"color":"#8892a4"}}, 8: {"label":"8%","style":{"color":"#8892a4"}}},
                               tooltip={"placement": "bottom", "always_visible": True, "style": {"color": "#0f1923", "backgroundColor": "#e2e8f0"}}),

                    html.Label("Prime Rate (%)", style={"fontSize": "11px", "color": MUTED, "marginTop": "12px", "display": "block"}),
                    dcc.Slider(id="s-prime", min=2, max=12, step=0.25, value=6.75,
                               marks={2: {"label":"2%","style":{"color":"#8892a4"}}, 6: {"label":"6%","style":{"color":"#8892a4"}}, 12: {"label":"12%","style":{"color":"#8892a4"}}},
                               tooltip={"placement": "bottom", "always_visible": True, "style": {"color": "#0f1923", "backgroundColor": "#e2e8f0"}}),

                    html.Label("Inflation Rate (%)", style={"fontSize": "11px", "color": MUTED, "marginTop": "12px", "display": "block"}),
                    dcc.Slider(id="s-inflation", min=0, max=10, step=0.25, value=2.83,
                               marks={0: {"label":"0%","style":{"color":"#8892a4"}}, 5: {"label":"5%","style":{"color":"#8892a4"}}, 10: {"label":"10%","style":{"color":"#8892a4"}}},
                               tooltip={"placement": "bottom", "always_visible": True, "style": {"color": "#0f1923", "backgroundColor": "#e2e8f0"}}),



                    html.Label("GDP Growth (%)", style={"fontSize": "11px", "color": MUTED, "marginTop": "12px", "display": "block"}),
                    dcc.Slider(id="s-gdp", min=-5, max=10, step=0.25, value=5.58,
                               marks={-5: {"label":"-5%","style":{"color":"#8892a4"}}, 0: {"label":"0%","style":{"color":"#8892a4"}}, 5: {"label":"5%","style":{"color":"#8892a4"}}, 10: {"label":"10%","style":{"color":"#8892a4"}}},
                               tooltip={"placement": "bottom", "always_visible": True, "style": {"color": "#0f1923", "backgroundColor": "#e2e8f0"}}),

                    html.Label("Retail Sales Growth (%)", style={"fontSize": "11px", "color": MUTED, "marginTop": "12px", "display": "block"}),
                    dcc.Slider(id="s-retail", min=-5, max=10, step=0.25, value=3.72,
                               marks={-5: {"label":"-5%","style":{"color":"#8892a4"}}, 0: {"label":"0%","style":{"color":"#8892a4"}}, 5: {"label":"5%","style":{"color":"#8892a4"}}, 10: {"label":"10%","style":{"color":"#8892a4"}}},
                               tooltip={"placement": "bottom", "always_visible": True, "style": {"color": "#0f1923", "backgroundColor": "#e2e8f0"}}),

                    html.Label("PCE Growth (%)", style={"fontSize": "11px", "color": MUTED, "marginTop": "12px", "display": "block"}),
                    dcc.Slider(id="s-pce", min=-5, max=8, step=0.25, value=-0.29,
                               marks={-5: {"label":"-5%","style":{"color":"#8892a4"}}, 0: {"label":"0%","style":{"color":"#8892a4"}}, 4: {"label":"4%","style":{"color":"#8892a4"}}, 8: {"label":"8%","style":{"color":"#8892a4"}}},
                               tooltip={"placement": "bottom", "always_visible": True, "style": {"color": "#0f1923", "backgroundColor": "#e2e8f0"}}),

                    html.Label("Disposable Income Growth (%)", style={"fontSize": "11px", "color": MUTED, "marginTop": "12px", "display": "block"}),
                    dcc.Slider(id="s-income", min=-5, max=8, step=0.25, value=0.6,
                               marks={-5: {"label":"-5%","style":{"color":"#8892a4"}}, 0: {"label":"0%","style":{"color":"#8892a4"}}, 4: {"label":"4%","style":{"color":"#8892a4"}}, 8: {"label":"8%","style":{"color":"#8892a4"}}},
                               tooltip={"placement": "bottom", "always_visible": True, "style": {"color": "#0f1923", "backgroundColor": "#e2e8f0"}}),

                    # Local Variables
                    html.P("LOCAL MARKET INDICATORS", style={
                        "fontSize": "9px", "color": YELLOW, "letterSpacing": "1px",
                        "fontWeight": "700", "margin": "16px 0 8px"
                    }),

                    html.Label("Local Unemployment (%)", style={"fontSize": "11px", "color": MUTED}),
                    dcc.Slider(id="s-unemployment", min=1, max=15, step=0.5, value=4.55,
                               marks={1: {"label":"1%","style":{"color":"#8892a4"}}, 5: {"label":"5%","style":{"color":"#8892a4"}}, 15: {"label":"15%","style":{"color":"#8892a4"}}},
                               tooltip={"placement": "bottom", "always_visible": True, "style": {"color": "#0f1923", "backgroundColor": "#e2e8f0"}}),

                    html.Label("Local Median Income ($)", style={"fontSize": "11px", "color": MUTED, "marginTop": "12px", "display": "block"}),
                    dcc.Slider(id="s-income-local", min=20000, max=150000, step=1000, value=58000,
                               marks={20000: {"label":"$20k","style":{"color":"#8892a4"}}, 74580: {"label":"$75k","style":{"color":"#8892a4"}}, 150000: {"label":"$150k","style":{"color":"#8892a4"}}},
                               tooltip={"placement": "bottom", "always_visible": True, "style": {"color": "#0f1923", "backgroundColor": "#e2e8f0"}}),

                    html.Label("Local Population", style={"fontSize": "11px", "color": MUTED, "marginTop": "12px", "display": "block"}),
                    dcc.Slider(id="s-population", min=5000, max=100000, step=1000, value=45000,
                               marks={5000: {"label":"5k","style":{"color":"#8892a4"}}, 50000: {"label":"50k","style":{"color":"#8892a4"}}, 100000: {"label":"100k","style":{"color":"#8892a4"}}},
                               tooltip={"placement": "bottom", "always_visible": True, "style": {"color": "#0f1923", "backgroundColor": "#e2e8f0"}}),

                    # ── NEW LOCAL INDICATORS ──────────────────────────────
                    html.P("LOCAL ENRICHMENT INDICATORS", style={
                        "fontSize": "9px", "color": GREEN, "letterSpacing": "1px",
                        "fontWeight": "700", "margin": "16px 0 8px"
                    }),

                    html.Label("Median Rent ($/month)", style={"fontSize": "11px", "color": MUTED}),
                    dcc.Slider(id="s-rent", min=500, max=3000, step=50, value=1200,
                               marks={500: {"label":"$500","style":{"color":"#8892a4"}},
                                      1500: {"label":"$1.5k","style":{"color":"#8892a4"}},
                                      3000: {"label":"$3k","style":{"color":"#8892a4"}}},
                               tooltip={"placement": "bottom", "always_visible": True,
                                        "style": {"color": "#0f1923", "backgroundColor": "#e2e8f0"}}),

                    html.Label("Competitor Density (per 10K people)", style={"fontSize": "11px", "color": MUTED, "marginTop": "12px", "display": "block"}),
                    dcc.Slider(id="s-density", min=0, max=50, step=0.5, value=10,
                               marks={0: {"label":"0","style":{"color":"#8892a4"}},
                                      10: {"label":"10","style":{"color":"#8892a4"}},
                                      30: {"label":"30","style":{"color":"#8892a4"}},
                                      50: {"label":"50","style":{"color":"#8892a4"}}},
                               tooltip={"placement": "bottom", "always_visible": True,
                                        "style": {"color": "#0f1923", "backgroundColor": "#e2e8f0"}}),

                    html.Label("% College Educated", style={"fontSize": "11px", "color": MUTED, "marginTop": "12px", "display": "block"}),
                    dcc.Slider(id="s-college", min=0, max=80, step=1, value=30,
                               marks={0: {"label":"0%","style":{"color":"#8892a4"}},
                                      40: {"label":"40%","style":{"color":"#8892a4"}},
                                      80: {"label":"80%","style":{"color":"#8892a4"}}},
                               tooltip={"placement": "bottom", "always_visible": True,
                                        "style": {"color": "#0f1923", "backgroundColor": "#e2e8f0"}}),

                    html.Label("Neighborhood Age Profile", style={"fontSize": "11px", "color": MUTED, "marginTop": "12px", "display": "block"}),
                    dcc.Dropdown(id="s-age-profile",
                        options=[
                            {"label": "Young / University Area", "value": "young"},
                            {"label": "Prime Age / Professional Area", "value": "prime"},
                            {"label": "Senior / Retirement Area", "value": "senior"},
                            {"label": "Mixed / Balanced", "value": "mixed"},
                        ],
                        value="mixed",
                        style={"backgroundColor": CARD, "borderRadius": "6px",
                               "color": "#0f1923", "marginTop": "4px"},
                    ),

                    html.Div(style={"marginTop": "16px", "display": "flex", "gap": "8px"}, children=[
                        html.Button("Apply Scenario", id="scenario-btn", style={
                            "flex": "1", "backgroundColor": YELLOW,
                            "color": "#000", "border": "none", "borderRadius": "6px",
                            "padding": "10px", "cursor": "pointer",
                            "fontSize": "12px", "fontWeight": "600",
                        }),
                        html.Button("Reset", id="reset-btn", style={
                            "backgroundColor": CARD, "color": MUTED,
                            "border": f"1px solid {BORDER}", "borderRadius": "6px",
                            "padding": "10px 14px", "cursor": "pointer",
                            "fontSize": "12px",
                        }),
                    ]),
                ]),

                # Agent Thinking Panel
                html.Div(style={
                    "backgroundColor": PANEL, "borderRadius": "10px",
                    "padding": "20px", "border": f"1px solid {BORDER}",
                }, children=[
                    html.P("AGENT REASONING TRACE", style={
                        "fontSize": "10px", "fontWeight": "700",
                        "color": GREEN, "letterSpacing": "1.5px", "margin": "0 0 12px",
                    }),
                    html.Div(id="reasoning-trace", style={
                        "fontFamily": "monospace", "fontSize": "11px",
                        "color": MUTED, "lineHeight": "1.8",
                    }, children=[
                        html.P("Waiting for analysis...", style={"color": MUTED, "fontStyle": "italic"})
                    ]),
                ]),

            ]),

            # ── RIGHT MAIN PANEL ──────────────────────────────────────────────
            html.Div([

                dcc.Loading(color=ACCENT, type="dot", children=[

                    # KPI Row
                    html.Div(id="kpi-row", style={
                        "display": "grid", "gridTemplateColumns": "1fr 1fr 1fr 1fr",
                        "gap": "12px", "marginBottom": "16px",
                    }),

                    # Recommendation Banner
                    html.Div(id="rec-banner", style={"marginBottom": "16px"}),

                    # Charts Row
                    html.Div(style={
                        "display": "grid", "gridTemplateColumns": "1fr 1fr",
                        "gap": "16px", "marginBottom": "16px",
                    }, children=[
                        html.Div(style={
                            "backgroundColor": PANEL, "borderRadius": "10px",
                            "padding": "16px", "border": f"1px solid {BORDER}",
                        }, children=[
                            html.P("NATIONAL MACRO INDICATORS", style={
                                "fontSize": "10px", "fontWeight": "700",
                                "color": MUTED, "letterSpacing": "1.5px", "margin": "0 0 8px",
                            }),
                            html.P("Source: FRED — Federal Reserve Bank of St. Louis", style={
                                "fontSize": "9px", "color": MUTED, "margin": "0 0 8px", "opacity": "0.7",
                            }),
                            dcc.Graph(id="macro-bar", config={"displayModeBar": False}),
                        ]),
                        html.Div(style={
                            "backgroundColor": PANEL, "borderRadius": "10px",
                            "padding": "16px", "border": f"1px solid {BORDER}",
                        }, children=[
                            html.P("LOCAL MARKET INDICATORS", style={
                                "fontSize": "10px", "fontWeight": "700",
                                "color": GREEN, "letterSpacing": "1.5px", "margin": "0 0 8px",
                            }),
                            html.P("Source: Census ACS 2023 | BLS LAUS | Zillow ZORI", style={
                                "fontSize": "9px", "color": MUTED, "margin": "0 0 8px", "opacity": "0.7",
                            }),
                            dcc.Graph(id="local-bar", config={"displayModeBar": False}),
                        ]),
                        html.Div(style={
                            "backgroundColor": PANEL, "borderRadius": "10px",
                            "padding": "16px", "border": f"1px solid {BORDER}",
                        }, children=[
                            html.P("SWOT ANALYSIS", style={
                                "fontSize": "10px", "fontWeight": "700",
                                "color": MUTED, "letterSpacing": "1.5px", "margin": "0 0 8px",
                            }),
                            html.Div(id="swot-chart"),
                        ]),
                        html.Div(style={
                            "backgroundColor": PANEL, "borderRadius": "10px",
                            "padding": "16px", "border": f"1px solid {BORDER}",
                        }, children=[
                            html.P("SCORE BREAKDOWN", style={
                                "fontSize": "10px", "fontWeight": "700",
                                "color": ACCENT, "letterSpacing": "1.5px", "margin": "0 0 8px",
                            }),
                            html.P("What is driving each score", style={
                                "fontSize": "9px", "color": MUTED, "margin": "0 0 12px",
                            }),
                            dcc.Tabs(id="breakdown-tabs", value="expansion", style={
                                "fontSize": "10px",
                            }, children=[
                                dcc.Tab(label="Expansion", value="expansion",
                                    style={"padding": "4px 8px", "fontSize": "10px"},
                                    selected_style={"padding": "4px 8px", "fontSize": "10px",
                                                   "backgroundColor": ACCENT, "color": "#fff",
                                                   "borderTop": f"2px solid {ACCENT}"}),
                                dcc.Tab(label="Local Market", value="local",
                                    style={"padding": "4px 8px", "fontSize": "10px"},
                                    selected_style={"padding": "4px 8px", "fontSize": "10px",
                                                   "backgroundColor": ACCENT, "color": "#fff",
                                                   "borderTop": f"2px solid {ACCENT}"}),
                                dcc.Tab(label="Pricing Power", value="pricing",
                                    style={"padding": "4px 8px", "fontSize": "10px"},
                                    selected_style={"padding": "4px 8px", "fontSize": "10px",
                                                   "backgroundColor": ACCENT, "color": "#fff",
                                                   "borderTop": f"2px solid {ACCENT}"}),
                                dcc.Tab(label="Entry Window", value="mew",
                                    style={"padding": "4px 8px", "fontSize": "10px"},
                                    selected_style={"padding": "4px 8px", "fontSize": "10px",
                                                   "backgroundColor": ACCENT, "color": "#fff",
                                                   "borderTop": f"2px solid {ACCENT}"}),
                            ]),
                            html.Div(id="breakdown-chart"),
                        ]),
                    ]),

                    # FOMC + Local Row
                    html.Div(style={
                        "display": "grid", "gridTemplateColumns": "1fr 1fr",
                        "gap": "16px", "marginBottom": "16px",
                    }, children=[
                        html.Div(id="fomc-panel", style={
                            "backgroundColor": PANEL, "borderRadius": "10px",
                            "padding": "20px", "border": f"1px solid {BORDER}",
                        }),
                        html.Div(id="local-panel", style={
                            "backgroundColor": PANEL, "borderRadius": "10px",
                            "padding": "20px", "border": f"1px solid {BORDER}",
                        }),
                    ]),
                    html.Div(id="nlp-panel", style={
                        "backgroundColor": PANEL, "borderRadius": "10px",
                        "padding": "20px", "border": f"1px solid {BORDER}",
                        "marginBottom": "16px",
                    }),

                    # Advisory Memo
                    html.Div(id="competitor-panel", style={"backgroundColor": PANEL, "borderRadius": "10px", "padding": "20px", "border": f"1px solid {BORDER}", "marginBottom": "16px"}),

                    html.Div(id="memo-panel", style={"marginBottom": "16px"}),
                    html.Div(id="history-panel", style={"marginBottom": "16px"}),

                ]),
            ]),
        ]),
    ]),
]) # end app.layout

# ── SCORE BREAKDOWN TAB CALLBACK ─────────────────────────────────────────────
@app.callback(
    Output("breakdown-chart", "children"),
    Input("breakdown-tabs", "value"),
    Input("breakdown-store", "data"),
)
def switch_breakdown_tab(tab, store_data):
    if not store_data or not tab:
        return html.Div("Run analysis to see score breakdown", style={
            "fontSize": "11px", "color": "#8892a4",
            "textAlign": "center", "padding": "20px", "fontStyle": "italic",
        })

    breakdown_dict = store_data.get(tab, {})
    score = store_data.get("scores", {}).get(tab, 0)

    print(f"DEBUG tab={tab} breakdown_dict={breakdown_dict} score={score}")

    if not breakdown_dict:
        return html.Div("Run analysis to see score breakdown", style={
            "fontSize": "11px", "color": "#8892a4",
            "textAlign": "center", "padding": "20px", "fontStyle": "italic",
        })

    breakdown_dict = store_data.get(tab, {})
    score = store_data.get("scores", {}).get(tab, 0)

    bars = []
    for label, value in breakdown_dict.items():
        if isinstance(value, list) and len(value) == 2:
            earned, maximum = value
        else:
            continue
        pct = earned / maximum if maximum > 0 else 0
        color = "#00d4aa" if pct >= 0.7 else ("#f5a623" if pct >= 0.4 else "#ff4757")
        bars.append(
            html.Div(style={"marginBottom": "8px"}, children=[
                html.Div(style={"display": "flex", "justifyContent": "space-between",
                               "marginBottom": "2px"}, children=[
                    html.Span(label, style={"fontSize": "9px", "color": "#8892a4"}),
                    html.Span(f"{earned}/{maximum}",
                             style={"fontSize": "9px", "color": color, "fontWeight": "600"}),
                ]),
                html.Div(style={"backgroundColor": "#1e2533", "borderRadius": "3px",
                               "height": "6px", "width": "100%"}, children=[
                    html.Div(style={
                        "backgroundColor": color,
                        "width": f"{max(2, round(pct*100))}%",
                        "height": "6px",
                        "borderRadius": "3px",
                    }),
                ]),
            ])
        )

    return html.Div([
        html.Div(style={"display": "flex", "justifyContent": "space-between",
                       "marginBottom": "12px", "paddingBottom": "8px",
                       "borderBottom": "1px solid #2d3748"}, children=[
            html.Span("Total Score", style={"fontSize": "11px", "color": "#8892a4"}),
            html.Span(f"{score}/100",
                     style={"fontSize": "14px", "fontWeight": "700",
                            "color": "#00d4aa" if score >= 60 else "#f5a623" if score >= 40 else "#ff4757"}),
        ]),
        *bars
    ])

# ── LANDING PAGE TRANSITION ──────────────────────────────────────────────────
@app.callback(
    Output("landing-page",      "style"),
    Output("business-type",     "value"),
    Output("zip-code",          "value"),
    Output("firm-type",         "value"),
    Output("price-tier",        "value"),
    Output("run-btn",           "n_clicks"),
    Input("landing-run-btn",    "n_clicks"),
    State("landing-business",   "value"),
    State("landing-zip",        "value"),
    State("landing-firm-type",  "value"),
    State("landing-price-tier", "value"),
    prevent_initial_call=True,
)
def go_to_dashboard(n, business, zip_code, firm_type, price_tier):
    if not n or not business or not zip_code:
        raise PreventUpdate
    return (
        {"display": "none"},
        business, zip_code, firm_type, price_tier, 1
    )

# ── CLOCK CALLBACK ────────────────────────────────────────────────────────────
@app.callback(Output("live-time", "children"), Input("clock", "n_intervals"))
def update_clock(n):
    now = datetime.now()
    return html.Div([
        html.Div(now.strftime("%A, %B %d %Y"), style={"fontSize": "12px", "color": MUTED}),
        html.Div(now.strftime("%H:%M:%S"), style={"fontSize": "12px", "color": MUTED}),
    ])

@app.callback(
    [Output("s-fed", "value"), Output("s-prime", "value"),
     Output("s-inflation", "value"), Output("s-gdp", "value"),
     Output("s-retail", "value"), Output("s-pce", "value"),
     Output("s-income", "value"), Output("s-unemployment", "value"),
     Output("s-income-local", "value"), Output("s-population", "value"),
     Output("s-rent", "value"), Output("s-density", "value"),
     Output("s-college", "value"), Output("s-age-profile", "value")],
    Input("reset-btn", "n_clicks"),
    prevent_initial_call=True,
)
def reset_sliders(n):
    return 3.64, 6.75, 2.83, 5.58, 3.72, -0.29, 0.6, 4.55, 58000, 45000, 1200, 10, 30, "mixed"

# ── MAIN ANALYSIS CALLBACK ────────────────────────────────────────────────────
@app.callback(
    [Output("kpi-row",          "children"),
     Output("rec-banner",       "children"),
     Output("macro-bar",        "figure"),
     Output("local-bar",        "figure"),
     Output("swot-chart",       "children"),
     Output("breakdown-tabs",   "value"),
     Output("breakdown-store",  "data"),
     Output("fomc-panel",       "children"),
     Output("local-panel",      "children"),
     Output("nlp-panel",        "children"),
     Output("memo-panel",       "children"),
     Output("reasoning-trace",  "children"),
     Output("competitor-panel", "children"),
     Output("zip-history-store",  "data"),
     Output("last-fetched-store", "data")],
    [Input("run-btn",      "n_clicks"),
     Input("scenario-btn", "n_clicks")],
    [State("business-type",   "value"),
     State("price-tier",      "value"),
     State("zip-code",        "value"),
     State("firm-type",        "value"),
     State("s-fed",           "value"),
     State("s-prime",         "value"),
     State("s-inflation",     "value"),
     State("s-gdp",           "value"),
     State("s-retail",        "value"),
     State("s-pce",           "value"),
     State("s-income",        "value"),
     State("s-unemployment",  "value"),
     State("s-income-local",  "value"),
     State("s-population",    "value"),
     State("s-rent",          "value"),
     State("s-density",       "value"),
     State("s-college",       "value"),
     State("s-age-profile",   "value"),
     State("zip-history-store", "data"),
     State("last-fetched-store", "data")],
    prevent_initial_call=True,
)
def run_analysis(run_clicks, scenario_clicks,
                 business_type, price_tier, zip_code, firm_type,
                 s_fed, s_prime, s_inflation, s_gdp,
                 s_retail, s_pce, s_income, s_unemployment, s_income_local, s_population,
                 s_rent, s_density, s_college, s_age_profile,
                 zip_history=None, last_fetched=None):

    ctx = callback_context
    triggered_id = ctx.triggered[0]["prop_id"] if ctx.triggered else ""
    is_scenario = triggered_id == "scenario-btn.n_clicks"
    if not run_clicks and not is_scenario:
        raise PreventUpdate

    # ── Build overrides if scenario mode ──────────────────────────────────
    overrides = {}
    if is_scenario:
        # Only override variables that differ from last fetched values
        # This ensures scenario changes ONLY the variable you moved
        overrides = {
            "fed_funds_rate":     s_fed,
            "prime_rate":         s_prime,
            "inflation_rate":     s_inflation,
            "gdp_growth":         s_gdp,
            "retail_sales_growth":s_retail,
            "pce_growth":         s_pce,
            "income_growth":      s_income,
        }

    # ── Fetch data in PARALLEL (MUSCLE) ──────────────────────────────────
    # ThreadPoolExecutor runs all IO-bound fetches simultaneously
    # Reduces load time from ~25s sequential to ~8s parallel
    import time
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=5) as executor:
        f_macro  = executor.submit(fetch_macro_data, overrides)
        f_fomc   = executor.submit(fetch_fomc_statement)
        f_local  = executor.submit(fetch_local_data, zip_code)
        f_zillow = executor.submit(fetch_zillow_rent, zip_code)
        f_state  = executor.submit(get_state_from_zip, zip_code)

        macro                    = f_macro.result()
        fomc_text, fomc_link     = f_fomc.result()
        local                    = f_local.result()
        zillow_rent, rent_trend, rent_date = f_zillow.result()
        state_abbr               = f_state.result()

    fomc = analyze_fomc_tone(fomc_text)
    print(f"First parallel fetch: {time.time()-t0:.1f}s")

    # Apply local overrides
    if is_scenario:
        # Only override if slider value differs from actual fetched value
        # This ensures we only change what the user intentionally moved
        # Use last fetched values as baseline if available
        actual_unemp = (last_fetched or {}).get("unemployment", local.get("unemployment_rate", 4.55))
        actual_income = (last_fetched or {}).get("income_local", local.get("median_income", 58000))
        actual_pop = (last_fetched or {}).get("population", local.get("population", 45000))
        actual_rent = (last_fetched or {}).get("rent", local.get("median_rent", 1200) or 1200)
        if abs(s_unemployment - actual_unemp) > 0.1:
            local["unemployment_rate"] = s_unemployment
        if abs(s_income_local - actual_income) > 100:
            local["median_income"] = s_income_local
        if abs(s_population - actual_pop) > 100:
            local["population"] = s_population
        if abs(s_rent - actual_rent) > 10:
            local["median_rent"] = s_rent

        # Age profile override
        age_profiles = {
            "young":  {"pct_children": 5,  "pct_young": 45, "pct_prime": 20, "pct_middle": 10, "pct_seniors": 10, "pct_college": 55},
            "prime":  {"pct_children": 15, "pct_young": 10, "pct_prime": 40, "pct_middle": 25, "pct_seniors": 10, "pct_college": 45},
            "senior": {"pct_children": 8,  "pct_young": 7,  "pct_prime": 20, "pct_middle": 25, "pct_seniors": 40, "pct_college": 25},
            "mixed":  {"pct_children": 20, "pct_young": 15, "pct_prime": 28, "pct_middle": 22, "pct_seniors": 15, "pct_college": 30},
        }
        profile = age_profiles.get(s_age_profile or "mixed", age_profiles["mixed"])
        for k, v in profile.items():
            local[k] = v
        local["pct_college"] = max(local["pct_college"], s_college)

    firm = {"business_type": business_type, "price_tier": price_tier, "target_zip": zip_code, "firm_type": firm_type}
    # Will be updated with scores after calculate_scores_deterministic runs
    # ── Apply Zillow result (fetched in first parallel block) ────────────
    if zillow_rent:
        local["median_rent"]    = zillow_rent
        local["rent_trend_pct"] = rent_trend
        local["rent_date"]      = rent_date
    else:
        local["median_rent"]    = None
        local["rent_trend_pct"] = 0
        local["rent_date"]      = "unavailable"

    # ── Second parallel fetch: BLS + Beige + News + Competitors ─────────
    t1 = time.time()
    district, beige_url = get_fed_district(state_abbr)
    city_name = local.get("city", zip_code)

    with ThreadPoolExecutor(max_workers=4) as executor:
        f_bls         = executor.submit(fetch_bls_unemployment, zip_code, state_abbr)
        f_beige       = executor.submit(fetch_beige_book_sentiment, district, beige_url)
        f_news        = executor.submit(fetch_local_news_sentiment, city_name, business_type)
        f_competitors = executor.submit(fetch_competitor_data_with_density, zip_code, business_type, local.get("population", 19326))

        bls_unemp, bls_period = f_bls.result()
        beige                 = f_beige.result()
        news                  = f_news.result()
        competitor_data       = f_competitors.result()

    # Competitor density override — AFTER fetch
    if is_scenario and s_density is not None:
        competitor_data["density_per_10k"] = s_density
        count = round(s_density * local.get("population", 45000) / 10000)
        competitor_data["competitor_count"] = count
        competitor_data["competitor_level"] = "Low" if s_density < 4 else ("Moderate" if s_density < 8 else "High")

    print(f"Second parallel fetch: {time.time()-t1:.1f}s")

    if bls_unemp is not None and not is_scenario:
        local["unemployment_rate"]   = bls_unemp
        local["unemployment_source"] = f"BLS LAUS {bls_period}"
    else:
        local["unemployment_source"] = "Census ACS 2023"

    local["beige_tone"]       = beige["tone"]
    local["beige_score"]      = beige["score"]
    local["beige_district"]   = beige.get("district", district)
    local["beige_summary"]    = beige.get("summary", "")
    local["beige_source_url"] = beige.get("source_url", "https://www.federalreserve.gov/monetarypolicy/beige-book-default.htm")
    local["beige_pos"]        = beige.get("pos_count", 0)
    local["beige_neg"]        = beige.get("neg_count", 0)

    local["news_tone"]       = news["tone"]
    local["news_score"]      = news["score"]
    local["news_headlines"]  = news.get("headlines", [])
    local["news_pos"]        = news.get("pos_count", 0)
    local["news_neg"]        = news.get("neg_count", 0)
    local["news_source_url"] = news.get("source_url", "https://news.google.com")
    local["news_summary"]    = news.get("summary", "")

    # ── Calculate scores deterministically (MUSCLE) ──────────────────────
    scores = calculate_scores_deterministic(macro, local, competitor_data, firm_type, price_tier)
    
    # ── Run strategy agent (BRAIN) ────────────────────────────────────────
    firm["expansion_score"] = scores["expansion_score"]
    firm["local_market_score"] = scores["local_market_score"]
    firm["pricing_power_score"] = scores["pricing_power_score"]
    firm["market_entry_score"] = scores["market_entry_score"]
    firm["market_entry_window"] = scores["market_entry_window"]
    result = run_strategy_agent(macro, fomc, local, firm, competitor_data, scores)
    
    # Override GPT scores with deterministic research-grounded scores
    result["expansion_score"] = scores["expansion_score"]
    result["local_market_score"] = scores["local_market_score"]
    result["pricing_power_score"] = scores["pricing_power_score"]
    result["market_entry_score"] = scores["market_entry_score"]
    result["market_entry_window"] = scores["market_entry_window"]

    # ── Reasoning Trace ───────────────────────────────────────────────────
    trace_items = []
    for step in result.get("reasoning_trace", []):
        step_parts = step.split(":", 1)
        label = step_parts[0] if len(step_parts) > 1 else "Step"
        detail= step_parts[1].strip() if len(step_parts) > 1 else step
        trace_items.append(html.Div([
            html.Span(label + ": ", style={"color": GREEN, "fontWeight": "600"}),
            html.Span(detail, style={"color": MUTED}),
        ], style={"marginBottom": "8px", "borderLeft": f"2px solid {BORDER}",
                  "paddingLeft": "8px"}))
    if is_scenario:
        trace_items.insert(0, html.Div(
            "SCENARIO MODE — overrides applied to simulation",
            style={"color": YELLOW, "fontWeight": "600",
                   "marginBottom": "12px", "fontSize": "10px", "letterSpacing": "1px"}
        ))

    # ── KPI Gauges ────────────────────────────────────────────────────────
    # Market Entry Window config
    entry_window = result.get("market_entry_window", "Cautious")
    entry_score  = result.get("market_entry_score", 50)
    entry_reason = result.get("market_entry_reason", "")
    entry_styles = {
        "Open":     {"bg": "rgba(0,212,170,0.1)",  "border": GREEN,  "color": GREEN,  "icon": "▲"},
        "Cautious": {"bg": "rgba(245,166,35,0.1)", "border": YELLOW, "color": YELLOW, "icon": "◆"},
        "Closed":   {"bg": "rgba(255,71,87,0.1)",  "border": RED,    "color": RED,    "icon": "▼"},
    }
    es = entry_styles.get(entry_window, entry_styles["Cautious"])

    # Fed tone — dynamic visual weight
    tone = fomc["tone"]
    fed_is_neutral = tone == "neutral"
    fed_border = tone_color(tone) if not fed_is_neutral else BORDER
    fed_bg     = f"rgba({','.join(str(int(tone_color(tone).lstrip('#')[i:i+2], 16)) for i in (0,2,4))},0.08)" if not fed_is_neutral else PANEL
    fed_font   = "14px" if not fed_is_neutral else "12px"
    fed_badge_size = "16px" if not fed_is_neutral else "13px"

    kpi_row = [
        html.Div([
            html.P("Should this firm enter this market? Based on competitor density and market concentration.",
                   style={"fontSize": "9px", "color": MUTED, "margin": "8px 8px 0", "opacity": "0.7", "lineHeight": "1.4", "textAlign": "center"}),
            dcc.Graph(figure=gauge_fig(result.get("expansion_score", 50), "Expansion Score"),
                            config={"displayModeBar": False})],
                 style={"backgroundColor": PANEL, "borderRadius": "10px",
                        "padding": "8px", "border": f"1px solid {BORDER}"}),
        html.Div([
            html.P("How attractive is this neighborhood? Based on income, population size and employment.",
                   style={"fontSize": "9px", "color": MUTED, "margin": "8px 8px 0", "opacity": "0.7", "lineHeight": "1.4", "textAlign": "center"}),
            dcc.Graph(figure=gauge_fig(result.get("local_market_score", 50), "Local Market Score"),
                            config={"displayModeBar": False})],
                 style={"backgroundColor": PANEL, "borderRadius": "10px",
                        "padding": "8px", "border": f"1px solid {BORDER}"}),
        html.Div([
            html.P("How much can this firm charge? Based on local purchasing power and brand strength.",
                   style={"fontSize": "9px", "color": MUTED, "margin": "8px 8px 0", "opacity": "0.7", "lineHeight": "1.4", "textAlign": "center"}),
            dcc.Graph(figure=gauge_fig(result.get("pricing_power_score", 50), "Pricing Power Score"),
                            config={"displayModeBar": False})],
                 style={"backgroundColor": PANEL, "borderRadius": "10px",
                        "padding": "8px", "border": f"1px solid {BORDER}"}),

        # Market Entry Window card
        html.Div([
            html.P("MARKET ENTRY WINDOW", style={
                "margin": "0 0 4px", "fontSize": "10px",
                "color": MUTED, "letterSpacing": "1.5px", "fontWeight": "700",
            }),
            html.Div(style={
                "backgroundColor": es["bg"],
                "border": f"1px solid {es['border']}",
                "borderRadius": "8px", "padding": "10px 14px",
                "marginBottom": "10px",
                "display": "flex", "alignItems": "center", "gap": "10px",
            }, children=[
                html.Span(es["icon"], style={"fontSize": "20px", "color": es["color"]}),
                html.Span(entry_window.upper(), style={
                    "fontSize": "18px", "fontWeight": "700",
                    "color": es["color"], "letterSpacing": "1px",
                }),
            ]),
            html.P(entry_reason, style={
                "fontSize": "10px", "color": MUTED,
                "margin": "0 0 12px", "lineHeight": "1.5",
            }),
            # Fed tone — subdued when neutral, prominent when hawkish/dovish
            html.Div(style={
                "backgroundColor": fed_bg,
                "border": f"1px solid {fed_border}",
                "borderRadius": "6px", "padding": "8px 12px",
                "display": "flex", "alignItems": "center",
                "justifyContent": "space-between",
            }, children=[
                html.Span("FED", style={
                    "fontSize": "9px", "color": MUTED,
                    "letterSpacing": "1px", "fontWeight": "700",
                }),
                html.Span(tone.upper(), style={
                    "fontSize": fed_badge_size,
                    "fontWeight": "700" if not fed_is_neutral else "400",
                    "color": tone_color(tone) if not fed_is_neutral else MUTED,
                    "letterSpacing": "0.5px",
                }),
            ]),
        ], style={
            "backgroundColor": PANEL, "borderRadius": "10px",
            "padding": "16px", "border": f"1px solid {BORDER}",
        }),
    ]

    # ── Recommendation Banner ─────────────────────────────────────────────
    rec_styles = {
        "Expand":         {"bg": "rgba(0,212,170,0.1)",  "border": GREEN,  "color": GREEN},
        "Hold":           {"bg": "rgba(245,166,35,0.1)", "border": YELLOW, "color": YELLOW},
        "Do Not Expand":  {"bg": "rgba(255,71,87,0.1)",  "border": RED,    "color": RED},
    }
    rs = rec_styles.get(result.get("expansion_recommendation", "Hold"),
                        {"bg": CARD, "border": BORDER, "color": TEXT})
    scenario_tag = html.Span(" [SCENARIO]", style={"color": YELLOW, "fontSize": "14px"}) if is_scenario else ""
    banner = html.Div(style={
        "backgroundColor": rs["bg"],
        "border": f"1px solid {rs['border']}",
        "borderRadius": "10px", "padding": "20px 24px",
        "display": "flex", "justifyContent": "space-between", "alignItems": "center",
    }, children=[
        html.Div([
            html.Div([
                html.Span(f"Recommendation: {result['expansion_recommendation']}",
                          style={"fontSize": "20px", "fontWeight": "700", "color": rs["color"]}),
                scenario_tag,
            ]),
            html.P(f"Confidence: {result['confidence_level']}  —  {result['confidence_reason']}",
                   style={"margin": "6px 0 0", "fontSize": "12px", "color": MUTED}),
        ]),
        html.Div([
            html.Div([
                html.Div(f"Recommended Price Tier: {scores['recommended_tier']}", style={
                    "backgroundColor": CARD, "border": f"1px solid {BORDER}",
                    "borderRadius": "6px", "padding": "8px 16px",
                    "fontSize": "13px", "fontWeight": "600", "color": TEXT,
                    "marginBottom": "4px",
                }),
                html.Div(
                    "Price tier correct for this market" if scores.get("pricing_direction") == "aligned"
                    else ("Selected tier exceeds what this market supports" if scores.get("pricing_direction") == "overpriced"
                    else "Selected tier is below this market's potential"),
                    style={
                        "fontSize": "11px", "fontWeight": "600",
                        "color": MUTED if scores.get("pricing_direction") == "aligned" else RED,
                        "marginBottom": "6px", "padding": "4px 8px",
                        "backgroundColor": CARD, "borderRadius": "4px",
                    }
                ),
            ]),
            html.Div(f"Macro Risk: {result['macro_risk']}", style={
                "backgroundColor": CARD, "border": f"1px solid {BORDER}",
                "borderRadius": "6px", "padding": "8px 16px",
                "fontSize": "13px", "fontWeight": "600", "color": TEXT,
            }),
        ]),
    ])
    # ── FOMC Panel ────────────────────────────────────────────────────────
    fomc_panel = html.Div([
        html.P("FEDERAL RESERVE SIGNAL", style={
            "fontSize": "10px", "fontWeight": "700",
            "color": MUTED, "letterSpacing": "1.5px", "margin": "0 0 16px",
        }),
        html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr 1fr",
                        "gap": "12px", "marginBottom": "16px"}, children=[
            html.Div([
                html.P("DIRECTION", style={"fontSize": "9px", "color": MUTED,
                                            "margin": "0 0 4px", "letterSpacing": "1px"}),
                html.P(fomc["policy_direction"].upper(),
                       style={"fontSize": "14px", "fontWeight": "700",
                               "color": TEXT, "margin": 0}),
            ]),
            html.Div([
                html.P("INFLATION", style={"fontSize": "9px", "color": MUTED,
                                            "margin": "0 0 4px", "letterSpacing": "1px"}),
                html.P(fomc["inflation_concern"].upper(),
                       style={"fontSize": "14px", "fontWeight": "700",
                               "color": TEXT, "margin": 0}),
            ]),
            html.Div([
                html.P("GROWTH", style={"fontSize": "9px", "color": MUTED,
                                         "margin": "0 0 4px", "letterSpacing": "1px"}),
                html.P(fomc["growth_concern"].upper(),
                       style={"fontSize": "14px", "fontWeight": "700",
                               "color": TEXT, "margin": 0}),
            ]),
        ]),
        html.P("KEY POLICY SIGNALS", style={
            "fontSize": "9px", "color": MUTED, "letterSpacing": "1px",
            "margin": "0 0 8px", "fontWeight": "700",
        }),
        html.Div([
            html.Div(sig, style={
                "backgroundColor": CARD,
                "borderLeft": f"2px solid {ACCENT}",
                "borderRadius": "0 4px 4px 0",
                "padding": "8px 12px",
                "fontSize": "11px", "color": TEXT,
                "marginBottom": "6px", "fontStyle": "italic",
            }) for sig in fomc.get("key_signals", [])
        ]),
        html.A("View Source Statement", href=fomc_link, target="_blank",
               style={"fontSize": "10px", "color": ACCENT, "textDecoration": "none",
                      "marginTop": "8px", "display": "block"}),
    ])

    # ── NLP Intelligence Panel ───────────────────────────────────────────
    beige_tone    = local.get("beige_tone", "neutral")
    beige_score   = local.get("beige_score", 0)
    beige_district= local.get("beige_district", "Unknown")
    beige_summary = local.get("beige_summary", "")
    news_tone     = local.get("news_tone", "neutral")
    news_score    = local.get("news_score", 0)
    news_headlines= local.get("news_headlines", [])
    city_name     = local.get("city", zip_code)

    tone_colors = {"positive": GREEN, "neutral": YELLOW, "negative": RED}
    beige_color = tone_colors.get(beige_tone, MUTED)
    news_color  = tone_colors.get(news_tone, MUTED)

    nlp_panel = html.Div([
        html.P("NLP INTELLIGENCE SIGNALS", style={
            "fontSize": "10px", "fontWeight": "700",
            "color": ACCENT, "letterSpacing": "1.5px", "margin": "0 0 12px",
        }),
        html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "12px"}, children=[
            html.Div(style={
                "backgroundColor": CARD, "borderRadius": "8px",
                "padding": "12px", "border": f"1px solid {beige_color}",
                "borderTop": f"3px solid {beige_color}",
            }, children=[
                html.P("BEIGE BOOK - REGIONAL", style={
                    "fontSize": "9px", "color": MUTED, "letterSpacing": "1px",
                    "margin": "0 0 6px", "fontWeight": "700",
                }),
                html.P(beige_tone.upper(), style={
                    "fontSize": "18px", "fontWeight": "700",
                    "color": beige_color, "margin": "0 0 4px",
                }),
                html.P(f"{beige_district} District", style={
                    "fontSize": "10px", "color": MUTED, "margin": "0 0 6px",
                }),
                html.P(beige_summary[:120] + "..." if len(beige_summary) > 120 else beige_summary, style={
                    "fontSize": "10px", "color": TEXT, "margin": "0 0 6px",
                    "lineHeight": "1.4", "fontStyle": "italic",
                }),
                html.P(f"Sentiment score: {beige_score:+.2f}  |  +{local.get('beige_pos',0)} positive / -{local.get('beige_neg',0)} negative signals", style={
                    "fontSize": "9px", "color": MUTED, "margin": "0 0 4px",
                }),
                html.A("View Source — Federal Reserve Beige Book",
                    href=local.get("beige_source_url", "https://www.federalreserve.gov/monetarypolicy/beige-book-default.htm"),
                    target="_blank",
                    style={"fontSize": "9px", "color": ACCENT, "textDecoration": "none"},
                ),
            ]),
            html.Div(style={
                "backgroundColor": CARD, "borderRadius": "8px",
                "padding": "12px", "border": f"1px solid {news_color}",
                "borderTop": f"3px solid {news_color}",
            }, children=[
                html.P("LOCAL NEWS - CITY LEVEL", style={
                    "fontSize": "9px", "color": MUTED, "letterSpacing": "1px",
                    "margin": "0 0 6px", "fontWeight": "700",
                }),
                html.P(news_tone.upper(), style={
                    "fontSize": "18px", "fontWeight": "700",
                    "color": news_color, "margin": "0 0 4px",
                }),
                html.P(f"{city_name} - Real-time", style={
                    "fontSize": "10px", "color": MUTED, "margin": "0 0 6px",
                }),
                html.Div([
                    html.P(f"- {h[:60]}..." if len(h) > 60 else f"- {h}", style={
                        "fontSize": "10px", "color": TEXT, "margin": "0 0 3px",
                        "lineHeight": "1.3",
                    }) for h in news_headlines[:3]
                ] if news_headlines else [
                    html.P("No local headlines available", style={
                        "fontSize": "10px", "color": MUTED, "fontStyle": "italic",
                    })
                ]),
                html.P(f"Sentiment score: {news_score:+.2f}  |  +{local.get('news_pos',0)} positive / -{local.get('news_neg',0)} negative signals", style={
                    "fontSize": "9px", "color": MUTED, "margin": "4px 0 0",
                }),
                html.A("View Source — Google News RSS (live)",
                    href=local.get("news_source_url", "https://news.google.com"),
                    target="_blank",
                    style={"fontSize": "9px", "color": ACCENT,
                           "textDecoration": "none", "marginTop": "4px", "display": "block"},
                ),
            ]),
        ]),
        html.P("Sources: Federal Reserve Beige Book (8x/year) | Google News RSS (live)", style={
            "fontSize": "9px", "color": MUTED, "margin": "8px 0 0",
            "opacity": "0.6", "fontStyle": "italic",
        }),
    ])

    # ── Local Panel ───────────────────────────────────────────────────────
    unemp_color = GREEN if local["unemployment_rate"] < 4 else YELLOW if local["unemployment_rate"] < 6 else RED
    local_panel = html.Div([
        html.P(f"LOCAL MARKET — ZIP {zip_code}", style={
            "fontSize": "10px", "fontWeight": "700",
            "color": MUTED, "letterSpacing": "1.5px", "margin": "0 0 16px",
        }),
        html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr",
                        "gap": "16px"}, children=[
            html.Div([
                html.P("MEDIAN INCOME", style={"fontSize": "9px", "color": MUTED,
                                                "margin": "0 0 4px", "letterSpacing": "1px"}),
                html.P(f"${local['median_income']:,}",
                       style={"fontSize": "22px", "fontWeight": "700",
                               "color": TEXT, "margin": 0}),
            ]),
            html.Div([
                html.P("POPULATION", style={"fontSize": "9px", "color": MUTED,
                                             "margin": "0 0 4px", "letterSpacing": "1px"}),
                html.P(f"{local['population']:,}",
                       style={"fontSize": "22px", "fontWeight": "700",
                               "color": TEXT, "margin": 0}),
            ]),
            html.Div([
                html.P("UNEMPLOYMENT", style={"fontSize": "9px", "color": MUTED,
                                               "margin": "0 0 4px", "letterSpacing": "1px"}),
                html.P(f"{local['unemployment_rate']}%",
                       style={"fontSize": "22px", "fontWeight": "700",
                               "color": unemp_color, "margin": 0}),
            ]),
            html.Div([
                html.P("LABOR FORCE", style={"fontSize": "9px", "color": MUTED,
                                              "margin": "0 0 4px", "letterSpacing": "1px"}),
                html.P(f"{local['labor_force']:,}",
                       style={"fontSize": "22px", "fontWeight": "700",
                               "color": TEXT, "margin": 0}),
            ]),
        ]),
    ])

    # ── Memo Panel ────────────────────────────────────────────────────────
    memo_panel = html.Div(style={
        "backgroundColor": PANEL, "borderRadius": "10px",
        "padding": "24px", "border": f"1px solid {BORDER}",
    }, children=[
        html.P("BOARD ADVISORY MEMO", style={
            "fontSize": "10px", "fontWeight": "700",
            "color": MUTED, "letterSpacing": "1.5px", "margin": "0 0 16px",
        }),
        html.P(result.get("advisory_memo", "Analysis unavailable."), style={
            "lineHeight": "1.9", "fontSize": "13px",
            "color": TEXT, "marginBottom": "16px",
            "borderLeft": f"3px solid {ACCENT}",
            "paddingLeft": "16px",
        }),
        html.Hr(style={"borderColor": BORDER, "margin": "16px 0"}),
        html.P("PRICING RATIONALE", style={
            "fontSize": "9px", "color": MUTED,
            "letterSpacing": "1px", "margin": "0 0 8px", "fontWeight": "700",
        }),
        html.P(result.get("pricing_rationale", "Pricing analysis not available."), style={
            "lineHeight": "1.7", "fontSize": "12px", "color": MUTED,
        }),
    ])

    # ── Competitor Panel ──────────────────────────────────────────────────
    comp_level = competitor_data.get("competitor_level", "Unknown")
    comp_count = competitor_data.get("competitor_count", 0)
    comp_colors = {"Low": GREEN, "Moderate": YELLOW, "High": RED, "Unknown": MUTED}
    comp_color  = comp_colors.get(comp_level, MUTED)
    proxy_hhi   = competitor_data.get("proxy_hhi", 0)
    hhi_level   = competitor_data.get("hhi_level", "Unknown")
    hhi_colors  = {"Unconcentrated": GREEN, "Moderately Concentrated": YELLOW, "Highly Concentrated": RED, "Unknown": MUTED}
    hhi_color   = hhi_colors.get(hhi_level, MUTED)
    
    competitor_panel = html.Div([
        html.Div(style={"display": "flex", "justifyContent": "space-between", 
                        "alignItems": "center", "marginBottom": "16px"}, children=[
            html.P("COMPETITIVE LANDSCAPE", style={
                "fontSize": "10px", "fontWeight": "700",
                "color": MUTED, "letterSpacing": "1.5px", "margin": 0,
            }),
            html.Div(style={"display": "flex", "alignItems": "center", "gap": "12px"}, children=[
                html.Span(f"{comp_count} competitors within 2 miles  |  {competitor_data.get('quality_note', '')}",
                         style={"fontSize": "12px", "color": MUTED}),
                html.Div(comp_level.upper(), style={
                    "backgroundColor": f"rgba({','.join(str(int(comp_color.lstrip('#')[i:i+2], 16)) for i in (0,2,4))},0.15)",
                    "color": comp_color, "border": f"1px solid {comp_color}",
                    "borderRadius": "4px", "padding": "4px 12px",
                    "fontSize": "11px", "fontWeight": "700", "letterSpacing": "1px",
                }),
            ]),
        ]),
        html.Div(style={"display": "grid", 
                        "gridTemplateColumns": f"repeat({min(len(competitor_data.get('nearby_competitors', [])), 5)}, 1fr)",
                        "gap": "10px"}, 
        children=[
            html.Div(style={
                "backgroundColor": CARD, "borderRadius": "6px",
                "padding": "12px", "border": f"1px solid {BORDER}",
                "borderLeft": f"2px solid {comp_color}",
            }, children=[
                html.P(c["name"], style={
                    "fontSize": "12px", "fontWeight": "600",
                    "color": TEXT, "margin": "0 0 4px",
                    "whiteSpace": "nowrap", "overflowX": "hidden",
                    "textOverflow": "ellipsis",
                }),
                html.P(f"Rating: {c['rating']} ⭐" if c["rating"] != "N/A" else "No rating",
                       style={"fontSize": "10px", "color": MUTED, "margin": "0 0 2px"}),
                html.P(c["vicinity"][:40] + "..." if len(c.get("vicinity","")) > 40 else c.get("vicinity",""),
                       style={"fontSize": "10px", "color": MUTED, "margin": 0}),
            ]) for c in competitor_data.get("nearby_competitors", [])
        ]) if competitor_data.get("nearby_competitors") else html.P(
            "No direct competitors found in this area — potential first-mover advantage.",
            style={"fontSize": "12px", "color": GREEN, "margin": 0}
        ),
    ])

    # Build score breakdown visualization
    def build_breakdown(breakdown_dict, score, max_score=100):
        bars = []
        for label, (earned, maximum) in breakdown_dict.items():
            pct = earned / maximum if maximum > 0 else 0
            color = "#00d4aa" if pct >= 0.7 else ("#f5a623" if pct >= 0.4 else "#ff4757")
            bar_width = f"{max(2, round(pct * 100))}%"
            bars.append(
                html.Div(style={"marginBottom": "8px"}, children=[
                    html.Div(style={"display": "flex", "justifyContent": "space-between",
                                   "marginBottom": "2px"}, children=[
                        html.Span(label, style={"fontSize": "9px", "color": MUTED}),
                        html.Span(f"{earned}/{maximum}",
                                 style={"fontSize": "9px", "color": color, "fontWeight": "600"}),
                    ]),
                    html.Div(style={"backgroundColor": CARD, "borderRadius": "3px",
                                   "height": "6px", "width": "100%"}, children=[
                        html.Div(style={
                            "backgroundColor": color,
                            "width": bar_width,
                            "height": "6px",
                            "borderRadius": "3px",
                            "transition": "width 0.3s ease",
                        }),
                    ]),
                ])
            )
        return html.Div([
            html.Div(style={"display": "flex", "justifyContent": "space-between",
                           "marginBottom": "12px", "paddingBottom": "8px",
                           "borderBottom": f"1px solid {BORDER}"}, children=[
                html.Span("Total Score", style={"fontSize": "11px", "color": MUTED}),
                html.Span(f"{score}/100",
                         style={"fontSize": "14px", "fontWeight": "700",
                                "color": GREEN if score >= 60 else YELLOW if score >= 40 else RED}),
            ]),
            *bars
        ])

    breakdown_panels = {
        "expansion": build_breakdown(
            result.get("expansion_breakdown", {}),
            result.get("expansion_score", 0)
        ),
        "local": build_breakdown(
            result.get("local_breakdown", {}),
            result.get("local_market_score", 0)
        ),
        "pricing": build_breakdown(
            result.get("pricing_breakdown", {}),
            result.get("pricing_power_score", 0)
        ),
        "mew": build_breakdown(
            result.get("mew_breakdown", {}),
            result.get("market_entry_score", 0)
        ),
    }

    # ── ZIP MEMORY: persist this run ─────────────────────────────────────
    if zip_history is None:
        zip_history = {}
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    run_record = {
        "timestamp":           timestamp,
        "zip_code":            zip_code,
        "business_type":       business_type,
        "firm_type":           firm_type,
        "price_tier":          price_tier,
        "expansion_score":     scores["expansion_score"],
        "local_market_score":  scores["local_market_score"],
        "pricing_power_score": scores["pricing_power_score"],
        "market_entry_score":  scores["market_entry_score"],
        "market_entry_window": scores["market_entry_window"],
        "recommendation":      result.get("expansion_recommendation", "Hold"),
        "median_income":       local.get("median_income", 0),
        "unemployment_rate":   local.get("unemployment_rate", 0),
        "competitor_count":    competitor_data.get("competitor_count", 0),
        "fed_funds_rate":      macro.get("fed_funds_rate", 0),
        "inflation_rate":      macro.get("inflation_rate", 0),
        "beige_tone":          local.get("beige_tone", "neutral"),
        "news_tone":           local.get("news_tone", "neutral"),
    }
    zip_key = zip_code.strip()
    if zip_key not in zip_history:
        zip_history[zip_key] = []
    zip_history[zip_key].append(run_record)
    zip_history[zip_key] = zip_history[zip_key][-12:]  # keep last 12 runs

    return (kpi_row, banner, macro_bar_fig(macro),
            local_bar_fig(local),
            swot_html(result.get("swot", {"strengths": [], "weaknesses": [], "opportunities": [], "threats": []})),
            "expansion",
            {
                "expansion": {k: list(v) for k,v in scores.get("expansion_breakdown", {}).items()},
                "local":     {k: list(v) for k,v in scores.get("local_breakdown", {}).items()},
                "pricing":   {k: list(v) for k,v in scores.get("pricing_breakdown", {}).items()},
                "mew":       {k: list(v) for k,v in scores.get("mew_breakdown", {}).items()},
                "scores": {
                    "expansion": scores.get("expansion_score", 0),
                    "local":     scores.get("local_market_score", 0),
                    "pricing":   scores.get("pricing_power_score", 0),
                    "mew":       scores.get("market_entry_score", 0),
                }
            },
            fomc_panel, local_panel,
            nlp_panel, memo_panel, trace_items, competitor_panel, zip_history,
            {
                "fed": macro.get("fed_funds_rate", 3.64),
                "prime": macro.get("prime_rate", 6.75),
                "inflation": macro.get("inflation_rate", 2.83),
                "gdp": macro.get("gdp_growth", 5.58),
                "retail": macro.get("retail_sales_growth", 3.72),
                "pce": macro.get("pce_growth", -0.29),
                "income": macro.get("income_growth", 0.03),
                "unemployment": local.get("unemployment_rate", 4.55),
                "income_local": local.get("median_income", 58000),
                "population": local.get("population", 45000),
                "rent": float(local.get("median_rent", 1200) or 1200),
            })


# ZIP MEMORY SYSTEM
def build_trend_chart(history_runs):
    if not history_runs or len(history_runs) < 2:
        return go.Figure().update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=200,
            annotations=[dict(text="Run at least 2 analyses on this ZIP to see trends",
                x=0.5, y=0.5, xref="paper", yref="paper",
                showarrow=False, font=dict(color="#8892a4", size=12))]
        )
    labels    = [r["timestamp"].split(" ")[0] for r in history_runs]
    expansion = [r["expansion_score"]         for r in history_runs]
    local_mkt = [r["local_market_score"]      for r in history_runs]
    pricing   = [r["pricing_power_score"]     for r in history_runs]
    mew       = [r["market_entry_score"]      for r in history_runs]
    fig = go.Figure()
    for name, values, color in [
        ("Expansion",     expansion, "#3d7aed"),
        ("Local Market",  local_mkt, "#00d4aa"),
        ("Pricing Power", pricing,   "#f5a623"),
        ("Entry Window",  mew,       "#7bed9f"),
    ]:
        fig.add_trace(go.Scatter(
            x=labels, y=values, name=name, mode="lines+markers",
            line=dict(color=color, width=2), marker=dict(size=6, color=color),
        ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#8892a4", height=220,
        margin=dict(t=10, b=10, l=10, r=10),
        legend=dict(orientation="h", y=-0.25, x=0, font=dict(size=10), bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(gridcolor="rgba(45,55,72,0.4)", tickfont=dict(size=9), tickangle=-20),
        yaxis=dict(range=[0, 105], gridcolor="rgba(45,55,72,0.4)", tickfont=dict(size=9)),
        hovermode="x unified",
    )
    return fig

def score_delta_badge(current, previous, label):
    delta = current - previous
    if abs(delta) < 1:
        arrow, color = "---", "#8892a4"
    elif delta > 0:
        arrow, color = f"+{delta:.0f}", "#00d4aa"
    else:
        arrow, color = f"{delta:.0f}", "#ff4757"
    return html.Div(style={
        "backgroundColor": CARD, "border": f"1px solid {BORDER}",
        "borderRadius": "6px", "padding": "10px 14px", "textAlign": "center",
    }, children=[
        html.P(label, style={"fontSize": "9px", "color": MUTED,
               "letterSpacing": "1px", "margin": "0 0 4px", "fontWeight": "700"}),
        html.P(f"{current:.0f}", style={"fontSize": "22px", "fontWeight": "700",
               "color": TEXT, "margin": "0"}),
        html.P(arrow, style={"fontSize": "11px", "color": color,
               "margin": "2px 0 0", "fontWeight": "600"}),
    ])

def generate_comparative_insight(current, previous, all_runs):
    try:
        metrics = [
            ("expansion_score",      "Expansion Score"),
            ("local_market_score",   "Local Market Score"),
            ("pricing_power_score",  "Pricing Power Score"),
            ("market_entry_score",   "Market Entry Score"),
        ]
        changes = []
        for key, label in metrics:
            delta = current[key] - previous[key]
            if abs(delta) >= 2:
                direction = "improved" if delta > 0 else "declined"
                changes.append(f"{label} {direction} by {abs(delta):.0f} points ({previous[key]:.0f} to {current[key]:.0f})")
        context_changes = []
        if abs(current["unemployment_rate"] - previous["unemployment_rate"]) >= 0.3:
            context_changes.append(f"unemployment {previous['unemployment_rate']:.1f}% to {current['unemployment_rate']:.1f}%")
        if abs(current["competitor_count"] - previous["competitor_count"]) >= 1:
            context_changes.append(f"competitors {previous['competitor_count']} to {current['competitor_count']}")
        if abs(current["fed_funds_rate"] - previous["fed_funds_rate"]) >= 0.1:
            context_changes.append(f"Fed rate {previous['fed_funds_rate']:.2f}% to {current['fed_funds_rate']:.2f}%")
        if current["market_entry_window"] != previous["market_entry_window"]:
            context_changes.append(f"Entry Window changed from {previous['market_entry_window']} to {current['market_entry_window']}")
        trend = "stable"
        if len(all_runs) >= 3:
            if all_runs[-1]["expansion_score"] > all_runs[-3]["expansion_score"]:
                trend = "improving"
            elif all_runs[-1]["expansion_score"] < all_runs[-3]["expansion_score"]:
                trend = "declining"
        prompt = (
            f"You are a market intelligence agent with memory of past analyses.\n"
            f"ZIP: {current['zip_code']} | Previous: {previous['recommendation']} | Current: {current['recommendation']}\n"
            f"Score changes: {', '.join(changes) if changes else 'scores largely unchanged'}\n"
            f"Context: {', '.join(context_changes) if context_changes else 'no significant shifts'}\n"
            f"Trend across {len(all_runs)} runs: {trend}\n"
            f"Write exactly 2 sentences. Sentence 1: what changed and by how much with specific numbers. "
            f"Sentence 2: what this means for the business decision."
        )
        r = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=130,
            system="Market intelligence agent. 2-sentence insights only.",
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(block.text for block in r.content if block.type == "text").strip()
    except Exception as e:
        print(f"Comparative insight error: {e}")
        if changes:
            return (f"Since the last analysis, {changes[0].lower()}. "
                    f"Current recommendation is {current.get('recommendation', 'Hold')}.")
        return f"Conditions in ZIP {current['zip_code']} remain stable since {previous['timestamp'].split(' ')[0]}."

def build_history_panel(zip_code, zip_history):
    runs = zip_history.get(zip_code.strip(), [])
    if not runs:
        return html.Div()
    current  = runs[-1]
    previous = runs[-2] if len(runs) >= 2 else None
    header = html.Div(style={
        "display": "flex", "justifyContent": "space-between",
        "alignItems": "center", "marginBottom": "16px",
    }, children=[
        html.Div([
            html.P("ZIP MEMORY", style={"fontSize": "10px", "fontWeight": "700",
                   "color": ACCENT, "letterSpacing": "1.5px", "margin": "0 0 4px"}),
            html.P(
                f"ZIP {zip_code}  |  {len(runs)} run{'s' if len(runs) != 1 else ''}  |  First: {runs[0]['timestamp'].split(' ')[0]}",
                style={"fontSize": "11px", "color": MUTED, "margin": 0}
            ),
        ]),
        html.P(f"Latest: {current['timestamp']}",
               style={"fontSize": "10px", "color": MUTED, "opacity": "0.6"}),
    ])
    comparative = html.Div()
    if previous:
        insight = generate_comparative_insight(current, previous, runs)
        comparative = html.Div(style={
            "backgroundColor": "rgba(61,122,237,0.05)",
            "border": f"1px solid {ACCENT}", "borderLeft": f"3px solid {ACCENT}",
            "borderRadius": "0 6px 6px 0", "padding": "14px 16px", "marginBottom": "14px",
        }, children=[
            html.P("AGENT MEMORY INSIGHT", style={"fontSize": "9px", "color": ACCENT,
                   "letterSpacing": "1px", "fontWeight": "700", "margin": "0 0 8px"}),
            html.P(insight, style={"fontSize": "12px", "color": TEXT,
                   "lineHeight": "1.8", "margin": 0}),
        ])
    delta_row = html.Div()
    if previous:
        delta_row = html.Div(style={
            "display": "grid", "gridTemplateColumns": "1fr 1fr 1fr 1fr",
            "gap": "10px", "marginBottom": "16px",
        }, children=[
            score_delta_badge(current["expansion_score"],     previous["expansion_score"],     "EXPANSION"),
            score_delta_badge(current["local_market_score"],  previous["local_market_score"],  "LOCAL MARKET"),
            score_delta_badge(current["pricing_power_score"], previous["pricing_power_score"], "PRICING"),
            score_delta_badge(current["market_entry_score"],  previous["market_entry_score"],  "ENTRY WINDOW"),
        ])
    trend_section = html.Div(style={"marginBottom": "16px"}, children=[
        html.P("SCORE TRENDS OVER TIME", style={"fontSize": "9px", "color": MUTED,
               "letterSpacing": "1px", "fontWeight": "700", "margin": "0 0 8px"}),
        dcc.Graph(figure=build_trend_chart(runs), config={"displayModeBar": False}),
    ])
    col_s = {"fontSize": "10px", "color": MUTED, "padding": "4px 8px", "textAlign": "right"}
    val_s = {"fontSize": "10px", "color": TEXT,  "padding": "4px 8px", "textAlign": "right"}
    table_rows = []
    for i, r in enumerate(reversed(runs)):
        rec_c = {"Expand": GREEN, "Hold": YELLOW, "Do Not Expand": RED}.get(r.get("recommendation","Hold"), MUTED)
        mew_c = {"Open": GREEN, "Cautious": YELLOW, "Closed": RED}.get(r.get("market_entry_window","Cautious"), MUTED)
        table_rows.append(html.Tr(style={
            "backgroundColor": CARD if i % 2 == 0 else "rgba(0,0,0,0)",
            "borderBottom": f"1px solid {BORDER}",
        }, children=[
            html.Td(r["timestamp"],                    style={**val_s, "textAlign": "left", "color": MUTED}),
            html.Td(r.get("business_type", "---"),     style={**val_s, "textAlign": "left"}),
            html.Td(r.get("firm_type", "---"),         style=val_s),
            html.Td(str(r["expansion_score"]),         style=val_s),
            html.Td(str(r["local_market_score"]),      style=val_s),
            html.Td(str(r["pricing_power_score"]),     style=val_s),
            html.Td(str(r["market_entry_score"]),      style=val_s),
            html.Td(r.get("recommendation", "---"),    style={**val_s, "color": rec_c, "fontWeight": "600"}),
            html.Td(r.get("market_entry_window","---"),style={**val_s, "color": mew_c}),
        ]))
    history_table = html.Div(style={"overflowX": "auto"}, children=[
        html.P("FULL RUN HISTORY", style={"fontSize": "9px", "color": MUTED,
               "letterSpacing": "1px", "fontWeight": "700", "margin": "0 0 8px"}),
        html.Table(style={"width": "100%", "borderCollapse": "collapse"}, children=[
            html.Thead(html.Tr(style={"borderBottom": f"2px solid {BORDER}"}, children=[
                html.Th("Date",     style={**col_s, "textAlign": "left"}),
                html.Th("Business", style={**col_s, "textAlign": "left"}),
                html.Th("Firm",     style=col_s),
                html.Th("Exp",      style=col_s),
                html.Th("Local",    style=col_s),
                html.Th("Pricing",  style=col_s),
                html.Th("Entry",    style=col_s),
                html.Th("Decision", style=col_s),
                html.Th("MEW",      style=col_s),
            ])),
            html.Tbody(table_rows),
        ]),
    ])
    return html.Div(style={
        "backgroundColor": PANEL, "borderRadius": "10px",
        "padding": "20px", "border": f"1px solid {BORDER}",
    }, children=[header, comparative, delta_row, trend_section, history_table])

@app.callback(
    Output("history-panel", "children"),
    Input("zip-history-store", "data"),
    State("zip-code", "value"),
    prevent_initial_call=True,
)
def update_history_panel(zip_history, zip_code):
    if not zip_history or not zip_code:
        return html.Div()
    return build_history_panel(zip_code, zip_history)


@app.callback(
    [Output("s-fed",          "value", allow_duplicate=True),
     Output("s-prime",        "value", allow_duplicate=True),
     Output("s-inflation",    "value", allow_duplicate=True),
     Output("s-gdp",          "value", allow_duplicate=True),
     Output("s-retail",       "value", allow_duplicate=True),
     Output("s-pce",          "value", allow_duplicate=True),
     Output("s-income",       "value", allow_duplicate=True),
     Output("s-unemployment", "value", allow_duplicate=True),
     Output("s-income-local", "value", allow_duplicate=True),
     Output("s-population",   "value", allow_duplicate=True),
     Output("s-rent",         "value", allow_duplicate=True)],
    Input("last-fetched-store", "data"),
    prevent_initial_call=True,
)
def update_sliders_from_fetch(last_fetched):
    if not last_fetched:
        raise PreventUpdate
    return (
        last_fetched.get("fed", 3.64),
        last_fetched.get("prime", 6.75),
        last_fetched.get("inflation", 2.83),
        last_fetched.get("gdp", 5.58),
        last_fetched.get("retail", 3.72),
        last_fetched.get("pce", -0.29),
        last_fetched.get("income", 0.03),
        last_fetched.get("unemployment", 4.55),
        last_fetched.get("income_local", 58000),
        last_fetched.get("population", 45000),
        last_fetched.get("rent", 1200),
    )

if __name__ == "__main__":
    app.run(debug=True, port=8050)
