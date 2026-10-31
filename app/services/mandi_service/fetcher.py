import os
import httpx
import logging
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv
from app.core.config import settings

# Force reload of .env so Uvicorn workers always pick up your personal key (...732324a3)
load_dotenv(override=True)

logger = logging.getLogger(__name__)

CROP_SYNONYM_MAP = {
    "aloo": "Potato",
    "potato": "Potato",
    "tamatar": "Tomato",
    "tomato": "Tomato",
    "pyaaz": "Onion",
    "pyaj": "Onion",
    "onion": "Onion",
    "dhan": "Rice",
    "paddy": "Rice",
    "chawal": "Rice",
    "rice": "Rice",
    "gehu": "Wheat",
    "wheat": "Wheat",
    "baingan": "Brinjal",
    "eggplant": "Brinjal",
    "gobhi": "Cauliflower",
    "sarson": "Mustard",
    "mirch": "Green Chilli",
    "chili": "Green Chilli",
}

def normalize_commodity(raw_commodity: str) -> str:
    key = raw_commodity.strip().lower()
    return CROP_SYNONYM_MAP.get(key, raw_commodity.strip().title())

def format_market_label(raw_market: str, raw_district: str) -> str:
    market_clean = raw_market.strip() if raw_market else "State"
    dist_clean = raw_district.strip() if raw_district else "Local"
    if "APMC" not in market_clean.upper():
        market_clean = f"{market_clean} APMC"
    return f"{market_clean} ({dist_clean})"

BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/129.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
}

def _select_best_record(records: List[Dict[str, Any]], clean_district: str, clean_state: str) -> Optional[Dict[str, Any]]:
    if not records:
        return None

    district_keywords = [
        w.lower() for w in clean_district.replace("(", " ").replace(")", " ").split()
        if len(w) > 3 and w.lower() not in {"west", "east", "north", "south"}
    ]

    # 1. Match user's district (e.g., "Medinipur" matches "Medinipur(W)")
    for rec in records:
        api_dist = str(rec.get("district", "")).lower()
        if clean_district.lower() == api_dist or any(kw in api_dist for kw in district_keywords):
            return rec

    # 2. Match user's state
    for rec in records:
        api_state = str(rec.get("state", "")).lower()
        if clean_state.lower() in api_state:
            return rec

    # 3. Fallback to first valid live commodity record in the state batch
    return records[0]

async def fetch_live_mandi_data(commodity: str, state: str, district: str) -> Optional[Dict[str, Any]]:
    url = f"https://api.data.gov.in/resource/{settings.DATA_GOV_RESOURCE_ID}"
    api_key = os.getenv("DATA_GOV_API_KEY", settings.DATA_GOV_API_KEY).strip()

    clean_state = state.strip().title()
    clean_district = district.strip().title()
    clean_commodity = normalize_commodity(commodity)

    # Force IPv4 (0.0.0.0) to bypass campus/ISP IPv6 blackholes
    transport = httpx.AsyncHTTPTransport(local_address="0.0.0.0", retries=1)
    timeout = httpx.Timeout(connect=5.0, read=10.0, write=5.0, pool=5.0)

    # Strictly filter by State + Commodity so out-of-state prices never distort local 7-day averages
    params = {
        "api-key": api_key,
        "format": "json",
        "filters[state.keyword]": clean_state,
        "filters[commodity]": clean_commodity,
        "limit": "100"
    }

    try:
        async with httpx.AsyncClient(
            transport=transport,
            headers=BROWSER_HEADERS,
            timeout=timeout,
            follow_redirects=True
        ) as client:
            resp = await client.get(url, params=params)

            if resp.status_code == 429:
                logger.warning("OGD API returned 429 Rate Limit. Failing over to SQLite cache.")
                return None

            resp.raise_for_status()
            payload = resp.json()

            if payload.get("status") == "error":
                logger.warning(f"OGD API validation error: {payload.get('validation_errors')}")
                return None

            records = payload.get("records", [])
            if not records:
                logger.info(f"No live OGD records today for {clean_commodity} in {clean_state}. Using SQLite cache.")
                return None

            target = _select_best_record(records, clean_district, clean_state)
            if not target:
                return None

            modal_price = float(target.get("modal_price", 0.0))
            if modal_price <= 0:
                return None

            # Find the highest-paying mandi in the state batch for spatial arbitrage
            best_rec = max(records, key=lambda r: float(r.get("modal_price", 0.0)))
            best_price = float(best_rec.get("modal_price", modal_price))
            best_market = format_market_label(
                best_rec.get("market", "State"),
                best_rec.get("district", clean_state)
            )

            return {
                "market": format_market_label(
                    target.get("market", "State"),
                    target.get("district", clean_state)
                ),
                "modal_price": modal_price,
                "arrival_date": target.get("arrival_date", "Today"),
                "best_state_market": best_market,
                "best_state_price": best_price
            }

    except httpx.TimeoutException:
        logger.warning("OGD API timed out. Switching to SQLite cache.")
        return None
    except Exception as e:
        logger.warning(f"Live OGD API error ({type(e).__name__}: {e}). Switching to SQLite cache.")
        return None