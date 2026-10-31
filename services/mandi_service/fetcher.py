import httpx
from typing import Optional, Dict, Any
from app.core.config import settings

async def fetch_live_mandi_data(commodity: str, state: str, district: str) -> Optional[Dict[str, Any]]:
    url = f"https://api.data.gov.in/resource/{settings.DATA_GOV_RESOURCE_ID}"
    params = {
        "api-key": settings.DATA_GOV_API_KEY,
        "format": "json",
        "filters[state.keyword]": state,
        "filters[district]": district,
        "filters[commodity]": commodity,
        "limit": 5
    }
    
    try:
        async with httpx.AsyncClient(timeout=settings.API_TIMEOUT_SECONDS) as client:
            response = await client.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            records = data.get("records", [])
            if not records:
                return None
                
            latest = records[0]
            return {
                "market": latest.get("market", f"{district} APMC"),
                "modal_price": float(latest.get("modal_price", 0.0)),
                "arrival_date": latest.get("arrival_date", "Today")
            }
    except Exception:
        # Any timeout, DNS failure, or 5xx error safely triggers fallback
        return None