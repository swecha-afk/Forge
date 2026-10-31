from typing import Literal

from app.core.config import settings
from app.schemas.mandi_schema import MandiQueryRequest, MandiScheduleResponse
from app.services.mandi_service.db_cache import (
    cache_live_price,
    get_cached_latest_and_avg,
)
from app.services.mandi_service.fetcher import (
    fetch_live_mandi_data,
    normalize_commodity,
)

PERISHABILITY_MAP = {
    "tomato": "HIGH",
    "brinjal": "HIGH",
    "cabbage": "HIGH",
    "cauliflower": "HIGH",
    "green chilli": "MEDIUM",
    "chili": "MEDIUM",
    "onion": "MEDIUM",
    "potato": "LOW",
    "rice": "LOW",
    "paddy": "LOW",
    "wheat": "LOW",
    "mustard": "LOW",
}


def get_perishability(commodity: str) -> Literal["HIGH", "MEDIUM", "LOW"]:
    crop_key = commodity.strip().lower()
    for k, tier in PERISHABILITY_MAP.items():
        if k in crop_key:
            return tier
    return "MEDIUM"


async def evaluate_harvest_schedule(req: MandiQueryRequest) -> MandiScheduleResponse:
    # 1. Normalize colloquial crop names ("Aloo" -> "Potato") BEFORE querying API or SQLite
    canonical_crop = normalize_commodity(req.commodity)

    live_data = await fetch_live_mandi_data(canonical_crop, req.state, req.district)

    if live_data and live_data["modal_price"] > 0:
        cache_live_price(
            canonical_crop,
            req.state,
            req.district,
            live_data["market"],
            live_data["modal_price"],
            live_data["arrival_date"],
        )
        source = "LIVE_OGD_API"
    else:
        source = "SQLITE_FALLBACK_CACHE"

    stats = get_cached_latest_and_avg(canonical_crop, req.district)

    if not stats:
        stats = {
            "market": f"{req.district} Central Mandi",
            "modal_price": 2000.0,
            "arrival_date": "Cached Baseline",
            "seven_day_avg": 1950.0,
        }

    current_price = stats["modal_price"]
    moving_avg = stats["seven_day_avg"]
    perishability = get_perishability(canonical_crop)
    rain_mm = req.rain_forecast_48h_mm or 0.0

    trend_delta_pct = (
        round(((current_price - moving_avg) / moving_avg) * 100, 2)
        if moving_avg > 0
        else 0.0
    )

    # Multi-variable override: High perishability + heavy rain forecast overrides price hold
    if perishability == "HIGH" and rain_mm >= 15.0:
        action_flag = "URGENT_HARVEST_WEATHER_RISK"
        reasoning = (
            f"SPOILAGE OVERRIDE: Although {canonical_crop} price at {stats['market']} is Rs. {current_price}/qtl "
            f"({trend_delta_pct}% vs 7-day avg of Rs. {moving_avg}/qtl), {canonical_crop} has HIGH perishability and "
            f"{rain_mm}mm rain is forecast in 48h. Harvest immediately to prevent field rot."
        )
    elif trend_delta_pct >= settings.HARVEST_THRESHOLD_PCT:
        action_flag = "RECOMMEND_HARVEST_TODAY"
        reasoning = (
            f"Current price in {stats['market']} (Rs. {current_price}/qtl) is {trend_delta_pct}% "
            f"above the 7-day average (Rs. {moving_avg}/qtl). Favorable window to harvest and sell."
        )
    elif trend_delta_pct <= -settings.HARVEST_THRESHOLD_PCT:
        if perishability == "HIGH":
            action_flag = "NEUTRAL_WINDOW"
            reasoning = (
                f"Price (Rs. {current_price}/qtl) is {abs(trend_delta_pct)}% below the 7-day average (Rs. {moving_avg}/qtl), "
                f"but {canonical_crop} is highly perishable. Do not delay harvest beyond 48 hours unless cold storage is available."
            )
        else:
            action_flag = "HOLD_FOR_RECOVERY"
            reasoning = (
                f"Current price (Rs. {current_price}/qtl) has dipped {abs(trend_delta_pct)}% below "
                f"the 7-day average (Rs. {moving_avg}/qtl). Since {canonical_crop} has {perishability} perishability, "
                f"hold harvest or store produce for price recovery."
            )
    else:
        action_flag = "NEUTRAL_WINDOW"
        reasoning = (
            f"Market price (Rs. {current_price}/qtl) is stable near the 7-day average (Rs. {moving_avg}/qtl). "
            f"Prioritize crop maturity and local weather for harvest timing."
        )

    return MandiScheduleResponse(
        commodity=canonical_crop,
        district=req.district,
        market_name=stats["market"],
        current_modal_price_inr_per_quintal=current_price,
        seven_day_moving_avg=moving_avg,
        trend_delta_pct=trend_delta_pct,
        perishability_tier=perishability,
        action_flag=action_flag,
        reasoning_summary=reasoning,
        best_state_market=live_data.get("best_state_market")
        if live_data
        else stats["market"],
        best_state_price_inr_per_quintal=live_data.get("best_state_price")
        if live_data
        else current_price,
        data_source=source,
        arrival_date=stats["arrival_date"],
    )
