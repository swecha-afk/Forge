from app.schemas.mandi_schema import MandiQueryRequest, MandiScheduleResponse
from app.services.mandi_service.fetcher import fetch_live_mandi_data
from app.services.mandi_service.db_cache import cache_live_price, get_cached_latest_and_avg
from app.core.config import settings

async def evaluate_harvest_schedule(req: MandiQueryRequest) -> MandiScheduleResponse:
    live_data = await fetch_live_mandi_data(req.commodity, req.state, req.district)
    
    if live_data and live_data["modal_price"] > 0:
        cache_live_price(
            req.commodity, req.state, req.district,
            live_data["market"], live_data["modal_price"], live_data["arrival_date"]
        )
        source = "LIVE_OGD_API"
    else:
        source = "SQLITE_FALLBACK_CACHE"

    stats = get_cached_latest_and_avg(req.commodity, req.district)
    
    # Fallback default if a user queries an unseeded exotic crop while offline
    if not stats:
        stats = {
            "market": f"{req.district} Central Mandi",
            "modal_price": 2000.0,
            "arrival_date": "Cached Baseline",
            "seven_day_avg": 1950.0
        }

    current_price = stats["modal_price"]
    moving_avg = stats["seven_day_avg"]
    
    trend_delta_pct = round(((current_price - moving_avg) / moving_avg) * 100, 2) if moving_avg > 0 else 0.0

    if trend_delta_pct >= settings.HARVEST_THRESHOLD_PCT:
        action_flag = "RECOMMEND_HARVEST_TODAY"
        reasoning = (
            f"Current price in {stats['market']} (Rs. {current_price}/qtl) is {trend_delta_pct}% "
            f"above the 7-day average (Rs. {moving_avg}/qtl). Favorable window to harvest and sell."
        )
    elif trend_delta_pct <= -settings.HARVEST_THRESHOLD_PCT:
        action_flag = "HOLD_FOR_RECOVERY"
        reasoning = (
            f"Current price (Rs. {current_price}/qtl) has dipped {abs(trend_delta_pct)}% below "
            f"the 7-day average (Rs. {moving_avg}/qtl). Delay harvest or store produce if weather permits."
        )
    else:
        action_flag = "NEUTRAL_WINDOW"
        reasoning = (
            f"Market price (Rs. {current_price}/qtl) is stable near the 7-day average (Rs. {moving_avg}/qtl). "
            f"Prioritize local weather and crop maturity for harvest timing."
        )

    return MandiScheduleResponse(
        commodity=req.commodity,
        district=req.district,
        market_name=stats["market"],
        current_modal_price_inr_per_quintal=current_price,
        seven_day_moving_avg=moving_avg,
        trend_delta_pct=trend_delta_pct,
        action_flag=action_flag,
        reasoning_summary=reasoning,
        data_source=source,
        arrival_date=stats["arrival_date"]
    )