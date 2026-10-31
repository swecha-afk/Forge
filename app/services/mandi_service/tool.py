from typing import Dict, Any
from app.schemas.mandi_schema import MandiQueryRequest
from app.services.mandi_service.scheduler import evaluate_harvest_schedule

async def mandi_harvest_advisor_tool(
    commodity: str,
    district: str,
    state: str = "West Bengal",
    rain_forecast_48h_mm: float = 0.0,
    farmer_id: str = "anon_01"
) -> Dict[str, Any]:
    """
    Fetches live government APMC Mandi prices (with automatic SQLite fallback),
    calculates the 7-day rolling price trend, evaluates crop perishability against
    upcoming 48-hour rainfall forecasts, and returns a deterministic harvest schedule.
    """
    req = MandiQueryRequest(
        farmer_id=farmer_id,
        commodity=commodity,
        state=state,
        district=district,
        rain_forecast_48h_mm=rain_forecast_48h_mm
    )
    result = await evaluate_harvest_schedule(req)
    return result.model_dump()