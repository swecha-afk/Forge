"""Pydantic v2 schemas for APMC Mandi price records, scheduling tools, and spatial arbitrage solving."""

from typing import Literal

from pydantic import BaseModel, Field


# --- Teammate 2 Mandi Service Contracts ---
class MandiQueryRequest(BaseModel):
    """Input payload for Mandi Harvest Scheduling Tool."""

    farmer_id: str | None = Field(
        default="anon_01", description="Unique farmer identifier"
    )
    commodity: str = Field(
        ..., examples=["Potato", "Tomato", "Rice"], description="Crop name"
    )
    state: str = Field(
        default="West Bengal", examples=["West Bengal"], description="State name"
    )
    district: str = Field(
        ..., examples=["Medinipur West", "Hooghly"], description="District name"
    )
    rain_forecast_48h_mm: float | None = Field(
        default=0.0,
        description="Predicted rainfall in mm over next 48 hours from Weather tool",
    )


class MandiScheduleResponse(BaseModel):
    """Output contract for Mandi Harvest Scheduling Tool."""

    commodity: str
    district: str
    market_name: str
    current_modal_price_inr_per_quintal: float
    seven_day_moving_avg: float
    trend_delta_pct: float
    perishability_tier: Literal["HIGH", "MEDIUM", "LOW"]
    action_flag: Literal[
        "RECOMMEND_HARVEST_TODAY",
        "HOLD_FOR_RECOVERY",
        "NEUTRAL_WINDOW",
        "URGENT_HARVEST_WEATHER_RISK",
    ]
    reasoning_summary: str
    best_state_market: str | None = None
    best_state_price_inr_per_quintal: float | None = None
    data_source: Literal["LIVE_OGD_API", "SQLITE_FALLBACK_CACHE"]
    arrival_date: str
