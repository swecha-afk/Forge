from pydantic import BaseModel, Field
from typing import Literal, Optional

class MandiQueryRequest(BaseModel):
    farmer_id: Optional[str] = Field(default="anon_01", description="Unique farmer identifier")
    commodity: str = Field(..., examples=["Potato", "Tomato", "Rice"], description="Crop name")
    state: str = Field(default="West Bengal", examples=["West Bengal"], description="State name")
    district: str = Field(..., examples=["Medinipur West", "Hooghly"], description="District name")
    rain_forecast_48h_mm: Optional[float] = Field(
        default=0.0, 
        description="Predicted rainfall in mm over next 48 hours from Weather tool"
    )

class MandiScheduleResponse(BaseModel):
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
        "URGENT_HARVEST_WEATHER_RISK"
    ]
    reasoning_summary: str
    best_state_market: Optional[str] = None
    best_state_price_inr_per_quintal: Optional[float] = None
    data_source: Literal["LIVE_OGD_API", "SQLITE_FALLBACK_CACHE"]
    arrival_date: str