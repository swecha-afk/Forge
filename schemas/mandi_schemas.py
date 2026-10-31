from pydantic import BaseModel, Field
from typing import Literal, Optional

class MandiQueryRequest(BaseModel):
    farmer_id: Optional[str] = Field(default="anon_01", description="Unique farmer identifier")
    commodity: str = Field(..., examples=["Potato", "Tomato", "Rice"], description="Crop name")
    state: str = Field(..., examples=["West Bengal"], description="State name")
    district: str = Field(..., examples=["Medinipur West", "Paschim Medinipur"], description="District name")

class MandiScheduleResponse(BaseModel):
    commodity: str
    district: str
    market_name: str
    current_modal_price_inr_per_quintal: float
    seven_day_moving_avg: float
    trend_delta_pct: float
    action_flag: Literal["RECOMMEND_HARVEST_TODAY", "HOLD_FOR_RECOVERY", "NEUTRAL_WINDOW"]
    reasoning_summary: str
    data_source: Literal["LIVE_OGD_API", "SQLITE_FALLBACK_CACHE"]
    arrival_date: str