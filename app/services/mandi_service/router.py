# app/services/mandi_service/router.py
from fastapi import APIRouter
from app.schemas.mandi_schema import MandiQueryRequest, MandiScheduleResponse
from app.services.mandi_service.scheduler import evaluate_harvest_schedule

router = APIRouter(prefix="/tools/mandi", tags=["Mandi & Scheduling Tool"])

@router.post("/schedule", response_model=MandiScheduleResponse)
async def get_mandi_schedule(payload: MandiQueryRequest) -> MandiScheduleResponse:
    return await evaluate_harvest_schedule(payload)