import asyncio
from app.services.mandi_service.db_cache import init_mandi_db
from app.services.mandi_service.tool import mandi_harvest_advisor_tool

async def run_checks():
    init_mandi_db()

    print("--- TEST 1: Live Mandi Price & Colloquial Name ('Aloo' -> 'Potato') ---")
    res1 = await mandi_harvest_advisor_tool(commodity="Aloo", district="Medinipur West")
    print(f"Source: {res1['data_source']} | Price: Rs.{res1['current_modal_price_inr_per_quintal']} | Action: {res1['action_flag']}")
    print(f"Reasoning: {res1['reasoning_summary']}\n")

    print("--- TEST 2: Perishable Crop + Storm Spoilage Override (Tomato + 35mm Rain) ---")
    res2 = await mandi_harvest_advisor_tool(commodity="Tomato", district="Medinipur West", rain_forecast_48h_mm=35.0)
    print(f"Perishability: {res2['perishability_tier']} | Action: {res2['action_flag']}")
    print(f"Reasoning: {res2['reasoning_summary']}\n")

    print("--- TEST 3: Unseeded Exotic Crop Edge Case ('Dragonfruit') ---")
    res3 = await mandi_harvest_advisor_tool(commodity="Dragonfruit", district="Bankura")
    print(f"Source: {res3['data_source']} | Action: {res3['action_flag']}")

if __name__ == "__main__":
    asyncio.run(run_checks())