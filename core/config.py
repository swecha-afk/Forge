import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Official Data.gov.in daily mandi prices resource ID
    DATA_GOV_RESOURCE_ID: str = "9ef84268-d588-465a-a308-a864a43d0070"
    # Sample test key from data.gov.in (replace with your free registered key in .env)
    DATA_GOV_API_KEY: str = os.getenv(
        "DATA_GOV_API_KEY", 
        "579b464db66ec23bdd000001cdd3946e44ce4aad7209ff7b23ac571b"
    )
    API_TIMEOUT_SECONDS: float = 3.0
    SQLITE_DB_PATH: str = "data/mandi_cache.db"
    HARVEST_THRESHOLD_PCT: float = 5.0  # Recommend harvest if price > 5% above 7-day avg

    class Config:
        env_file = ".env"

settings = Settings()