from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Primary: Current Daily Price of Various Commodities (Today's table)
    DATA_GOV_RESOURCE_ID: str = "9ef84268-d588-465a-a308-a864a43d0070"
    # Secondary: Variety-wise Daily Market Prices Data (Historical + Recent Agmarknet ledger)
    DATA_GOV_SECONDARY_RESOURCE_ID: str = "35985678-0d79-46b4-9ed6-6f13308a1d24"

    # Pydantic BaseSettings automatically reads DATA_GOV_API_KEY from .env
    DATA_GOV_API_KEY: str = "579b464db66ec23bdd000001cdd3946e44ce4aad7209ff7b23ac571b"
    API_TIMEOUT_SECONDS: float = 8.0
    SQLITE_DB_PATH: str = "data/mandi_cache.db"
    HARVEST_THRESHOLD_PCT: float = 5.0

    # Bhashini ULCA Speech-to-Text & Translation Settings
    BHASHINI_USER_ID: str = ""
    BHASHINI_API_KEY: str = ""
    BHASHINI_PIPELINE_ID: str = "64392f96daac500b55c543d0"
    BHASHINI_AUTH_URL: str = "https://meity-auth.ulcacontrib.org/ulca/apis/v0/model/getModelsPipeline"
    BHASHINI_INFERENCE_URL: str = "https://dhruva-api.bhashini.gov.in/services/inference/pipeline"
    VOICE_CACHE_DB_PATH: str = "data/voice_cache.db"

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )


settings = Settings()
