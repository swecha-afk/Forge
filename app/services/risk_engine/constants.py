"""Constants and locked taxonomy for the Agritech Risk & RAG Engine.

Theme 2: Climate, Resource Infrastructure & Global Supply Resilience
Problem 1: Multilingual Agritech & Resource Advisory Assistant
"""

# Supported Crops & Taxonomy
SUPPORTED_CROPS: list[str] = ["potato", "rice", "wheat", "tomato"]
CROP_TO_ID: dict[str, int] = {
    "potato": 0,
    "rice": 1,
    "wheat": 2,
    "tomato": 3,
}
ID_TO_CROP: dict[int, str] = {v: k for k, v in CROP_TO_ID.items()}

# Supported Disease Classes
SUPPORTED_DISEASES: list[str] = [
    "healthy",
    "late_blight",
    "early_blight",
    "rice_blast",
    "bacterial_leaf_blight",
    "yellow_rust",
    "powdery_mildew",
    "tomato_leaf_curl",
]
DISEASE_TO_ID: dict[str, int] = {
    "healthy": 0,
    "late_blight": 1,
    "early_blight": 2,
    "rice_blast": 3,
    "bacterial_leaf_blight": 4,
    "yellow_rust": 5,
    "powdery_mildew": 6,
    "tomato_leaf_curl": 7,
}
ID_TO_DISEASE: dict[int, str] = {v: k for k, v in DISEASE_TO_ID.items()}

# Mathematical Feature Vector Specification (x in R^6)
FEATURE_ORDER: list[str] = [
    "temp_min_c",
    "temp_max_c",
    "humidity_3d_avg",
    "rain_7d_mm",
    "soil_moisture",
    "crop_id",
]

# Guardrails & Deterministic Safety Thresholds
SPRAY_RAIN_THRESHOLD_MM: float = (
    10.0  # Precipitation > 10 mm in next 24h triggers spray lock
)
SPRAY_WIND_THRESHOLD_KMH: float = (
    18.0  # Wind speed > 18 km/h triggers spray lock (drift prevention)
)
SOIL_MOISTURE_LEACHING_THRESHOLD: float = (
    0.32  # > 0.32 m^3/m^3 triggers irrigation/urea lock (root hypoxia/leaching)
)
CONFIDENCE_GATE_TAU: float = 0.60  # Cosine similarity < 0.60 blocks chemical synthesis
RRF_K: int = 60  # Standard Reciprocal Rank Fusion constant (1 / (60 + rank))

# Risk Level Classification Thresholds
RISK_THRESHOLD_MODERATE: float = 0.35
RISK_THRESHOLD_HIGH: float = 0.65
RISK_THRESHOLD_SEVERE: float = 0.85

# Artifact Storage & Model Paths
DEFAULT_DB_PATH: str = "artifacts/icar_knowledge.db"
DEFAULT_FAISS_PATH: str = "artifacts/icar_faiss.index"
EMBEDDING_MODEL_NAME: str = "BAAI/bge-small-en-v1.5"
