"""FastAPI Application Entrypoint: Agritech Swarm Execution Engine.

Combines:
- Member 1: Core Intelligence, Predictive ML, TreeSHAP, and Subspace ICAR RAG
- Member 2: Agmarknet Mandi Price Harvester and Spoilage Scheduler
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers.advisory_router import router as advisory_router
from app.services.mandi_service.db_cache import init_mandi_db
from app.services.mandi_service.router import router as mandi_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize SQLite database cache on startup
    init_mandi_db()
    yield


app = FastAPI(
    title="Agritech Swarm Execution Engine",
    description="Multilingual Agritech & Resource Advisory Assistant (Forge Buildathon - Theme 2, Problem 1)",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for Member 3's Streamlit / Gradio frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Routers
app.include_router(mandi_router)
app.include_router(advisory_router)


@app.get("/health", tags=["Health & Diagnostics"])
async def health_check():
    return {
        "status": "HEALTHY",
        "services": {
            "mandi_engine": "ACTIVE",
            "risk_engine": "ACTIVE",
            "subspace_rag": "ACTIVE",
            "weather_connector": "ACTIVE",
        },
    }
