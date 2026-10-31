# app/main.py
from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.services.mandi_service.db_cache import init_mandi_db
from app.services.mandi_service.router import router as mandi_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_mandi_db()
    yield

app = FastAPI(
    title="Agritech Swarm Execution Engine",
    version="1.0.0",
    lifespan=lifespan
)

# Your router
app.include_router(mandi_router)

# Teammates will mount their routers below without touching your files:
# app.include_router(risk_router)
# app.include_router(rag_router)