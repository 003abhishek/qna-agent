"""Main FastAPI application."""

from contextlib import asynccontextmanager
from fastapi import FastAPI

from app.config import get_settings
from app.database import init_db
from app.routes import router

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown."""
    # Startup
    await init_db()
    print(f"Database initialized at {settings.database_path}")
    print(f"Knowledge base path: {settings.knowledge_path}")
    
    yield
    
    # Shutdown
    print("Application shutting down")


app = FastAPI(
    title="QnA Agent API",
    description="RAG-based Question Answering API",
    version="0.1.0",
    lifespan=lifespan
)

# Include routes
app.include_router(router, prefix="/api/v1")


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "QnA Agent API",
        "version": "0.1.0",
        "docs": "/docs"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )