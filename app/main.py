"""FastAPI application factory — mirrors Spring Boot @SpringBootApplication."""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core import settings, engine, close_redis


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Startup/shutdown lifecycle — mirrors Spring Boot lifecycle."""
    # Startup
    yield
    # Shutdown
    await close_redis()
    await engine.dispose()


def create_app() -> FastAPI:
    app = FastAPI(
        title="Honghu AI",
        description="Enterprise Multi-Model Intelligent Chat & RAG Platform (FastAPI + LangChain)",
        version="0.0.1",
        lifespan=lifespan,
        docs_url="/swagger-ui.html",
        openapi_url="/api-docs",
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register routers
    from app.api import (
        health_router,
        chat_router,
        user_router,
        rag_router,
        sessions_router,
        admin_router,
        monitor_router,
        capability_router,
    )

    app.include_router(health_router)
    app.include_router(chat_router)
    app.include_router(user_router)
    app.include_router(rag_router)
    app.include_router(sessions_router)
    app.include_router(admin_router)
    app.include_router(monitor_router)
    app.include_router(capability_router)

    return app


app = create_app()
