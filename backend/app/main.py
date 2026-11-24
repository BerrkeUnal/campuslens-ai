from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api.routes import (
    documents,
    evaluations,
    health,
    questions,
)
from backend.app.core.config import Settings, get_settings
from backend.app.core.error_handlers import register_error_handlers
from backend.app.core.middleware import RequestIDMiddleware
from backend.app.db.database import SQLiteDatabase


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    database = SQLiteDatabase(settings.database_url)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await database.initialize()
        yield

    app = FastAPI(
        title="CampusLens AI API",
        version="0.1.0",
        description="University document question-answering API",
        lifespan=lifespan,
    )

    app.state.database = database

    if settings.app_env == "development":
        app.add_middleware(
            CORSMiddleware,
            allow_origins=[settings.frontend_origin],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    app.add_middleware(RequestIDMiddleware)
    register_error_handlers(app)

    app.include_router(health.router, prefix="/api")
    app.include_router(documents.router, prefix="/api")
    app.include_router(questions.router, prefix="/api")
    app.include_router(evaluations.router, prefix="/api")

    return app


app = create_app()
