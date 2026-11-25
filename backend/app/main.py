import asyncio
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
from backend.app.rag.model_readiness import ModelReadinessManager
from backend.app.rag.model_settings import (
    validate_model_settings_compatibility,
)
from backend.app.rag.model_warmup import (
    create_model_readiness_manager,
)


def create_app(
    settings: Settings | None = None,
    *,
    readiness_manager: ModelReadinessManager | None = None,
) -> FastAPI:
    settings = settings or get_settings()
    database = SQLiteDatabase(settings.database_url)

    if settings.model_warmup_enabled:
        if readiness_manager is None:
            validate_model_settings_compatibility(
                settings,
                get_settings(),
            )
            readiness_manager = create_model_readiness_manager()
    else:
        readiness_manager = None

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await database.initialize()

        warmup_task: asyncio.Task[None] | None = None

        if readiness_manager is not None:
            warmup_task = asyncio.create_task(
                readiness_manager.warm_up(),
                name="campuslens-model-warmup",
            )

        app.state.model_warmup_task = warmup_task

        try:
            yield
        finally:
            try:
                if warmup_task is not None:
                    await warmup_task
            finally:
                app.state.model_warmup_task = None

    app = FastAPI(
        title="CampusLens AI API",
        version="0.1.0",
        description="University document question-answering API",
        lifespan=lifespan,
    )

    app.state.database = database
    app.state.model_readiness = readiness_manager
    app.state.model_warmup_task = None
    app.state.evaluation_lock = asyncio.Lock()

    cors_allowed_origins = (
        (settings.frontend_origin,) if settings.app_env == "development" else ()
    )
    app.state.cors_allowed_origins = cors_allowed_origins

    if cors_allowed_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(cors_allowed_origins),
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
