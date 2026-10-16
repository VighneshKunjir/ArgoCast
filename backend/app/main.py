import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from workspace root or backend directory
root_env = Path(__file__).resolve().parents[2] / ".env"
if root_env.exists():
    load_dotenv(dotenv_path=root_env)
else:
    load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.routing import APIRoute

from app.api import health, floats, forecast, chat, benchmark
from app.db.session import init_db
from app.api.forecast import load_model_and_preprocessor


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="FloatChat X-RAG Forecasting Engine",
        version="2.0.0",
        description="Explainable Evidence-Linked Argo Ocean Profile Forecasting & Physics Validation Engine",
    )

    # CORS for local development
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers with manual prefix handling (workaround for FastAPI 0.141 include_router prefix bug)
    def include_router_with_prefix(router, prefix: str):
        for route in router.routes:
            new_route = type(route)(
                path=prefix + route.path,
                endpoint=route.endpoint,
                methods=route.methods,
                name=route.name,
                response_model=route.response_model,
                response_class=route.response_class,
                status_code=route.status_code,
                summary=route.summary,
                description=route.description,
                tags=route.tags,
                deprecated=route.deprecated,
                operation_id=route.operation_id,
            )
            app.router.routes.append(new_route)

    include_router_with_prefix(health.router, "/api")
    include_router_with_prefix(floats.router, "/api")
    include_router_with_prefix(forecast.router, "/api")
    include_router_with_prefix(chat.router, "/api")
    include_router_with_prefix(benchmark.router, "/api")

    @app.on_event("startup")
    async def startup():
        """Initialize database and load ML model on startup."""
        init_db()
        # Load ML model and preprocessor
        from app.api.forecast import load_model_and_preprocessor
        if not load_model_and_preprocessor():
            print("WARNING: Failed to load ML model and preprocessor")
        else:
            print("ML model and preprocessor loaded successfully")
        # Build the evidence-link search index once so the first forecast is not slow.
        try:
            from app.core.evidence import _candidate_index
            from app.db.session import SessionLocal
            db = SessionLocal()
            try:
                _candidate_index(db)
            finally:
                db.close()
        except Exception as exc:  # empty or missing database: index is built on first use
            print(f"Evidence index not built at startup: {exc}")

    @app.get("/")
    async def root():
        """Root endpoint."""
        return {
            "message": "FloatChat X-RAG Forecasting Engine",
            "version": "2.0.0",
            "docs": "/docs",
        }

    return app


app = create_app()