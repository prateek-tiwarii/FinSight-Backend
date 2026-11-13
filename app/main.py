from dotenv import load_dotenv

load_dotenv()

import logging

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.analysis_routes import router as analysis_router
from app.api.auth_routes import router as auth_router
from app.api.guest_routes import router as guest_router
from app.api.portfolio_routes import router as portfolio_router
from app.api.report_routes import router as report_router
from app.api.routes import router as api_router
from app.core.config import settings
from app.core.csrf import CsrfProtectionMiddleware
from app.db.session import DatabaseNotConfiguredError

logger = logging.getLogger(__name__)

app = FastAPI(title="FinSight AI API", version="0.2.0")

# CSRF runs inside CORS so preflight/error responses keep CORS headers.
app.add_middleware(CsrfProtectionMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-CSRF-Token"],
)


@app.exception_handler(DatabaseNotConfiguredError)
async def database_not_configured_handler(
    request: Request, exc: DatabaseNotConfiguredError
) -> JSONResponse:
    return JSONResponse(
        status_code=503,
        content={
            "detail": (
                "Database is not configured. Set DATABASE_URL to a PostgreSQL/Neon "
                "connection string and run migrations."
            )
        },
    )


app.include_router(auth_router, prefix="/api")
app.include_router(guest_router, prefix="/api")
app.include_router(portfolio_router, prefix="/api")
app.include_router(analysis_router, prefix="/api")
app.include_router(report_router, prefix="/api")
app.include_router(api_router, prefix="/api")
