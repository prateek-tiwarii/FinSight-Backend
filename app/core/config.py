from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Existing
    google_api_key: str | None = None
    gemini_model: str = "gemini-2.5-flash"
    max_upload_size_mb: int = 10

    # Runtime
    environment: str = "development"
    log_sql: bool = False
    api_base_url: str = "http://localhost:8000"
    frontend_base_url: str = "http://localhost:3000"
    cors_origins: str = "http://localhost:3000,http://127.0.0.1:3000"

    # Database (Neon PostgreSQL in production; SQLite is supported only for tests)
    database_url: str = ""
    direct_url: str | None = None

    # Auth / guest sessions
    auth_required: bool = True
    auth_session_days: int = 7
    guest_session_days: int = 7
    guest_max_portfolios: int = 10
    guest_max_analyses: int = 25
    cookie_secure: bool | None = None
    cookie_samesite: str = "lax"
    cookie_domain: str | None = None
    cookie_prefix: str = "finsight"

    # Rate limiting (requests per window per client IP)
    rate_limit_window_seconds: int = 60
    rate_limit_register: int = 5
    rate_limit_login: int = 10
    rate_limit_guest: int = 20

    # LangGraph checkpointer: auto|postgres|memory
    checkpoint_backend: str = "auto"
    checkpoint_dsn: str | None = None

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    @property
    def is_production(self) -> bool:
        return self.environment.strip().lower() == "production"

    @property
    def is_test(self) -> bool:
        return self.environment.strip().lower() in {"test", "testing"}

    @property
    def secure_cookies(self) -> bool:
        if self.cookie_secure is not None:
            return self.cookie_secure
        return self.is_production

    @property
    def allowed_origins(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @property
    def session_cookie_name(self) -> str:
        return f"{self.cookie_prefix}_session"

    @property
    def guest_cookie_name(self) -> str:
        return f"{self.cookie_prefix}_guest"

    @property
    def csrf_cookie_name(self) -> str:
        return f"{self.cookie_prefix}_csrf"


settings = Settings()
