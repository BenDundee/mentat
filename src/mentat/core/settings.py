"""Centralised application settings loaded from environment / .env file."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with validation.

    Values are read from environment variables (case-insensitive) and from a
    ``.env`` file in the working directory.  The ``.env`` file is optional —
    environment variables take precedence.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Required
    openrouter_api_key: str

    # Optional with sensible defaults
    log_level: str = "INFO"
    mentat_debug: bool = False
    environment: str = "development"
    data_dir: str = "data"
    port: int = 8000
    host: str = "0.0.0.0"


settings = Settings()
