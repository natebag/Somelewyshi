"""Pydantic settings validation for all Miro Fish configuration."""

from __future__ import annotations

import tomli
from pathlib import Path
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# ── TOML sub-models ──────────────────────────────────────────────

class GeneralConfig(BaseModel):
    log_level: str = "INFO"


class MiroFishConfig(BaseModel):
    base_url: str = "http://localhost:5001"
    simulation_rounds: int = 30
    poll_interval_seconds: int = 5
    poll_timeout_seconds: int = 300


class TradingConfig(BaseModel):
    chain_id: int = 137
    clob_url: str = "https://clob.polymarket.com"
    signature_type: int = 0
    bankroll_usd: float = 500.0
    max_position_usd: float = 50.0
    kelly_fraction: float = 0.25
    min_ev_threshold: float = 0.05
    dry_run: bool = True


class BroadcastConfig(BaseModel):
    twitter_enabled: bool = True
    youtube_enabled: bool = False
    discord_enabled: bool = False
    twitter_account_id: str = "1"
    youtube_niche: str = "finance predictions"


class SchedulerConfig(BaseModel):
    enabled: bool = False
    interval_hours: int = 6


# ── Secrets from .env ────────────────────────────────────────────

class Settings(BaseSettings):
    """Loads secrets from .env and config from settings.toml."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Secrets
    anthropic_api_key: str = ""
    llm_api_key: str = ""
    llm_base_url: str = "https://api.openai.com/v1"
    llm_model_name: str = "gpt-4o-mini"
    polymarket_private_key: str = ""
    polymarket_funder_address: str = ""
    zep_api_key: str = ""
    gemini_api_key: str = ""
    discord_webhook_url: str = ""

    # TOML config sections
    general: GeneralConfig = Field(default_factory=GeneralConfig)
    mirofish: MiroFishConfig = Field(default_factory=MiroFishConfig)
    trading: TradingConfig = Field(default_factory=TradingConfig)
    broadcast: BroadcastConfig = Field(default_factory=BroadcastConfig)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._load_toml()

    def _load_toml(self):
        """Overlay settings.toml onto defaults."""
        toml_path = Path(__file__).parent / "settings.toml"
        if not toml_path.exists():
            return
        with open(toml_path, "rb") as f:
            data = tomli.load(f)

        if "general" in data:
            self.general = GeneralConfig(**data["general"])
        if "mirofish" in data:
            self.mirofish = MiroFishConfig(**data["mirofish"])
        if "trading" in data:
            self.trading = TradingConfig(**data["trading"])
        if "broadcast" in data:
            self.broadcast = BroadcastConfig(**data["broadcast"])
        if "scheduler" in data:
            self.scheduler = SchedulerConfig(**data["scheduler"])


def load_settings() -> Settings:
    """Load and validate all settings."""
    return Settings()
