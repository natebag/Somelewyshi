"""Pydantic models for trading data."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class MarketSnapshot:
    """Point-in-time snapshot of a Polymarket market."""

    condition_id: str
    question: str
    yes_price: float
    no_price: float
    volume: float
    end_date: str
    token_id_yes: str = ""
    token_id_no: str = ""
    fetched_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class TradeOrder:
    """An order to be placed on Polymarket."""

    token_id: str
    side: str  # "BUY_YES" or "BUY_NO"
    amount_usd: float
    market_price: float
    estimated_prob: float
    ev_per_dollar: float
    kelly_fraction: float
    dry_run: bool = True


@dataclass
class TradeResult:
    """Result of a trade execution (or dry-run)."""

    order_id: str | None
    status: str  # "EXECUTED", "DRY_RUN", "FAILED", "SKIPPED"
    fill_price: float | None = None
    amount_filled: float | None = None
    error: str | None = None
