"""SQLModel ORM definitions for pipeline state tracking."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from sqlmodel import Field, SQLModel


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _uuid() -> str:
    return str(uuid.uuid4())


# ── Enums ────────────────────────────────────────────────────────

class PipelineStatus(str, Enum):
    PENDING = "PENDING"
    PREDICTING = "PREDICTING"
    TRADING = "TRADING"
    BROADCASTING = "BROADCASTING"
    DONE = "DONE"
    FAILED = "FAILED"


class TradeStatus(str, Enum):
    EXECUTED = "EXECUTED"
    SKIPPED = "SKIPPED"
    FAILED = "FAILED"
    DRY_RUN = "DRY_RUN"


class BroadcastStatus(str, Enum):
    POSTED = "POSTED"
    SKIPPED = "SKIPPED"
    FAILED = "FAILED"


class PositionStatus(str, Enum):
    OPEN = "OPEN"
    CLOSED = "CLOSED"


# ── Tables ───────────────────────────────────────────────────────

class PipelineRun(SQLModel, table=True):
    __tablename__ = "pipeline_runs"

    id: str = Field(default_factory=_uuid, primary_key=True)
    market_query: str
    market_id: Optional[str] = None
    status: PipelineStatus = PipelineStatus.PENDING
    error_message: Optional[str] = None
    started_at: datetime = Field(default_factory=_utcnow)
    completed_at: Optional[datetime] = None


class Prediction(SQLModel, table=True):
    __tablename__ = "predictions"

    id: str = Field(default_factory=_uuid, primary_key=True)
    pipeline_run_id: str = Field(foreign_key="pipeline_runs.id")
    probability: float
    confidence: float
    reasoning: Optional[str] = None
    report_text: Optional[str] = None
    mirofish_session_id: Optional[str] = None
    created_at: datetime = Field(default_factory=_utcnow)


class Trade(SQLModel, table=True):
    __tablename__ = "trades"

    id: str = Field(default_factory=_uuid, primary_key=True)
    pipeline_run_id: str = Field(foreign_key="pipeline_runs.id")
    prediction_id: str = Field(foreign_key="predictions.id")
    market_id: Optional[str] = None
    token_id: Optional[str] = None
    side: str = "YES"  # YES or NO
    size_usd: float = 0.0
    expected_value: float = 0.0
    kelly_fraction: float = 0.0
    market_price_at_entry: float = 0.0
    order_id: Optional[str] = None
    status: TradeStatus = TradeStatus.SKIPPED
    fill_price: Optional[float] = None
    skip_reason: Optional[str] = None
    created_at: datetime = Field(default_factory=_utcnow)


class Broadcast(SQLModel, table=True):
    __tablename__ = "broadcasts"

    id: str = Field(default_factory=_uuid, primary_key=True)
    pipeline_run_id: str = Field(foreign_key="pipeline_runs.id")
    platform: str = "TWITTER"  # TWITTER or YOUTUBE
    content_text: Optional[str] = None
    post_url: Optional[str] = None
    status: BroadcastStatus = BroadcastStatus.SKIPPED
    created_at: datetime = Field(default_factory=_utcnow)


class Position(SQLModel, table=True):
    __tablename__ = "positions"

    id: str = Field(default_factory=_uuid, primary_key=True)
    market_id: str
    token_id: Optional[str] = None
    side: str = "YES"
    entry_price: float
    current_price: Optional[float] = None
    size_usd: float
    unrealized_pnl: Optional[float] = None
    status: PositionStatus = PositionStatus.OPEN
    opened_at: datetime = Field(default_factory=_utcnow)
    closed_at: Optional[datetime] = None
