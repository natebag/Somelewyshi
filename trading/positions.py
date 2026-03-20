"""Position tracker — monitors open positions, calculates P&L."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import datetime, timezone

from sqlmodel import Session, select

from db.models import Position, PositionStatus, Trade, TradeStatus

logger = logging.getLogger(__name__)


@dataclass
class PositionSummary:
    """Summary of a tracked position."""

    position_id: str
    market_id: str
    side: str
    entry_price: float
    current_price: float
    size_usd: float
    unrealized_pnl: float
    pnl_pct: float
    log_return: float
    status: str


@dataclass
class PortfolioSummary:
    """Aggregate portfolio stats."""

    total_invested: float
    total_unrealized_pnl: float
    total_realized_pnl: float
    open_positions: int
    closed_positions: int
    win_rate: float
    total_log_return: float


class PositionTracker:
    """Tracks and monitors positions from executed trades."""

    def __init__(self, session: Session):
        self.session = session

    def create_from_trade(self, trade: Trade) -> Position | None:
        """Create a tracked position from an executed trade."""
        if trade.status not in (TradeStatus.EXECUTED, TradeStatus.DRY_RUN):
            return None

        position = Position(
            market_id=trade.market_id or "",
            token_id=trade.token_id,
            side=trade.side,
            entry_price=trade.market_price_at_entry,
            current_price=trade.market_price_at_entry,
            size_usd=trade.size_usd,
            unrealized_pnl=0.0,
            status=PositionStatus.OPEN,
        )
        self.session.add(position)
        self.session.commit()
        logger.info(f"Position opened: {position.side} ${position.size_usd:.2f} at {position.entry_price:.2%}")
        return position

    def update_price(self, position_id: str, current_price: float):
        """Update a position's current price and recalculate P&L."""
        position = self.session.get(Position, position_id)
        if not position or position.status != PositionStatus.OPEN:
            return

        position.current_price = current_price

        # Calculate unrealized P&L
        if position.side in ("BUY_YES", "YES"):
            # Bought YES: profit if price goes up
            shares = position.size_usd / position.entry_price
            position.unrealized_pnl = shares * (current_price - position.entry_price)
        else:
            # Bought NO: profit if YES price goes down
            no_entry = 1 - position.entry_price
            no_current = 1 - current_price
            shares = position.size_usd / no_entry
            position.unrealized_pnl = shares * (no_current - no_entry)

        self.session.add(position)
        self.session.commit()

    def close_position(self, position_id: str, exit_price: float):
        """Close a position and lock in P&L."""
        position = self.session.get(Position, position_id)
        if not position or position.status != PositionStatus.OPEN:
            return

        self.update_price(position_id, exit_price)
        position.status = PositionStatus.CLOSED
        position.closed_at = datetime.now(timezone.utc)

        self.session.add(position)
        self.session.commit()
        logger.info(
            f"Position closed: {position.side} P&L=${position.unrealized_pnl:.2f}"
        )

    def get_open_positions(self) -> list[PositionSummary]:
        """Get all open positions with current P&L."""
        positions = self.session.exec(
            select(Position).where(Position.status == PositionStatus.OPEN)
        ).all()

        return [self._to_summary(p) for p in positions]

    def get_all_positions(self) -> list[PositionSummary]:
        """Get all positions (open + closed)."""
        positions = self.session.exec(select(Position)).all()
        return [self._to_summary(p) for p in positions]

    def get_portfolio_summary(self) -> PortfolioSummary:
        """Calculate aggregate portfolio statistics."""
        all_positions = self.session.exec(select(Position)).all()

        if not all_positions:
            return PortfolioSummary(
                total_invested=0, total_unrealized_pnl=0, total_realized_pnl=0,
                open_positions=0, closed_positions=0, win_rate=0, total_log_return=0,
            )

        open_pos = [p for p in all_positions if p.status == PositionStatus.OPEN]
        closed_pos = [p for p in all_positions if p.status == PositionStatus.CLOSED]

        total_invested = sum(p.size_usd for p in all_positions)
        total_unrealized = sum(p.unrealized_pnl or 0 for p in open_pos)
        total_realized = sum(p.unrealized_pnl or 0 for p in closed_pos)

        # Win rate (closed positions only)
        wins = sum(1 for p in closed_pos if (p.unrealized_pnl or 0) > 0)
        win_rate = wins / len(closed_pos) if closed_pos else 0

        # Total log return
        total_log = 0
        for p in closed_pos:
            if p.entry_price > 0 and p.current_price and p.current_price > 0:
                total_log += math.log(p.current_price / p.entry_price)

        return PortfolioSummary(
            total_invested=total_invested,
            total_unrealized_pnl=total_unrealized,
            total_realized_pnl=total_realized,
            open_positions=len(open_pos),
            closed_positions=len(closed_pos),
            win_rate=win_rate,
            total_log_return=total_log,
        )

    def _to_summary(self, p: Position) -> PositionSummary:
        entry = p.entry_price or 0.001
        current = p.current_price or entry
        pnl = p.unrealized_pnl or 0
        pnl_pct = (pnl / p.size_usd * 100) if p.size_usd else 0
        lr = math.log(current / entry) if entry > 0 and current > 0 else 0

        return PositionSummary(
            position_id=p.id,
            market_id=p.market_id,
            side=p.side,
            entry_price=entry,
            current_price=current,
            size_usd=p.size_usd,
            unrealized_pnl=pnl,
            pnl_pct=pnl_pct,
            log_return=lr,
            status=p.status.value,
        )
