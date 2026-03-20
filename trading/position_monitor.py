"""Realistic position monitor — tracks real market prices over time.

Instead of instant dry-run closes, this module:
1. Records entry price + timestamp
2. Polls real Polymarket prices on each daemon cycle
3. Applies repricing exit logic on REAL price movements
4. Only closes when exit conditions are actually met
5. Tracks how long positions take to reprice in practice

This gives accurate P&L that reflects what live trading would actually look like.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from sqlmodel import Session, select

from db.models import Position, PositionStatus
from trading.client import PolymarketClient
from trading.repricing import RepricingEngine
from trading.slippage import estimate_fill_price

logger = logging.getLogger(__name__)


class PositionMonitor:
    """Monitors open positions against real market prices."""

    def __init__(
        self,
        session: Session,
        poly_client: PolymarketClient,
        repricing_engine: RepricingEngine | None = None,
    ):
        self.session = session
        self.poly_client = poly_client
        self.repricing = repricing_engine or RepricingEngine()

    def update_all_positions(self) -> list[dict]:
        """Poll real prices and evaluate exits for all open positions.

        Returns list of actions taken (updates, closes).
        """
        positions = self.session.exec(
            select(Position).where(Position.status == PositionStatus.OPEN)
        ).all()

        if not positions:
            return []

        actions = []

        for pos in positions:
            try:
                action = self._update_position(pos)
                if action:
                    actions.append(action)
            except Exception as e:
                logger.debug(f"Failed to update position {pos.id}: {e}")

        if actions:
            self.session.commit()

        return actions

    def _update_position(self, pos: Position) -> dict | None:
        """Update a single position with real market data."""
        # Get real current price
        token_id = pos.token_id
        if not token_id:
            return None

        # Try order book first for more accurate pricing
        book = self.poly_client.get_order_book(token_id)
        if book:
            current_price = book["mid_price"]
            best_bid = book["best_bid"]
            best_ask = book["best_ask"]
        else:
            midpoint = self.poly_client.get_midpoint(token_id)
            if midpoint is None:
                return None
            current_price = midpoint
            best_bid = None
            best_ask = None

        old_price = pos.current_price or pos.entry_price

        # Update current price
        pos.current_price = current_price

        # Calculate unrealized P&L with slippage on potential exit
        exit_fill = estimate_fill_price(
            side="BUY_NO" if pos.side in ("BUY_YES", "YES") else "BUY_YES",
            mid_price=current_price,
            order_size_usd=pos.size_usd,
            best_bid=best_bid,
            best_ask=best_ask,
        )

        # P&L based on realistic exit fill
        if pos.side in ("BUY_YES", "YES"):
            shares = pos.size_usd / pos.entry_price if pos.entry_price > 0 else 0
            pos.unrealized_pnl = shares * (exit_fill - pos.entry_price)
        else:
            no_entry = 1 - pos.entry_price
            no_exit = 1 - exit_fill
            shares = pos.size_usd / no_entry if no_entry > 0 else 0
            pos.unrealized_pnl = shares * (no_exit - no_entry)

        self.session.add(pos)

        # Calculate position age
        age_seconds = 0
        if pos.opened_at:
            age_seconds = int((datetime.now(timezone.utc) - pos.opened_at).total_seconds())

        # Check if we should exit using repricing engine
        # Use a rough estimate for our original probability
        # (entry_price + edge that triggered the trade)
        our_estimate = pos.entry_price  # Fallback — will be refined when we store estimates
        if pos.side in ("BUY_YES", "YES"):
            our_estimate = min(pos.entry_price + 0.15, 0.95)  # We thought YES was underpriced
        else:
            our_estimate = max(pos.entry_price - 0.15, 0.05)  # We thought YES was overpriced

        # Time since last meaningful price change
        price_change = abs(current_price - old_price)
        last_change_seconds = 0 if price_change > 0.005 else age_seconds

        exit_signal = self.repricing.check_exit(
            direction=pos.side,
            entry_price=pos.entry_price,
            current_price=current_price,
            our_estimate=our_estimate,
            position_age_seconds=age_seconds,
            last_price_change_seconds=last_change_seconds,
            size_usd=pos.size_usd,
        )

        if exit_signal.should_exit:
            # Close the position at realistic exit price
            pos.status = PositionStatus.CLOSED
            pos.closed_at = datetime.now(timezone.utc)
            pos.current_price = exit_fill  # Record actual exit fill
            self.session.add(pos)

            logger.info(
                f"[POSITION CLOSED] {pos.side} | "
                f"entry={pos.entry_price:.0%} exit={exit_fill:.0%} | "
                f"P&L=${pos.unrealized_pnl:+.2f} | "
                f"held {age_seconds}s | "
                f"reason: {exit_signal.reason}"
            )

            return {
                "action": "CLOSED",
                "position_id": pos.id,
                "entry": pos.entry_price,
                "exit": exit_fill,
                "pnl": pos.unrealized_pnl,
                "hold_seconds": age_seconds,
                "reason": exit_signal.reason,
            }

        # Just an update, not a close
        if abs(current_price - old_price) > 0.005:
            logger.debug(
                f"[POSITION UPDATE] {pos.side} | "
                f"entry={pos.entry_price:.0%} now={current_price:.0%} | "
                f"P&L=${pos.unrealized_pnl:+.2f}"
            )
            return {
                "action": "UPDATED",
                "position_id": pos.id,
                "old_price": old_price,
                "new_price": current_price,
                "pnl": pos.unrealized_pnl,
            }

        return None
