"""Market resolution tracker — checks if markets resolved and validates predictions.

This is the critical validation layer:
- Polls Polymarket for resolved markets
- Records actual YES/NO outcome
- Compares our prediction vs reality
- Tracks repricing exit P&L vs hold-to-resolution P&L
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone

import httpx
from sqlmodel import Session, select

from db.models import Position, PositionStatus, Prediction, Trade
from trading.client import PolymarketClient

logger = logging.getLogger(__name__)

GAMMA_API = "https://gamma-api.polymarket.com"


@dataclass
class ResolutionResult:
    """Result of checking a market resolution."""

    market_id: str
    question: str
    resolved: bool
    outcome: str  # "YES", "NO", "SPLIT", "UNRESOLVED"
    winning_token_id: str = ""
    resolution_price: float = 0.0  # 1.0 for winner, 0.0 for loser


@dataclass
class ValidationResult:
    """Comparison of predicted vs actual outcome."""

    prediction_id: str
    market_question: str
    predicted_prob: float
    actual_outcome: str
    prediction_correct: bool
    repricing_exit_pnl: float  # What we made with repricing exit
    hold_to_resolution_pnl: float  # What we WOULD have made holding
    pnl_difference: float  # repricing - hold (positive = repricing was better)


class ResolutionTracker:
    """Tracks market resolutions and validates predictions."""

    def __init__(self, session: Session, poly_client: PolymarketClient):
        self.session = session
        self.poly_client = poly_client

    def check_resolutions(self) -> list[ValidationResult]:
        """Check all open positions for resolved markets.

        Returns validation results for any newly resolved markets.
        """
        # Get all positions (open and closed) that don't have resolution recorded
        positions = self.session.exec(select(Position)).all()

        if not positions:
            return []

        results = []
        checked_markets = set()

        for pos in positions:
            if not pos.market_id or pos.market_id in checked_markets:
                continue
            checked_markets.add(pos.market_id)

            resolution = self._check_market_resolution(pos.market_id)
            if not resolution or not resolution.resolved:
                continue

            logger.info(
                f"[RESOLUTION] Market resolved: {resolution.question[:50]}... → {resolution.outcome}"
            )

            # Find the prediction for this market
            validation = self._validate_prediction(pos, resolution)
            if validation:
                results.append(validation)

                # Update prediction record with resolution
                self._record_resolution(pos, resolution)

        return results

    def _check_market_resolution(self, condition_id: str) -> ResolutionResult | None:
        """Check if a specific market has resolved via Gamma API."""
        try:
            with httpx.Client(timeout=15) as http:
                resp = http.get(
                    f"{GAMMA_API}/markets",
                    params={"conditionId": condition_id},
                )
                resp.raise_for_status()
                markets = resp.json()

            if not markets:
                return None

            m = markets[0] if isinstance(markets, list) else markets

            # Check resolution status
            is_closed = m.get("closed", False)
            if not is_closed:
                return ResolutionResult(
                    market_id=condition_id,
                    question=m.get("question", ""),
                    resolved=False,
                    outcome="UNRESOLVED",
                )

            # Determine outcome from token prices
            # Resolved market: winning token price = 1.0, losing = 0.0
            import json
            outcome_prices = m.get("outcomePrices", [])
            if isinstance(outcome_prices, str):
                try:
                    outcome_prices = json.loads(outcome_prices)
                except (json.JSONDecodeError, ValueError):
                    outcome_prices = []

            outcomes = m.get("outcomes", [])
            if isinstance(outcomes, str):
                try:
                    outcomes = json.loads(outcomes)
                except (json.JSONDecodeError, ValueError):
                    outcomes = ["Yes", "No"]

            if len(outcome_prices) >= 2:
                yes_price = float(outcome_prices[0])
                no_price = float(outcome_prices[1])

                if yes_price > 0.9:
                    outcome = "YES"
                elif no_price > 0.9:
                    outcome = "NO"
                else:
                    outcome = "SPLIT"
            else:
                outcome = "UNRESOLVED"

            clob_ids = m.get("clobTokenIds", [])
            if isinstance(clob_ids, str):
                try:
                    clob_ids = json.loads(clob_ids)
                except (json.JSONDecodeError, ValueError):
                    clob_ids = []

            winning_token = ""
            if outcome == "YES" and len(clob_ids) >= 1:
                winning_token = clob_ids[0]
            elif outcome == "NO" and len(clob_ids) >= 2:
                winning_token = clob_ids[1]

            return ResolutionResult(
                market_id=condition_id,
                question=m.get("question", ""),
                resolved=True,
                outcome=outcome,
                winning_token_id=winning_token,
                resolution_price=1.0 if outcome in ("YES", "NO") else 0.5,
            )

        except Exception as e:
            logger.debug(f"Failed to check resolution for {condition_id}: {e}")
            return None

    def _validate_prediction(
        self, pos: Position, resolution: ResolutionResult
    ) -> ValidationResult | None:
        """Compare our prediction and trade against the actual outcome."""
        # Find the prediction associated with this position's trade
        trade = self.session.exec(
            select(Trade).where(Trade.market_id == pos.market_id)
        ).first()

        if not trade:
            return None

        pred = self.session.exec(
            select(Prediction).where(Prediction.id == trade.prediction_id)
        ).first()

        if not pred:
            return None

        # Was the prediction directionally correct?
        if resolution.outcome == "YES":
            prediction_correct = pred.probability > 0.5
        elif resolution.outcome == "NO":
            prediction_correct = pred.probability < 0.5
        else:
            prediction_correct = False

        # Calculate hold-to-resolution P&L
        # If we held: we get $1 per share if we bet on the winner, $0 if loser
        entry_price = pos.entry_price
        size_usd = pos.size_usd

        if pos.side in ("BUY_YES", "YES"):
            shares = size_usd / entry_price if entry_price > 0 else 0
            if resolution.outcome == "YES":
                hold_pnl = shares * (1.0 - entry_price)  # Won: get $1/share
            else:
                hold_pnl = shares * (0.0 - entry_price)  # Lost: get $0/share
        else:  # BUY_NO
            no_entry = 1 - entry_price
            shares = size_usd / no_entry if no_entry > 0 else 0
            if resolution.outcome == "NO":
                hold_pnl = shares * (1.0 - no_entry)
            else:
                hold_pnl = shares * (0.0 - no_entry)

        # Repricing exit P&L (what we actually made)
        repricing_pnl = pos.unrealized_pnl or 0.0

        # If position is still open, close it now at resolution price
        if pos.status == PositionStatus.OPEN:
            if pos.side in ("BUY_YES", "YES"):
                res_price = 1.0 if resolution.outcome == "YES" else 0.0
                repricing_pnl = shares * (res_price - entry_price)
            else:
                res_price = 0.0 if resolution.outcome == "NO" else 1.0
                no_exit = 1 - res_price
                repricing_pnl = shares * (no_exit - no_entry)

        pnl_diff = repricing_pnl - hold_pnl

        logger.info(
            f"[VALIDATION] {resolution.question[:40]}... | "
            f"Predicted: {pred.probability:.0%} | Actual: {resolution.outcome} | "
            f"{'CORRECT' if prediction_correct else 'WRONG'} | "
            f"Repricing P&L: ${repricing_pnl:+.2f} | "
            f"Hold P&L: ${hold_pnl:+.2f} | "
            f"Diff: ${pnl_diff:+.2f} ({'repricing better' if pnl_diff > 0 else 'hold better'})"
        )

        return ValidationResult(
            prediction_id=pred.id,
            market_question=resolution.question,
            predicted_prob=pred.probability,
            actual_outcome=resolution.outcome,
            prediction_correct=prediction_correct,
            repricing_exit_pnl=repricing_pnl,
            hold_to_resolution_pnl=hold_pnl,
            pnl_difference=pnl_diff,
        )

    def _record_resolution(self, pos: Position, resolution: ResolutionResult):
        """Record the resolution in our database."""
        # Update prediction with resolution
        trade = self.session.exec(
            select(Trade).where(Trade.market_id == pos.market_id)
        ).first()

        if trade:
            pred = self.session.exec(
                select(Prediction).where(Prediction.id == trade.prediction_id)
            ).first()
            if pred:
                pred.resolution = resolution.outcome
                pred.resolved_at = datetime.now(timezone.utc)
                self.session.add(pred)

        # Close position if still open
        if pos.status == PositionStatus.OPEN:
            if pos.side in ("BUY_YES", "YES"):
                pos.current_price = 1.0 if resolution.outcome == "YES" else 0.0
            else:
                pos.current_price = 0.0 if resolution.outcome == "NO" else 1.0

            # Recalc P&L at resolution price
            if pos.side in ("BUY_YES", "YES"):
                shares = pos.size_usd / pos.entry_price if pos.entry_price > 0 else 0
                pos.unrealized_pnl = shares * (pos.current_price - pos.entry_price)
            else:
                no_entry = 1 - pos.entry_price
                no_current = 1 - pos.current_price
                shares = pos.size_usd / no_entry if no_entry > 0 else 0
                pos.unrealized_pnl = shares * (no_current - no_entry)

            pos.status = PositionStatus.CLOSED
            pos.closed_at = datetime.now(timezone.utc)
            self.session.add(pos)

        self.session.commit()
