"""Repricing exit engine — exit before resolution, capture the spread.

Core insight from the Polymarket leaderboard analysis:
  The real money isn't in predicting outcomes.
  It's in buying mispriced probability and exiting when it corrects.

This module monitors positions and triggers exits based on
probability repricing, NOT outcome prediction.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


@dataclass
class RepricingExit:
    """Exit signal from the repricing engine."""

    should_exit: bool
    reason: str
    entry_price: float
    current_price: float
    profit_if_exit: float  # in dollars per share
    log_return: float
    hold_duration_seconds: int = 0


class RepricingEngine:
    """Monitors positions for repricing opportunities to capture spreads.

    Exit triggers (in priority order):
    1. Target repricing reached — probability corrected toward our estimate
    2. Edge disappeared — market and our estimate converged
    3. Adverse repricing — market moved against us past stop-loss
    4. Time-based exit — approaching resolution, reduce settlement risk
    5. Stale position — no meaningful price movement for too long
    """

    def __init__(
        self,
        target_capture_pct: float = 0.60,
        stop_loss_pct: float = 0.30,
        time_exit_seconds: int = 60,
        stale_threshold_seconds: int = 600,
        min_profit_to_exit: float = 0.02,
    ):
        """
        Args:
            target_capture_pct: Exit when we've captured this fraction of the
                                total expected repricing (e.g., 0.60 = exit at
                                60% of the way from entry to our estimate)
            stop_loss_pct: Exit if price moves this far against us
            time_exit_seconds: Exit this many seconds before resolution
            stale_threshold_seconds: Exit if no movement for this long
            min_profit_to_exit: Minimum profit per share to trigger target exit
        """
        self.target_capture_pct = target_capture_pct
        self.stop_loss_pct = stop_loss_pct
        self.time_exit_seconds = time_exit_seconds
        self.stale_threshold_seconds = stale_threshold_seconds
        self.min_profit_to_exit = min_profit_to_exit

    def check_exit(
        self,
        direction: str,
        entry_price: float,
        current_price: float,
        our_estimate: float,
        time_to_expiry_seconds: int | None = None,
        position_age_seconds: int = 0,
        last_price_change_seconds: int = 0,
        size_usd: float = 0.0,
    ) -> RepricingExit:
        """Check if a position should be exited based on repricing.

        Args:
            direction: "BUY_YES" or "BUY_NO"
            entry_price: Price we entered at
            current_price: Current market price
            our_estimate: Our probability estimate
            time_to_expiry_seconds: Seconds until resolution (None if no expiry)
            position_age_seconds: How long we've held
            last_price_change_seconds: Time since last meaningful price movement
            size_usd: Position size in dollars
        """
        # Calculate profit metrics
        if direction in ("BUY_YES", "YES"):
            profit_per_share = current_price - entry_price
            expected_repricing = our_estimate - entry_price
            adverse_move = entry_price - current_price
        else:  # BUY_NO
            profit_per_share = entry_price - current_price
            expected_repricing = entry_price - our_estimate
            adverse_move = current_price - entry_price

        log_ret = math.log(current_price / entry_price) if entry_price > 0 and current_price > 0 else 0

        base = RepricingExit(
            should_exit=False,
            reason="",
            entry_price=entry_price,
            current_price=current_price,
            profit_if_exit=profit_per_share,
            log_return=log_ret,
            hold_duration_seconds=position_age_seconds,
        )

        # Priority 1: Time exit — approaching resolution
        if time_to_expiry_seconds is not None and time_to_expiry_seconds < self.time_exit_seconds:
            base.should_exit = True
            base.reason = f"Time exit: {time_to_expiry_seconds}s to expiry (threshold: {self.time_exit_seconds}s)"
            return base

        # Priority 2: Stop loss — adverse repricing
        if adverse_move > self.stop_loss_pct:
            base.should_exit = True
            base.reason = f"Stop loss: {adverse_move:.1%} adverse move (threshold: {self.stop_loss_pct:.0%})"
            return base

        # Priority 3: Target repricing captured
        if expected_repricing > 0 and profit_per_share > self.min_profit_to_exit:
            capture_ratio = profit_per_share / expected_repricing
            if capture_ratio >= self.target_capture_pct:
                base.should_exit = True
                base.reason = (
                    f"Target captured: {capture_ratio:.0%} of expected repricing "
                    f"(+{profit_per_share:.2f}/share, target: {self.target_capture_pct:.0%})"
                )
                return base

        # Priority 4: Edge disappeared — market converged to our estimate
        remaining_edge = abs(our_estimate - current_price)
        if remaining_edge < 0.03:
            if profit_per_share > 0:
                base.should_exit = True
                base.reason = f"Edge converged: market at {current_price:.0%}, estimate at {our_estimate:.0%} — locking in +{profit_per_share:.2f}/share"
                return base

        # Priority 5: Stale position — no movement
        if last_price_change_seconds > self.stale_threshold_seconds and position_age_seconds > self.stale_threshold_seconds:
            base.should_exit = True
            base.reason = f"Stale: no price movement for {last_price_change_seconds}s — freeing capital"
            return base

        return base

    def calculate_optimal_exit(
        self,
        entry_price: float,
        our_estimate: float,
        direction: str,
    ) -> dict:
        """Calculate the optimal exit price targets for a position.

        Returns target and stop prices for order placement.
        """
        if direction in ("BUY_YES", "YES"):
            expected_move = our_estimate - entry_price
            target_price = entry_price + expected_move * self.target_capture_pct
            stop_price = max(entry_price - self.stop_loss_pct, 0.01)
        else:
            expected_move = entry_price - our_estimate
            target_price = entry_price - expected_move * self.target_capture_pct
            stop_price = min(entry_price + self.stop_loss_pct, 0.99)

        reward = abs(target_price - entry_price)
        risk = abs(stop_price - entry_price)
        rr_ratio = reward / risk if risk > 0 else 0

        return {
            "target_price": round(target_price, 4),
            "stop_price": round(stop_price, 4),
            "expected_profit_per_share": round(abs(target_price - entry_price), 4),
            "max_loss_per_share": round(risk, 4),
            "reward_risk_ratio": round(rr_ratio, 2),
        }
