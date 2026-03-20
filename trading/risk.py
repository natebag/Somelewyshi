"""Risk management: position limits, exposure caps, stop-loss."""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RiskLimits:
    """Configurable risk parameters."""

    max_position_usd: float = 50.0
    max_total_exposure_usd: float = 200.0
    max_positions: int = 10
    max_kelly_fraction: float = 0.25
    min_bankroll_reserve: float = 0.20  # keep 20% in reserve


class RiskManager:
    """Enforces risk limits on trade decisions."""

    def __init__(self, limits: RiskLimits, current_exposure: float = 0.0, open_positions: int = 0):
        self.limits = limits
        self.current_exposure = current_exposure
        self.open_positions = open_positions

    def check_trade(self, position_size: float, bankroll: float) -> tuple[float, str | None]:
        """
        Validate and potentially reduce a position size.

        Returns:
            (adjusted_size, rejection_reason)
            If rejected, adjusted_size is 0 and reason explains why.
        """
        # Cap individual position
        size = min(position_size, self.limits.max_position_usd)

        # Check total exposure
        if self.current_exposure + size > self.limits.max_total_exposure_usd:
            remaining = self.limits.max_total_exposure_usd - self.current_exposure
            if remaining <= 0:
                return 0.0, f"Max total exposure reached (${self.limits.max_total_exposure_usd:.0f})"
            size = remaining
            logger.warning(f"Position reduced to ${size:.2f} due to exposure limits")

        # Check position count
        if self.open_positions >= self.limits.max_positions:
            return 0.0, f"Max positions reached ({self.limits.max_positions})"

        # Reserve check
        min_reserve = bankroll * self.limits.min_bankroll_reserve
        available = bankroll - self.current_exposure - min_reserve
        if size > available:
            if available <= 0:
                return 0.0, "Bankroll reserve limit reached"
            size = available
            logger.warning(f"Position reduced to ${size:.2f} due to reserve requirement")

        return round(size, 2), None
