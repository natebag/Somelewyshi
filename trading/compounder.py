"""BTC Compounder — starts small, compounds gains on 5-min markets.

The $100 → $1,000+ strategy:
- Start with a fixed bankroll (e.g., $100)
- Size every position off CURRENT balance, not starting balance
- Compound: $100 → win → $115 → win → $132 → ...
- Strict risk rules: daily loss limit, consecutive loss stop
- Focus exclusively on BTC 5-min / 15-min markets
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


@dataclass
class CompoundState:
    """Live state of the compounding bot."""

    starting_balance: float
    current_balance: float
    peak_balance: float  # High water mark
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    consecutive_losses: int = 0
    daily_pnl: float = 0.0
    daily_trades: int = 0
    last_reset_date: str = ""  # YYYY-MM-DD
    total_pnl: float = 0.0
    is_halted: bool = False
    halt_reason: str = ""

    @property
    def growth_pct(self) -> float:
        return ((self.current_balance - self.starting_balance) / self.starting_balance * 100)

    @property
    def win_rate(self) -> float:
        return self.winning_trades / self.total_trades if self.total_trades > 0 else 0

    @property
    def drawdown_pct(self) -> float:
        if self.peak_balance <= 0:
            return 0
        return (self.peak_balance - self.current_balance) / self.peak_balance * 100

    def summary(self) -> str:
        return (
            f"Balance: ${self.current_balance:.2f} "
            f"({'+' if self.growth_pct >= 0 else ''}{self.growth_pct:.1f}% from ${self.starting_balance:.2f}) | "
            f"Trades: {self.total_trades} ({self.win_rate:.0%} win) | "
            f"Today: ${self.daily_pnl:+.2f} ({self.daily_trades} trades) | "
            f"DD: {self.drawdown_pct:.1f}%"
        )


class Compounder:
    """Manages bankroll compounding with risk controls.

    Risk rules:
    1. Max daily loss: stop trading if down X% today
    2. Consecutive loss limit: pause after N losses in a row
    3. Max drawdown: halt if balance drops X% from peak
    4. Position sizing: always use quarter-Kelly off CURRENT balance
    5. Max single position: never risk more than X% of balance
    """

    def __init__(
        self,
        starting_balance: float = 100.0,
        max_daily_loss_pct: float = 5.0,    # Stop if down 5% today
        max_consecutive_losses: int = 4,     # Pause after 4 losses in a row
        max_drawdown_pct: float = 20.0,      # Halt if 20% below peak
        max_position_pct: float = 10.0,      # Max 10% of balance per trade
        kelly_fraction: float = 0.25,        # Quarter-Kelly
        min_ev_threshold: float = 0.05,      # Min 5% edge
    ):
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_consecutive_losses = max_consecutive_losses
        self.max_drawdown_pct = max_drawdown_pct
        self.max_position_pct = max_position_pct
        self.kelly_fraction = kelly_fraction
        self.min_ev_threshold = min_ev_threshold

        self.state = CompoundState(
            starting_balance=starting_balance,
            current_balance=starting_balance,
            peak_balance=starting_balance,
            last_reset_date=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        )

    def can_trade(self) -> tuple[bool, str]:
        """Check if we're allowed to trade right now."""
        # Reset daily counters if new day
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if today != self.state.last_reset_date:
            self.state.daily_pnl = 0.0
            self.state.daily_trades = 0
            self.state.last_reset_date = today
            self.state.is_halted = False
            self.state.halt_reason = ""
            self.state.consecutive_losses = 0
            logger.info(f"[COMPOUND] New day — daily counters reset")

        if self.state.is_halted:
            return False, f"HALTED: {self.state.halt_reason}"

        # Check daily loss limit
        daily_loss_pct = abs(self.state.daily_pnl) / self.state.current_balance * 100 if self.state.daily_pnl < 0 else 0
        if daily_loss_pct >= self.max_daily_loss_pct:
            self.state.is_halted = True
            self.state.halt_reason = f"Daily loss limit hit ({daily_loss_pct:.1f}%)"
            return False, self.state.halt_reason

        # Check consecutive losses
        if self.state.consecutive_losses >= self.max_consecutive_losses:
            self.state.is_halted = True
            self.state.halt_reason = f"{self.state.consecutive_losses} consecutive losses"
            return False, self.state.halt_reason

        # Check max drawdown
        if self.state.drawdown_pct >= self.max_drawdown_pct:
            self.state.is_halted = True
            self.state.halt_reason = f"Max drawdown hit ({self.state.drawdown_pct:.1f}%)"
            return False, self.state.halt_reason

        # Check minimum balance
        if self.state.current_balance < 5.0:
            self.state.is_halted = True
            self.state.halt_reason = "Balance too low (<$5)"
            return False, self.state.halt_reason

        return True, ""

    def calculate_position_size(self, ev_per_dollar: float, market_price: float) -> float:
        """Calculate position size based on CURRENT balance + Kelly.

        This is the compounding magic — as balance grows, positions grow.
        """
        if ev_per_dollar < self.min_ev_threshold:
            return 0.0

        # Kelly sizing off current balance
        if market_price <= 0 or market_price >= 1:
            return 0.0

        b = (1 - market_price) / market_price
        true_prob = ev_per_dollar + market_price  # rough estimate
        f = max((true_prob * b - (1 - true_prob)) / b, 0) * self.kelly_fraction

        position = self.state.current_balance * f

        # Cap at max position percentage
        max_pos = self.state.current_balance * (self.max_position_pct / 100)
        position = min(position, max_pos)

        # Minimum $1 position
        if position < 1.0:
            return 0.0

        return round(position, 2)

    def record_trade(self, pnl: float):
        """Record a trade result and update compounding state."""
        self.state.total_trades += 1
        self.state.daily_trades += 1
        self.state.daily_pnl += pnl
        self.state.total_pnl += pnl
        self.state.current_balance += pnl

        if pnl > 0:
            self.state.winning_trades += 1
            self.state.consecutive_losses = 0
        elif pnl < 0:
            self.state.losing_trades += 1
            self.state.consecutive_losses += 1

        # Update high water mark
        if self.state.current_balance > self.state.peak_balance:
            self.state.peak_balance = self.state.current_balance

        logger.info(
            f"[COMPOUND] Trade #{self.state.total_trades}: "
            f"${pnl:+.2f} | {self.state.summary()}"
        )

    def get_state(self) -> CompoundState:
        return self.state
