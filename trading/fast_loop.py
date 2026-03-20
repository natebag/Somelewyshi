"""BTC Fast Loop strategy — 5-minute market repricing arbitrage.

The core insight: don't predict the outcome, capture the repricing spread.
Buy mispriced probability, exit when it corrects — before resolution.

Inspired by Nunchi-trade/auto-researchtrading ensemble voting pattern.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


@dataclass
class FastLoopSignal:
    """Signal from the fast loop strategy."""

    direction: str  # "BUY_YES", "BUY_NO", "SKIP"
    confidence: float  # 0.0 - 1.0
    edge: float  # expected repricing spread
    entry_price: float
    target_exit_price: float
    stop_loss_price: float
    signals_agree: int  # how many of N signals agree
    total_signals: int
    reasoning: str
    time_to_expiry_seconds: int = 0


@dataclass
class PriceBar:
    """A single price observation for a market."""

    timestamp: datetime
    yes_price: float
    volume: float = 0.0


class FastLoopStrategy:
    """5-minute BTC market strategy using ensemble voting + repricing exits.

    Uses N-of-M signal agreement (inspired by Nunchi's 6-signal ensemble):
    1. Momentum — price movement direction on Polymarket
    2. Edge signal — our probability estimate vs market price
    3. Mean reversion — extreme prices tend to correct
    4. Volume signal — high volume = conviction
    5. Time decay — probability should approach 0 or 1 near expiry

    Exits based on repricing, NOT outcome prediction.
    """

    def __init__(
        self,
        min_signals_agree: int = 3,
        min_edge: float = 0.08,
        stop_loss_pct: float = 0.15,
        target_reprice_pct: float = 0.10,
    ):
        self.min_signals_agree = min_signals_agree
        self.min_edge = min_edge
        self.stop_loss_pct = stop_loss_pct
        self.target_reprice_pct = target_reprice_pct
        self.price_history: list[PriceBar] = []

    def add_price(self, bar: PriceBar):
        """Add a price observation."""
        self.price_history.append(bar)
        # Keep last 100 observations
        if len(self.price_history) > 100:
            self.price_history = self.price_history[-100:]

    def evaluate(
        self,
        current_price: float,
        estimated_prob: float,
        time_to_expiry_seconds: int = 300,
        current_volume: float = 0.0,
    ) -> FastLoopSignal:
        """Evaluate whether to enter a fast loop trade.

        Args:
            current_price: Current YES price on Polymarket
            estimated_prob: Our probability estimate (from Claude/MiroFish)
            time_to_expiry_seconds: Seconds until market resolves
            current_volume: Current trading volume
        """
        signals = []

        # Signal 1: Edge signal (our estimate vs market)
        edge = estimated_prob - current_price
        edge_signal = self._edge_signal(edge)
        signals.append(("edge", edge_signal))

        # Signal 2: Momentum (price direction from recent history)
        momentum_signal = self._momentum_signal(current_price)
        signals.append(("momentum", momentum_signal))

        # Signal 3: Mean reversion (extreme prices tend to correct)
        reversion_signal = self._mean_reversion_signal(current_price)
        signals.append(("reversion", reversion_signal))

        # Signal 4: Volume confirmation
        volume_signal = self._volume_signal(current_volume)
        signals.append(("volume", volume_signal))

        # Signal 5: Time decay (don't enter too close to expiry)
        time_signal = self._time_decay_signal(
            current_price, time_to_expiry_seconds
        )
        signals.append(("time", time_signal))

        # Count agreement
        bullish = sum(1 for _, s in signals if s > 0)
        bearish = sum(1 for _, s in signals if s < 0)
        total = len(signals)

        # Determine direction based on edge + ensemble
        if edge > self.min_edge and bullish >= self.min_signals_agree:
            direction = "BUY_YES"
            target_exit = min(current_price + self.target_reprice_pct, 0.95)
            stop_loss = max(current_price - self.stop_loss_pct, 0.05)
        elif edge < -self.min_edge and bearish >= self.min_signals_agree:
            direction = "BUY_NO"
            target_exit = max(current_price - self.target_reprice_pct, 0.05)
            stop_loss = min(current_price + self.stop_loss_pct, 0.95)
        else:
            direction = "SKIP"
            target_exit = current_price
            stop_loss = current_price

        # Build reasoning
        signal_details = ", ".join(
            f"{name}={'↑' if val > 0 else '↓' if val < 0 else '—'}"
            for name, val in signals
        )

        return FastLoopSignal(
            direction=direction,
            confidence=max(bullish, bearish) / total,
            edge=abs(edge),
            entry_price=current_price,
            target_exit_price=target_exit,
            stop_loss_price=stop_loss,
            signals_agree=max(bullish, bearish),
            total_signals=total,
            reasoning=f"Ensemble [{signal_details}] | {max(bullish, bearish)}/{total} agree",
            time_to_expiry_seconds=time_to_expiry_seconds,
        )

    def should_exit(
        self,
        entry_price: float,
        current_price: float,
        direction: str,
        time_to_expiry_seconds: int,
    ) -> tuple[bool, str]:
        """Decide whether to exit a fast loop position.

        Key insight: exit on repricing, not on outcome.
        """
        # Exit 15 seconds before resolution (avoid settlement risk)
        if time_to_expiry_seconds < 15:
            return True, "Time exit: <15s to expiry"

        if direction == "BUY_YES":
            # Take profit: price repriced up
            gain = current_price - entry_price
            if gain >= self.target_reprice_pct:
                return True, f"Target hit: +{gain:.0%} repricing captured"

            # Stop loss
            loss = entry_price - current_price
            if loss >= self.stop_loss_pct:
                return True, f"Stop loss: -{loss:.0%}"

        elif direction == "BUY_NO":
            # Take profit: YES price dropped (NO gained)
            gain = entry_price - current_price
            if gain >= self.target_reprice_pct:
                return True, f"Target hit: +{gain:.0%} repricing captured"

            # Stop loss
            loss = current_price - entry_price
            if loss >= self.stop_loss_pct:
                return True, f"Stop loss: -{loss:.0%}"

        return False, ""

    # ── Individual signals ───────────────────────────────────

    def _edge_signal(self, edge: float) -> int:
        """Our estimate vs market price. Core signal."""
        if edge > self.min_edge:
            return 1  # bullish — market underpricing YES
        elif edge < -self.min_edge:
            return -1  # bearish — market overpricing YES
        return 0

    def _momentum_signal(self, current_price: float) -> int:
        """Price direction from recent history."""
        if len(self.price_history) < 3:
            return 0

        recent = [b.yes_price for b in self.price_history[-5:]]
        avg_recent = sum(recent) / len(recent)
        older = [b.yes_price for b in self.price_history[-10:-5]] if len(self.price_history) >= 10 else recent
        avg_older = sum(older) / len(older)

        diff = avg_recent - avg_older
        if diff > 0.02:
            return 1  # upward momentum
        elif diff < -0.02:
            return -1  # downward momentum
        return 0

    def _mean_reversion_signal(self, current_price: float) -> int:
        """Extreme prices tend to correct toward 0.50."""
        if current_price > 0.85:
            return -1  # likely to revert down
        elif current_price < 0.15:
            return 1  # likely to revert up
        return 0

    def _volume_signal(self, current_volume: float) -> int:
        """High volume = conviction behind the move."""
        if not self.price_history:
            return 0

        volumes = [b.volume for b in self.price_history if b.volume > 0]
        if not volumes:
            return 0

        avg_vol = sum(volumes) / len(volumes)
        if current_volume > avg_vol * 1.5:
            return 1  # high volume confirms direction
        elif current_volume < avg_vol * 0.5:
            return -1  # low volume = weak move
        return 0

    def _time_decay_signal(
        self, current_price: float, time_to_expiry_seconds: int
    ) -> int:
        """Time decay: near expiry, probability should be near 0 or 1."""
        if time_to_expiry_seconds < 30:
            return 0  # too close, don't enter

        # If far from expiry and price is mid-range, neutral
        if time_to_expiry_seconds > 180 and 0.30 < current_price < 0.70:
            return 0

        # Near expiry but price still mid-range = opportunity
        if time_to_expiry_seconds < 120 and 0.35 < current_price < 0.65:
            return 1 if current_price > 0.50 else -1

        return 0
