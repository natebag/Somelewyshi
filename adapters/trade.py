"""Trading adapter — connects prediction output to Polymarket execution."""

from __future__ import annotations

import logging

from adapters.predict import PredictionResult
from trading.client import PolymarketClient
from trading.models import TradeOrder, TradeResult
from trading.risk import RiskManager
from trading.strategy import TradeSignal, evaluate_trade

logger = logging.getLogger(__name__)


class TradeAdapter:
    """Evaluates predictions and executes trades on Polymarket."""

    def __init__(
        self,
        client: PolymarketClient,
        risk_manager: RiskManager,
        bankroll: float = 500.0,
        kelly_fraction: float = 0.25,
        min_ev_threshold: float = 0.05,
        max_position_usd: float = 50.0,
        dry_run: bool = True,
    ):
        self.client = client
        self.risk = risk_manager
        self.bankroll = bankroll
        self.kelly_fraction = kelly_fraction
        self.min_ev_threshold = min_ev_threshold
        self.max_position_usd = max_position_usd
        self.dry_run = dry_run

    def evaluate(
        self,
        prediction: PredictionResult,
        market_price: float,
        token_id: str = "",
    ) -> TradeSignal:
        """Evaluate whether a prediction warrants a trade."""

        # Skip low-confidence predictions
        if prediction.confidence == "low":
            return TradeSignal(
                should_trade=False,
                direction="SKIP",
                ev_per_dollar=0.0,
                kelly_fraction=0.0,
                position_size=0.0,
                market_price=market_price,
                estimated_prob=prediction.probability,
                reasoning="Prediction confidence too low",
            )

        signal = evaluate_trade(
            market_price=market_price,
            true_prob=prediction.probability,
            bankroll=self.bankroll,
            kelly_multiplier=self.kelly_fraction,
            min_ev_threshold=self.min_ev_threshold,
            max_position_usd=self.max_position_usd,
        )

        # Risk check
        if signal.should_trade:
            adjusted_size, reject_reason = self.risk.check_trade(
                signal.position_size, self.bankroll
            )
            if reject_reason:
                signal.should_trade = False
                signal.position_size = 0.0
                signal.reasoning = reject_reason
            else:
                signal.position_size = adjusted_size

        return signal

    def execute(self, signal: TradeSignal, token_id: str = "") -> TradeResult:
        """Execute a trade based on the signal."""
        if not signal.should_trade:
            return TradeResult(
                order_id=None,
                status="SKIPPED",
                error=signal.reasoning,
            )

        order = TradeOrder(
            token_id=token_id,
            side=signal.direction,
            amount_usd=signal.position_size,
            market_price=signal.market_price,
            estimated_prob=signal.estimated_prob,
            ev_per_dollar=signal.ev_per_dollar,
            kelly_fraction=signal.kelly_fraction,
            dry_run=self.dry_run,
        )

        result = self.client.execute_order(order)

        if result.status == "DRY_RUN":
            logger.info(
                f"[DRY RUN] {signal.direction} ${signal.position_size:.2f} | "
                f"EV: {signal.ev_per_dollar:.4f} | Kelly: {signal.kelly_fraction:.1%}"
            )
        elif result.status == "EXECUTED":
            logger.info(
                f"[LIVE] {signal.direction} ${signal.position_size:.2f} | "
                f"Order: {result.order_id}"
            )

        return result
