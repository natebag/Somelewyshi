"""Main Predict → Trade → Broadcast pipeline."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone

from adapters.broadcast import BroadcastAdapter, BroadcastResult
from adapters.predict import PredictAdapter, PredictionResult
from adapters.trade import TradeAdapter
from db.engine import get_session, init_db
from db.models import (
    Broadcast,
    BroadcastStatus,
    PipelineRun,
    PipelineStatus,
    Prediction,
    Trade,
    TradeStatus,
)
from trading.models import TradeResult
from trading.positions import PositionTracker
from trading.strategy import TradeSignal

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Full output of a pipeline run."""

    run_id: str
    prediction: PredictionResult
    trade_signal: TradeSignal
    trade_result: TradeResult
    broadcasts: list[BroadcastResult]
    status: str


class Pipeline:
    """
    Orchestrates the full Predict → Trade → Broadcast loop.

    State machine: IDLE → PREDICTING → TRADING → BROADCASTING → DONE
    """

    def __init__(
        self,
        predict: PredictAdapter,
        trade: TradeAdapter,
        broadcast: BroadcastAdapter,
    ):
        self.predict = predict
        self.trade = trade
        self.broadcast = broadcast
        self.engine = init_db()

    def run(
        self,
        market_question: str,
        market_id: str = "",
        market_price: float | None = None,
        token_id: str = "",
        skip_broadcast: bool = False,
    ) -> PipelineResult:
        """Execute the full pipeline for a market question."""

        session = get_session(self.engine)

        # Create pipeline run record
        run = PipelineRun(market_query=market_question, market_id=market_id)
        session.add(run)
        session.commit()

        try:
            # ── Phase 1: PREDICT ─────────────────────────────────
            run.status = PipelineStatus.PREDICTING
            session.add(run)
            session.commit()

            logger.info(f"[PREDICT] Analyzing: {market_question}")
            prediction = self.predict.run_prediction(market_question)

            # Save prediction
            pred_record = Prediction(
                pipeline_run_id=run.id,
                probability=prediction.probability,
                confidence={"high": 0.9, "medium": 0.7, "low": 0.4}.get(
                    prediction.confidence, 0.5
                ),
                reasoning=prediction.reasoning,
                report_text=prediction.report_text,
                mirofish_session_id=prediction.session_id,
            )
            session.add(pred_record)
            session.commit()

            logger.info(
                f"[PREDICT] Result: {prediction.probability:.0%} "
                f"(confidence: {prediction.confidence})"
            )

            # ── Phase 2: TRADE ───────────────────────────────────
            run.status = PipelineStatus.TRADING
            session.add(run)
            session.commit()

            # Get market price if not provided
            if market_price is None:
                if token_id and self.trade.client:
                    market_price = self.trade.client.get_midpoint(token_id) or 0.5
                else:
                    market_price = 0.5
                    logger.warning("No market price available — using 0.5 default")

            signal = self.trade.evaluate(prediction, market_price, token_id)
            trade_result = self.trade.execute(signal, token_id)

            # Save trade
            trade_record = Trade(
                pipeline_run_id=run.id,
                prediction_id=pred_record.id,
                market_id=market_id,
                token_id=token_id,
                side=signal.direction,
                size_usd=signal.position_size,
                expected_value=signal.ev_per_dollar,
                kelly_fraction=signal.kelly_fraction,
                market_price_at_entry=market_price,
                order_id=trade_result.order_id,
                status=TradeStatus(trade_result.status),
                fill_price=trade_result.fill_price,
                skip_reason=signal.reasoning if not signal.should_trade else None,
            )
            session.add(trade_record)
            session.commit()

            logger.info(
                f"[TRADE] {signal.direction} | "
                f"Size: ${signal.position_size:.2f} | "
                f"EV: {signal.ev_per_dollar:.4f} | "
                f"Status: {trade_result.status}"
            )

            # Track position if trade was executed
            if trade_result.status in ("EXECUTED", "DRY_RUN") and signal.should_trade:
                tracker = PositionTracker(session)
                tracker.create_from_trade(trade_record)

            # ── Phase 3: BROADCAST ───────────────────────────────
            broadcasts = []
            if not skip_broadcast:
                run.status = PipelineStatus.BROADCASTING
                session.add(run)
                session.commit()

                pred_data = {
                    "market_question": market_question,
                    "probability": prediction.probability,
                    "confidence": prediction.confidence,
                    "reasoning": prediction.reasoning,
                }
                trade_data = {
                    "direction": signal.direction,
                    "position_size": signal.position_size,
                    "ev_per_dollar": signal.ev_per_dollar,
                    "kelly_fraction": signal.kelly_fraction,
                    "market_price": market_price,
                    "status": trade_result.status,
                }

                broadcasts = self.broadcast.broadcast(pred_data, trade_data)

                for bc in broadcasts:
                    bc_record = Broadcast(
                        pipeline_run_id=run.id,
                        platform=bc.platform,
                        content_text=bc.content_text,
                        post_url=bc.post_url,
                        status=BroadcastStatus(bc.status) if bc.status in BroadcastStatus.__members__ else BroadcastStatus.SKIPPED,
                    )
                    session.add(bc_record)
                session.commit()

            # ── Done ─────────────────────────────────────────────
            run.status = PipelineStatus.DONE
            run.completed_at = datetime.now(timezone.utc)
            session.add(run)
            session.commit()

            return PipelineResult(
                run_id=run.id,
                prediction=prediction,
                trade_signal=signal,
                trade_result=trade_result,
                broadcasts=broadcasts,
                status="DONE",
            )

        except Exception as e:
            run.status = PipelineStatus.FAILED
            run.error_message = str(e)
            run.completed_at = datetime.now(timezone.utc)
            session.add(run)
            session.commit()
            logger.error(f"Pipeline failed: {e}")
            raise
        finally:
            session.close()
