"""APScheduler-based automated pipeline execution with market discovery."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from apscheduler.schedulers.blocking import BlockingScheduler

from adapters.llm import LLMClient
from adapters.predict import PredictAdapter
from db.engine import get_session, init_db
from db.models import PipelineRun, PipelineStatus, Prediction, Trade, TradeStatus
from orchestrator.pipeline import Pipeline
from trading.client import PolymarketClient
from trading.discovery import DiscoveryConfig, MarketDiscovery
from trading.positions import PositionTracker
from trading.scanner import MarketScanner
from trading.strategy import bayesian_update, evaluate_trade

logger = logging.getLogger(__name__)


class PipelineScheduler:
    """Runs automated discover + predict + trade cycles on a schedule."""

    def __init__(
        self,
        pipeline: Pipeline,
        poly_client: PolymarketClient,
        predict: PredictAdapter,
        interval_hours: int = 6,
        discovery_config: DiscoveryConfig | None = None,
        trading_config: dict | None = None,
    ):
        self.pipeline = pipeline
        self.poly_client = poly_client
        self.predict = predict
        self.interval_hours = interval_hours
        self.discovery_config = discovery_config or DiscoveryConfig()
        self.trading_config = trading_config or {}
        self.scheduler = BlockingScheduler()
        self.engine = init_db()

        # Manual market overrides
        self.tracked_markets: list[dict] = []

    def add_market(self, question: str, market_id: str = "", token_id: str = ""):
        """Manually add a market to track alongside auto-discovery."""
        self.tracked_markets.append({
            "question": question,
            "market_id": market_id,
            "token_id": token_id,
        })

    def _run_cycle(self):
        """Execute one full discover + analyze + trade cycle."""
        cycle_start = datetime.now(timezone.utc)
        logger.info("=" * 60)
        logger.info(f"CYCLE START: {cycle_start.strftime('%Y-%m-%d %H:%M UTC')}")
        logger.info("=" * 60)

        # Phase 1: Discover markets
        logger.info("[DISCOVER] Scanning Polymarket for opportunities...")
        scanner = MarketScanner(client=self.poly_client)
        discovery = MarketDiscovery(scanner, self.discovery_config)
        discovered = discovery.discover()

        # Combine with manually tracked markets
        markets_to_analyze = []
        for d in discovered:
            markets_to_analyze.append({
                "question": d.snapshot.question,
                "market_id": d.snapshot.condition_id,
                "token_id": d.snapshot.token_id_yes,
                "yes_price": d.snapshot.yes_price,
                "volume": d.snapshot.volume,
            })
        for m in self.tracked_markets:
            markets_to_analyze.append(m)

        logger.info(f"[DISCOVER] {len(markets_to_analyze)} markets to analyze")

        # Phase 2: Analyze + evaluate each market
        opportunities = []
        skipped = 0

        for market in markets_to_analyze:
            try:
                result = self.pipeline.run(
                    market_question=market["question"],
                    market_id=market.get("market_id", ""),
                    market_price=market.get("yes_price"),
                    token_id=market.get("token_id", ""),
                    skip_broadcast=True,  # only broadcast winners
                )

                if result.trade_signal.should_trade:
                    opportunities.append((market, result))
                    logger.info(
                        f"  [EDGE] {market['question'][:50]}... | "
                        f"Edge: {result.trade_signal.ev_per_dollar:.1%} | "
                        f"Bet: ${result.trade_signal.position_size:.2f}"
                    )
                else:
                    skipped += 1

            except Exception as e:
                logger.error(f"  [FAIL] {market['question'][:50]}... | {e}")
                skipped += 1

        # Phase 3: Broadcast top opportunities
        if opportunities:
            logger.info(f"\n[BROADCAST] {len(opportunities)} trades found, broadcasting top...")
            # Broadcast the best trade only (to avoid spam)
            best_market, best_result = max(
                opportunities, key=lambda x: x[1].trade_signal.ev_per_dollar
            )
            try:
                pred_data = {
                    "market_question": best_market["question"],
                    "probability": best_result.prediction.probability,
                    "confidence": best_result.prediction.confidence,
                    "reasoning": best_result.prediction.reasoning,
                }
                trade_data = {
                    "direction": best_result.trade_signal.direction,
                    "position_size": best_result.trade_signal.position_size,
                    "ev_per_dollar": best_result.trade_signal.ev_per_dollar,
                    "kelly_fraction": best_result.trade_signal.kelly_fraction,
                    "market_price": best_result.trade_signal.market_price,
                    "status": best_result.trade_result.status,
                }
                self.pipeline.broadcast.broadcast(pred_data, trade_data)
            except Exception as e:
                logger.error(f"  Broadcast failed: {e}")

        # Phase 4: Update existing positions
        self._update_positions()

        # Summary
        logger.info("")
        logger.info(f"CYCLE COMPLETE: {datetime.now(timezone.utc).strftime('%H:%M UTC')}")
        logger.info(f"  Analyzed:  {len(markets_to_analyze)}")
        logger.info(f"  Trades:    {len(opportunities)}")
        logger.info(f"  Skipped:   {skipped}")
        logger.info("=" * 60)

    def _update_positions(self):
        """Update positions with real prices + repricing exit logic."""
        session = get_session(self.engine)

        from trading.position_monitor import PositionMonitor
        from trading.repricing import RepricingEngine

        monitor = PositionMonitor(
            session=session,
            poly_client=self.poly_client,
            repricing_engine=RepricingEngine(),
        )

        actions = monitor.update_all_positions()

        closes = [a for a in actions if a.get("action") == "CLOSED"]
        updates = [a for a in actions if a.get("action") == "UPDATED"]

        if closes:
            logger.info(f"[POSITIONS] Closed {len(closes)} positions:")
            for c in closes:
                logger.info(
                    f"  P&L: ${c['pnl']:+.2f} | held {c['hold_seconds']}s | {c['reason']}"
                )
        if updates:
            logger.info(f"[POSITIONS] Updated {len(updates)} positions")
            except Exception as e:
                logger.debug(f"  Failed to update {pos.position_id}: {e}")

        session.close()

    def start(self):
        """Start the scheduler daemon."""
        logger.info(
            f"Miro Fish daemon starting | "
            f"Interval: {self.interval_hours}h | "
            f"Manual markets: {len(self.tracked_markets)} | "
            f"Auto-discover: {self.discovery_config.max_markets_to_analyze} markets"
        )

        # Run immediately
        self._run_cycle()

        # Schedule recurring
        self.scheduler.add_job(
            self._run_cycle,
            "interval",
            hours=self.interval_hours,
            id="pipeline_cycle",
        )

        try:
            self.scheduler.start()
        except (KeyboardInterrupt, SystemExit):
            logger.info("Daemon stopped")
