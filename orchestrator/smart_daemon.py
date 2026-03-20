"""Smart daemon — unified scanner with per-market-type frequency.

Instead of one interval for everything:
- BTC fast markets: every 2 minutes
- Sports: every 30 minutes
- Political/weather: every hour

Runs all loops in one process using APScheduler.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from apscheduler.schedulers.blocking import BlockingScheduler

from adapters.predict import PredictAdapter
from db.engine import get_session, init_db
from orchestrator.pipeline import Pipeline
from trading.client import PolymarketClient
from trading.market_classifier import MarketType, classify_market
from trading.scanner import MarketScanner, ScanFilter

logger = logging.getLogger(__name__)


class SmartDaemon:
    """Unified daemon with smart scan frequency per market type."""

    def __init__(
        self,
        pipeline: Pipeline,
        poly_client: PolymarketClient,
        max_markets: int = 20,
    ):
        self.pipeline = pipeline
        self.poly_client = poly_client
        self.max_markets = max_markets
        self.scheduler = BlockingScheduler()
        self.engine = init_db()

        # Track last scan time per market type
        self._last_scan: dict[MarketType, float] = {}
        self._cycle_count = 0

    def _run_fast_loop(self):
        """Scan BTC fast markets (every 2 minutes)."""
        self.pipeline.predict.reset_mirofish()
        self._scan_markets_of_type(
            [MarketType.BTC_FAST],
            max_markets=3,
            label="BTC FAST",
        )

    def _run_medium_loop(self):
        """Scan crypto + sports markets (every 15-30 minutes)."""
        self.pipeline.predict.reset_mirofish()
        self._scan_markets_of_type(
            [MarketType.CRYPTO, MarketType.SPORTS],
            max_markets=5,
            label="CRYPTO/SPORTS",
        )

    def _run_slow_loop(self):
        """Scan political, weather, economic markets (every hour)."""
        self.pipeline.predict.reset_mirofish()
        self._scan_markets_of_type(
            [MarketType.POLITICAL, MarketType.ECONOMIC, MarketType.WEATHER, MarketType.OTHER],
            max_markets=5,
            label="SLOW",
        )

    def _run_maintenance(self):
        """Position monitoring, resolution tracking, auto-optimize (every 5 min)."""
        self._cycle_count += 1
        now = datetime.now(timezone.utc).strftime("%H:%M:%S")

        # Monitor positions
        session = get_session(self.engine)
        try:
            from trading.position_monitor import PositionMonitor
            from trading.repricing import RepricingEngine

            monitor = PositionMonitor(
                session=session,
                poly_client=self.poly_client,
                repricing_engine=RepricingEngine(),
            )
            actions = monitor.update_all_positions()

            closes = [a for a in actions if a.get("action") == "CLOSED"]
            if closes:
                logger.info(f"[{now}] [POSITIONS] Closed {len(closes)}:")
                for c in closes:
                    logger.info(f"  P&L: ${c['pnl']:+.2f} | {c['reason']}")

            # Check resolutions
            from trading.resolution import ResolutionTracker

            resolver = ResolutionTracker(session, self.poly_client)
            validations = resolver.check_resolutions()

            if validations:
                logger.info(f"[{now}] [RESOLUTION] {len(validations)} markets resolved:")
                for v in validations:
                    correct_str = "CORRECT" if v.prediction_correct else "WRONG"
                    logger.info(
                        f"  {v.market_question[:40]}... | {correct_str} | "
                        f"Repricing: ${v.repricing_exit_pnl:+.2f} vs Hold: ${v.hold_to_resolution_pnl:+.2f}"
                    )

            # Auto-optimize (check every 10 cycles = ~50 minutes)
            if self._cycle_count % 10 == 0:
                self._check_auto_optimize(session)

        except Exception as e:
            logger.error(f"Maintenance error: {e}")
        finally:
            session.close()

    def _scan_markets_of_type(
        self, types: list[MarketType], max_markets: int, label: str
    ):
        """Scan and trade markets matching given types."""
        now = datetime.now(timezone.utc).strftime("%H:%M:%S")
        logger.info(f"[{now}] [{label}] Scanning...")

        try:
            scanner = MarketScanner(
                self.poly_client,
                ScanFilter(min_volume=1000, max_markets=self.max_markets),
            )
            all_results = scanner.scan()

            # Filter by market type
            typed_results = []
            for r in all_results:
                profile = classify_market(r.snapshot.question)
                if profile.market_type in types:
                    typed_results.append((r, profile))

            typed_results.sort(key=lambda x: x[1].priority, reverse=True)
            typed_results = typed_results[:max_markets]

            if not typed_results:
                logger.info(f"[{now}] [{label}] No markets found")
                return

            logger.info(f"[{now}] [{label}] Found {len(typed_results)} markets")

            opportunities = 0
            for r, profile in typed_results:
                snap = r.snapshot
                try:
                    result = self.pipeline.run(
                        market_question=snap.question,
                        market_id=snap.condition_id,
                        market_price=snap.yes_price,
                        token_id=snap.token_id_yes,
                        skip_broadcast=True,
                    )

                    if result.trade_signal.should_trade:
                        opportunities += 1
                        logger.info(
                            f"  [{profile.market_type.value}] {snap.question[:45]}... | "
                            f"Edge: {result.trade_signal.ev_per_dollar:.0%} | "
                            f"${result.trade_signal.position_size:.2f}"
                        )

                except Exception as e:
                    logger.error(f"  Error on {snap.question[:30]}...: {e}")

            logger.info(f"[{now}] [{label}] {opportunities} trades from {len(typed_results)} markets")

        except Exception as e:
            logger.error(f"[{label}] Scan error: {e}")

    def _check_auto_optimize(self, session):
        """Check win rate and auto-optimize if needed."""
        from trading.positions import PositionTracker

        tracker = PositionTracker(session)
        summary = tracker.get_portfolio_summary()

        if summary.closed_positions < 10:
            return

        if summary.win_rate < 0.55:
            logger.warning(
                f"[AUTO-OPTIMIZE] Win rate {summary.win_rate:.0%} — triggering optimization"
            )

            from db.models import Trade, TradeStatus
            from trading.optimizer import HistoricalTrade, StrategyOptimizer
            from sqlmodel import select

            trades = session.exec(
                select(Trade).where(
                    Trade.status.in_([TradeStatus.DRY_RUN, TradeStatus.EXECUTED])
                )
            ).all()

            if len(trades) < 10:
                return

            hist_trades = [
                HistoricalTrade(
                    market_id=t.market_id or "",
                    question="",
                    entry_price=t.market_price_at_entry,
                    exit_price=t.fill_price or t.market_price_at_entry,
                    our_estimate=t.market_price_at_entry + t.expected_value,
                    direction=t.side,
                    resolution="YES" if t.expected_value > 0 else "NO",
                    time_to_expiry_at_entry=300,
                    hold_duration_seconds=120,
                    size_usd=t.size_usd,
                )
                for t in trades
            ]

            optimizer = StrategyOptimizer(historical_trades=hist_trades)
            best = optimizer.run_optimization(num_experiments=30)
            logger.info(
                f"[AUTO-OPTIMIZE] New params: ev={best.min_ev_threshold:.3f}, "
                f"kelly={best.kelly_fraction:.3f}"
            )

    def start(self):
        """Start the smart daemon with all scan loops."""
        logger.info("=" * 60)
        logger.info("MIRO FISH SMART DAEMON STARTING")
        logger.info("=" * 60)
        logger.info("Scan loops:")
        logger.info("  BTC Fast:      every 2 minutes")
        logger.info("  Crypto/Sports: every 15 minutes")
        logger.info("  Political/Wx:  every 60 minutes")
        logger.info("  Maintenance:   every 5 minutes")
        logger.info("=" * 60)

        # Run initial scans immediately — fast first (time-sensitive)
        self._run_fast_loop()
        self._run_medium_loop()
        self._run_slow_loop()
        self._run_maintenance()

        # Schedule recurring loops
        self.scheduler.add_job(
            self._run_fast_loop, "interval", minutes=2, id="fast_loop"
        )
        self.scheduler.add_job(
            self._run_medium_loop, "interval", minutes=15, id="medium_loop"
        )
        self.scheduler.add_job(
            self._run_slow_loop, "interval", minutes=60, id="slow_loop"
        )
        self.scheduler.add_job(
            self._run_maintenance, "interval", minutes=5, id="maintenance"
        )

        try:
            self.scheduler.start()
        except (KeyboardInterrupt, SystemExit):
            logger.info("Smart daemon stopped")
