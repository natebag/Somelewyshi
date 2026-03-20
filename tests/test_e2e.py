"""End-to-end dry-run test — exercises the full pipeline without API keys."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from adapters.broadcast import BroadcastAdapter
from adapters.predict import PredictAdapter, PredictionResult
from adapters.trade import TradeAdapter
from config.schemas import Settings, TradingConfig
from db.engine import init_db, get_session
from orchestrator.pipeline import Pipeline
from trading.client import PolymarketClient
from trading.models import TradeResult
from trading.positions import PositionTracker


def _mock_predict() -> PredictAdapter:
    adapter = PredictAdapter.__new__(PredictAdapter)
    adapter.llm = MagicMock()
    adapter.base_url = "http://localhost:5001"
    adapter.run_prediction = MagicMock(return_value=PredictionResult(
        probability=0.65,
        confidence="high",
        reasoning="Test: strong edge based on recent data",
    ))
    adapter.is_mirofish_available = MagicMock(return_value=False)
    return adapter


def _mock_trade(settings: Settings) -> TradeAdapter:
    client = MagicMock(spec=PolymarketClient)
    client.get_midpoint.return_value = 0.42
    client.execute_order.return_value = TradeResult(
        status="DRY_RUN", order_id="test-123", fill_price=0.42
    )
    from trading.risk import RiskLimits, RiskManager
    risk = RiskManager(
        limits=RiskLimits(
            max_position_usd=settings.trading.max_position_usd,
            max_total_exposure_usd=settings.trading.bankroll_usd,
        ),
    )
    adapter = TradeAdapter(
        client=client,
        risk_manager=risk,
        bankroll=settings.trading.bankroll_usd,
        kelly_fraction=settings.trading.kelly_fraction,
        min_ev_threshold=settings.trading.min_ev_threshold,
        max_position_usd=settings.trading.max_position_usd,
        dry_run=True,
    )
    return adapter


def test_full_pipeline_dry_run():
    """Full predict -> trade -> position tracking cycle."""
    settings = Settings()
    settings.trading = TradingConfig(
        bankroll_usd=1000,
        max_position_usd=100,
        kelly_fraction=0.25,
        min_ev_threshold=0.05,
        dry_run=True,
    )

    predict = _mock_predict()
    trade = _mock_trade(settings)
    broadcast = BroadcastAdapter(llm=MagicMock(), twitter_enabled=False, youtube_enabled=False)

    pipeline = Pipeline(predict=predict, trade=trade, broadcast=broadcast)

    result = pipeline.run(
        market_question="Will BTC exceed $100K by end of March 2026?",
        market_id="test-market-btc",
        market_price=0.42,
        token_id="test-token",
        skip_broadcast=True,
    )

    # Verify prediction was made
    assert result.prediction.probability == 0.65
    assert result.prediction.confidence == "high"

    # Verify trade signal was calculated correctly
    sig = result.trade_signal
    assert sig.should_trade is True
    assert sig.direction == "BUY_YES"
    assert sig.ev_per_dollar > 0.05  # above threshold
    assert sig.position_size > 0
    assert sig.position_size <= 100  # within max

    # Verify trade was "executed" (dry run)
    assert result.trade_result.status == "DRY_RUN"


def test_pipeline_skips_low_edge():
    """Pipeline should skip when Claude estimate is close to market price."""
    settings = Settings()
    settings.trading = TradingConfig(
        bankroll_usd=1000,
        min_ev_threshold=0.05,
        dry_run=True,
    )

    predict = PredictAdapter.__new__(PredictAdapter)
    predict.llm = MagicMock()
    predict.base_url = "http://localhost:5001"
    predict.run_prediction = MagicMock(return_value=PredictionResult(
        probability=0.43,  # Very close to market price of 0.42
        confidence="low",
        reasoning="Not confident",
    ))
    predict.is_mirofish_available = MagicMock(return_value=False)

    trade = _mock_trade(settings)
    broadcast = BroadcastAdapter(llm=MagicMock(), twitter_enabled=False, youtube_enabled=False)

    pipeline = Pipeline(predict=predict, trade=trade, broadcast=broadcast)

    result = pipeline.run(
        market_question="Test low edge market?",
        market_id="test-low-edge",
        market_price=0.42,
        token_id="test-token",
        skip_broadcast=True,
    )

    # Should not trade — edge too small
    assert result.trade_signal.should_trade is False
    assert result.trade_signal.direction == "SKIP"


def test_position_tracking_lifecycle():
    """Test creating, updating, and closing a position."""
    from sqlmodel import SQLModel, Session, create_engine
    engine = create_engine("sqlite://", echo=False)
    SQLModel.metadata.create_all(engine)
    session = Session(engine)

    tracker = PositionTracker(session)

    # Simulate a trade record
    from db.models import Trade, TradeStatus
    trade = Trade(
        pipeline_run_id="test-run",
        prediction_id="test-pred",
        market_id="btc-100k-march",
        token_id="test-token",
        side="BUY_YES",
        size_usd=83.0,
        expected_value=0.19,
        kelly_fraction=0.083,
        market_price_at_entry=0.42,
        status=TradeStatus.DRY_RUN,
    )
    session.add(trade)
    session.commit()

    # Create position
    position = tracker.create_from_trade(trade)
    assert position is not None
    assert position.unrealized_pnl == 0.0

    # Update price — market moved in our favor
    tracker.update_price(position.id, 0.55)
    open_pos = tracker.get_open_positions()
    assert len(open_pos) == 1
    assert open_pos[0].unrealized_pnl > 0  # profit

    # Close position
    tracker.close_position(position.id, 0.60)
    open_pos = tracker.get_open_positions()
    assert len(open_pos) == 0

    # Portfolio summary
    summary = tracker.get_portfolio_summary()
    assert summary.closed_positions == 1
    assert summary.total_realized_pnl > 0

    session.close()


def test_discord_webhook_format():
    """Test Discord embed construction (without actually sending)."""
    from adapters.discord import DiscordWebhook

    # No URL = disabled, should return False
    webhook = DiscordWebhook(webhook_url=None)
    result = webhook.send_trade_alert(
        market_question="Test?",
        direction="BUY_YES",
        market_price=0.42,
        claude_estimate=0.65,
        ev_per_dollar=0.19,
        position_size=83.0,
        kelly_fraction=0.083,
    )
    assert result is False  # disabled, no error
