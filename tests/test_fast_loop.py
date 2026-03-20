"""Tests for BTC Fast Loop strategy."""

from trading.fast_loop import FastLoopStrategy, PriceBar
from datetime import datetime, timezone


def _make_bar(price, volume=100):
    return PriceBar(timestamp=datetime.now(timezone.utc), yes_price=price, volume=volume)


def test_strong_edge_triggers_buy():
    strategy = FastLoopStrategy(min_signals_agree=2, min_edge=0.08)
    # Feed some price history showing upward momentum
    for p in [0.40, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49]:
        strategy.add_price(_make_bar(p, 200))

    signal = strategy.evaluate(
        current_price=0.42,
        estimated_prob=0.62,  # 20% edge
        time_to_expiry_seconds=200,
        current_volume=300,
    )
    assert signal.direction == "BUY_YES"
    assert signal.edge > 0.15


def test_no_edge_skips():
    strategy = FastLoopStrategy(min_edge=0.08)
    signal = strategy.evaluate(
        current_price=0.50,
        estimated_prob=0.52,  # only 2% edge
        time_to_expiry_seconds=200,
    )
    assert signal.direction == "SKIP"


def test_exit_on_target_repricing():
    strategy = FastLoopStrategy(target_reprice_pct=0.10, stop_loss_pct=0.15)
    should_exit, reason = strategy.should_exit(
        entry_price=0.40,
        current_price=0.51,  # +11% repricing
        direction="BUY_YES",
        time_to_expiry_seconds=120,
    )
    assert should_exit is True
    assert "target" in reason.lower()


def test_exit_on_stop_loss():
    strategy = FastLoopStrategy(target_reprice_pct=0.10, stop_loss_pct=0.15)
    should_exit, reason = strategy.should_exit(
        entry_price=0.40,
        current_price=0.24,  # -16% loss
        direction="BUY_YES",
        time_to_expiry_seconds=120,
    )
    assert should_exit is True
    assert "stop" in reason.lower()


def test_exit_near_expiry():
    strategy = FastLoopStrategy()
    should_exit, reason = strategy.should_exit(
        entry_price=0.40,
        current_price=0.45,
        direction="BUY_YES",
        time_to_expiry_seconds=10,  # 10 seconds left
    )
    assert should_exit is True
    assert "time" in reason.lower()


def test_no_exit_in_normal_conditions():
    strategy = FastLoopStrategy(target_reprice_pct=0.10, stop_loss_pct=0.15)
    should_exit, reason = strategy.should_exit(
        entry_price=0.40,
        current_price=0.44,  # only +4%
        direction="BUY_YES",
        time_to_expiry_seconds=120,
    )
    assert should_exit is False
