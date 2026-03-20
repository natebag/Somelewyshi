"""Tests for repricing exit engine."""

from trading.repricing import RepricingEngine


def test_target_capture_exit():
    engine = RepricingEngine(target_capture_pct=0.60, min_profit_to_exit=0.02)
    result = engine.check_exit(
        direction="BUY_YES",
        entry_price=0.40,
        current_price=0.55,  # captured 75% of edge (0.40 → 0.60 estimate)
        our_estimate=0.60,
    )
    assert result.should_exit is True
    assert "target" in result.reason.lower()


def test_stop_loss_exit():
    engine = RepricingEngine(stop_loss_pct=0.15)
    result = engine.check_exit(
        direction="BUY_YES",
        entry_price=0.40,
        current_price=0.22,  # -18% adverse
        our_estimate=0.60,
    )
    assert result.should_exit is True
    assert "stop" in result.reason.lower()


def test_edge_converged_exit():
    # Use high target_capture so target trigger doesn't fire first
    engine = RepricingEngine(target_capture_pct=0.99)
    result = engine.check_exit(
        direction="BUY_YES",
        entry_price=0.40,
        current_price=0.58,  # market caught up to our estimate
        our_estimate=0.60,
    )
    assert result.should_exit is True
    assert "converge" in result.reason.lower()


def test_time_exit():
    engine = RepricingEngine(time_exit_seconds=60)
    result = engine.check_exit(
        direction="BUY_YES",
        entry_price=0.40,
        current_price=0.45,
        our_estimate=0.60,
        time_to_expiry_seconds=30,
    )
    assert result.should_exit is True
    assert "time" in result.reason.lower()


def test_stale_position_exit():
    engine = RepricingEngine(stale_threshold_seconds=600)
    result = engine.check_exit(
        direction="BUY_YES",
        entry_price=0.40,
        current_price=0.41,
        our_estimate=0.60,
        position_age_seconds=700,
        last_price_change_seconds=650,
    )
    assert result.should_exit is True
    assert "stale" in result.reason.lower()


def test_hold_when_healthy():
    engine = RepricingEngine(target_capture_pct=0.60, stop_loss_pct=0.30)
    result = engine.check_exit(
        direction="BUY_YES",
        entry_price=0.40,
        current_price=0.48,  # some profit but not at target yet
        our_estimate=0.60,
        time_to_expiry_seconds=300,
    )
    assert result.should_exit is False


def test_optimal_exit_calculation():
    engine = RepricingEngine(target_capture_pct=0.60, stop_loss_pct=0.20)
    targets = engine.calculate_optimal_exit(
        entry_price=0.40,
        our_estimate=0.60,
        direction="BUY_YES",
    )
    assert targets["target_price"] == 0.52  # 0.40 + (0.20 * 0.60)
    assert targets["stop_price"] == 0.20  # 0.40 - 0.20
    assert targets["reward_risk_ratio"] > 0
