"""Tests for risk management."""

from trading.risk import RiskManager, RiskLimits


def test_position_capped():
    """Position gets capped at max_position_usd."""
    rm = RiskManager(limits=RiskLimits(max_position_usd=50))
    size, reason = rm.check_trade(100, bankroll=1000)
    assert size == 50
    assert reason is None


def test_exposure_limit():
    """Total exposure blocks new trades."""
    rm = RiskManager(
        limits=RiskLimits(max_total_exposure_usd=100),
        current_exposure=100,
    )
    size, reason = rm.check_trade(50, bankroll=1000)
    assert size == 0
    assert reason is not None


def test_position_count_limit():
    """Max positions blocks new trades."""
    rm = RiskManager(
        limits=RiskLimits(max_positions=3),
        open_positions=3,
    )
    size, reason = rm.check_trade(50, bankroll=1000)
    assert size == 0


def test_reserve_limit():
    """Bankroll reserve prevents over-betting."""
    rm = RiskManager(
        limits=RiskLimits(
            max_position_usd=500,
            min_bankroll_reserve=0.50,  # keep 50% reserve
        ),
        current_exposure=0,
    )
    # Bankroll $100, reserve 50% -> only $50 available
    size, reason = rm.check_trade(80, bankroll=100)
    assert size <= 50
