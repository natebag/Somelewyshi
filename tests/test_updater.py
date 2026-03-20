"""Tests for Bayesian re-updater."""

from trading.updater import market_move_update, should_exit_position


def test_confirming_move_boosts_confidence():
    """Market moving toward our estimate should increase confidence."""
    # We think 70%, market was at 50%, now moved to 55%
    updated = market_move_update(
        prior_estimate=0.70,
        old_market_price=0.50,
        new_market_price=0.55,
    )
    # Should stay above our prior or close to it
    assert updated > 0.60


def test_disagreeing_move_reduces_confidence():
    """Market moving away from our estimate should reduce confidence."""
    # We think 70%, market was at 50%, now dropped to 40%
    updated = market_move_update(
        prior_estimate=0.70,
        old_market_price=0.50,
        new_market_price=0.40,
    )
    # Should be lower than our original estimate
    assert updated < 0.70


def test_trivial_move_no_change():
    """Tiny market move should not change estimate."""
    updated = market_move_update(
        prior_estimate=0.60,
        old_market_price=0.50,
        new_market_price=0.505,
    )
    assert abs(updated - 0.60) < 0.01


def test_should_exit_edge_eroded():
    """Position should exit when edge disappears."""
    exit, reason = should_exit_position(
        entry_price=0.40,
        current_price=0.58,
        original_estimate=0.60,
        updated_estimate=0.59,
        min_edge=0.03,
    )
    assert exit is True
    assert "eroded" in reason.lower()


def test_should_not_exit_strong_edge():
    """Position should hold when edge is strong."""
    exit, reason = should_exit_position(
        entry_price=0.40,
        current_price=0.45,
        original_estimate=0.60,
        updated_estimate=0.62,
        min_edge=0.03,
    )
    assert exit is False


def test_should_exit_stop_loss():
    """Position should exit on large loss."""
    exit, reason = should_exit_position(
        entry_price=0.80,
        current_price=0.30,
        original_estimate=0.90,
        updated_estimate=0.50,
    )
    assert exit is True
    assert "stop-loss" in reason.lower()
