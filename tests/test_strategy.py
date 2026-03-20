"""Tests for trading strategy formulas."""

import math
from trading.strategy import (
    expected_value,
    kelly_fraction,
    bayesian_update,
    log_return,
    total_log_return,
    lmsr_cost,
    lmsr_price,
    evaluate_trade,
)


def test_expected_value_positive_edge():
    """Market at 40%, we think 60% -> 20 cent edge."""
    ev = expected_value(0.40, 0.60)
    assert abs(ev - 0.20) < 0.001


def test_expected_value_no_edge():
    """Market at 50%, we think 50% -> zero edge."""
    ev = expected_value(0.50, 0.50)
    assert abs(ev) < 0.001


def test_expected_value_negative_edge():
    """Market at 60%, we think 40% -> negative edge."""
    ev = expected_value(0.60, 0.40)
    assert ev < 0


def test_kelly_fraction_basic():
    """60% true prob, 40% market price -> 33.3% Kelly."""
    kf = kelly_fraction(0.60, 0.40)
    assert abs(kf - 1 / 3) < 0.001


def test_kelly_fraction_no_edge():
    """No edge -> zero Kelly."""
    kf = kelly_fraction(0.40, 0.40)
    assert kf == 0.0


def test_kelly_fraction_negative_edge():
    """Negative edge -> clamped to zero."""
    kf = kelly_fraction(0.30, 0.40)
    assert kf == 0.0


def test_bayesian_update():
    """Fed rate cut example from the article."""
    posterior = bayesian_update(prior=0.55, likelihood=0.80, evidence=0.50)
    assert abs(posterior - 0.88) < 0.01


def test_bayesian_update_clamps():
    """Extreme updates get clamped to [0.01, 0.99]."""
    assert bayesian_update(0.99, 1.0, 0.01) == 0.99
    assert bayesian_update(0.01, 0.001, 1.0) == 0.01


def test_log_return_roundtrip():
    """0.80 -> 0.40 -> 0.80 should net zero in log space."""
    leg1 = log_return(0.80, 0.40)
    leg2 = log_return(0.40, 0.80)
    assert abs(leg1 + leg2) < 1e-10


def test_total_log_return():
    """Price series that returns to start = zero total log return."""
    prices = [0.80, 0.40, 0.80]
    assert abs(total_log_return(prices)) < 1e-10


def test_lmsr_prices_sum_to_one():
    """LMSR prices for all outcomes should sum to 1."""
    quantities = [100, 50]
    total = sum(lmsr_price(quantities, b=100, outcome_index=i) for i in range(2))
    assert abs(total - 1.0) < 1e-10


def test_lmsr_equal_quantities():
    """Equal quantities -> equal prices (50/50)."""
    p0 = lmsr_price([50, 50], b=100, outcome_index=0)
    p1 = lmsr_price([50, 50], b=100, outcome_index=1)
    assert abs(p0 - 0.5) < 1e-10
    assert abs(p1 - 0.5) < 1e-10


def test_evaluate_trade_buy_yes():
    """Strong positive edge -> BUY_YES signal."""
    signal = evaluate_trade(
        market_price=0.40,
        true_prob=0.60,
        bankroll=1000,
        kelly_multiplier=0.25,
        min_ev_threshold=0.05,
        max_position_usd=100,
    )
    assert signal.should_trade is True
    assert signal.direction == "BUY_YES"
    assert signal.ev_per_dollar > 0.15
    assert signal.position_size > 0
    assert signal.position_size <= 100  # max cap


def test_evaluate_trade_skip_low_ev():
    """Small edge below threshold -> SKIP."""
    signal = evaluate_trade(
        market_price=0.48,
        true_prob=0.52,
        bankroll=1000,
        kelly_multiplier=0.25,
        min_ev_threshold=0.05,
    )
    assert signal.should_trade is False
    assert signal.direction == "SKIP"


def test_evaluate_trade_position_capped():
    """Position size gets capped at max_position_usd."""
    signal = evaluate_trade(
        market_price=0.20,
        true_prob=0.80,
        bankroll=10000,
        kelly_multiplier=0.25,
        max_position_usd=50,
    )
    assert signal.position_size <= 50
