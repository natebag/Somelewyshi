"""Bayesian re-updater — adjusts probability estimates based on market movement."""

from __future__ import annotations

import logging
import math

from trading.strategy import bayesian_update

logger = logging.getLogger(__name__)


def market_move_update(
    prior_estimate: float,
    old_market_price: float,
    new_market_price: float,
    market_weight: float = 0.6,
) -> float:
    """
    Update our probability estimate when the market price changes.

    Uses Bayesian updating where:
    - prior = our previous estimate
    - likelihood = P(seeing this market move | our hypothesis is correct)
    - evidence = P(seeing this market move in general)

    market_weight controls how much we trust the market's new information.
    0.0 = ignore market completely (stubborn)
    1.0 = fully adopt market price (no edge)
    0.5-0.7 = healthy balance (default 0.6)
    """
    if old_market_price <= 0 or new_market_price <= 0:
        return prior_estimate

    # Direction and magnitude of market move
    move = new_market_price - old_market_price
    move_magnitude = abs(move)

    if move_magnitude < 0.01:
        # Trivial move, don't update
        return prior_estimate

    # If market moved toward our estimate, our confidence should increase
    # If market moved away, we should partially update toward the market
    market_agrees = (move > 0 and prior_estimate > old_market_price) or \
                    (move < 0 and prior_estimate < old_market_price)

    if market_agrees:
        # Market is confirming our view — slight confidence boost
        # likelihood high: we expected this if our view is right
        likelihood = 0.7 + 0.3 * min(move_magnitude / 0.10, 1.0)
        evidence = 0.5
    else:
        # Market is disagreeing — we need to update
        # The bigger the move, the more we should update
        likelihood = 0.3 - 0.2 * min(move_magnitude / 0.10, 1.0)
        likelihood = max(likelihood, 0.1)
        evidence = 0.5

    posterior = bayesian_update(prior_estimate, likelihood, evidence)

    # Blend with market price based on market_weight
    # This prevents us from being too stubborn against large moves
    blended = posterior * (1 - market_weight * move_magnitude) + \
              new_market_price * (market_weight * move_magnitude)

    # Clamp
    result = min(max(blended, 0.01), 0.99)

    logger.info(
        f"Bayes update: prior={prior_estimate:.2%} | "
        f"market {old_market_price:.2%}->{new_market_price:.2%} | "
        f"posterior={result:.2%} | "
        f"{'confirmed' if market_agrees else 'disagreed'}"
    )

    return result


def should_exit_position(
    entry_price: float,
    current_price: float,
    original_estimate: float,
    updated_estimate: float,
    min_edge: float = 0.03,
) -> tuple[bool, str]:
    """
    Decide whether to close a position based on updated estimates.

    Returns (should_exit, reason).
    """
    # Edge has disappeared
    current_edge = updated_estimate - current_price
    if abs(current_edge) < min_edge:
        return True, f"Edge eroded: {current_edge:.1%} < {min_edge:.1%} threshold"

    # Estimate flipped to wrong side
    if original_estimate > entry_price and updated_estimate < current_price:
        return True, "Estimate flipped below market price"
    if original_estimate < entry_price and updated_estimate > current_price:
        return True, "Estimate flipped above market price"

    # Large loss (stop-loss at -30% of position value)
    if entry_price > 0:
        log_ret = math.log(current_price / entry_price) if current_price > 0 else -999
        if log_ret < -0.36:  # ~30% loss
            return True, f"Stop-loss triggered: log return {log_ret:.2f}"

    return False, ""
