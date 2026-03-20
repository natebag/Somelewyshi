"""Core trading formulas: EV, Kelly Criterion, Bayesian updating, log returns, LMSR."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class TradeSignal:
    """Output of trade evaluation."""

    should_trade: bool
    direction: str  # "BUY_YES", "BUY_NO", or "SKIP"
    ev_per_dollar: float
    kelly_fraction: float
    position_size: float
    market_price: float
    estimated_prob: float
    reasoning: str = ""


# ── Formula 1: Expected Value ────────────────────────────────────

def expected_value(market_price: float, true_prob: float) -> float:
    """
    EV = P(win) × Profit − P(lose) × Loss

    On Polymarket: buying YES at market_price
    - If YES wins: profit = (1 - market_price) per share
    - If YES loses: loss = market_price per share
    """
    return true_prob * (1 - market_price) - (1 - true_prob) * market_price


# ── Formula 2: Kelly Criterion ───────────────────────────────────

def kelly_fraction(true_prob: float, market_price: float) -> float:
    """
    f* = (p × b − q) / b

    Where b = payout ratio = (1 - market_price) / market_price
    Returns the optimal fraction of bankroll to bet.
    """
    if market_price <= 0 or market_price >= 1:
        return 0.0
    b = (1 - market_price) / market_price
    p = true_prob
    q = 1 - p
    f = (p * b - q) / b
    return max(f, 0.0)


# ── Formula 3: Bayesian Updating ────────────────────────────────

def bayesian_update(prior: float, likelihood: float, evidence: float) -> float:
    """
    P(H|E) = P(E|H) × P(H) / P(E)

    Updates probability estimate when new information arrives.
    Clamps result to [0.01, 0.99] to avoid certainty.
    """
    if evidence <= 0:
        return prior
    posterior = (likelihood * prior) / evidence
    return min(max(posterior, 0.01), 0.99)


# ── Formula 4: Log Returns ──────────────────────────────────────

def log_return(price_start: float, price_end: float) -> float:
    """
    log_return = ln(P₁ / P₀)

    Log returns sum correctly across multiple trades,
    unlike arithmetic returns which systematically overstate results.
    """
    if price_start <= 0 or price_end <= 0:
        return 0.0
    return math.log(price_end / price_start)


def total_log_return(prices: list[float]) -> float:
    """Sum of sequential log returns from a price series."""
    if len(prices) < 2:
        return 0.0
    return sum(
        log_return(prices[i], prices[i + 1])
        for i in range(len(prices) - 1)
    )


# ── LMSR Cost Function ──────────────────────────────────────────

def lmsr_cost(quantities: list[float], b: float) -> float:
    """
    C(q) = b · ln Σ e^(qi/b)

    Logarithmic Market Scoring Rule — the pricing mechanism
    Polymarket uses under the hood.
    """
    if b <= 0:
        return 0.0
    exp_sum = sum(math.exp(q / b) for q in quantities)
    return b * math.log(exp_sum)


def lmsr_price(quantities: list[float], b: float, outcome_index: int) -> float:
    """
    Price of a specific outcome under LMSR.
    p_i = e^(qi/b) / Σ e^(qj/b)

    This is the softmax function — same math Claude uses to pick the next word.
    """
    if b <= 0:
        return 0.0
    exps = [math.exp(q / b) for q in quantities]
    total = sum(exps)
    return exps[outcome_index] / total


# ── Evaluate Trade ───────────────────────────────────────────────

def evaluate_trade(
    market_price: float,
    true_prob: float,
    bankroll: float,
    kelly_multiplier: float = 0.25,
    min_ev_threshold: float = 0.05,
    max_position_usd: float = 50.0,
) -> TradeSignal:
    """
    Full trade evaluation: EV → Kelly → position sizing.

    Uses quarter-Kelly by default for conservative sizing.
    """
    # Determine direction
    if true_prob > market_price:
        direction = "BUY_YES"
        ev = expected_value(market_price, true_prob)
        kf = kelly_fraction(true_prob, market_price)
    elif true_prob < (1 - market_price):
        # Edge on the NO side
        no_price = 1 - market_price
        direction = "BUY_NO"
        ev = expected_value(no_price, 1 - true_prob)
        kf = kelly_fraction(1 - true_prob, no_price)
    else:
        return TradeSignal(
            should_trade=False,
            direction="SKIP",
            ev_per_dollar=0.0,
            kelly_fraction=0.0,
            position_size=0.0,
            market_price=market_price,
            estimated_prob=true_prob,
            reasoning="No edge found on either side",
        )

    # Apply Kelly multiplier (quarter-Kelly default)
    adjusted_kelly = kf * kelly_multiplier
    position_size = bankroll * adjusted_kelly

    # Cap position size
    position_size = min(position_size, max_position_usd)

    # Check minimum EV threshold
    if ev < min_ev_threshold:
        return TradeSignal(
            should_trade=False,
            direction="SKIP",
            ev_per_dollar=ev,
            kelly_fraction=adjusted_kelly,
            position_size=0.0,
            market_price=market_price,
            estimated_prob=true_prob,
            reasoning=f"Edge too small: {ev:.1%} < {min_ev_threshold:.1%} threshold",
        )

    return TradeSignal(
        should_trade=True,
        direction=direction,
        ev_per_dollar=ev,
        kelly_fraction=adjusted_kelly,
        position_size=round(position_size, 2),
        market_price=market_price,
        estimated_prob=true_prob,
    )
