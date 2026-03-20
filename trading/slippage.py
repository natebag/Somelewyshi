"""Realistic slippage simulation for dry-run trades.

In real trading:
- You buy at the ASK (higher than mid), sell at the BID (lower than mid)
- Large orders walk the book, getting progressively worse fills
- Thin markets have wider spreads

This module estimates realistic fill prices so dry-run P&L
doesn't overstate actual performance.
"""

from __future__ import annotations

import logging
import math

logger = logging.getLogger(__name__)


def estimate_fill_price(
    side: str,
    mid_price: float,
    order_size_usd: float,
    best_bid: float | None = None,
    best_ask: float | None = None,
    spread_pct: float | None = None,
    market_volume: float = 0,
) -> float:
    """Estimate realistic fill price accounting for spread and book depth.

    Args:
        side: "BUY_YES" or "BUY_NO"
        mid_price: Current midpoint price
        order_size_usd: How much we're trying to trade
        best_bid: Best bid price (if available from order book)
        best_ask: Best ask price (if available from order book)
        spread_pct: Spread as percentage (if available)
        market_volume: Total market volume (used to estimate depth)

    Returns:
        Estimated fill price (always worse than mid for the trader)
    """
    # Step 1: Base spread cost (you cross the spread)
    if best_bid is not None and best_ask is not None:
        # Use real order book data
        spread = best_ask - best_bid
        if side in ("BUY_YES", "YES"):
            base_fill = best_ask  # You pay the ask
        else:
            base_fill = best_bid  # You sell at the bid (buying NO = selling YES)
    elif spread_pct is not None:
        # Estimate from spread percentage
        half_spread = mid_price * (spread_pct / 100) / 2
        if side in ("BUY_YES", "YES"):
            base_fill = mid_price + half_spread
        else:
            base_fill = mid_price - half_spread
        spread = mid_price * (spread_pct / 100)
    else:
        # Default: assume 2% spread on Polymarket (conservative)
        half_spread = mid_price * 0.01
        if side in ("BUY_YES", "YES"):
            base_fill = mid_price + half_spread
        else:
            base_fill = mid_price - half_spread
        spread = mid_price * 0.02

    # Step 2: Market impact (large orders walk the book)
    # Rough model: impact scales with order_size / market_depth
    # Market depth ~= daily_volume * 0.01 (1% of daily volume is typical depth)
    if market_volume > 0:
        estimated_depth = market_volume * 0.01
        if estimated_depth > 0:
            impact_ratio = order_size_usd / estimated_depth
            # Impact follows square root model (Kyle's lambda)
            impact = spread * 0.5 * math.sqrt(min(impact_ratio, 10))
        else:
            impact = spread * 0.1
    else:
        # Unknown volume — assume moderate impact
        impact = spread * 0.15

    # Apply impact (always worse for the trader)
    if side in ("BUY_YES", "YES"):
        fill_price = base_fill + impact
    else:
        fill_price = base_fill - impact

    # Clamp to valid range
    fill_price = max(0.01, min(0.99, fill_price))

    # Log the breakdown
    slippage_from_mid = abs(fill_price - mid_price)
    slippage_pct = (slippage_from_mid / mid_price * 100) if mid_price > 0 else 0

    logger.info(
        f"[SLIPPAGE] mid={mid_price:.4f} → fill={fill_price:.4f} | "
        f"spread={spread:.4f} impact={impact:.4f} | "
        f"total slippage={slippage_pct:.2f}%"
    )

    return round(fill_price, 4)


def adjust_ev_for_slippage(
    ev_per_dollar: float,
    mid_price: float,
    fill_price: float,
    true_prob: float,
) -> float:
    """Recalculate EV using the realistic fill price instead of mid.

    This is the EV you'd actually get in a real trade.
    """
    # EV = P(win) × profit_per_share - P(lose) × loss_per_share
    # For BUY YES: profit = (1 - fill_price), loss = fill_price
    adjusted_ev = true_prob * (1 - fill_price) - (1 - true_prob) * fill_price
    return adjusted_ev
