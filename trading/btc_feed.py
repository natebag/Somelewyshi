"""Live BTC price feed — real-time spot price + short-term delta.

Feeds BTC price context into predictions for crypto markets.
Uses CryptoCompare (free, no API key) like Nunchi's autoresearch setup.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone

import httpx

logger = logging.getLogger(__name__)

CRYPTOCOMPARE_URL = "https://min-api.cryptocompare.com/data"


@dataclass
class BTCSnapshot:
    """Point-in-time BTC price data."""

    price: float
    delta_5m_pct: float  # 5-minute price change %
    delta_1h_pct: float  # 1-hour price change %
    high_24h: float
    low_24h: float
    volume_24h: float
    timestamp: datetime

    def to_prompt_context(self) -> str:
        """Format for injection into LLM prompts."""
        direction_5m = "+" if self.delta_5m_pct >= 0 else ""
        direction_1h = "+" if self.delta_1h_pct >= 0 else ""

        return (
            f"BTC spot: ${self.price:,.2f} "
            f"(Δ5min: {direction_5m}{self.delta_5m_pct:.2f}%, "
            f"Δ1h: {direction_1h}{self.delta_1h_pct:.2f}%) | "
            f"24h range: ${self.low_24h:,.0f}-${self.high_24h:,.0f} | "
            f"24h vol: ${self.volume_24h:,.0f}"
        )


class BTCFeed:
    """Fetches live BTC price data from CryptoCompare."""

    def __init__(self, cache_seconds: int = 30):
        self.cache_seconds = cache_seconds
        self._last_snapshot: BTCSnapshot | None = None
        self._last_fetch: float = 0
        self._price_history: list[tuple[float, float]] = []  # (timestamp, price)

    def get_snapshot(self) -> BTCSnapshot | None:
        """Get current BTC snapshot. Caches for cache_seconds."""
        now = time.time()
        if self._last_snapshot and (now - self._last_fetch) < self.cache_seconds:
            return self._last_snapshot

        try:
            snapshot = self._fetch()
            self._last_snapshot = snapshot
            self._last_fetch = now

            # Track price history for delta calculations
            self._price_history.append((now, snapshot.price))
            # Keep last 100 observations
            if len(self._price_history) > 100:
                self._price_history = self._price_history[-100:]

            return snapshot
        except Exception as e:
            logger.warning(f"BTC feed fetch failed: {e}")
            return self._last_snapshot  # Return stale data

    def _fetch(self) -> BTCSnapshot:
        """Fetch fresh BTC data from CryptoCompare."""
        with httpx.Client(timeout=10) as http:
            # Get current price + 24h stats
            price_resp = http.get(
                f"{CRYPTOCOMPARE_URL}/pricemultifull",
                params={"fsyms": "BTC", "tsyms": "USD"},
            )
            price_resp.raise_for_status()
            data = price_resp.json()
            raw = data["RAW"]["BTC"]["USD"]

            current_price = float(raw["PRICE"])
            high_24h = float(raw["HIGH24HOUR"])
            low_24h = float(raw["LOW24HOUR"])
            volume_24h = float(raw["VOLUME24HOURTO"])

            # Get minute-level data for delta calculations
            minute_resp = http.get(
                f"{CRYPTOCOMPARE_URL}/v2/histominute",
                params={"fsym": "BTC", "tsym": "USD", "limit": 60},
            )
            minute_resp.raise_for_status()
            minute_data = minute_resp.json()
            candles = minute_data.get("Data", {}).get("Data", [])

            # 5-minute delta
            delta_5m = 0.0
            if len(candles) >= 6:
                price_5m_ago = float(candles[-6]["close"])
                if price_5m_ago > 0:
                    delta_5m = ((current_price - price_5m_ago) / price_5m_ago) * 100

            # 1-hour delta
            delta_1h = 0.0
            if len(candles) >= 60:
                price_1h_ago = float(candles[0]["close"])
                if price_1h_ago > 0:
                    delta_1h = ((current_price - price_1h_ago) / price_1h_ago) * 100

            return BTCSnapshot(
                price=current_price,
                delta_5m_pct=round(delta_5m, 3),
                delta_1h_pct=round(delta_1h, 3),
                high_24h=high_24h,
                low_24h=low_24h,
                volume_24h=volume_24h,
                timestamp=datetime.now(timezone.utc),
            )

    def is_btc_market(self, question: str) -> bool:
        """Check if a market question is about BTC."""
        q = question.lower()
        return any(kw in q for kw in ["btc", "bitcoin", "₿"])


# Global singleton
_btc_feed: BTCFeed | None = None


def get_btc_feed() -> BTCFeed:
    global _btc_feed
    if _btc_feed is None:
        _btc_feed = BTCFeed()
    return _btc_feed
