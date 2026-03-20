"""Market discovery — finds markets worth analyzing based on heuristics."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

from trading.scanner import MarketScanner, ScanFilter, ScanResult

logger = logging.getLogger(__name__)


@dataclass
class DiscoveryConfig:
    """Tunable params for market discovery."""

    min_volume: float = 5000.0
    max_markets_to_scan: int = 50
    max_markets_to_analyze: int = 10
    # Prefer markets with prices between these bounds (most potential edge)
    sweet_spot_low: float = 0.15
    sweet_spot_high: float = 0.85
    # Prefer markets closing within this many days
    max_days_to_close: int = 30
    # Volume spike detection — rank markets where recent volume is high
    prefer_high_volume: bool = True


class MarketDiscovery:
    """
    Discovers the most promising markets to analyze.

    Strategy:
    1. Fetch all markets from Polymarket
    2. Filter by volume, price range, and time to close
    3. Rank by a composite score (volume + price sweetness + recency)
    4. Return top N for Claude/MiroFish analysis
    """

    def __init__(self, scanner: MarketScanner, config: DiscoveryConfig | None = None):
        self.scanner = scanner
        self.config = config or DiscoveryConfig()

    def discover(self) -> list[ScanResult]:
        """Find the best markets to analyze right now."""
        # Override scanner filters with discovery config
        self.scanner.filters = ScanFilter(
            min_volume=self.config.min_volume,
            min_price=0.03,
            max_price=0.97,
            max_markets=self.config.max_markets_to_scan,
        )

        all_markets = self.scanner.scan()

        # Re-rank with discovery-specific scoring
        scored = []
        for result in all_markets:
            score = self._discovery_score(result)
            result.score = score
            scored.append(result)

        scored.sort(key=lambda r: r.score, reverse=True)

        top = scored[: self.config.max_markets_to_analyze]

        logger.info(
            f"Discovery: {len(all_markets)} scanned, "
            f"{len(top)} selected for analysis"
        )

        return top

    def _discovery_score(self, result: ScanResult) -> float:
        """
        Score a market for discovery priority.

        Components:
        - Volume (log-scaled) — high volume = liquid, real interest
        - Price sweetness — prices near 50% have most room for edge
        - Sweet spot bonus — prices in the 15-85% range get a bonus
        """
        snap = result.snapshot

        # Volume component (log-scaled, normalized roughly to 0-5)
        volume_score = math.log10(max(snap.volume, 1))

        # Price sweetness — distance from 50% (closer = better)
        price_distance = abs(snap.yes_price - 0.5)
        sweetness = 1.0 - (price_distance * 2)  # 1.0 at 50%, 0.0 at 0% or 100%

        # Sweet spot bonus — markets in the interesting range
        in_sweet_spot = (
            self.config.sweet_spot_low <= snap.yes_price <= self.config.sweet_spot_high
        )
        sweet_bonus = 1.5 if in_sweet_spot else 0.5

        return volume_score * sweetness * sweet_bonus
