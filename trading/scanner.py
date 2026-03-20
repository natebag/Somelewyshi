"""Market scanner — discovers and filters Polymarket markets for opportunities."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from trading.client import PolymarketClient
from trading.models import MarketSnapshot

logger = logging.getLogger(__name__)


@dataclass
class ScanFilter:
    """Filters for market scanning."""

    min_volume: float = 1000.0  # minimum $ volume
    min_price: float = 0.05  # ignore near-zero contracts
    max_price: float = 0.95  # ignore near-certain contracts
    max_markets: int = 50  # cap on markets to analyze


@dataclass
class ScanResult:
    """A market that passed the scan filter."""

    snapshot: MarketSnapshot
    score: float = 0.0  # ranking score for prioritization


class MarketScanner:
    """Scans Polymarket for tradeable markets."""

    def __init__(self, client: PolymarketClient, filters: ScanFilter | None = None):
        self.client = client
        self.filters = filters or ScanFilter()

    def scan(self) -> list[ScanResult]:
        """
        Fetch markets from Polymarket and filter for tradeable ones.

        Returns markets sorted by volume (highest first).
        """
        logger.info("Scanning Polymarket for markets...")

        try:
            raw_markets = self.client.get_markets()
            if isinstance(raw_markets, dict):
                raw_markets = raw_markets.get("data", [])
        except Exception as e:
            logger.error(f"Failed to fetch markets: {e}")
            return []

        results = []
        for market in raw_markets:
            snapshot = self._parse_market(market)
            if snapshot and self._passes_filter(snapshot):
                score = self._rank_score(snapshot)
                results.append(ScanResult(snapshot=snapshot, score=score))

        # Sort by score (volume-weighted)
        results.sort(key=lambda r: r.score, reverse=True)

        # Cap results
        results = results[: self.filters.max_markets]

        logger.info(
            f"Scan complete: {len(raw_markets)} total, "
            f"{len(results)} passed filters"
        )
        return results

    def _parse_market(self, raw: dict) -> MarketSnapshot | None:
        """Parse a Gamma API market dict into a MarketSnapshot.

        Gamma API format (ported from polyjones):
        - outcomePrices: array or JSON string of [yes_price, no_price]
        - clobTokenIds: array or JSON string of [yes_token, no_token]
        - volume/volumeNum: total volume
        - endDate: expiry
        """
        try:
            if not raw.get("question"):
                return None
            if raw.get("closed"):
                return None

            # Parse prices (Gamma uses outcomePrices)
            from trading.client import PolymarketClient
            yes_price, no_price = PolymarketClient._parse_prices(raw)
            token_yes, token_no = PolymarketClient._parse_token_ids(raw)

            volume = float(raw.get("volume", 0) or raw.get("volumeNum", 0) or 0)

            return MarketSnapshot(
                condition_id=raw.get("conditionId", raw.get("condition_id", raw.get("id", ""))),
                question=raw.get("question", "Unknown"),
                yes_price=yes_price,
                no_price=no_price,
                volume=volume,
                end_date=raw.get("endDate", raw.get("end_date_iso", "")),
                token_id_yes=token_yes,
                token_id_no=token_no,
            )
        except (ValueError, KeyError, TypeError) as e:
            logger.debug(f"Failed to parse market: {e}")
            return None

    def _get_token_id(self, raw: dict, outcome: str) -> str:
        """Extract token ID for a specific outcome."""
        for token in raw.get("tokens", []):
            if token.get("outcome", "").upper() == outcome:
                return token.get("token_id", "")
        return ""

    def _passes_filter(self, snap: MarketSnapshot) -> bool:
        """Check if a market passes the scan filters."""
        if snap.volume < self.filters.min_volume:
            return False
        if snap.yes_price < self.filters.min_price or snap.yes_price > self.filters.max_price:
            return False
        if not snap.question:
            return False
        return True

    def _rank_score(self, snap: MarketSnapshot) -> float:
        """
        Score a market for trading potential.

        Prefers: high volume, prices near 0.5 (most room for edge),
        not too close to expiry.
        """
        # Volume component (log-scaled)
        import math
        volume_score = math.log10(max(snap.volume, 1))

        # Price component — markets near 50/50 have most potential edge
        price_distance = abs(snap.yes_price - 0.5)
        price_score = 1.0 - price_distance  # max at 0.5

        return volume_score * price_score
