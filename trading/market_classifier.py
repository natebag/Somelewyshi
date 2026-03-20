"""Market classifier — categorizes markets and assigns scan frequency.

Different market types need different scan speeds:
- BTC 5-min: scan every 2-3 minutes (expires fast)
- Sports: scan every 30 minutes (game-driven)
- Political/weather: scan every 1-2 hours (slow-moving)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class MarketType(str, Enum):
    BTC_FAST = "btc_fast"        # 5-min / 15-min BTC markets
    CRYPTO = "crypto"            # General crypto (daily/weekly)
    SPORTS = "sports"            # Game-based markets
    WEATHER = "weather"          # Temperature brackets
    POLITICAL = "political"      # Elections, policy, geopolitics
    ECONOMIC = "economic"        # Fed, GDP, inflation
    OTHER = "other"


@dataclass
class MarketProfile:
    """Classification result with recommended scan parameters."""

    market_type: MarketType
    scan_interval_seconds: int  # How often to scan
    max_hold_seconds: int       # Max time to hold a position
    exit_before_seconds: int    # Exit this many seconds before resolution
    priority: int               # Higher = scan first (1-10)

    @property
    def scan_interval_minutes(self) -> float:
        return self.scan_interval_seconds / 60


# Scan profiles per market type
PROFILES = {
    MarketType.BTC_FAST: MarketProfile(
        market_type=MarketType.BTC_FAST,
        scan_interval_seconds=120,     # Every 2 minutes
        max_hold_seconds=240,          # Max 4 minutes
        exit_before_seconds=15,        # Exit 15s before expiry
        priority=10,
    ),
    MarketType.CRYPTO: MarketProfile(
        market_type=MarketType.CRYPTO,
        scan_interval_seconds=900,     # Every 15 minutes
        max_hold_seconds=3600,         # Max 1 hour
        exit_before_seconds=120,       # Exit 2min before
        priority=7,
    ),
    MarketType.SPORTS: MarketProfile(
        market_type=MarketType.SPORTS,
        scan_interval_seconds=1800,    # Every 30 minutes
        max_hold_seconds=7200,         # Max 2 hours
        exit_before_seconds=300,       # Exit 5min before
        priority=5,
    ),
    MarketType.WEATHER: MarketProfile(
        market_type=MarketType.WEATHER,
        scan_interval_seconds=3600,    # Every hour
        max_hold_seconds=14400,        # Max 4 hours
        exit_before_seconds=600,       # Exit 10min before
        priority=4,
    ),
    MarketType.POLITICAL: MarketProfile(
        market_type=MarketType.POLITICAL,
        scan_interval_seconds=3600,    # Every hour
        max_hold_seconds=86400,        # Max 24 hours
        exit_before_seconds=1800,      # Exit 30min before
        priority=3,
    ),
    MarketType.ECONOMIC: MarketProfile(
        market_type=MarketType.ECONOMIC,
        scan_interval_seconds=3600,    # Every hour
        max_hold_seconds=86400,        # Max 24 hours
        exit_before_seconds=1800,      # Exit 30min before
        priority=3,
    ),
    MarketType.OTHER: MarketProfile(
        market_type=MarketType.OTHER,
        scan_interval_seconds=3600,    # Every hour
        max_hold_seconds=43200,        # Max 12 hours
        exit_before_seconds=600,       # Exit 10min before
        priority=2,
    ),
}

# Keyword patterns for classification
_PATTERNS: list[tuple[MarketType, list[str]]] = [
    (MarketType.BTC_FAST, [
        r"btc.*(?:5[- ]?min|15[- ]?min|up.?or.?down|above|below)",
        r"bitcoin.*(?:5[- ]?min|15[- ]?min|up.?or.?down)",
        r"btc.*(?:\$\d+[kK]?\s*(?:by|at|in)\s*\d+:\d+)",
    ]),
    (MarketType.CRYPTO, [
        r"\b(?:btc|bitcoin|eth|ethereum|sol|solana|crypto|token|coin)\b",
        r"\$\d+[kK]",
    ]),
    (MarketType.SPORTS, [
        r"\b(?:nba|nfl|mlb|nhl|ncaa|premier league|champions league|ufc|mma)\b",
        r"\b(?:vs\.?|versus|o/?u|over.?under|spread|moneyline)\b",
        r"\b(?:warriors|lakers|celtics|yankees|dodgers|chiefs|eagles)\b",
        r"\b(?:game|match|bout|fight|race)\b.*\b(?:win|score|points)\b",
    ]),
    (MarketType.WEATHER, [
        r"\b(?:temperature|temp|°[fFcC]|weather|rain|snow|hurricane|tornado)\b",
        r"\b(?:noaa|forecast|high.?of|low.?of)\b",
    ]),
    (MarketType.POLITICAL, [
        r"\b(?:president|election|congress|senate|governor|vote|ballot)\b",
        r"\b(?:trump|biden|democrat|republican|gop|dnc|rnc)\b",
        r"\b(?:war|invasion|sanctions|nato|un\b|military)\b",
        r"\b(?:iran|russia|ukraine|china|taiwan|israel)\b",
    ]),
    (MarketType.ECONOMIC, [
        r"\b(?:fed|fomc|interest rate|rate cut|rate hike|inflation|cpi|gdp)\b",
        r"\b(?:unemployment|jobs report|nonfarm|treasury|yield|recession)\b",
    ]),
]


def classify_market(question: str) -> MarketProfile:
    """Classify a market question and return its scan profile."""
    q = question.lower()

    for market_type, patterns in _PATTERNS:
        for pattern in patterns:
            if re.search(pattern, q, re.IGNORECASE):
                return PROFILES[market_type]

    return PROFILES[MarketType.OTHER]
