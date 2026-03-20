"""Polymarket client — uses Gamma API for discovery, CLOB API for trading.

Ported from polyjones (F:/coding/crypto/polyjones) which already solved
the Polymarket API integration.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

import httpx

from trading.models import MarketSnapshot, TradeOrder, TradeResult

logger = logging.getLogger(__name__)

GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_API = "https://clob.polymarket.com"


class PolymarketClient:
    """Wraps Gamma API for discovery + CLOB API for trading."""

    def __init__(
        self,
        clob_url: str = CLOB_API,
        private_key: str = "",
        chain_id: int = 137,
        signature_type: int = 0,
        funder: str = "",
    ):
        self.clob_url = clob_url
        self.private_key = private_key
        self.chain_id = chain_id
        self._clob_client = None

        if private_key:
            self._init_authenticated(signature_type, funder)
        else:
            self._init_readonly()

    def _init_readonly(self):
        from py_clob_client.client import ClobClient

        self._clob_client = ClobClient(self.clob_url)
        logger.info("Polymarket client initialized (read-only)")

    def _init_authenticated(self, signature_type: int, funder: str):
        from py_clob_client.client import ClobClient

        self._clob_client = ClobClient(
            self.clob_url,
            key=self.private_key,
            chain_id=self.chain_id,
            signature_type=signature_type,
            funder=funder or None,
        )
        self._clob_client.set_api_creds(self._clob_client.create_or_derive_api_creds())
        logger.info("Polymarket client initialized (authenticated)")

    # ── Market Discovery (Gamma API) ────────────────────────────

    def get_markets(self, limit: int = 500) -> list[dict]:
        """Fetch active markets from Gamma API, sorted by volume.

        This is the same approach used in polyjones — the Gamma API returns
        active markets with prices, volume, and token IDs in one call.
        """
        url = f"{GAMMA_API}/markets?closed=false&limit={limit}&order=volume&ascending=false"

        try:
            with httpx.Client(timeout=30) as http:
                resp = http.get(url)
                resp.raise_for_status()
                raw_markets = resp.json()

            if not isinstance(raw_markets, list):
                logger.warning(f"Unexpected Gamma API response type: {type(raw_markets)}")
                return []

            logger.info(f"Fetched {len(raw_markets)} active markets from Gamma API")
            return raw_markets

        except Exception as e:
            logger.error(f"Gamma API fetch failed: {e}")
            return []

    def get_market_snapshot(self, condition_id: str) -> Optional[MarketSnapshot]:
        """Fetch current state of a specific market."""
        try:
            url = f"{GAMMA_API}/markets?conditionId={condition_id}"
            with httpx.Client(timeout=15) as http:
                resp = http.get(url)
                resp.raise_for_status()
                markets = resp.json()

            if not markets:
                return None

            m = markets[0] if isinstance(markets, list) else markets
            yes_price, no_price = self._parse_prices(m)
            token_yes, token_no = self._parse_token_ids(m)

            return MarketSnapshot(
                condition_id=condition_id,
                question=m.get("question", ""),
                yes_price=yes_price,
                no_price=no_price,
                volume=float(m.get("volume", 0) or m.get("volumeNum", 0) or 0),
                end_date=m.get("endDate", m.get("end_date_iso", "")),
                token_id_yes=token_yes,
                token_id_no=token_no,
            )
        except Exception as e:
            logger.error(f"Failed to fetch market {condition_id}: {e}")
            return None

    def get_midpoint(self, token_id: str) -> Optional[float]:
        """Get the midpoint price for a token from the order book."""
        try:
            url = f"{self.clob_url}/book?token_id={token_id}"
            with httpx.Client(timeout=10) as http:
                resp = http.get(url)
                resp.raise_for_status()
                data = resp.json()

            bids = data.get("bids", [])
            asks = data.get("asks", [])

            # CLOB returns bids ascending, asks descending
            best_bid = float(bids[-1]["price"]) if bids else 0
            best_ask = float(asks[-1]["price"]) if asks else 1

            return (best_bid + best_ask) / 2

        except Exception as e:
            logger.error(f"Failed to get midpoint for {token_id}: {e}")
            return None

    def get_order_book(self, token_id: str) -> Optional[dict]:
        """Get full order book info for a token."""
        try:
            url = f"{self.clob_url}/book?token_id={token_id}"
            with httpx.Client(timeout=10) as http:
                resp = http.get(url)
                resp.raise_for_status()
                data = resp.json()

            bids = data.get("bids", [])
            asks = data.get("asks", [])

            best_bid = float(bids[-1]["price"]) if bids else 0
            best_ask = float(asks[-1]["price"]) if asks else 1
            mid = (best_bid + best_ask) / 2
            spread = best_ask - best_bid

            return {
                "best_bid": best_bid,
                "best_ask": best_ask,
                "mid_price": mid,
                "spread": spread,
                "spread_pct": (spread / mid * 100) if mid > 0 else 0,
            }
        except Exception as e:
            logger.error(f"Failed to get order book for {token_id}: {e}")
            return None

    # ── Trading (CLOB API) ──────────────────────────────────────

    def execute_order(self, order: TradeOrder) -> TradeResult:
        """Execute a trade order on Polymarket."""
        if order.dry_run:
            # Simulate realistic fill with slippage
            from trading.slippage import estimate_fill_price, adjust_ev_for_slippage

            # Try to get real order book for accurate spread
            book = None
            if order.token_id:
                book = self.get_order_book(order.token_id)

            fill_price = estimate_fill_price(
                side=order.side,
                mid_price=order.market_price,
                order_size_usd=order.amount_usd,
                best_bid=book["best_bid"] if book else None,
                best_ask=book["best_ask"] if book else None,
                spread_pct=book["spread_pct"] if book else None,
            )

            adjusted_ev = adjust_ev_for_slippage(
                ev_per_dollar=order.ev_per_dollar,
                mid_price=order.market_price,
                fill_price=fill_price,
                true_prob=order.estimated_prob,
            )

            logger.info(
                f"[DRY RUN] {order.side} ${order.amount_usd:.2f} "
                f"at {fill_price:.4f} (mid: {order.market_price:.4f}) | "
                f"EV: {adjusted_ev:.4f} (raw: {order.ev_per_dollar:.4f}) | "
                f"Kelly: {order.kelly_fraction:.1%}"
            )

            # If slippage kills the edge, mark as skipped
            if adjusted_ev < 0.02:
                logger.warning(
                    f"[DRY RUN] Slippage kills edge: EV {order.ev_per_dollar:.4f} → {adjusted_ev:.4f} — SKIP"
                )
                return TradeResult(
                    order_id=None,
                    status="SKIPPED",
                    fill_price=fill_price,
                    amount_filled=0,
                    error=f"Slippage reduced EV from {order.ev_per_dollar:.4f} to {adjusted_ev:.4f}",
                )

            return TradeResult(
                order_id=None,
                status="DRY_RUN",
                fill_price=fill_price,
                amount_filled=order.amount_usd,
            )

        if not self.private_key:
            return TradeResult(
                order_id=None,
                status="FAILED",
                error="No private key configured — cannot execute live trades",
            )

        try:
            from py_clob_client.clob_types import MarketOrderArgs, OrderType
            from py_clob_client.order_builder.constants import BUY

            mo = MarketOrderArgs(
                token_id=order.token_id,
                amount=order.amount_usd,
                side=BUY,
                order_type=OrderType.FOK,
            )
            signed = self._clob_client.create_market_order(mo)
            resp = self._clob_client.post_order(signed, OrderType.FOK)

            logger.info(f"Order executed: {resp}")
            return TradeResult(
                order_id=resp.get("orderID"),
                status="EXECUTED",
                fill_price=order.market_price,
                amount_filled=order.amount_usd,
            )
        except Exception as e:
            logger.error(f"Trade execution failed: {e}")
            return TradeResult(
                order_id=None,
                status="FAILED",
                error=str(e),
            )

    # ── Helpers ──────────────────────────────────────────────────

    @staticmethod
    def _parse_prices(m: dict) -> tuple[float, float]:
        """Parse prices from Gamma API market data.

        Gamma uses outcomePrices which can be an array or JSON string.
        """
        outcome_prices = m.get("outcomePrices", ["0.50", "0.50"])

        if isinstance(outcome_prices, str):
            try:
                outcome_prices = json.loads(outcome_prices)
            except (json.JSONDecodeError, ValueError):
                outcome_prices = ["0.50", "0.50"]

        if isinstance(outcome_prices, list) and len(outcome_prices) >= 2:
            yes_price = float(outcome_prices[0])
            no_price = float(outcome_prices[1])
        elif m.get("bestAsk"):
            yes_price = float(m["bestAsk"])
            no_price = 1 - yes_price
        else:
            yes_price = 0.5
            no_price = 0.5

        return yes_price, no_price

    @staticmethod
    def _parse_token_ids(m: dict) -> tuple[str, str]:
        """Parse CLOB token IDs from Gamma API data.

        Gamma uses clobTokenIds which can be an array or JSON string.
        """
        clob_ids = m.get("clobTokenIds", [])

        if isinstance(clob_ids, str):
            try:
                clob_ids = json.loads(clob_ids)
            except (json.JSONDecodeError, ValueError):
                clob_ids = []

        if isinstance(clob_ids, list) and len(clob_ids) >= 2:
            return clob_ids[0], clob_ids[1]
        elif isinstance(clob_ids, list) and len(clob_ids) == 1:
            return clob_ids[0], ""

        return "", ""
