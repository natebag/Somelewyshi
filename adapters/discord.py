"""Discord webhook adapter — sends trade alerts to a Discord channel."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import httpx

logger = logging.getLogger(__name__)


class DiscordWebhook:
    """Sends formatted trade alerts to Discord via webhook."""

    def __init__(self, webhook_url: str | None = None):
        self.webhook_url = webhook_url
        self.enabled = bool(webhook_url)

    def send_trade_alert(
        self,
        market_question: str,
        direction: str,
        market_price: float,
        claude_estimate: float,
        ev_per_dollar: float,
        position_size: float,
        kelly_fraction: float,
        confidence: str = "medium",
        reasoning: str = "",
        dry_run: bool = True,
    ) -> bool:
        """Send a trade alert embed to Discord."""
        if not self.enabled:
            logger.debug("Discord webhook not configured, skipping")
            return False

        edge = claude_estimate - market_price
        edge_pct = ev_per_dollar * 100

        # Color: green for BUY_YES, red for BUY_NO, grey for SKIP
        if direction == "BUY_YES":
            color = 0x00D26A
            dir_emoji = "🟢"
        elif direction == "BUY_NO":
            color = 0xFF4757
            dir_emoji = "🔴"
        else:
            color = 0x808080
            dir_emoji = "⏭️"

        mode_tag = "🧪 DRY RUN" if dry_run else "🔴 LIVE"

        embed = {
            "title": f"{dir_emoji} {direction} — {market_question[:80]}",
            "color": color,
            "fields": [
                {"name": "Market Price", "value": f"`{market_price:.0%}`", "inline": True},
                {"name": "Claude Estimate", "value": f"`{claude_estimate:.0%}`", "inline": True},
                {"name": "Edge", "value": f"`{edge_pct:+.1f}%`", "inline": True},
                {"name": "Kelly (¼)", "value": f"`{kelly_fraction:.1%}`", "inline": True},
                {"name": "Position Size", "value": f"`${position_size:.2f}`", "inline": True},
                {"name": "Confidence", "value": f"`{confidence}`", "inline": True},
            ],
            "footer": {"text": f"Miro Fish • {mode_tag} • {datetime.now(timezone.utc).strftime('%H:%M UTC')}"},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        if reasoning:
            embed["description"] = reasoning[:300]

        payload = {
            "username": "Miro Fish 🐟",
            "embeds": [embed],
        }

        try:
            resp = httpx.post(self.webhook_url, json=payload, timeout=10)
            resp.raise_for_status()
            logger.info(f"Discord alert sent: {direction} {market_question[:40]}")
            return True
        except httpx.HTTPError as e:
            logger.error(f"Discord webhook failed: {e}")
            return False

    def send_scan_summary(
        self,
        markets_analyzed: int,
        opportunities_found: int,
        top_trades: list[dict],
    ) -> bool:
        """Send a scan cycle summary to Discord."""
        if not self.enabled:
            return False

        lines = [f"**Markets analyzed:** {markets_analyzed}", f"**Opportunities:** {opportunities_found}"]

        if top_trades:
            lines.append("")
            for i, t in enumerate(top_trades[:5], 1):
                lines.append(
                    f"`#{i}` {t.get('question', 'N/A')[:45]}\n"
                    f"    Market: `{t.get('market_price', 0):.0%}` → "
                    f"Claude: `{t.get('claude_estimate', 0):.0%}` | "
                    f"Bet: `${t.get('position_size', 0):.0f}`"
                )

        embed = {
            "title": "📊 Scan Cycle Complete",
            "description": "\n".join(lines),
            "color": 0x6C5CE7,
            "footer": {"text": f"Miro Fish • {datetime.now(timezone.utc).strftime('%H:%M UTC')}"},
        }

        payload = {"username": "Miro Fish 🐟", "embeds": [embed]}

        try:
            resp = httpx.post(self.webhook_url, json=payload, timeout=10)
            resp.raise_for_status()
            return True
        except httpx.HTTPError as e:
            logger.error(f"Discord summary failed: {e}")
            return False

    def send_portfolio_update(
        self,
        realized_pnl: float,
        unrealized_pnl: float,
        win_rate: float,
        open_positions: int,
        total_log_return: float,
    ) -> bool:
        """Send a portfolio status update."""
        if not self.enabled:
            return False

        pnl_emoji = "📈" if realized_pnl >= 0 else "📉"

        embed = {
            "title": f"{pnl_emoji} Portfolio Update",
            "color": 0x00D26A if realized_pnl >= 0 else 0xFF4757,
            "fields": [
                {"name": "Realized P&L", "value": f"`${realized_pnl:+.2f}`", "inline": True},
                {"name": "Unrealized P&L", "value": f"`${unrealized_pnl:+.2f}`", "inline": True},
                {"name": "Win Rate", "value": f"`{win_rate:.0%}`", "inline": True},
                {"name": "Open Positions", "value": f"`{open_positions}`", "inline": True},
                {"name": "Log Return", "value": f"`{total_log_return:+.4f}`", "inline": True},
            ],
            "footer": {"text": f"Miro Fish • {datetime.now(timezone.utc).strftime('%H:%M UTC')}"},
        }

        payload = {"username": "Miro Fish 🐟", "embeds": [embed]}

        try:
            resp = httpx.post(self.webhook_url, json=payload, timeout=10)
            resp.raise_for_status()
            return True
        except httpx.HTTPError as e:
            logger.error(f"Discord portfolio update failed: {e}")
            return False
