"""Broadcast adapter — wraps MoneyPrinterV2 for content posting."""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path

from adapters.llm import LLMClient

logger = logging.getLogger(__name__)

# Add MoneyPrinterV2 to path for direct imports
_VENDOR_PATH = Path(__file__).parent.parent / "vendor" / "moneyprinterv2" / "src"
if _VENDOR_PATH.exists() and str(_VENDOR_PATH) not in sys.path:
    sys.path.insert(0, str(_VENDOR_PATH))


@dataclass
class BroadcastResult:
    """Result of a broadcast operation."""

    platform: str
    status: str  # "POSTED", "FAILED", "SKIPPED"
    content_text: str = ""
    post_url: str = ""
    error: str = ""


class BroadcastAdapter:
    """Generates and posts content from prediction/trade data."""

    def __init__(
        self,
        llm: LLMClient,
        twitter_enabled: bool = True,
        youtube_enabled: bool = False,
        twitter_account_id: str = "1",
    ):
        self.llm = llm
        self.twitter_enabled = twitter_enabled
        self.youtube_enabled = youtube_enabled
        self.twitter_account_id = twitter_account_id

    def broadcast(
        self,
        prediction: dict,
        trade: dict,
    ) -> list[BroadcastResult]:
        """Generate content and post to enabled platforms."""
        results = []

        if self.twitter_enabled:
            results.append(self._post_twitter(prediction, trade))

        if self.youtube_enabled:
            results.append(self._post_youtube(prediction, trade))

        if not results:
            results.append(BroadcastResult(
                platform="NONE",
                status="SKIPPED",
                content_text="No broadcast platforms enabled",
            ))

        return results

    def _post_twitter(self, prediction: dict, trade: dict) -> BroadcastResult:
        """Generate and post a tweet about the trade."""
        try:
            tweet_text = self.llm.generate_tweet(prediction, trade)
            logger.info(f"Generated tweet ({len(tweet_text)} chars): {tweet_text}")

            try:
                from classes.Twitter import Twitter

                twitter = Twitter(self.twitter_account_id)
                twitter.post(custom_text=tweet_text)

                return BroadcastResult(
                    platform="TWITTER",
                    status="POSTED",
                    content_text=tweet_text,
                )
            except ImportError:
                logger.warning(
                    "MoneyPrinterV2 Twitter module not available — tweet generated but not posted"
                )
                return BroadcastResult(
                    platform="TWITTER",
                    status="SKIPPED",
                    content_text=tweet_text,
                    error="MoneyPrinterV2 not installed",
                )

        except Exception as e:
            logger.error(f"Twitter broadcast failed: {e}")
            return BroadcastResult(
                platform="TWITTER",
                status="FAILED",
                error=str(e),
            )

    def _post_youtube(self, prediction: dict, trade: dict) -> BroadcastResult:
        """Generate and upload a YouTube Short about the trade."""
        # Deferred to Phase 5
        return BroadcastResult(
            platform="YOUTUBE",
            status="SKIPPED",
            error="YouTube Shorts not yet implemented",
        )
