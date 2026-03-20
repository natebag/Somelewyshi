"""Unified LLM client — OpenAI-compatible API (works with GPT, Claude via proxy, etc.)."""

from __future__ import annotations

import json
import logging

from openai import OpenAI

logger = logging.getLogger(__name__)


class LLMClient:
    """OpenAI-compatible LLM wrapper for probability extraction and content generation."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-4o-mini",
    ):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def complete(self, prompt: str, max_tokens: int = 500) -> str:
        """Raw completion — returns text response."""
        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content or ""

    def extract_probability(self, report_text: str, market_question: str) -> dict:
        """
        Extract a structured probability estimate from a MiroFish report.

        Returns: {"probability": float, "confidence": str, "reasoning": str}
        """
        prompt = f"""You are a quantitative prediction market analyst.

Given this simulation report and market question, extract a probability estimate.

Market Question: {market_question}

Simulation Report:
{report_text[:8000]}

Your task:
1. Estimate the probability of YES (0.00 - 1.00)
2. Rate your confidence: high / medium / low
3. Provide brief reasoning

Respond STRICTLY in JSON:
{{"probability": 0.XX, "confidence": "high/medium/low", "reasoning": "..."}}"""

        text = self.complete(prompt, max_tokens=500)
        return self._parse_json(text)

    def estimate_probability(self, market_question: str, market_data: dict) -> dict:
        """
        Direct probability estimation.

        Returns: {"probability": float, "confidence": str, "reasoning": str}
        """
        research_ctx = ""
        if market_data.get("research_context"):
            research_ctx = f"""
Research Context:
{market_data['research_context'][:4000]}
"""

        prompt = f"""You are a quantitative prediction market analyst.

Analyze this Polymarket market:

Question: {market_question}
Current YES price: {market_data.get('yes_price', 'N/A')}
Current NO price: {market_data.get('no_price', 'N/A')}
Volume: {market_data.get('volume', 'N/A')}
Close date: {market_data.get('end_date', 'N/A')}
{research_ctx}
Your task:
1. Estimate the REAL probability of the event (0.00 - 1.00)
2. Consider the base rate — how often do similar events happen?
3. Factor in all data you currently know
4. Be calibrated — if you say 70%, ~7 out of 10 such assessments should come true

Respond STRICTLY in JSON:
{{"probability": 0.XX, "confidence": "high/medium/low", "reasoning": "..."}}"""

        text = self.complete(prompt, max_tokens=500)
        return self._parse_json(text)

    def generate_tweet(self, prediction: dict, trade: dict) -> str:
        """Generate a tweet about a prediction/trade."""
        prompt = f"""Write a concise, engaging tweet (max 260 chars) about this prediction market trade.

Market: {prediction.get('market_question', 'Unknown')}
My estimate: {prediction.get('probability', 0):.0%}
Market price: {trade.get('market_price', 0):.0%}
Edge: {prediction.get('probability', 0) - trade.get('market_price', 0):.0%}
Action: {trade.get('direction', 'SKIP')}
Bet size: ${trade.get('position_size', 0):.2f}

Style: data-driven, confident but not arrogant. Include the key numbers.
No hashtags. No emojis. Just signal.

Return ONLY the tweet text, nothing else."""

        return self.complete(prompt, max_tokens=300).strip()

    def _parse_json(self, text: str) -> dict:
        """Parse JSON from LLM response, handling markdown wrapping."""
        try:
            # Handle ```json ... ``` wrapping
            clean = text.strip()
            if "```" in clean:
                clean = clean.split("```")[1].strip()
                if clean.startswith("json"):
                    clean = clean[4:].strip()
            return json.loads(clean)
        except (json.JSONDecodeError, IndexError) as e:
            logger.error(f"Failed to parse LLM response: {e}\nRaw: {text[:200]}")
            return {"probability": 0.5, "confidence": "low", "reasoning": "Parse error"}
