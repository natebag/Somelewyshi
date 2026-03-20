"""Auto-research module — gathers evidence on market topics to feed MiroFish.

Inspired by karpathy/autoresearch pattern:
  hypothesis → gather evidence → synthesize → score → iterate

Generates structured research documents that MiroFish can ingest
for its OASIS social simulation pipeline.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

from adapters.llm import LLMClient

logger = logging.getLogger(__name__)


@dataclass
class ResearchSource:
    """A single piece of evidence."""

    title: str
    content: str
    source_type: str  # "news", "data", "analysis", "historical"
    relevance_score: float = 0.0


@dataclass
class ResearchReport:
    """Structured research output for a market question."""

    question: str
    summary: str
    base_rate: float  # historical frequency of similar events
    key_factors_for: list[str]  # evidence supporting YES
    key_factors_against: list[str]  # evidence supporting NO
    probability_estimate: float
    confidence: str  # "high", "medium", "low"
    sources: list[ResearchSource] = field(default_factory=list)
    generated_at: str = ""
    research_strategy: str = ""

    def to_document(self) -> str:
        """Convert to a text document suitable for MiroFish ingestion."""
        lines = [
            f"# Research Report: {self.question}",
            f"Generated: {self.generated_at}",
            f"Strategy: {self.research_strategy}",
            "",
            "## Executive Summary",
            self.summary,
            "",
            f"## Base Rate Analysis",
            f"Historical frequency of similar events: {self.base_rate:.0%}",
            "",
            "## Evidence FOR (YES)",
        ]
        for i, factor in enumerate(self.key_factors_for, 1):
            lines.append(f"{i}. {factor}")

        lines.extend(["", "## Evidence AGAINST (NO)"])
        for i, factor in enumerate(self.key_factors_against, 1):
            lines.append(f"{i}. {factor}")

        lines.extend([
            "",
            f"## Probability Estimate: {self.probability_estimate:.0%}",
            f"## Confidence: {self.confidence}",
            "",
            "## Sources",
        ])
        for src in self.sources:
            lines.append(f"- [{src.source_type}] {src.title}: {src.content[:200]}")

        return "\n".join(lines)


class AutoResearcher:
    """Autonomous research agent that gathers evidence for market predictions.

    Uses the autoresearch pattern:
    1. Decompose market question into research queries
    2. Gather evidence from multiple angles
    3. Synthesize into structured report
    4. Score and iterate on research strategy
    """

    def __init__(self, llm: LLMClient):
        self.llm = llm
        self.research_history: list[dict] = []

    def research(
        self,
        market_question: str,
        market_data: dict | None = None,
        max_depth: int = 2,
    ) -> ResearchReport:
        """Run autonomous research on a market question.

        Args:
            market_question: The prediction market question
            market_data: Optional current market data (price, volume, etc.)
            max_depth: How many research iterations to run
        """
        logger.info(f"[RESEARCH] Starting research on: {market_question}")

        # Step 1: Decompose into research angles
        angles = self._decompose_question(market_question, market_data)
        logger.info(f"[RESEARCH] Identified {len(angles)} research angles")

        # Step 2: Research each angle
        sources: list[ResearchSource] = []
        for angle in angles:
            evidence = self._research_angle(angle, market_question)
            sources.extend(evidence)

        logger.info(f"[RESEARCH] Gathered {len(sources)} pieces of evidence")

        # Step 3: Synthesize into report
        report = self._synthesize(market_question, sources, market_data)

        # Step 4: Log for strategy improvement tracking
        self.research_history.append({
            "question": market_question,
            "num_sources": len(sources),
            "estimate": report.probability_estimate,
            "confidence": report.confidence,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

        return report

    def _decompose_question(
        self, question: str, market_data: dict | None
    ) -> list[str]:
        """Break a market question into research angles."""
        market_context = ""
        if market_data:
            market_context = f"""
Current market data:
- YES price: {market_data.get('yes_price', 'N/A')}
- Volume: {market_data.get('volume', 'N/A')}
- End date: {market_data.get('end_date', 'N/A')}
"""

        prompt = f"""You are a research analyst decomposing a prediction market question into research angles.

Question: {question}
{market_context}

Break this into 3-5 specific research angles. Each angle should be a specific
factual question that helps estimate the probability of the event.

Think about:
1. Base rate — how often do similar events happen?
2. Current conditions — what recent data is relevant?
3. Expert/institutional signals — what do authorities say?
4. Historical analogies — what happened in similar past situations?
5. Contrarian view — what could make the consensus wrong?

Respond as a JSON array of strings, each being one research angle.
Example: ["What is the historical base rate of Fed rate cuts in similar economic conditions?", ...]
"""

        response = self.llm.complete(prompt, max_tokens=500)
        try:
            # Extract JSON array from response
            text = response.strip()
            if "```" in text:
                text = text.split("```")[1].strip()
                if text.startswith("json"):
                    text = text[4:].strip()
            angles = json.loads(text)
            if isinstance(angles, list):
                return angles[:5]
        except (json.JSONDecodeError, IndexError):
            pass

        # Fallback: generate basic angles
        return [
            f"What is the base rate for events like: {question}?",
            f"What are the most recent developments related to: {question}?",
            f"What factors could cause {question} to resolve YES vs NO?",
        ]

    def _research_angle(
        self, angle: str, original_question: str
    ) -> list[ResearchSource]:
        """Research a specific angle using LLM knowledge."""
        prompt = f"""You are researching a specific angle for a prediction market question.

Original question: {original_question}
Research angle: {angle}

Provide 2-3 specific, factual pieces of evidence related to this angle.
Include concrete data points, dates, numbers, and sources where possible.

Respond as a JSON array of objects with fields:
- "title": short description
- "content": the evidence (2-3 sentences with specific data)
- "source_type": one of "news", "data", "analysis", "historical"
- "relevance_score": 0.0 to 1.0

Example: [{{"title": "Fed dot plot March 2026", "content": "The March FOMC meeting...", "source_type": "data", "relevance_score": 0.8}}]
"""

        response = self.llm.complete(prompt, max_tokens=800)
        try:
            text = response.strip()
            if "```" in text:
                text = text.split("```")[1].strip()
                if text.startswith("json"):
                    text = text[4:].strip()
            items = json.loads(text)
            if isinstance(items, list):
                return [
                    ResearchSource(
                        title=item.get("title", ""),
                        content=item.get("content", ""),
                        source_type=item.get("source_type", "analysis"),
                        relevance_score=item.get("relevance_score", 0.5),
                    )
                    for item in items[:3]
                ]
        except (json.JSONDecodeError, IndexError):
            pass

        return [
            ResearchSource(
                title=f"Analysis: {angle[:50]}",
                content=response[:500],
                source_type="analysis",
                relevance_score=0.5,
            )
        ]

    def _synthesize(
        self,
        question: str,
        sources: list[ResearchSource],
        market_data: dict | None,
    ) -> ResearchReport:
        """Synthesize all evidence into a structured report."""
        evidence_text = "\n".join(
            f"- [{s.source_type}] {s.title}: {s.content}" for s in sources
        )

        market_context = ""
        if market_data:
            market_context = f"""
Market currently prices YES at {market_data.get('yes_price', 'N/A')}.
Volume: {market_data.get('volume', 'N/A')}.
"""

        prompt = f"""You are a quantitative analyst synthesizing research into a probability estimate.

Question: {question}
{market_context}

Evidence gathered:
{evidence_text}

Synthesize this into a final assessment. Respond as JSON:
{{
    "summary": "2-3 sentence executive summary",
    "base_rate": 0.XX,
    "key_factors_for": ["factor 1", "factor 2", ...],
    "key_factors_against": ["factor 1", "factor 2", ...],
    "probability_estimate": 0.XX,
    "confidence": "high/medium/low",
    "research_strategy": "brief description of what research approach was used"
}}

Be calibrated. If you say 70%, that means ~7 out of 10 similar assessments should be correct.
Consider the base rate before adjusting based on evidence.
"""

        response = self.llm.complete(prompt, max_tokens=800)
        try:
            text = response.strip()
            if "```" in text:
                text = text.split("```")[1].strip()
                if text.startswith("json"):
                    text = text[4:].strip()
            data = json.loads(text)

            return ResearchReport(
                question=question,
                summary=data.get("summary", ""),
                base_rate=data.get("base_rate", 0.5),
                key_factors_for=data.get("key_factors_for", []),
                key_factors_against=data.get("key_factors_against", []),
                probability_estimate=data.get("probability_estimate", 0.5),
                confidence=data.get("confidence", "medium"),
                sources=sources,
                generated_at=datetime.now(timezone.utc).isoformat(),
                research_strategy=data.get("research_strategy", "multi-angle analysis"),
            )
        except (json.JSONDecodeError, KeyError):
            return ResearchReport(
                question=question,
                summary="Research synthesis failed — using evidence summary",
                base_rate=0.5,
                key_factors_for=[],
                key_factors_against=[],
                probability_estimate=0.5,
                confidence="low",
                sources=sources,
                generated_at=datetime.now(timezone.utc).isoformat(),
            )
