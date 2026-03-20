"""Prediction calibration tracker — measures if the bot is actually right.

The most important question: when the bot says 60%, does YES happen ~60% of the time?

Tracks:
- Brier score (lower = better, 0 = perfect)
- Calibration curve (predicted vs actual)
- Resolution tracking (did the market actually resolve YES or NO?)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone

from sqlmodel import Session, select

from db.models import Prediction

logger = logging.getLogger(__name__)


@dataclass
class CalibrationBucket:
    """A bucket for calibration analysis (e.g., all predictions between 0.5-0.6)."""

    range_low: float
    range_high: float
    predictions: int = 0
    actual_yes: int = 0

    @property
    def predicted_avg(self) -> float:
        return (self.range_low + self.range_high) / 2

    @property
    def actual_rate(self) -> float:
        return self.actual_yes / self.predictions if self.predictions > 0 else 0

    @property
    def error(self) -> float:
        """How far off the prediction was from reality."""
        return abs(self.predicted_avg - self.actual_rate)


@dataclass
class CalibrationReport:
    """Full calibration analysis."""

    total_predictions: int = 0
    resolved_predictions: int = 0
    unresolved_predictions: int = 0
    brier_score: float = 0.0  # 0 = perfect, 1 = worst
    accuracy: float = 0.0  # % of correct binary calls
    avg_confidence: float = 0.0
    buckets: list[CalibrationBucket] = field(default_factory=list)
    overconfident: bool = False  # Predicts too extreme (70% but only 55% right)
    underconfident: bool = False  # Predicts too mild (55% but actually 70% right)

    def summary(self) -> str:
        lines = [
            f"Calibration Report ({self.resolved_predictions} resolved / {self.total_predictions} total)",
            f"  Brier Score:     {self.brier_score:.4f} (0=perfect, 0.25=coin flip)",
            f"  Binary Accuracy: {self.accuracy:.0%}",
            f"  Avg Confidence:  {self.avg_confidence:.0%}",
        ]
        if self.overconfident:
            lines.append("  WARNING: Model is OVERCONFIDENT — predictions are too extreme")
        if self.underconfident:
            lines.append("  NOTE: Model is underconfident — could size positions larger")

        if self.buckets:
            lines.append("\n  Calibration Curve:")
            lines.append(f"  {'Predicted':>10} {'Actual':>8} {'Count':>6} {'Error':>7}")
            for b in self.buckets:
                if b.predictions > 0:
                    marker = " *" if b.error > 0.15 else ""
                    lines.append(
                        f"  {b.predicted_avg:>9.0%} {b.actual_rate:>7.0%} {b.predictions:>6} {b.error:>6.0%}{marker}"
                    )

        return "\n".join(lines)


@dataclass
class PredictionOutcome:
    """A prediction paired with its actual outcome."""

    prediction_id: str
    market_question: str
    predicted_prob: float
    confidence: str
    actual_outcome: str  # "YES", "NO", "UNRESOLVED"
    was_correct: bool | None = None  # None if unresolved


class CalibrationTracker:
    """Tracks and analyzes prediction calibration."""

    def __init__(self, session: Session):
        self.session = session

    def record_outcome(self, prediction_id: str, actual_outcome: str):
        """Record the actual outcome for a prediction.

        Call this when a market resolves.
        actual_outcome: "YES" or "NO"
        """
        pred = self.session.get(Prediction, prediction_id)
        if not pred:
            logger.warning(f"Prediction {prediction_id} not found")
            return

        # Store outcome in the prediction's metadata
        # We'll add a resolution field
        pred.resolution = actual_outcome
        pred.resolved_at = datetime.now(timezone.utc)
        self.session.add(pred)
        self.session.commit()

        # Check if prediction was directionally correct
        if actual_outcome == "YES":
            correct = pred.probability > 0.5
        else:
            correct = pred.probability < 0.5

        logger.info(
            f"[CALIBRATION] {pred.market_query[:40]}... | "
            f"Predicted: {pred.probability:.0%} | "
            f"Actual: {actual_outcome} | "
            f"{'CORRECT' if correct else 'WRONG'}"
        )

    def get_all_outcomes(self) -> list[PredictionOutcome]:
        """Get all predictions with their outcomes."""
        preds = self.session.exec(select(Prediction)).all()

        outcomes = []
        for p in preds:
            resolution = getattr(p, "resolution", None) or "UNRESOLVED"
            was_correct = None
            if resolution == "YES":
                was_correct = p.probability > 0.5
            elif resolution == "NO":
                was_correct = p.probability < 0.5

            outcomes.append(PredictionOutcome(
                prediction_id=p.id,
                market_question=p.market_query,
                predicted_prob=p.probability,
                confidence=p.confidence,
                actual_outcome=resolution,
                was_correct=was_correct,
            ))

        return outcomes

    def analyze(self) -> CalibrationReport:
        """Run full calibration analysis on all resolved predictions."""
        outcomes = self.get_all_outcomes()

        total = len(outcomes)
        resolved = [o for o in outcomes if o.actual_outcome in ("YES", "NO")]
        unresolved = total - len(resolved)

        if not resolved:
            return CalibrationReport(
                total_predictions=total,
                unresolved_predictions=unresolved,
            )

        # Brier score: mean of (predicted - actual)^2
        brier_sum = 0
        correct_count = 0
        confidence_sum = 0

        for o in resolved:
            actual = 1.0 if o.actual_outcome == "YES" else 0.0
            brier_sum += (o.predicted_prob - actual) ** 2
            if o.was_correct:
                correct_count += 1
            confidence_sum += abs(o.predicted_prob - 0.5)

        brier_score = brier_sum / len(resolved)
        accuracy = correct_count / len(resolved)
        avg_confidence = confidence_sum / len(resolved) + 0.5

        # Build calibration buckets (10% intervals)
        buckets = []
        for i in range(10):
            low = i * 0.1
            high = (i + 1) * 0.1
            bucket = CalibrationBucket(range_low=low, range_high=high)

            for o in resolved:
                if low <= o.predicted_prob < high:
                    bucket.predictions += 1
                    if o.actual_outcome == "YES":
                        bucket.actual_yes += 1

            buckets.append(bucket)

        # Detect over/under confidence
        active_buckets = [b for b in buckets if b.predictions >= 3]
        if active_buckets:
            avg_error = sum(b.error for b in active_buckets) / len(active_buckets)
            overconfident = sum(
                1 for b in active_buckets
                if b.predicted_avg > 0.5 and b.actual_rate < b.predicted_avg
            )
            underconfident = sum(
                1 for b in active_buckets
                if b.predicted_avg > 0.5 and b.actual_rate > b.predicted_avg
            )
        else:
            overconfident = 0
            underconfident = 0

        return CalibrationReport(
            total_predictions=total,
            resolved_predictions=len(resolved),
            unresolved_predictions=unresolved,
            brier_score=brier_score,
            accuracy=accuracy,
            avg_confidence=avg_confidence,
            buckets=buckets,
            overconfident=overconfident > underconfident,
            underconfident=underconfident > overconfident,
        )
