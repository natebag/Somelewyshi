"""Strategy optimizer — autoresearch loop for self-improving trading strategy.

Inspired by karpathy/autoresearch and Nunchi-trade/auto-researchtrading:
  Modify strategy → backtest → measure → keep/revert → repeat

Instead of optimizing an LLM training loop, we optimize:
  - EV thresholds
  - Kelly fraction
  - Exit timing
  - Ensemble signal weights
  - Repricing capture targets
"""

from __future__ import annotations

import copy
import json
import logging
import math
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


@dataclass
class StrategyParams:
    """Tunable strategy parameters."""

    # Entry
    min_ev_threshold: float = 0.05
    min_confidence: float = 0.5
    min_signals_agree: int = 3

    # Sizing (Kelly)
    kelly_fraction: float = 0.25
    max_position_pct: float = 0.15

    # Repricing exits
    target_capture_pct: float = 0.60
    stop_loss_pct: float = 0.30
    stale_timeout_seconds: int = 600

    # Time
    time_exit_seconds: int = 60
    min_time_to_expiry: int = 30

    def to_dict(self) -> dict:
        return {
            "min_ev_threshold": self.min_ev_threshold,
            "min_confidence": self.min_confidence,
            "min_signals_agree": self.min_signals_agree,
            "kelly_fraction": self.kelly_fraction,
            "max_position_pct": self.max_position_pct,
            "target_capture_pct": self.target_capture_pct,
            "stop_loss_pct": self.stop_loss_pct,
            "stale_timeout_seconds": self.stale_timeout_seconds,
            "time_exit_seconds": self.time_exit_seconds,
            "min_time_to_expiry": self.min_time_to_expiry,
        }


@dataclass
class BacktestResult:
    """Result of a single backtest run."""

    total_trades: int = 0
    winning_trades: int = 0
    total_pnl: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    win_rate: float = 0.0
    avg_profit_per_trade: float = 0.0
    avg_hold_time_seconds: float = 0.0
    total_log_return: float = 0.0
    score: float = 0.0  # composite score


@dataclass
class HistoricalTrade:
    """A historical trade for backtesting."""

    market_id: str
    question: str
    entry_price: float
    exit_price: float
    our_estimate: float
    direction: str
    resolution: str  # "YES" or "NO"
    time_to_expiry_at_entry: int
    hold_duration_seconds: int
    size_usd: float


@dataclass
class ExperimentResult:
    """Result of one optimization experiment."""

    experiment_id: int
    params: StrategyParams
    backtest: BacktestResult
    description: str
    status: str  # "keep", "discard", "baseline"
    timestamp: str = ""


class StrategyOptimizer:
    """Self-improving strategy optimizer using the autoresearch pattern.

    Loop:
    1. Start with current best params (baseline)
    2. Propose a mutation (change one or more params)
    3. Backtest the mutated strategy
    4. If score improved → keep (new baseline)
    5. If score worse → revert
    6. Repeat
    """

    def __init__(
        self,
        historical_trades: list[HistoricalTrade] | None = None,
        base_params: StrategyParams | None = None,
    ):
        self.historical_trades = historical_trades or []
        self.best_params = base_params or StrategyParams()
        self.best_score = 0.0
        self.experiments: list[ExperimentResult] = []
        self.experiment_counter = 0

    def add_trade(self, trade: HistoricalTrade):
        """Add a historical trade for backtesting."""
        self.historical_trades.append(trade)

    def run_optimization(self, num_experiments: int = 50) -> StrategyParams:
        """Run the autoresearch optimization loop.

        Returns the best strategy parameters found.
        """
        if len(self.historical_trades) < 10:
            logger.warning(
                f"Only {len(self.historical_trades)} historical trades — "
                f"need at least 10 for meaningful optimization"
            )
            return self.best_params

        # Establish baseline
        baseline = self._backtest(self.best_params)
        self.best_score = baseline.score
        self._log_experiment(self.best_params, baseline, "baseline", "baseline")

        logger.info(f"[OPTIMIZER] Baseline score: {self.best_score:.4f}")

        # Optimization loop
        for i in range(num_experiments):
            # Propose mutation
            mutated = self._mutate(self.best_params)
            description = self._describe_mutation(self.best_params, mutated)

            # Backtest
            result = self._backtest(mutated)

            # Keep or discard
            if result.score > self.best_score:
                self.best_params = mutated
                self.best_score = result.score
                status = "keep"
                logger.info(
                    f"[OPTIMIZER] Exp {i+1}: {description} → "
                    f"score {result.score:.4f} (↑ KEEP)"
                )
            else:
                status = "discard"
                logger.debug(
                    f"[OPTIMIZER] Exp {i+1}: {description} → "
                    f"score {result.score:.4f} (discard)"
                )

            self._log_experiment(mutated, result, description, status)

        kept = sum(1 for e in self.experiments if e.status == "keep")
        logger.info(
            f"[OPTIMIZER] Complete: {num_experiments} experiments, "
            f"{kept} improvements, final score: {self.best_score:.4f}"
        )

        return self.best_params

    def _backtest(self, params: StrategyParams) -> BacktestResult:
        """Run a backtest with given parameters against historical trades."""
        if not self.historical_trades:
            return BacktestResult()

        trades_taken = []
        equity = 1000.0
        peak_equity = equity
        max_dd = 0.0

        for trade in self.historical_trades:
            # Would we enter this trade with these params?
            edge = abs(trade.our_estimate - trade.entry_price)
            if edge < params.min_ev_threshold:
                continue

            if trade.time_to_expiry_at_entry < params.min_time_to_expiry:
                continue

            # Size the position
            position_size = min(
                equity * params.kelly_fraction * edge,
                equity * params.max_position_pct,
            )
            if position_size < 1:
                continue

            # Simulate exit based on repricing rules
            pnl = self._simulate_exit(trade, params, position_size)
            equity += pnl
            trades_taken.append(pnl)

            # Track drawdown
            peak_equity = max(peak_equity, equity)
            dd = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0
            max_dd = max(max_dd, dd)

            if equity <= 0:
                break

        if not trades_taken:
            return BacktestResult(score=-999)

        wins = sum(1 for p in trades_taken if p > 0)
        total_pnl = sum(trades_taken)
        avg_pnl = total_pnl / len(trades_taken)

        # Sharpe approximation
        if len(trades_taken) > 1:
            mean = avg_pnl
            variance = sum((p - mean) ** 2 for p in trades_taken) / (len(trades_taken) - 1)
            std = math.sqrt(variance) if variance > 0 else 1
            sharpe = (mean / std) * math.sqrt(252) if std > 0 else 0
        else:
            sharpe = 0

        win_rate = wins / len(trades_taken)

        # Composite score (inspired by Nunchi)
        trade_factor = math.sqrt(min(len(trades_taken) / 20, 1.0))
        dd_penalty = max(0, max_dd - 0.15) * 0.5
        score = sharpe * trade_factor - dd_penalty

        # Hard cutoffs
        if len(trades_taken) < 5:
            score = -999
        if max_dd > 0.50:
            score = -999

        return BacktestResult(
            total_trades=len(trades_taken),
            winning_trades=wins,
            total_pnl=total_pnl,
            max_drawdown_pct=max_dd,
            sharpe_ratio=sharpe,
            win_rate=win_rate,
            avg_profit_per_trade=avg_pnl,
            score=score,
        )

    def _simulate_exit(
        self, trade: HistoricalTrade, params: StrategyParams, size: float
    ) -> float:
        """Simulate a trade exit based on repricing parameters."""
        entry = trade.entry_price
        exit_p = trade.exit_price
        estimate = trade.our_estimate

        if trade.direction in ("BUY_YES", "YES"):
            expected_move = estimate - entry
            target_price = entry + expected_move * params.target_capture_pct
            stop_price = max(entry - params.stop_loss_pct, 0.01)

            # Did we hit target or stop first?
            if exit_p >= target_price:
                # Hit target — capture partial repricing
                actual_exit = target_price
            elif exit_p <= stop_price:
                # Hit stop
                actual_exit = stop_price
            else:
                # Neither — exited at actual price
                actual_exit = exit_p

            shares = size / entry if entry > 0 else 0
            return shares * (actual_exit - entry)
        else:
            expected_move = entry - estimate
            target_price = entry - expected_move * params.target_capture_pct
            stop_price = min(entry + params.stop_loss_pct, 0.99)

            if exit_p <= target_price:
                actual_exit = target_price
            elif exit_p >= stop_price:
                actual_exit = stop_price
            else:
                actual_exit = exit_p

            no_entry = 1 - entry
            no_exit = 1 - actual_exit
            shares = size / no_entry if no_entry > 0 else 0
            return shares * (no_exit - no_entry)

    def _mutate(self, params: StrategyParams) -> StrategyParams:
        """Propose a random mutation to strategy parameters."""
        mutated = copy.deepcopy(params)

        # Pick 1-2 random parameters to mutate
        mutations = random.sample([
            "min_ev_threshold",
            "kelly_fraction",
            "max_position_pct",
            "target_capture_pct",
            "stop_loss_pct",
            "min_signals_agree",
            "time_exit_seconds",
        ], k=random.randint(1, 2))

        for param in mutations:
            current = getattr(mutated, param)
            if isinstance(current, float):
                # Mutate by ±10-30%
                factor = 1 + random.uniform(-0.3, 0.3)
                new_val = current * factor
                # Clamp to reasonable ranges
                if param in ("min_ev_threshold", "kelly_fraction", "max_position_pct",
                             "target_capture_pct", "stop_loss_pct"):
                    new_val = max(0.01, min(0.95, new_val))
                setattr(mutated, param, round(new_val, 4))
            elif isinstance(current, int):
                delta = random.randint(-1, 1)
                new_val = max(1, current + delta)
                setattr(mutated, param, new_val)

        return mutated

    def _describe_mutation(self, old: StrategyParams, new: StrategyParams) -> str:
        """Describe what changed between two parameter sets."""
        changes = []
        for field_name in old.to_dict():
            old_val = getattr(old, field_name)
            new_val = getattr(new, field_name)
            if old_val != new_val:
                changes.append(f"{field_name}: {old_val} → {new_val}")
        return " | ".join(changes) if changes else "no change"

    def _log_experiment(
        self, params: StrategyParams, result: BacktestResult,
        description: str, status: str,
    ):
        self.experiment_counter += 1
        self.experiments.append(ExperimentResult(
            experiment_id=self.experiment_counter,
            params=copy.deepcopy(params),
            backtest=result,
            description=description,
            status=status,
            timestamp=datetime.now(timezone.utc).isoformat(),
        ))

    def get_results_summary(self) -> str:
        """Get a TSV-like summary of all experiments (autoresearch style)."""
        lines = ["exp\tscore\ttrades\twin_rate\tmax_dd\tstatus\tdescription"]
        for e in self.experiments:
            lines.append(
                f"{e.experiment_id}\t{e.backtest.score:.4f}\t"
                f"{e.backtest.total_trades}\t{e.backtest.win_rate:.0%}\t"
                f"{e.backtest.max_drawdown_pct:.1%}\t{e.status}\t"
                f"{e.description}"
            )
        return "\n".join(lines)
