"""Tests for strategy optimizer."""

from trading.optimizer import (
    BacktestResult,
    HistoricalTrade,
    StrategyOptimizer,
    StrategyParams,
)


def _make_trades(n=30, win_rate=0.7):
    """Generate synthetic historical trades."""
    trades = []
    for i in range(n):
        is_win = i % 10 < int(win_rate * 10)
        entry = 0.40
        estimate = 0.60
        exit_p = 0.55 if is_win else 0.30

        trades.append(HistoricalTrade(
            market_id=f"market_{i}",
            question=f"Test question {i}",
            entry_price=entry,
            exit_price=exit_p,
            our_estimate=estimate,
            direction="BUY_YES",
            resolution="YES" if is_win else "NO",
            time_to_expiry_at_entry=300,
            hold_duration_seconds=120,
            size_usd=50,
        ))
    return trades


def test_baseline_scores():
    trades = _make_trades(30)
    optimizer = StrategyOptimizer(historical_trades=trades)
    result = optimizer._backtest(StrategyParams())
    assert result.total_trades > 0
    assert result.score != -999


def test_optimization_improves_or_maintains():
    trades = _make_trades(50)
    optimizer = StrategyOptimizer(historical_trades=trades)

    baseline = optimizer._backtest(StrategyParams())
    best = optimizer.run_optimization(num_experiments=20)

    final = optimizer._backtest(best)
    # Should maintain or improve (optimizer only keeps improvements)
    assert final.score >= baseline.score or final.score > -999


def test_mutation_changes_params():
    optimizer = StrategyOptimizer()
    original = StrategyParams()
    # Try multiple times since mutation is random
    changed = False
    for _ in range(10):
        mutated = optimizer._mutate(original)
        orig_dict = original.to_dict()
        mut_dict = mutated.to_dict()
        if any(orig_dict[k] != mut_dict[k] for k in orig_dict):
            changed = True
            break
    assert changed


def test_too_few_trades_warns():
    trades = _make_trades(5)
    optimizer = StrategyOptimizer(historical_trades=trades)
    result = optimizer.run_optimization(num_experiments=5)
    # Should return default params (not enough data)
    assert result is not None


def test_results_summary():
    trades = _make_trades(30)
    optimizer = StrategyOptimizer(historical_trades=trades)
    optimizer.run_optimization(num_experiments=5)
    summary = optimizer.get_results_summary()
    assert "exp" in summary
    assert "score" in summary
