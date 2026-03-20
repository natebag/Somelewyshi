"""Quick verification that all modules load correctly."""

import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    print("=== Miro Fish — Verification ===\n")

    # 1. Config
    print("[1] Config...")
    from config.schemas import load_settings
    s = load_settings()
    print(f"    Trading: dry_run={s.trading.dry_run}, kelly={s.trading.kelly_fraction}")
    print(f"    MiroFish: {s.mirofish.base_url}")
    print(f"    Broadcast: twitter={s.broadcast.twitter_enabled}")
    print(f"    OK\n")

    # 2. Database
    print("[2] Database...")
    from db.engine import init_db
    from db.models import PipelineRun, Prediction, Trade, Broadcast, Position
    engine = init_db()
    print(f"    Tables: pipeline_runs, predictions, trades, broadcasts, positions")
    print(f"    OK\n")

    # 3. Trading strategy
    print("[3] Trading strategy...")
    from trading.strategy import evaluate_trade, expected_value, kelly_fraction
    ev = expected_value(0.40, 0.60)
    kf = kelly_fraction(0.60, 0.40)
    signal = evaluate_trade(
        market_price=0.40,
        true_prob=0.60,
        bankroll=1000,
        kelly_multiplier=0.25,
    )
    print(f"    EV(market=40%, true=60%) = {ev:.2f}")
    print(f"    Kelly(60%, 40%) = {kf:.1%}")
    print(f"    Signal: {signal.direction} ${signal.position_size:.2f}")
    print(f"    OK\n")

    # 4. LMSR
    print("[4] LMSR pricing...")
    from trading.strategy import lmsr_price, lmsr_cost
    prices = [lmsr_price([100, 50], b=100, outcome_index=i) for i in range(2)]
    print(f"    LMSR prices [q=100,50, b=100]: YES={prices[0]:.3f}, NO={prices[1]:.3f}")
    print(f"    Sum={sum(prices):.3f} (should be 1.000)")
    print(f"    OK\n")

    # 5. Adapters
    print("[5] Adapters...")
    from adapters.llm import LLMClient
    from adapters.predict import PredictAdapter
    from adapters.trade import TradeAdapter
    from adapters.broadcast import BroadcastAdapter
    print(f"    All adapters imported OK\n")

    # 6. CLI
    print("[6] CLI...")
    from orchestrator.cli import app
    print(f"    CLI app loaded OK\n")

    print("=== All checks passed ===")


if __name__ == "__main__":
    main()
