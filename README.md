# Miro Fish

Automated Polymarket prediction & trading bot powered by AI research.

**Scan markets → Auto-research → Predict probabilities → Size bets → Execute trades**

## How It Works

```
Polymarket (Gamma API)          GPT-4o / MiroFish OASIS
       │                               │
       ▼                               ▼
  Market Scanner ──────────► Auto-Research Engine
       │                         │
       │                    Decompose question
       │                    Gather evidence (15+ sources)
       │                    Synthesize report
       │                         │
       ▼                         ▼
  EV Calculator ◄──── Probability Estimate
       │
       ├── Expected Value (entry signal)
       ├── Kelly Criterion (position sizing)
       ├── Bayesian Updating (re-evaluate on new data)
       ├── Repricing Exit Engine (exit before resolution)
       │
       ▼
  Trade Executor (dry-run or live)
       │
       ▼
  Position Tracker + Strategy Optimizer
```

## Quick Start

```bash
# Clone
git clone https://github.com/natebag/Somelewyshi.git
cd Somelewyshi

# Set up Python environment
python -m venv .venv
.venv/Scripts/activate      # Windows
# source .venv/bin/activate  # Mac/Linux

# Install
pip install -e ".[dev]"

# Configure
cp .env.example .env
# Edit .env with your API keys (at minimum: LLM_API_KEY)

# Test
pytest tests/ -v

# Run
mirofish status
```

## Commands

| Command | What it does |
|---------|-------------|
| `mirofish predict "Question"` | Get AI probability estimate |
| `mirofish trade "Question" -p 0.42` | Predict + evaluate trade |
| `mirofish run "Question" -p 0.42` | Full pipeline: predict + trade + broadcast |
| `mirofish scan --analyze` | Scan Polymarket for mispriced markets |
| `mirofish smart` | **Smart daemon** — auto-adjusts scan frequency per market type |
| `mirofish daemon --interval 1` | Simple daemon — fixed interval for all markets |
| `mirofish fastloop --interval 5` | BTC 5-min market scanner with live price feed |
| `mirofish history --portfolio` | View trades, positions, P&L |
| `mirofish calibration` | Check prediction accuracy (Brier score, win rate) |
| `mirofish validate` | Check resolved markets, compare repricing vs hold P&L |
| `mirofish status` | System health check |

## Example

```bash
# Single trade analysis
mirofish trade "Will the Fed cut rates in June 2026?" -p 0.42

# Output:
#   Market Price:    42%
#   AI Estimate:     30% (medium confidence)
#   Edge:            12.0%
#   Direction:       BUY NO
#   Kelly (1/4):     7.1% of bankroll
#   Position Size:   $35.71
#   Status:          DRY_RUN

# Scan real markets
mirofish scan -n 10

# Start the daemon (scans every hour, all market types)
mirofish daemon --interval 1 --max-markets 5

# Smart daemon (recommended — auto-adjusts scan frequency)
mirofish smart

# Or simple daemon (fixed hourly interval)
mirofish daemon --interval 1 --max-markets 5

# Or BTC-only fast-loop
mirofish fastloop --interval 5

# Check if predictions are actually correct
mirofish calibration

# Compare repricing exit vs hold-to-resolution P&L
mirofish validate
```

## Smart Daemon

`mirofish smart` is the recommended way to run. It auto-classifies markets and scans at the right frequency:

```
┌─────────────────┬───────────┬──────────────────────────────────────┐
│ Market Type      │ Frequency │ Examples                             │
├─────────────────┼───────────┼──────────────────────────────────────┤
│ BTC Fast (5min)  │ 2 min     │ "BTC above $95K in 5 minutes?"       │
│ Crypto / Sports  │ 15 min    │ "ETH > $4K by Friday?", NBA games    │
│ Political / Wx   │ 60 min    │ "Fed rate cut?", "NYC temp > 60°F?"  │
│ Maintenance      │ 5 min     │ Position monitoring + resolutions     │
└─────────────────┴───────────┴──────────────────────────────────────┘
```

Each cycle:
```
SCAN     → Gamma API, filtered by market type
    ↓
RESEARCH → Auto-research engine (10-15 evidence sources)
    ↓
PREDICT  → GPT-4o + BTC price feed (for crypto markets)
    ↓
EVALUATE → EV → slippage simulation → Kelly sizing → risk cap
    ↓
TRADE    → Realistic dry-run or live execution
    ↓
MONITOR  → Real price polling + repricing exit engine
    ↓
RESOLVE  → Track market outcomes, validate predictions
    ↓
OPTIMIZE → Auto-retune strategy when win rate drops
```

## Architecture

```
adapters/
  llm.py          — OpenAI-compatible LLM client (GPT-4o, Claude, etc.)
  predict.py      — 3-tier prediction: MiroFish → Research+LLM → LLM-only
  trade.py        — Trade evaluation + execution adapter
  broadcast.py    — Twitter/Discord broadcast adapter

trading/
  strategy.py     — EV, Kelly Criterion, Bayesian updating, LMSR, log returns
  client.py       — Polymarket client (Gamma API discovery + CLOB execution)
  scanner.py      — Market scanner + filters
  discovery.py    — Auto-discover trending/mispriced markets
  research.py     — Auto-research engine (decompose → gather → synthesize)
  fast_loop.py    — BTC 5-min strategy with ensemble voting
  btc_feed.py     — Live BTC price feed (CryptoCompare, spot + deltas)
  repricing.py    — Exit engine: capture repricing spread before resolution
  optimizer.py    — Self-improving strategy (autoresearch pattern)
  slippage.py     — Realistic fill price simulation (spread + market impact)
  calibration.py  — Prediction accuracy tracking (Brier score)
  position_monitor.py — Real-time position monitoring with exit logic
  positions.py    — Position tracker + P&L
  updater.py      — Bayesian re-updating on market movements
  risk.py         — Position caps, exposure limits, bankroll reserve

orchestrator/
  pipeline.py     — Predict → Trade → Broadcast pipeline
  scheduler.py    — Automated daemon with market discovery
  cli.py          — Typer CLI entry point

config/
  schemas.py      — Pydantic settings + TOML config
  settings.toml   — Default configuration

db/
  models.py       — SQLModel schemas (predictions, trades, positions)
  engine.py       — SQLite database engine

vendor/
  mirofish/       — OASIS multi-agent social simulation engine
  moneyprinterv2/ — Content generation for broadcasts
```

## The Math

Four formulas power the trading decisions:

**1. Expected Value** — when to enter
```
EV = P(win) × Profit − P(lose) × Loss
```

**2. Kelly Criterion** — how much to bet
```
f* = (p × b − q) / b    (we use quarter-Kelly for safety)
```

**3. Bayesian Updating** — how to change your mind
```
P(H|E) = P(E|H) × P(H) / P(E)
```

**4. Log Returns** — how to measure profit correctly
```
log_return = ln(P₁ / P₀)    (sums correctly across trades)
```

## Key Insight

> The real money isn't in predicting outcomes. It's in buying mispriced probability and exiting when it reprices. Exit before resolution, capture the spread.

The repricing exit engine monitors positions and exits when:
- Target repricing captured (default: 60% of expected move)
- Edge disappeared (market converged to our estimate)
- Stop loss hit (adverse move > 30%)
- Time exit (approaching resolution)
- Stale position (no movement)

## Configuration

Copy `.env.example` to `.env` and fill in:

```env
# Required — at least one LLM key
LLM_API_KEY=sk-...              # OpenAI key
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL_NAME=gpt-4o-mini

# Optional — for live trading
POLYMARKET_PRIVATE_KEY=0x...    # Polygon wallet private key

# Optional — for MiroFish simulations
ZEP_API_KEY=z_...               # getzep.com

# Optional — for alerts
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...
```

Settings in `config/settings.toml`:
```toml
[trading]
bankroll_usd = 500.0
kelly_fraction = 0.25
min_ev_threshold = 0.05
max_position_usd = 50.0
dry_run = true
```

## MiroFish (Optional)

MiroFish runs OASIS social simulations — hundreds of AI agents debating on simulated Twitter/Reddit to generate probability estimates. Much richer signal than single-LLM predictions.

```bash
# Requires Docker
cd vendor/mirofish
docker compose up -d

# Verify
curl http://localhost:5001/health
```

## Testing

```bash
# Run all tests (47 tests)
pytest tests/ -v

# Run specific module
pytest tests/test_strategy.py -v
pytest tests/test_fast_loop.py -v
pytest tests/test_repricing.py -v
pytest tests/test_optimizer.py -v
```

## Backtesting & Self-Improvement

The strategy optimizer uses the autoresearch pattern (inspired by [karpathy/autoresearch](https://github.com/karpathy/autoresearch) and [Nunchi-trade/auto-researchtrading](https://github.com/Nunchi-trade/auto-researchtrading)):

1. Start with current strategy params
2. Propose a mutation (tweak EV threshold, Kelly fraction, exit targets)
3. Backtest against historical trades
4. Keep if score improved, revert if worse
5. Repeat

After enough trades accumulate, run:
```python
from trading.optimizer import StrategyOptimizer
optimizer = StrategyOptimizer(historical_trades=trades)
best_params = optimizer.run_optimization(num_experiments=100)
```

## Credits

- [MiroFish](https://github.com/666ghj/MiroFish) — OASIS multi-agent simulation
- [MoneyPrinterV2](https://github.com/FujiwaraChoki/MoneyPrinterV2) — Content generation
- [karpathy/autoresearch](https://github.com/karpathy/autoresearch) — Self-improving research pattern
- [Nunchi-trade/auto-researchtrading](https://github.com/Nunchi-trade/auto-researchtrading) — Strategy optimization loop

## License

MIT
