# Miro Fish

Automated Polymarket prediction & trading bot powered by AI research.

**Scan markets > Auto-research > Predict probabilities > Size bets > Execute trades**

## Setup (5 minutes)

### 1. Clone the repo

```bash
git clone https://github.com/natebag/Somelewyshi.git
cd Somelewyshi
```

### 2. Create Python environment

**Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**Mac/Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -e ".[dev]"
```

### 4. Configure API keys

```bash
cp .env.example .env
```

Open `.env` in any text editor and add your OpenAI key:

```env
LLM_API_KEY=sk-your-openai-key-here
```

That's the only required key. Everything else is optional.

### 5. Verify it works

```bash
pytest tests/ -v          # Should show 47 passed
mirofish status           # Should show LLM configured
```

### 6. Start the bot

```bash
mirofish smart
```

That's it. The bot is now scanning Polymarket and making dry-run trades.

## Checking Your Results

Open a **second terminal** (keep the bot running in the first one):

```bash
cd Somelewyshi
.venv\Scripts\activate    # Windows
# source .venv/bin/activate  # Mac/Linux

# See all trades
mirofish history

# Portfolio summary (P&L, win rate)
mirofish history --portfolio

# Open positions
mirofish history --positions

# Are predictions actually correct?
mirofish calibration

# Repricing exit vs hold-to-resolution comparison
mirofish validate

# System health
mirofish status
```

## All Commands

| Command | What it does |
|---------|-------------|
| `mirofish smart` | **Start the bot** - auto-adjusts scan speed per market type |
| `mirofish history` | View all trades |
| `mirofish history --portfolio` | P&L summary, win rate, total return |
| `mirofish history --positions` | Open/closed positions |
| `mirofish calibration` | Prediction accuracy (Brier score) |
| `mirofish validate` | Compare repricing exit vs hold-to-resolution P&L |
| `mirofish status` | System health check |
| `mirofish predict "Question"` | One-off probability estimate |
| `mirofish trade "Question" -p 0.42` | One-off trade analysis |
| `mirofish scan --analyze` | Scan Polymarket without trading |
| `mirofish daemon --interval 1` | Simple daemon (fixed hourly, all markets) |
| `mirofish fastloop --interval 5` | BTC-only 5-min scanner |

## How `mirofish smart` Works

The smart daemon runs 4 loops simultaneously, each scanning at the right speed for its market type:

| Loop | Frequency | Markets |
|------|-----------|---------|
| Fast | Every 2 min | BTC 5-min / 15-min price markets |
| Medium | Every 15 min | Crypto, sports, NBA, UFC |
| Slow | Every 60 min | Political, weather, economic |
| Maintenance | Every 5 min | Position monitoring, resolutions, auto-optimizer |

Each scan cycle:

```
SCAN     > Find markets on Polymarket (Gamma API)
RESEARCH > Auto-gather 10-15 evidence sources per market
PREDICT  > GPT-4o estimates probability (+ live BTC price for crypto)
EVALUATE > Calculate edge > simulate slippage > Kelly size > risk cap
TRADE    > Execute (dry-run or live) with realistic fill price
MONITOR  > Poll real prices, close positions on repricing triggers
RESOLVE  > Track actual outcomes, validate predictions
OPTIMIZE > Auto-retune strategy if win rate drops below 55%
```

## The Trading Math

**Expected Value** - should I trade this?
```
EV = P(win) x Profit - P(lose) x Loss
Only trade when EV > 5% (configurable)
```

**Kelly Criterion** - how much to bet?
```
f* = (p x b - q) / b
We use quarter-Kelly (aggressive enough to grow, safe enough to survive)
```

**Repricing Exit** - when to close?
```
Don't wait for the market to resolve.
Buy mispriced probability, exit when it corrects.
Capture the spread, move on.
```

Exit triggers:
- Target repricing captured (60% of expected move)
- Edge disappeared (market caught up to our estimate)
- Stop loss (adverse move > 30%)
- Time exit (approaching resolution)
- Stale position (no price movement)

**Slippage Simulation** - is the edge real?
```
Dry-run trades get realistic fill prices (spread + market impact).
If slippage kills the edge, the trade is SKIPPED, not fake-executed.
```

## Configuration

### Required

| Key | Where to get it |
|-----|----------------|
| `LLM_API_KEY` | [platform.openai.com/api-keys](https://platform.openai.com/api-keys) |

### Optional

| Key | What it enables |
|-----|----------------|
| `POLYMARKET_PRIVATE_KEY` | Live trading (Polygon wallet private key) |
| `ZEP_API_KEY` | MiroFish OASIS simulations ([getzep.com](https://getzep.com)) |
| `DISCORD_WEBHOOK_URL` | Trade alerts to Discord |

### Trading Settings

Edit `config/settings.toml`:

```toml
[trading]
bankroll_usd = 500.0       # Starting capital
kelly_fraction = 0.25       # Quarter-Kelly (safe)
min_ev_threshold = 0.05     # Minimum 5% edge to trade
max_position_usd = 50.0     # Cap per position
dry_run = true              # Set to false for live trading
```

## Going Live Checklist

Don't go live until you've validated in dry-run:

- [ ] Run `mirofish smart` for at least a few days
- [ ] Accumulate 50+ dry-run trades
- [ ] Check `mirofish calibration` - are predictions accurate?
- [ ] Check `mirofish validate` - is repricing exit profitable?
- [ ] Check `mirofish history --portfolio` - is P&L positive after slippage?
- [ ] Fund a Polygon wallet with USDC
- [ ] Add `POLYMARKET_PRIVATE_KEY` to `.env`
- [ ] Set `dry_run = false` in `config/settings.toml`
- [ ] Start with small bankroll ($50-100)
- [ ] Scale up only after real P&L matches dry-run P&L

## MiroFish (Optional - Advanced)

MiroFish runs hundreds of AI agents in a social simulation (Twitter/Reddit style) to generate richer probability estimates. Requires Docker.

```bash
# Start MiroFish
cd vendor/mirofish
docker compose up -d

# Verify it's running
curl http://localhost:5001/health
# Should return: {"status": "ok"}
```

When MiroFish is running, the bot automatically uses it instead of direct GPT-4o estimation.

## Project Structure

```
adapters/           API clients (LLM, Polymarket, Discord)
trading/            Core logic (strategy, scanner, research, exits)
orchestrator/       Pipeline, daemon, CLI
config/             Settings (TOML + env)
db/                 SQLite database (trades, positions, predictions)
tests/              47 tests
vendor/             MiroFish + MoneyPrinterV2 (git submodules)
```

## Credits

- [MiroFish](https://github.com/666ghj/MiroFish) - OASIS multi-agent simulation
- [MoneyPrinterV2](https://github.com/FujiwaraChoki/MoneyPrinterV2) - Content generation
- [karpathy/autoresearch](https://github.com/karpathy/autoresearch) - Self-improving research pattern
- [Nunchi-trade/auto-researchtrading](https://github.com/Nunchi-trade/auto-researchtrading) - Strategy optimization loop

## License

MIT
