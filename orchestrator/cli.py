"""Typer CLI -- the main entry point for Miro Fish."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.schemas import load_settings
from adapters.llm import LLMClient
from adapters.predict import PredictAdapter
from adapters.trade import TradeAdapter
from adapters.broadcast import BroadcastAdapter
from orchestrator.pipeline import Pipeline
from trading.client import PolymarketClient
from trading.risk import RiskLimits, RiskManager
from trading.scanner import MarketScanner, ScanFilter

app = typer.Typer(name="mirofish", help="Miro Fish -- Predict > Trade > Broadcast")
console = Console()


def _setup_logging(level: str = "INFO"):
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _build_components(settings=None):
    """Wire up all components from settings. Returns (settings, llm, predict, trade, broadcast)."""
    if settings is None:
        settings = load_settings()

    # Use OpenAI-compatible LLM (supports GPT, Claude via proxy, etc.)
    api_key = settings.llm_api_key or settings.anthropic_api_key
    llm = LLMClient(
        api_key=api_key,
        base_url=settings.llm_base_url,
        model=settings.llm_model_name,
    )

    predict = PredictAdapter(
        base_url=settings.mirofish.base_url,
        llm=llm,
        poll_interval=settings.mirofish.poll_interval_seconds,
        poll_timeout=settings.mirofish.poll_timeout_seconds,
    )

    poly_client = PolymarketClient(
        clob_url=settings.trading.clob_url,
        private_key=settings.polymarket_private_key,
        chain_id=settings.trading.chain_id,
        signature_type=settings.trading.signature_type,
        funder=settings.polymarket_funder_address,
    )

    risk_mgr = RiskManager(
        limits=RiskLimits(max_position_usd=settings.trading.max_position_usd),
    )

    trade = TradeAdapter(
        client=poly_client,
        risk_manager=risk_mgr,
        bankroll=settings.trading.bankroll_usd,
        kelly_fraction=settings.trading.kelly_fraction,
        min_ev_threshold=settings.trading.min_ev_threshold,
        max_position_usd=settings.trading.max_position_usd,
        dry_run=settings.trading.dry_run,
    )

    broadcast = BroadcastAdapter(
        llm=llm,
        twitter_enabled=settings.broadcast.twitter_enabled,
        youtube_enabled=settings.broadcast.youtube_enabled,
        twitter_account_id=settings.broadcast.twitter_account_id,
    )

    return settings, llm, predict, trade, broadcast, poly_client


def _build_pipeline(settings=None) -> Pipeline:
    """Wire up the full pipeline."""
    s, llm, predict, trade, broadcast, _ = _build_components(settings)
    return Pipeline(predict=predict, trade=trade, broadcast=broadcast)


# ── Commands ─────────────────────────────────────────────────────


@app.command()
def predict(
    question: str = typer.Argument(..., help="Market question to predict"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed output"),
):
    """Predict the probability of a market event using Claude or MiroFish."""
    _setup_logging("DEBUG" if verbose else "WARNING")
    settings = load_settings()
    # Use OpenAI-compatible LLM (supports GPT, Claude via proxy, etc.)
    api_key = settings.llm_api_key or settings.anthropic_api_key
    llm = LLMClient(
        api_key=api_key,
        base_url=settings.llm_base_url,
        model=settings.llm_model_name,
    )
    predictor = PredictAdapter(base_url=settings.mirofish.base_url, llm=llm)

    console.print(f"\n[bold]Analyzing:[/bold] {question}\n")

    result = predictor.run_prediction(question)

    table = Table(title="Prediction Result", show_lines=True)
    table.add_column("", style="cyan", width=14)
    table.add_column("", style="green")
    table.add_row("Probability", f"[bold]{result.probability:.0%}[/bold]")
    table.add_row("Confidence", result.confidence.upper())
    table.add_row("Source", result.source.upper())
    table.add_row("Reasoning", result.reasoning[:300])

    console.print(table)


@app.command()
def trade(
    question: str = typer.Argument(..., help="Market question"),
    market_price: float = typer.Option(..., "--price", "-p", help="Current YES price (0-1)"),
    market_id: str = typer.Option("", "--market-id", help="Polymarket condition ID"),
    token_id: str = typer.Option("", "--token-id", help="Token ID for execution"),
    dry_run: bool = typer.Option(True, "--dry-run/--live", help="Dry run mode"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Predict + evaluate a trade for a specific market."""
    _setup_logging("DEBUG" if verbose else "WARNING")
    settings = load_settings()
    if dry_run:
        settings.trading.dry_run = True

    pipeline = _build_pipeline(settings)
    result = pipeline.run(
        market_question=question,
        market_id=market_id,
        market_price=market_price,
        token_id=token_id,
        skip_broadcast=True,
    )

    _print_trade_result(result, settings.trading.dry_run)


@app.command()
def run(
    question: str = typer.Argument(..., help="Market question"),
    market_price: float = typer.Option(..., "--price", "-p", help="Current YES price (0-1)"),
    market_id: str = typer.Option("", "--market-id"),
    token_id: str = typer.Option("", "--token-id"),
    dry_run: bool = typer.Option(True, "--dry-run/--live"),
    broadcast: bool = typer.Option(True, "--broadcast/--no-broadcast"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Full pipeline: Predict > Trade > Broadcast."""
    _setup_logging("DEBUG" if verbose else "WARNING")
    settings = load_settings()
    if dry_run:
        settings.trading.dry_run = True

    pipeline = _build_pipeline(settings)
    result = pipeline.run(
        market_question=question,
        market_id=market_id,
        market_price=market_price,
        token_id=token_id,
        skip_broadcast=not broadcast,
    )

    _print_trade_result(result, settings.trading.dry_run)

    if result.broadcasts:
        console.print("\n[bold]Broadcast:[/bold]")
        for bc in result.broadcasts:
            status_color = "green" if bc.status == "POSTED" else "yellow"
            console.print(f"  {bc.platform}: [{status_color}]{bc.status}[/{status_color}]")
            if bc.content_text:
                console.print(Panel(bc.content_text, title="Tweet", width=70))


@app.command()
def scan(
    top: int = typer.Option(10, "--top", "-n", help="Number of top markets to show"),
    min_volume: float = typer.Option(10000, "--min-volume", help="Minimum volume filter"),
    analyze: bool = typer.Option(False, "--analyze", "-a", help="Run Claude prediction on top markets"),
    dry_run: bool = typer.Option(True, "--dry-run/--live"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Scan Polymarket for trading opportunities."""
    _setup_logging("DEBUG" if verbose else "WARNING")
    settings, llm, predict, trade_adapter, broadcast, poly_client = _build_components()
    if dry_run:
        settings.trading.dry_run = True

    scanner = MarketScanner(
        client=poly_client,
        filters=ScanFilter(min_volume=min_volume, max_markets=top),
    )

    console.print("\n[bold]Scanning Polymarket...[/bold]\n")
    results = scanner.scan()

    if not results:
        console.print("[yellow]No markets found matching filters.[/yellow]")
        return

    # Display scan results
    table = Table(title=f"Top {len(results)} Markets", show_lines=True)
    table.add_column("#", style="dim", width=3)
    table.add_column("Market", style="white", max_width=50)
    table.add_column("YES", style="green", justify="right", width=6)
    table.add_column("Volume", style="cyan", justify="right", width=12)
    table.add_column("Score", style="yellow", justify="right", width=6)

    for i, r in enumerate(results, 1):
        table.add_row(
            str(i),
            r.snapshot.question[:50],
            f"{r.snapshot.yes_price:.0%}",
            f"${r.snapshot.volume:,.0f}",
            f"{r.score:.1f}",
        )

    console.print(table)

    # Optionally analyze top markets with Claude
    if analyze:
        console.print(f"\n[bold]Analyzing top {min(len(results), 5)} markets with Claude...[/bold]\n")

        opportunities = []
        for r in results[:5]:
            pred = predict.run_prediction(
                r.snapshot.question,
                market_data={
                    "yes_price": r.snapshot.yes_price,
                    "no_price": r.snapshot.no_price,
                    "volume": r.snapshot.volume,
                    "end_date": r.snapshot.end_date,
                },
            )

            from trading.strategy import evaluate_trade
            signal = evaluate_trade(
                market_price=r.snapshot.yes_price,
                true_prob=pred.probability,
                bankroll=settings.trading.bankroll_usd,
                kelly_multiplier=settings.trading.kelly_fraction,
                min_ev_threshold=settings.trading.min_ev_threshold,
                max_position_usd=settings.trading.max_position_usd,
            )

            if signal.should_trade:
                opportunities.append((r, pred, signal))

        _print_scan_analysis(opportunities, settings)


@app.command()
def daemon(
    interval: int = typer.Option(6, help="Hours between scan cycles"),
    max_markets: int = typer.Option(10, "--max-markets", "-n", help="Max markets to analyze per cycle"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Start the automated discover + predict + trade daemon."""
    _setup_logging("DEBUG" if verbose else "INFO")
    from orchestrator.scheduler import PipelineScheduler
    from trading.discovery import DiscoveryConfig

    settings, llm, predict_adapter, trade_adapter, broadcast, poly_client = _build_components()

    pipeline = Pipeline(predict=predict_adapter, trade=trade_adapter, broadcast=broadcast)

    discovery_config = DiscoveryConfig(max_markets_to_analyze=max_markets)
    scheduler = PipelineScheduler(
        pipeline=pipeline,
        poly_client=poly_client,
        predict=predict_adapter,
        interval_hours=interval,
        discovery_config=discovery_config,
    )

    console.print("[bold]Starting Miro Fish daemon[/bold]")
    console.print(f"  Interval:    {interval}h")
    console.print(f"  Max markets: {max_markets}")
    console.print(f"  Mode:        {'DRY RUN' if settings.trading.dry_run else 'LIVE'}")
    console.print("  Press Ctrl+C to stop\n")

    scheduler.start()


@app.command()
def fastloop(
    interval_minutes: int = typer.Option(5, "--interval", "-i", help="Minutes between scans"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """BTC fast-loop: 5-min market scanner with live price feed."""
    _setup_logging("DEBUG" if verbose else "INFO")
    import time as _time
    from trading.btc_feed import get_btc_feed
    from trading.fast_loop import FastLoopStrategy, PriceBar
    from datetime import datetime, timezone

    settings, llm, predict_adapter, trade_adapter, broadcast, poly_client = _build_components()
    pipeline = Pipeline(predict=predict_adapter, trade=trade_adapter, broadcast=broadcast)
    btc_feed = get_btc_feed()
    strategy = FastLoopStrategy(min_signals_agree=3, min_edge=0.08)

    console.print("[bold]Starting BTC Fast Loop[/bold]")
    console.print(f"  Scan interval: {interval_minutes}min")
    console.print(f"  Mode:          {'DRY RUN' if settings.trading.dry_run else 'LIVE'}")
    console.print("  Press Ctrl+C to stop\n")

    cycle = 0
    while True:
        try:
            cycle += 1
            now = datetime.now(timezone.utc).strftime("%H:%M:%S")

            # Get live BTC price
            btc = btc_feed.get_snapshot()
            if btc:
                console.print(f"\n[dim]{now}[/dim] [bold]Cycle {cycle}[/bold] | {btc.to_prompt_context()}")
            else:
                console.print(f"\n[dim]{now}[/dim] [bold]Cycle {cycle}[/bold] | BTC feed unavailable")

            # Scan for BTC markets
            from trading.scanner import MarketScanner, ScanFilter
            scanner = MarketScanner(poly_client, ScanFilter(min_volume=1000, max_markets=50))
            results = scanner.scan()

            btc_markets = [r for r in results if btc_feed.is_btc_market(r.snapshot.question)]

            if not btc_markets:
                console.print(f"  [dim]No BTC markets found[/dim]")
                _time.sleep(interval_minutes * 60)
                continue

            console.print(f"  Found {len(btc_markets)} BTC markets")

            for r in btc_markets[:3]:  # Top 3 BTC markets
                snap = r.snapshot
                console.print(f"\n  [cyan]{snap.question[:60]}[/cyan]")
                console.print(f"  YES: {snap.yes_price:.0%} | NO: {snap.no_price:.0%} | Vol: ${snap.volume:,.0f}")

                # Add BTC price as history for momentum signals
                strategy.add_price(PriceBar(
                    timestamp=datetime.now(timezone.utc),
                    yes_price=snap.yes_price,
                    volume=snap.volume,
                ))

                # Run prediction with BTC context
                try:
                    result = pipeline.run(
                        market_question=snap.question,
                        market_id=snap.condition_id,
                        market_price=snap.yes_price,
                        token_id=snap.token_id_yes,
                        skip_broadcast=True,
                    )

                    sig = result.trade_signal
                    pred = result.prediction

                    # Color-coded output
                    edge_color = "green" if sig.ev_per_dollar > 0.10 else "yellow" if sig.ev_per_dollar > 0.05 else "dim"
                    dir_display = {
                        "BUY_YES": "[green]BUY YES[/green]",
                        "BUY_NO": "[red]BUY NO[/red]",
                    }.get(sig.direction, "[dim]SKIP[/dim]")

                    console.print(
                        f"  P(up): {pred.probability:.2f} | "
                        f"Edge: [{edge_color}]{sig.ev_per_dollar:.0%}[/{edge_color}] | "
                        f"Conf: {pred.confidence} | "
                        f"Signal: {dir_display} | "
                        f"Size: ${sig.position_size:.2f}"
                    )
                except Exception as e:
                    console.print(f"  [red]Error: {e}[/red]")

            _time.sleep(interval_minutes * 60)

        except KeyboardInterrupt:
            console.print("\n[bold]Fast loop stopped[/bold]")
            break
        except Exception as e:
            console.print(f"[red]Cycle error: {e}[/red]")
            _time.sleep(30)


@app.command()
def history(
    limit: int = typer.Option(20, "--limit", "-n", help="Number of records to show"),
    positions: bool = typer.Option(False, "--positions", "-p", help="Show positions instead of trades"),
    portfolio: bool = typer.Option(False, "--portfolio", help="Show portfolio summary"),
):
    """View past trades, positions, and P&L."""
    _setup_logging("WARNING")

    from db.engine import get_engine, get_session
    from db.models import PipelineRun, Trade
    from sqlmodel import select
    from trading.positions import PositionTracker

    engine = get_engine()
    session = get_session(engine)

    if portfolio:
        tracker = PositionTracker(session)
        summary = tracker.get_portfolio_summary()

        table = Table(title="Portfolio Summary", show_lines=True)
        table.add_column("Metric", style="cyan", width=22)
        table.add_column("Value", style="green")

        table.add_row("Total Invested", f"${summary.total_invested:.2f}")
        pnl_color = "green" if summary.total_unrealized_pnl >= 0 else "red"
        table.add_row("Unrealized P&L", f"[{pnl_color}]${summary.total_unrealized_pnl:+.2f}[/{pnl_color}]")
        rpnl_color = "green" if summary.total_realized_pnl >= 0 else "red"
        table.add_row("Realized P&L", f"[{rpnl_color}]${summary.total_realized_pnl:+.2f}[/{rpnl_color}]")
        table.add_row("Open Positions", str(summary.open_positions))
        table.add_row("Closed Positions", str(summary.closed_positions))
        table.add_row("Win Rate", f"{summary.win_rate:.0%}")
        table.add_row("Total Log Return", f"{summary.total_log_return:+.4f}")

        console.print(table)
        session.close()
        return

    if positions:
        tracker = PositionTracker(session)
        all_pos = tracker.get_all_positions()

        if not all_pos:
            console.print("[dim]No positions yet.[/dim]")
            session.close()
            return

        table = Table(title="Positions", show_lines=True)
        table.add_column("Market", max_width=30)
        table.add_column("Side", width=8)
        table.add_column("Entry", justify="right", width=6)
        table.add_column("Current", justify="right", width=7)
        table.add_column("Size", justify="right", width=8)
        table.add_column("P&L", justify="right", width=10)
        table.add_column("Status", width=7)

        for p in all_pos[:limit]:
            pnl_color = "green" if p.unrealized_pnl >= 0 else "red"
            status_color = "green" if p.status == "OPEN" else "dim"
            table.add_row(
                p.market_id[:30] if p.market_id else "N/A",
                p.side,
                f"{p.entry_price:.0%}",
                f"{p.current_price:.0%}",
                f"${p.size_usd:.2f}",
                f"[{pnl_color}]${p.unrealized_pnl:+.2f}[/{pnl_color}]",
                f"[{status_color}]{p.status}[/{status_color}]",
            )

        console.print(table)
        session.close()
        return

    # Default: show recent trades
    trades = session.exec(
        select(Trade).order_by(Trade.created_at.desc()).limit(limit)
    ).all()

    if not trades:
        console.print("[dim]No trades yet.[/dim]")
        session.close()
        return

    table = Table(title="Trade History", show_lines=True)
    table.add_column("Time", style="dim", width=19)
    table.add_column("Side", width=8)
    table.add_column("Price", justify="right", width=6)
    table.add_column("EV", justify="right", width=6)
    table.add_column("Kelly", justify="right", width=6)
    table.add_column("Size", justify="right", width=8)
    table.add_column("Status", width=9)

    for t in trades:
        ev_color = "green" if t.expected_value > 0.10 else "yellow" if t.expected_value > 0.05 else "dim"
        status_color = {"EXECUTED": "green", "DRY_RUN": "yellow", "SKIPPED": "dim", "FAILED": "red"}.get(t.status.value, "white")

        table.add_row(
            t.created_at.strftime("%Y-%m-%d %H:%M:%S"),
            t.side,
            f"{t.market_price_at_entry:.0%}",
            f"[{ev_color}]{t.expected_value:.1%}[/{ev_color}]",
            f"{t.kelly_fraction:.1%}",
            f"${t.size_usd:.2f}",
            f"[{status_color}]{t.status.value}[/{status_color}]",
        )

    console.print(table)
    session.close()


@app.command()
def calibration():
    """Check prediction accuracy — is the bot actually right?"""
    _setup_logging("WARNING")

    from db.engine import get_engine, get_session
    from trading.calibration import CalibrationTracker

    engine = get_engine()
    session = get_session(engine)

    tracker = CalibrationTracker(session)
    report = tracker.analyze()

    if report.resolved_predictions == 0:
        console.print("[yellow]No resolved predictions yet.[/yellow]")
        console.print(f"[dim]{report.total_predictions} predictions waiting for resolution.[/dim]")
        console.print("\n[dim]Markets need to resolve (close) before we can measure accuracy.[/dim]")
        session.close()
        return

    console.print(Panel(report.summary(), title="Prediction Calibration", width=70))
    session.close()


@app.command()
def status():
    """Show current system status and recent pipeline runs."""
    _setup_logging("WARNING")

    settings = load_settings()

    # Config status
    table = Table(title="Miro Fish Status", show_lines=True)
    table.add_column("Component", style="cyan", width=16)
    table.add_column("Status", style="green")

    llm_key = settings.llm_api_key or settings.anthropic_api_key
    llm_label = f"configured ({settings.llm_model_name})" if llm_key else "[red]missing key[/red]"
    table.add_row("LLM", llm_label)
    table.add_row("Polymarket", "configured" if settings.polymarket_private_key else "read-only (no key)")
    table.add_row("Trading Mode", "[yellow]DRY RUN[/yellow]" if settings.trading.dry_run else "[red]LIVE[/red]")
    table.add_row("Bankroll", f"${settings.trading.bankroll_usd:.0f}")
    table.add_row("Kelly Fraction", f"{settings.trading.kelly_fraction:.0%}")
    table.add_row("Min EV", f"{settings.trading.min_ev_threshold:.0%}")
    table.add_row("Twitter", "enabled" if settings.broadcast.twitter_enabled else "disabled")

    # Check MiroFish
    predict = PredictAdapter(base_url=settings.mirofish.base_url)
    mf_status = "running" if predict.is_mirofish_available() else "[yellow]offline (Claude fallback)[/yellow]"
    table.add_row("MiroFish", mf_status)

    console.print(table)

    # Recent runs
    try:
        from db.engine import get_engine, get_session
        from db.models import PipelineRun
        from sqlmodel import select

        engine = get_engine()
        session = get_session(engine)
        runs = session.exec(
            select(PipelineRun).order_by(PipelineRun.started_at.desc()).limit(5)
        ).all()

        if runs:
            run_table = Table(title="Recent Pipeline Runs", show_lines=True)
            run_table.add_column("ID", style="dim", width=8)
            run_table.add_column("Market", max_width=40)
            run_table.add_column("Status", width=14)
            run_table.add_column("Started", width=19)

            for r in runs:
                status_color = {"DONE": "green", "FAILED": "red"}.get(r.status.value, "yellow")
                run_table.add_row(
                    r.id[:8],
                    r.market_query[:40],
                    f"[{status_color}]{r.status.value}[/{status_color}]",
                    r.started_at.strftime("%Y-%m-%d %H:%M:%S"),
                )
            console.print(run_table)
        else:
            console.print("\n[dim]No pipeline runs yet.[/dim]")

        session.close()
    except Exception:
        console.print("\n[dim]No pipeline runs yet.[/dim]")


# ── Display Helpers ──────────────────────────────────────────────


def _print_trade_result(result, dry_run: bool):
    """Pretty-print a pipeline result."""
    sig = result.trade_signal
    pred = result.prediction

    # Determine edge color
    edge = sig.ev_per_dollar
    edge_color = "green" if edge > 0.10 else "yellow" if edge > 0.05 else "red"

    # Direction indicator
    if sig.direction == "BUY_YES":
        dir_display = "[green]BUY YES[/green]"
    elif sig.direction == "BUY_NO":
        dir_display = "[red]BUY NO[/red]"
    else:
        dir_display = "[dim]SKIP[/dim]"

    table = Table(
        title=f"{'[DRY RUN] ' if dry_run else ''}Trade Analysis",
        show_lines=True,
    )
    table.add_column("", style="cyan", width=16)
    table.add_column("", min_width=30)

    table.add_row("Market Price", f"{sig.market_price:.0%}")
    table.add_row("Claude Estimate", f"[bold]{pred.probability:.0%}[/bold] ({pred.confidence})")
    table.add_row("Edge", f"[{edge_color}]{edge:.1%}[/{edge_color}]")
    table.add_row("Direction", dir_display)
    table.add_row("Kelly (1/4)", f"{sig.kelly_fraction:.1%} of bankroll")
    table.add_row("Position Size", f"[bold]${sig.position_size:.2f}[/bold]")
    table.add_row("Status", result.trade_result.status)
    table.add_row("Reasoning", pred.reasoning[:200])

    console.print(table)
    console.print(f"\n  [dim]Run ID: {result.run_id}[/dim]")


def _print_scan_analysis(opportunities, settings):
    """Pretty-print scan + analysis results like the article's output."""
    if not opportunities:
        console.print("[yellow]No opportunities found above EV threshold.[/yellow]")
        return

    console.print()
    header = (
        f"[bold]SCAN COMPLETE[/bold]\n"
        f"  Edge found: {len(opportunities)}\n"
    )
    console.print(Panel(header, title="Miro Fish Scanner", width=70))

    table = Table(show_lines=True, width=70)
    table.add_column("#", style="dim", width=3)
    table.add_column("Market", max_width=35)
    table.add_column("Mkt", justify="right", width=5)
    table.add_column("Est", justify="right", width=5)
    table.add_column("Edge", justify="right", width=6)
    table.add_column("Bet", justify="right", width=8)
    table.add_column("Signal", width=8)

    for i, (scan_result, pred, signal) in enumerate(opportunities, 1):
        edge = signal.ev_per_dollar
        edge_color = "green" if edge > 0.10 else "yellow"
        sig_display = "[green]BUY YES[/green]" if signal.direction == "BUY_YES" else "[red]BUY NO[/red]"

        table.add_row(
            str(i),
            scan_result.snapshot.question[:35],
            f"{scan_result.snapshot.yes_price:.0%}",
            f"{pred.probability:.0%}",
            f"[{edge_color}]{edge:.0%}[/{edge_color}]",
            f"${signal.position_size:.0f}",
            sig_display,
        )

    console.print(table)


if __name__ == "__main__":
    _setup_logging()
    app()
