"""
CLI Demo - Live Terminal Visualization.

Simulates streaming predictions with fuel gauge display.
"""

import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rich import box
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from opencr.logging import get_logger

logger = get_logger(__name__)
console = Console()


@dataclass(frozen=True)
class DemoSymbols:
    trend_up: str
    trend_down: str
    trend_flat: str
    ok: str
    fail: str
    bar_filled: str
    bar_empty: str


@dataclass
class DemoConfig:
    """Configuration for demo display."""

    delay_ms: int = 500  # Delay between updates in milliseconds
    gauge_width: int = 40  # Width of fuel gauge bar
    window_size: int = 5  # Moving average window for trend
    ascii_only: bool | None = None  # Force ASCII output (auto if None)


def _auto_ascii() -> bool:
    encoding = (sys.stdout.encoding or "").lower()
    return os.name == "nt" and "utf-8" not in encoding


def _resolve_ascii(ascii_only: bool | None) -> bool:
    return _auto_ascii() if ascii_only is None else ascii_only


def _get_symbols(ascii_only: bool) -> DemoSymbols:
    if ascii_only:
        return DemoSymbols(
            trend_up="^",
            trend_down="v",
            trend_flat="-",
            ok="OK",
            fail="X",
            bar_filled="#",
            bar_empty="-",
        )
    return DemoSymbols(
        trend_up="\u2191",
        trend_down="\u2193",
        trend_flat="\u2192",
        ok="\u2714",
        fail="\u2716",
        bar_filled="\u2588",
        bar_empty="\u2591",
    )


def _panel_kwargs(ascii_only: bool) -> dict[str, object]:
    return {"box": box.ASCII} if ascii_only else {}


def run_demo(
    run_dir: Path,
    subject_id: str | None = None,
    config: DemoConfig | None = None,
) -> None:
    """
    Run live demo visualization.

    Args:
        run_dir: Path to baseline run directory.
        subject_id: Subject to visualize (optional, uses first if not specified).
        config: Demo configuration.
    """
    if config is None:
        config = DemoConfig()

    ascii_only = _resolve_ascii(config.ascii_only)
    symbols = _get_symbols(ascii_only)
    panel_kwargs = _panel_kwargs(ascii_only)

    run_dir = Path(run_dir)

    results_path = run_dir / "results.json"
    if not results_path.exists():
        console.print(f"[red]Error:[/red] results.json not found in {run_dir}")
        return

    with open(results_path) as f:
        results = json.load(f)

    folds = results.get("folds", [])

    if not folds:
        console.print("[red]Error:[/red] No folds found in results")
        return

    target_fold = None
    if subject_id:
        for fold in folds:
            if fold["test_subject"] == subject_id or subject_id in fold["test_subject"]:
                target_fold = fold
                break
        if not target_fold:
            console.print(f"[yellow]Subject {subject_id} not found, using first fold[/yellow]")
            target_fold = folds[0]
    else:
        target_fold = folds[0]

    subject_name = target_fold["test_subject"]
    fold_idx = target_fold["fold_idx"]

    pred_path = run_dir / f"fold_{fold_idx:02d}" / "predictions.npz"
    if not pred_path.exists():
        console.print(f"[red]Error:[/red] Predictions not found: {pred_path}")
        return

    with np.load(pred_path, allow_pickle=True) as data:
        y_true = data["y_true"]
        y_pred = data["y_pred"]

    n_samples = len(y_true)

    console.print(
        Panel(
            f"[bold cyan]OpenCR Demo[/bold cyan]\n\n"
            f"Subject: [green]{subject_name}[/green]\n"
            f"Samples: {n_samples}\n"
            f"Delay: {config.delay_ms}ms",
            title="[bold]Live Fuel Gauge[/bold]",
            **panel_kwargs,
        )
    )
    console.print()

    time.sleep(1)

    history = []
    trend_history = []

    with Live(console=console, refresh_per_second=4) as live:
        for i in range(n_samples):
            true_val = int(y_true[i])
            pred_val = int(y_pred[i])
            correct = true_val == pred_val

            history.append(correct)

            accuracy = sum(history) / len(history)
            gauge_pct = accuracy * 100
            trend_history.append(gauge_pct)
            if len(trend_history) > config.window_size:
                trend_history.pop(0)

            trend = trend_history[-1] - trend_history[0] if len(trend_history) >= 2 else 0.0

            if trend > 2:
                trend_arrow = symbols.trend_up
                trend_text = "Improving"
                trend_color = "green"
            elif trend < -2:
                trend_arrow = symbols.trend_down
                trend_text = "Declining"
                trend_color = "red"
            else:
                trend_arrow = symbols.trend_flat
                trend_text = "Stable"
                trend_color = "yellow"

            if accuracy >= 0.8:
                confidence = "[bold green]HIGH CONFIDENCE[/bold green]"
                conf_color = "green"
            elif accuracy >= 0.6:
                confidence = "[bold yellow]MODERATE[/bold yellow]"
                conf_color = "yellow"
            else:
                confidence = "[bold red]LOW CONFIDENCE[/bold red]"
                conf_color = "red"

            display = _build_fuel_gauge_display(
                sample_idx=i + 1,
                n_samples=n_samples,
                true_val=true_val,
                pred_val=pred_val,
                correct=correct,
                gauge_pct=gauge_pct,
                gauge_width=config.gauge_width,
                trend_arrow=trend_arrow,
                trend_text=trend_text,
                trend_color=trend_color,
                confidence=confidence,
                conf_color=conf_color,
                subject=subject_name,
                symbols=symbols,
                panel_kwargs=panel_kwargs,
            )

            live.update(display)
            time.sleep(config.delay_ms / 1000)

    # Final summary
    final_accuracy = sum(history) / len(history)
    console.print()
    console.print(
        Panel(
            f"[bold]Demo Complete[/bold]\n\n"
            f"Subject: [cyan]{subject_name}[/cyan]\n"
            f"Total Samples: {n_samples}\n"
            f"Final Accuracy: [{'green' if final_accuracy >= 0.7 else 'red'}]"
            f"{final_accuracy:.1%}[/]\n"
            f"Correct: {sum(history)}/{n_samples}",
            title="[green]Summary[/green]",
            **panel_kwargs,
        )
    )


def _build_fuel_gauge_display(
    sample_idx: int,
    n_samples: int,
    true_val: int,
    pred_val: int,
    correct: bool,
    gauge_pct: float,
    gauge_width: int,
    trend_arrow: str,
    trend_text: str,
    trend_color: str,
    confidence: str,
    conf_color: str,
    subject: str,
    symbols: DemoSymbols,
    panel_kwargs: dict[str, object],
) -> Panel:
    """Build the fuel gauge display panel."""
    filled = int(gauge_width * gauge_pct / 100)
    empty = gauge_width - filled

    if gauge_pct >= 80:
        bar_color = "green"
    elif gauge_pct >= 60:
        bar_color = "yellow"
    else:
        bar_color = "red"

    if correct:
        result = f"[green]{symbols.ok}[/green]"
    else:
        result = f"[red]{symbols.fail}[/red]"

    table = Table.grid(padding=(0, 2))
    table.add_column(justify="right", style="dim")
    table.add_column(justify="left")

    table.add_row("Subject", f"[cyan]{subject}[/cyan]")
    table.add_row("Sample", f"{sample_idx}/{n_samples}")
    table.add_row("", "")
    table.add_row("True", f"[blue]{true_val}[/blue]")
    table.add_row("Predicted", f"[magenta]{pred_val}[/magenta]  {result}")
    table.add_row("", "")
    table.add_row("Accuracy", f"{gauge_pct:.1f}%")
    table.add_row("", "")

    gauge_text = Text()
    gauge_text.append("  ")
    gauge_text.append(symbols.bar_filled * filled, style=bar_color)
    gauge_text.append(symbols.bar_empty * empty, style="dim")
    gauge_text.append(f"  {gauge_pct:.1f}%", style="bold")

    content = Table.grid()
    content.add_column()
    content.add_row(table)
    content.add_row("")
    content.add_row(Text("  FUEL GAUGE", style="bold"))
    content.add_row(gauge_text)
    content.add_row("")
    trend_line = Text("  Trend: ")
    trend_line.append(trend_arrow, style=trend_color)
    trend_line.append(f" {trend_text}")
    content.add_row(trend_line)
    content.add_row("")
    content.add_row(Text("  Status: ") + Text.from_markup(confidence))

    return Panel(
        content,
        title="[bold cyan]OpenCR Live Demo[/bold cyan]",
        border_style=conf_color,
        **panel_kwargs,
    )


def run_fuel_gauge_demo(
    threshold: float = 0.7,
    duration_sec: int = 10,
    ascii_only: bool | None = None,
) -> None:
    """
    Run a simple fuel gauge demo with simulated data.

    Args:
        threshold: Confidence threshold for alerts.
        duration_sec: Demo duration in seconds.
    """
    ascii_only = _resolve_ascii(ascii_only)
    symbols = _get_symbols(ascii_only)
    panel_kwargs = _panel_kwargs(ascii_only)

    console.print(
        Panel(
            f"[bold]Fuel Gauge Demo[/bold]\n\n"
            f"Threshold: {threshold:.0%}\n"
            f"Duration: {duration_sec}s\n\n"
            f"[dim]Simulating predictions...[/dim]",
            title="[cyan]OpenCR[/cyan]",
            **panel_kwargs,
        )
    )

    np.random.seed(42)
    n_steps = duration_sec * 2  # 2 updates per second

    history = []

    with Live(console=console, refresh_per_second=4) as live:
        for i in range(n_steps):
            base_accuracy = 0.5 + 0.4 * (i / n_steps)
            correct = np.random.rand() < base_accuracy
            history.append(correct)

            accuracy = sum(history) / len(history)
            gauge_pct = accuracy * 100

            if accuracy >= threshold:
                conf = "[bold green]HIGH[/bold green]"
                border = "green"
            else:
                conf = "[bold red]LOW[/bold red]"
                border = "red"
            width = 30
            filled = int(width * gauge_pct / 100)
            bar = symbols.bar_filled * filled + symbols.bar_empty * (width - filled)

            panel = Panel(
                f"Sample: {i + 1}/{n_steps}\n\n"
                f"Accuracy: {gauge_pct:.1f}%\n"
                f"[{'green' if accuracy >= threshold else 'red'}]{bar}[/]\n\n"
                f"Confidence: {conf}",
                title="[cyan]Fuel Gauge[/cyan]",
                border_style=border,
                **panel_kwargs,
            )

            live.update(panel)
            time.sleep(0.5)

    console.print("\n[green]Demo complete![/green]")
