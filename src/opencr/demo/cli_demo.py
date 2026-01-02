"""
CLI Demo - Live Terminal Visualization.

Simulates streaming predictions with fuel gauge display.
"""

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from opencr.logging import get_logger

logger = get_logger(__name__)
console = Console()


@dataclass
class DemoConfig:
    """Configuration for demo display."""

    delay_ms: int = 500  # Delay between updates in milliseconds
    gauge_width: int = 40  # Width of fuel gauge bar
    window_size: int = 5  # Moving average window for trend


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

    run_dir = Path(run_dir)

    # Load results
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

    # Find matching fold/subject
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

    # Load predictions
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
        )
    )
    console.print()

    time.sleep(1)

    # Streaming simulation
    history = []
    trend_history = []

    with Live(console=console, refresh_per_second=4) as live:
        for i in range(n_samples):
            true_val = int(y_true[i])
            pred_val = int(y_pred[i])
            correct = true_val == pred_val

            history.append(correct)

            # Compute metrics
            accuracy = sum(history) / len(history)
            gauge_pct = accuracy * 100

            # Trend (moving average of recent accuracy change)
            trend_history.append(gauge_pct)
            if len(trend_history) > config.window_size:
                trend_history.pop(0)

            if len(trend_history) >= 2:
                trend = trend_history[-1] - trend_history[0]
            else:
                trend = 0

            # Determine trend arrow
            if trend > 2:
                trend_arrow = "[green]↑[/green]"
                trend_text = "Improving"
            elif trend < -2:
                trend_arrow = "[red]↓[/red]"
                trend_text = "Declining"
            else:
                trend_arrow = "[yellow]→[/yellow]"
                trend_text = "Stable"

            # Confidence state based on accuracy
            if accuracy >= 0.8:
                confidence = "[bold green]HIGH CONFIDENCE[/bold green]"
                conf_color = "green"
            elif accuracy >= 0.6:
                confidence = "[bold yellow]MODERATE[/bold yellow]"
                conf_color = "yellow"
            else:
                confidence = "[bold red]LOW CONFIDENCE[/bold red]"
                conf_color = "red"

            # Build display
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
                confidence=confidence,
                conf_color=conf_color,
                subject=subject_name,
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
    confidence: str,
    conf_color: str,
    subject: str,
) -> Panel:
    """Build the fuel gauge display panel."""
    # Gauge bar
    filled = int(gauge_width * gauge_pct / 100)
    empty = gauge_width - filled

    if gauge_pct >= 80:
        bar_color = "green"
    elif gauge_pct >= 60:
        bar_color = "yellow"
    else:
        bar_color = "red"

    # Result indicator
    if correct:
        result = "[green]✓[/green]"
    else:
        result = "[red]✗[/red]"

    # Build table
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

    # Gauge visualization
    gauge_text = Text()
    gauge_text.append("  ")
    gauge_text.append("█" * filled, style=bar_color)
    gauge_text.append("░" * empty, style="dim")
    gauge_text.append(f"  {gauge_pct:.1f}%", style="bold")

    content = Table.grid()
    content.add_column()
    content.add_row(table)
    content.add_row("")
    content.add_row(Text("  FUEL GAUGE", style="bold"))
    content.add_row(gauge_text)
    content.add_row("")
    content.add_row(Text(f"  Trend: {trend_arrow} {trend_text}"))
    content.add_row("")
    content.add_row(Text("  Status: ") + Text.from_markup(confidence))

    return Panel(
        content,
        title="[bold cyan]OpenCR Live Demo[/bold cyan]",
        border_style=conf_color,
    )


def run_fuel_gauge_demo(
    threshold: float = 0.7,
    duration_sec: int = 10,
) -> None:
    """
    Run a simple fuel gauge demo with simulated data.

    Args:
        threshold: Confidence threshold for alerts.
        duration_sec: Demo duration in seconds.
    """
    console.print(
        Panel(
            f"[bold]Fuel Gauge Demo[/bold]\n\n"
            f"Threshold: {threshold:.0%}\n"
            f"Duration: {duration_sec}s\n\n"
            f"[dim]Simulating predictions...[/dim]",
            title="[cyan]OpenCR[/cyan]",
        )
    )

    np.random.seed(42)
    n_steps = duration_sec * 2  # 2 updates per second

    history = []

    with Live(console=console, refresh_per_second=4) as live:
        for i in range(n_steps):
            # Simulate prediction (accuracy improves over time)
            base_accuracy = 0.5 + 0.4 * (i / n_steps)
            correct = np.random.rand() < base_accuracy
            history.append(correct)

            accuracy = sum(history) / len(history)
            gauge_pct = accuracy * 100

            # Confidence
            if accuracy >= threshold:
                conf = "[bold green]HIGH[/bold green]"
                border = "green"
            else:
                conf = "[bold red]LOW[/bold red]"
                border = "red"

            # Bar
            width = 30
            filled = int(width * gauge_pct / 100)
            bar = "█" * filled + "░" * (width - filled)

            panel = Panel(
                f"Sample: {i + 1}/{n_steps}\n\n"
                f"Accuracy: {gauge_pct:.1f}%\n"
                f"[{'green' if accuracy >= threshold else 'red'}]{bar}[/]\n\n"
                f"Confidence: {conf}",
                title="[cyan]Fuel Gauge[/cyan]",
                border_style=border,
            )

            live.update(panel)
            time.sleep(0.5)

    console.print("\n[green]Demo complete![/green]")
