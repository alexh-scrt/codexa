"""Typer-based CLI entry point for Codexa.

Provides three commands:
  - generate: Walk a directory tree and produce CODEXA.md files.
  - preview:  Print the generated CODEXA.md for a directory without writing.
  - clean:    Remove all CODEXA.md files from a directory tree.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from codexa import __version__

app = typer.Typer(
    name="codexa",
    help=(
        "Codexa analyzes codebases and generates structured CODEXA.md context "
        "files to help AI coding agents navigate large projects efficiently."
    ),
    add_completion=True,
    no_args_is_help=True,
)

console = Console()
err_console = Console(stderr=True, style="bold red")


def version_callback(value: bool) -> None:
    """Print the current Codexa version and exit."""
    if value:
        console.print(f"codexa version {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Optional[bool] = typer.Option(
        None,
        "--version",
        "-v",
        help="Print the version and exit.",
        callback=version_callback,
        is_eager=True,
    ),
) -> None:
    """Codexa CLI root callback."""


@app.command("generate")
def generate(
    root: Path = typer.Argument(
        Path("."),
        help="Root directory to analyze (defaults to current directory).",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        resolve_path=True,
    ),
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to a .codexa.toml config file (auto-detected if omitted).",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Print what would be generated without writing any files.",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Regenerate all CODEXA.md files even if content hashes are unchanged.",
    ),
    depth: Optional[int] = typer.Option(
        None,
        "--depth",
        "-d",
        help="Maximum directory recursion depth (overrides config).",
        min=0,
    ),
) -> None:
    """Analyze a codebase and generate CODEXA.md files per directory.

    Walks the directory tree starting from ROOT, extracts structural metadata,
    calls the configured LLM to produce summaries, and writes a CODEXA.md file
    to each directory that contains source files.
    """
    console.print(
        f"[bold green]codexa generate[/bold green] — analyzing [cyan]{root}[/cyan]"
    )
    if dry_run:
        console.print("[yellow]Dry-run mode: no files will be written.[/yellow]")

    # Placeholder until later phases wire up the full pipeline.
    console.print(
        "[dim]Pipeline not yet wired — implement in later phases.[/dim]"
    )
    raise typer.Exit(code=0)


@app.command("preview")
def preview(
    directory: Path = typer.Argument(
        Path("."),
        help="Directory whose CODEXA.md to preview (defaults to current directory).",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        resolve_path=True,
    ),
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to a .codexa.toml config file (auto-detected if omitted).",
    ),
) -> None:
    """Preview the CODEXA.md that would be generated for a single directory.

    Runs analysis and LLM summarization for DIRECTORY and prints the resulting
    markdown to stdout without writing any files to disk.
    """
    console.print(
        f"[bold blue]codexa preview[/bold blue] — "
        f"previewing [cyan]{directory}[/cyan]"
    )
    console.print(
        "[dim]Pipeline not yet wired — implement in later phases.[/dim]"
    )
    raise typer.Exit(code=0)


@app.command("clean")
def clean(
    root: Path = typer.Argument(
        Path("."),
        help="Root directory from which to remove CODEXA.md files.",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        resolve_path=True,
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Skip the confirmation prompt and delete immediately.",
    ),
) -> None:
    """Remove all CODEXA.md files under a directory tree.

    Recursively searches ROOT for files named CODEXA.md and deletes them.
    Prompts for confirmation unless --yes is passed.
    """
    codexa_files = list(root.rglob("CODEXA.md"))

    if not codexa_files:
        console.print(
            f"[green]No CODEXA.md files found under [cyan]{root}[/cyan].[/green]"
        )
        raise typer.Exit(code=0)

    console.print(
        f"Found [bold]{len(codexa_files)}[/bold] CODEXA.md file(s) under "
        f"[cyan]{root}[/cyan]:"
    )
    for f in codexa_files:
        console.print(f"  [dim]{f}[/dim]")

    if not yes:
        confirmed = typer.confirm(
            f"Delete all {len(codexa_files)} CODEXA.md file(s)?"
        )
        if not confirmed:
            console.print("[yellow]Aborted — no files deleted.[/yellow]")
            raise typer.Exit(code=0)

    deleted = 0
    errors = 0
    for f in codexa_files:
        try:
            f.unlink()
            deleted += 1
        except OSError as exc:
            err_console.print(f"Failed to delete {f}: {exc}")
            errors += 1

    console.print(
        f"[green]Deleted {deleted} file(s).[/green]"
        + (f" [red]{errors} error(s).[/red]" if errors else "")
    )
    if errors:
        raise typer.Exit(code=1)
    raise typer.Exit(code=0)


if __name__ == "__main__":
    app()
