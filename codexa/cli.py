"""Typer-based CLI entry point for Codexa.

Provides three commands:
  - generate: Walk a directory tree and produce CODEXA.md files.
  - preview:  Print the generated CODEXA.md for a directory without writing.
  - clean:    Remove all CODEXA.md files from a directory tree.

All commands use Rich for styled terminal output and progress reporting.
Errors from the analysis, LLM, or rendering pipeline are caught and
displayed with clear messages before exiting with a non-zero code.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich.text import Text

from codexa import __version__

app = typer.Typer(
    name="codexa",
    help=(
        "Codexa analyzes codebases and generates structured CODEXA.md context "
        "files to help AI coding agents navigate large projects efficiently."
    ),
    add_completion=True,
    no_args_is_help=True,
    rich_markup_mode="rich",
)

console = Console()
err_console = Console(stderr=True)

logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s: %(message)s",
    stream=sys.stderr,
)


# ---------------------------------------------------------------------------
# Version callback
# ---------------------------------------------------------------------------


def version_callback(value: bool) -> None:
    """Print the current Codexa version and exit."""
    if value:
        console.print(f"[bold cyan]codexa[/bold cyan] version [green]{__version__}[/green]")
        raise typer.Exit()


# ---------------------------------------------------------------------------
# Root callback
# ---------------------------------------------------------------------------


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
    verbose: bool = typer.Option(
        False,
        "--verbose",
        help="Enable verbose logging output.",
    ),
) -> None:
    """Codexa — auto-generate structured CODEXA.md context files for your codebase."""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.getLogger("codexa").setLevel(logging.DEBUG)


# ---------------------------------------------------------------------------
# generate command
# ---------------------------------------------------------------------------


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
    mock_llm: bool = typer.Option(
        False,
        "--mock-llm",
        help="Use a deterministic mock LLM (no API calls). Useful for testing.",
        hidden=False,
    ),
) -> None:
    """Analyze a codebase and generate CODEXA.md files per directory.

    Walks the directory tree starting from ROOT, extracts structural metadata,
    calls the configured LLM to produce summaries, and writes a CODEXA.md file
    to each directory that contains source files.

    [bold]Examples:[/bold]

      codexa generate
      codexa generate ./src --depth 3
      codexa generate . --dry-run --mock-llm
      codexa generate . --force --config .codexa.toml
    """
    from codexa.analyzer import AnalyzerError, analyze_tree
    from codexa.config import ConfigError, build_ignore_spec, load_config
    from codexa.llm import LLMError, create_llm_client
    from codexa.models import ModuleSummary
    from codexa.renderer import Renderer, RendererError, build_template_context

    # ------------------------------------------------------------------ #
    # Header banner
    # ------------------------------------------------------------------ #
    console.print()
    console.print(
        Panel.fit(
            Text.from_markup(
                f"[bold cyan]codexa generate[/bold cyan]\n"
                f"Root: [green]{root}[/green]" +
                ("  [yellow]\[dry-run][/yellow]" if dry_run else ""),
            ),
            border_style="cyan",
        )
    )
    console.print()

    # ------------------------------------------------------------------ #
    # Load config
    # ------------------------------------------------------------------ #
    try:
        cfg = load_config(config_path=config, root=root)
    except ConfigError as exc:
        err_console.print(f"[bold red]Config error:[/bold red] {exc}")
        raise typer.Exit(code=1)

    # CLI --depth overrides config
    effective_depth = depth if depth is not None else cfg.max_depth

    # ------------------------------------------------------------------ #
    # Build ignore spec
    # ------------------------------------------------------------------ #
    try:
        ignore_spec = build_ignore_spec(cfg.ignore)
    except ConfigError as exc:
        err_console.print(f"[bold red]Ignore pattern error:[/bold red] {exc}")
        raise typer.Exit(code=1)

    # ------------------------------------------------------------------ #
    # Walk & analyze
    # ------------------------------------------------------------------ #
    console.print("[dim]Scanning directory tree…[/dim]")
    try:
        contexts = analyze_tree(
            root=root,
            ignore_spec=ignore_spec,
            max_depth=effective_depth,
        )
    except AnalyzerError as exc:
        err_console.print(f"[bold red]Analysis error:[/bold red] {exc}")
        raise typer.Exit(code=1)
    except Exception as exc:  # noqa: BLE001
        err_console.print(f"[bold red]Unexpected error during analysis:[/bold red] {exc}")
        raise typer.Exit(code=1)

    if not contexts:
        console.print("[yellow]No directories with Python files found.[/yellow]")
        raise typer.Exit(code=0)

    console.print(
        f"Found [bold]{len(contexts)}[/bold] director"
        f"{'y' if len(contexts) == 1 else 'ies'} to process.\n"
    )

    # ------------------------------------------------------------------ #
    # Build LLM client
    # ------------------------------------------------------------------ #
    use_mock = mock_llm or dry_run

    if not use_mock:
        api_key = cfg.effective_api_key
        if not api_key:
            err_console.print(
                "[bold red]No API key configured.[/bold red]\n"
                "Set [cyan]OPENAI_API_KEY[/cyan] env var or add "
                "[cyan]api_key[/cyan] to [cyan].codexa.toml[/cyan].\n"
                "Use [cyan]--mock-llm[/cyan] to skip real API calls."
            )
            raise typer.Exit(code=1)

    llm_client = create_llm_client(
        api_key=cfg.effective_api_key if not use_mock else "",
        model=cfg.model,
        base_url=cfg.effective_base_url,
        max_tokens=cfg.max_tokens,
        mock=use_mock,
    )

    # ------------------------------------------------------------------ #
    # Renderer
    # ------------------------------------------------------------------ #
    template_path = cfg.template_path
    renderer = Renderer(template_path=template_path)

    # ------------------------------------------------------------------ #
    # Process each directory with progress bar
    # ------------------------------------------------------------------ #
    written = 0
    skipped = 0
    errors = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    ) as progress:
        task = progress.add_task(
            "[cyan]Generating CODEXA.md files…", total=len(contexts)
        )

        for ctx in contexts:
            dir_label = str(ctx.path.relative_to(root)) if ctx.path != root else "."
            progress.update(task, description=f"[cyan]{dir_label}[/cyan]")

            # Summarize via LLM
            try:
                summary = llm_client.summarize_directory(ctx)
                ctx.summary = summary
            except LLMError as exc:
                err_console.print(
                    f"[yellow]LLM error for {dir_label}: {exc}[/yellow]"
                )
                ctx.summary = ModuleSummary.empty()
                errors += 1
            except Exception as exc:  # noqa: BLE001
                err_console.print(
                    f"[yellow]Unexpected LLM error for {dir_label}: {exc}[/yellow]"
                )
                ctx.summary = ModuleSummary.empty()
                errors += 1

            if dry_run:
                context = build_template_context(ctx)
                rendered = renderer.render(context)
                console.print(
                    Panel(
                        rendered[:800] + ("\n[dim]…[truncated][/dim]" if len(rendered) > 800 else ""),
                        title=f"[cyan]Preview: {dir_label}/CODEXA.md[/cyan]",
                        border_style="dim",
                    )
                )
                skipped += 1
                progress.advance(task)
                continue

            # Write CODEXA.md
            try:
                context = build_template_context(ctx)
                did_write = renderer.write(
                    directory=ctx.path,
                    context=context,
                    force=force,
                )
                if did_write:
                    written += 1
                else:
                    skipped += 1
            except RendererError as exc:
                err_console.print(
                    f"[yellow]Render error for {dir_label}: {exc}[/yellow]"
                )
                errors += 1
            except Exception as exc:  # noqa: BLE001
                err_console.print(
                    f"[yellow]Unexpected render error for {dir_label}: {exc}[/yellow]"
                )
                errors += 1

            progress.advance(task)

    # ------------------------------------------------------------------ #
    # Summary
    # ------------------------------------------------------------------ #
    console.print()
    _print_summary_table(
        written=written,
        skipped=skipped,
        errors=errors,
        dry_run=dry_run,
    )

    if errors > 0:
        raise typer.Exit(code=1)
    raise typer.Exit(code=0)


# ---------------------------------------------------------------------------
# preview command
# ---------------------------------------------------------------------------


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
    mock_llm: bool = typer.Option(
        False,
        "--mock-llm",
        help="Use a deterministic mock LLM (no API calls).",
    ),
    no_llm: bool = typer.Option(
        False,
        "--no-llm",
        help="Skip LLM summarization entirely and render with extracted metadata only.",
    ),
) -> None:
    """Preview the CODEXA.md that would be generated for a single directory.

    Runs analysis and (optionally) LLM summarization for DIRECTORY and prints
    the resulting markdown to stdout without writing any files to disk.

    [bold]Examples:[/bold]

      codexa preview
      codexa preview ./src/auth --no-llm
      codexa preview . --mock-llm
    """
    from codexa.analyzer import AnalyzerError, analyze_directory
    from codexa.config import ConfigError, build_ignore_spec, load_config
    from codexa.llm import LLMError, create_llm_client
    from codexa.models import ModuleSummary
    from codexa.renderer import Renderer, RendererError, build_template_context

    # ------------------------------------------------------------------ #
    # Load config
    # ------------------------------------------------------------------ #
    try:
        cfg = load_config(config_path=config, root=directory)
    except ConfigError as exc:
        err_console.print(f"[bold red]Config error:[/bold red] {exc}")
        raise typer.Exit(code=1)

    # ------------------------------------------------------------------ #
    # Build ignore spec
    # ------------------------------------------------------------------ #
    try:
        ignore_spec = build_ignore_spec(cfg.ignore)
    except ConfigError as exc:
        err_console.print(f"[bold red]Ignore pattern error:[/bold red] {exc}")
        raise typer.Exit(code=1)

    # ------------------------------------------------------------------ #
    # Analyze single directory
    # ------------------------------------------------------------------ #
    err_console.print(f"[dim]Analyzing [cyan]{directory}[/cyan]…[/dim]")
    try:
        ctx = analyze_directory(
            directory=directory,
            ignore_spec=ignore_spec,
        )
    except AnalyzerError as exc:
        err_console.print(f"[bold red]Analysis error:[/bold red] {exc}")
        raise typer.Exit(code=1)
    except Exception as exc:  # noqa: BLE001
        err_console.print(f"[bold red]Unexpected analysis error:[/bold red] {exc}")
        raise typer.Exit(code=1)

    # ------------------------------------------------------------------ #
    # LLM summarization
    # ------------------------------------------------------------------ #
    if not no_llm:
        use_mock = mock_llm

        if not use_mock:
            api_key = cfg.effective_api_key
            if not api_key:
                err_console.print(
                    "[bold yellow]Warning:[/bold yellow] No API key configured. "
                    "Use [cyan]--mock-llm[/cyan] or [cyan]--no-llm[/cyan] to skip."
                )
                ctx.summary = ModuleSummary.empty()
            else:
                llm_client = create_llm_client(
                    api_key=api_key,
                    model=cfg.model,
                    base_url=cfg.effective_base_url,
                    max_tokens=cfg.max_tokens,
                    mock=False,
                )
                try:
                    err_console.print("[dim]Calling LLM for summary…[/dim]")
                    ctx.summary = llm_client.summarize_directory(ctx)
                except LLMError as exc:
                    err_console.print(f"[yellow]LLM error:[/yellow] {exc}")
                    ctx.summary = ModuleSummary.empty()
                except Exception as exc:  # noqa: BLE001
                    err_console.print(f"[yellow]Unexpected LLM error:[/yellow] {exc}")
                    ctx.summary = ModuleSummary.empty()
        else:
            llm_client = create_llm_client(mock=True)
            try:
                ctx.summary = llm_client.summarize_directory(ctx)
            except Exception as exc:  # noqa: BLE001
                err_console.print(f"[yellow]Mock LLM error:[/yellow] {exc}")
                ctx.summary = ModuleSummary.empty()
    else:
        ctx.summary = ModuleSummary.empty()

    # ------------------------------------------------------------------ #
    # Render and print
    # ------------------------------------------------------------------ #
    template_path = cfg.template_path
    renderer = Renderer(template_path=template_path)

    try:
        context = build_template_context(ctx)
        rendered = renderer.render(context)
    except RendererError as exc:
        err_console.print(f"[bold red]Render error:[/bold red] {exc}")
        raise typer.Exit(code=1)
    except Exception as exc:  # noqa: BLE001
        err_console.print(f"[bold red]Unexpected render error:[/bold red] {exc}")
        raise typer.Exit(code=1)

    # Print to stdout (not using Console to allow piping)
    print(rendered)
    raise typer.Exit(code=0)


# ---------------------------------------------------------------------------
# clean command
# ---------------------------------------------------------------------------


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
    Prompts for confirmation unless [cyan]--yes[/cyan] is passed.

    [bold]Examples:[/bold]

      codexa clean
      codexa clean ./src --yes
    """
    codexa_files = list(root.rglob("CODEXA.md"))

    if not codexa_files:
        console.print(
            f"[green]No CODEXA.md files found under [cyan]{root}[/cyan].[/green]"
        )
        raise typer.Exit(code=0)

    console.print(
        f"Found [bold]{len(codexa_files)}[/bold] CODEXA.md file(s) under "
        f"[cyan]{root}[/cyan]:\n"
    )
    for f in sorted(codexa_files):
        try:
            rel = f.relative_to(root)
        except ValueError:
            rel = f
        console.print(f"  [dim]{rel}[/dim]")

    console.print()

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
            err_console.print(f"[red]Failed to delete {f}: {exc}[/red]")
            errors += 1

    if errors:
        console.print(
            f"[green]Deleted {deleted} file(s).[/green] "
            f"[red]{errors} file(s) could not be deleted.[/red]"
        )
        raise typer.Exit(code=1)
    else:
        console.print(f"[green]Deleted {deleted} file(s) successfully.[/green]")
        raise typer.Exit(code=0)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _print_summary_table(
    written: int,
    skipped: int,
    errors: int,
    dry_run: bool,
) -> None:
    """Print a Rich table summarising the generate run results.

    Args:
        written: Number of CODEXA.md files written (or previewed in dry-run).
        skipped: Number of directories skipped due to unchanged hash.
        errors: Number of errors encountered.
        dry_run: Whether the run was in dry-run mode.
    """
    table = Table(title="Run Summary", border_style="cyan", show_header=True)
    table.add_column("Metric", style="bold")
    table.add_column("Count", justify="right")

    if dry_run:
        table.add_row("Previewed", f"[cyan]{written + skipped}[/cyan]")
    else:
        table.add_row(
            "Written",
            f"[green]{written}[/green]" if written else f"[dim]{written}[/dim]",
        )
        table.add_row(
            "Skipped (unchanged)",
            f"[dim]{skipped}[/dim]",
        )

    table.add_row(
        "Errors",
        f"[red]{errors}[/red]" if errors else f"[dim]{errors}[/dim]",
    )

    console.print(table)

    if errors == 0 and not dry_run:
        console.print("\n[bold green]:white_check_mark: Done![/bold green]")
    elif dry_run:
        console.print("\n[bold yellow]:information: Dry-run complete — no files written.[/bold yellow]")
    else:
        console.print(
            "\n[bold yellow]:warning: Completed with errors. "
            "Check the output above for details.[/bold yellow]"
        )


if __name__ == "__main__":
    app()
