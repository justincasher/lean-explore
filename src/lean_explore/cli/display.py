# src/lean_explore/cli/display.py

"""Display and formatting utilities for CLI output."""

import textwrap

from rich.console import Console
from rich.panel import Panel

from lean_explore.models import SearchResponse


def _wrap_line(line: str, width: int) -> list[str]:
    """Wraps a single line of text to the specified width.

    Args:
        line: The line to wrap.
        width: The target width for wrapped text.

    Returns:
        List of wrapped line segments, each padded to the target width.
    """
    empty_line = " " * width
    if not line.strip():
        return [empty_line]

    segments = textwrap.wrap(
        line,
        width=width,
        replace_whitespace=True,
        drop_whitespace=True,
        break_long_words=True,
        break_on_hyphens=True,
    )
    return [segment.ljust(width) for segment in segments] if segments else [empty_line]


def _format_text_for_panel(text_content: str | None, width: int = 80) -> str:
    """Wraps text and pads lines to ensure fixed content width for a Panel.

    Splits text into paragraphs (by double newline), wraps each line within
    paragraphs, and pads all lines to the target width.

    Args:
        text_content: The text to format.
        width: The target width for wrapped text.

    Returns:
        Formatted text with proper line wrapping and padding.
    """
    empty_line = " " * width
    if not text_content:
        return empty_line

    output_lines = []
    paragraphs = text_content.split("\n\n")

    for index, paragraph in enumerate(paragraphs):
        # Handle empty paragraphs (preserve blank lines between paragraphs)
        if not paragraph.strip():
            if index < len(paragraphs) - 1:
                output_lines.append(empty_line)
            continue

        # Process each line within the paragraph
        for line in paragraph.splitlines():
            output_lines.extend(_wrap_line(line, width))

        # Add separator between paragraphs (except after the last one)
        if index < len(paragraphs) - 1:
            output_lines.append(empty_line)

    return "\n".join(output_lines) if output_lines else empty_line


def display_search_results(
    response: SearchResponse,
    display_limit: int = 5,
    console: Console | None = None,
) -> None:
    """Displays search results using fixed-width Panels for each item.

    Args:
        response: The search response containing results to display.
        display_limit: Maximum number of results to show.
        console: The Rich console to use for output. If None, creates a new one.
    """
    if console is None:
        console = Console()

    console.print(
        Panel(
            f"[bold cyan]Search Query:[/bold cyan] {response.query}",
            expand=False,
            border_style="dim",
        )
    )

    num_results_to_show = min(len(response.results), display_limit)
    time_info = (
        f"Time: {response.processing_time_ms}ms" if response.processing_time_ms else ""
    )
    console.print(
        f"Showing {num_results_to_show} of {response.count} results. {time_info}"
    )

    if not response.results:
        console.print("[yellow]No results found.[/yellow]")
        return

    console.print("")

    for i, item in enumerate(response.results):
        if i >= display_limit:
            break

        console.rule(f"[bold]Result {i + 1}[/bold]", style="dim")
        console.print(f"[bold cyan]ID:[/bold cyan] [dim]{item.id}[/dim]")
        console.print(f"[bold cyan]Name:[/bold cyan] {item.name}")
        console.print(f"[bold cyan]Module:[/bold cyan] [green]{item.module}[/green]")
        source_formatted = (
            f"[bold cyan]Source:[/bold cyan] "
            f"[link={item.source_link}]{item.source_link}[/link]"
        )
        console.print(source_formatted)

        if item.source_text:
            formatted_code = _format_text_for_panel(item.source_text)
            console.print(
                Panel(
                    formatted_code,
                    title="[bold green]Code[/bold green]",
                    border_style="green",
                    expand=False,
                    padding=(0, 1),
                )
            )

        if item.docstring:
            formatted_doc = _format_text_for_panel(item.docstring)
            console.print(
                Panel(
                    formatted_doc,
                    title="[bold blue]Docstring[/bold blue]",
                    border_style="blue",
                    expand=False,
                    padding=(0, 1),
                )
            )

        if item.informalization:
            formatted_informal = _format_text_for_panel(item.informalization)
            console.print(
                Panel(
                    formatted_informal,
                    title="[bold magenta]Informalization[/bold magenta]",
                    border_style="magenta",
                    expand=False,
                    padding=(0, 1),
                )
            )

        if i < num_results_to_show - 1:
            console.print("")

    console.rule(style="dim")
    if len(response.results) > num_results_to_show:
        console.print(
            f"...and {len(response.results) - num_results_to_show} more results "
            "received but not shown due to limit."
        )
