"""Typer CLI entry point and REPL."""

import logging
import sys
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.panel import Panel

from app_strategist.config import get_api_key
from app_strategist.models.session import AnalysisSession
from app_strategist.rendering.console import ConsoleRenderer
from app_strategist.services.analysis import AnalysisService
from app_strategist.utils import JSONExtractionError

app = typer.Typer(
    name="app-strategist",
    help="Job search assistance - compare resume/cover letter against job descriptions",
)
console = Console()

REPL_SYSTEM_PROMPT = """You are a helpful job search assistant. The user has received an evaluation comparing their resume/cover letter to a job description.

Use the full context below to answer their follow-up questions. Never fabricate qualifications, experience, or achievements. Only suggest improvements based on what is present in the documents.

Context:
{context}
"""


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )


@app.command()
def analyze(
    resume: Annotated[
        Path,
        typer.Option("--resume", "-r", help="Path to resume (.txt or .md)"),
    ],
    job: Annotated[
        Path,
        typer.Option("--job", "-j", help="Path to job description (.txt or .md)"),
    ],
    cover_letter: Annotated[
        Optional[Path],
        typer.Option("--cover-letter", "-c", help="Path to cover letter (.txt or .md)"),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable debug logging"),
    ] = False,
) -> None:
    """Compare resume/cover letter against job description and get actionable feedback."""
    _setup_logging(verbose)

    try:
        get_api_key()  # Fail fast with clear error if missing
        service = AnalysisService()
        session = service.analyze(
            resume_path=resume,
            job_path=job,
            cover_letter_path=cover_letter,
        )

        renderer = ConsoleRenderer()
        renderer.render(session)

        # REPL loop
        _run_repl(session)
    except FileNotFoundError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
    except PermissionError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
    except JSONExtractionError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
    except Exception as e:
        logging.getLogger(__name__).exception("Unexpected error")
        console.print(f"[red]Unexpected error:[/red] {e}")
        raise typer.Exit(1)


def _run_repl(session: AnalysisSession) -> None:
    """Run follow-up question REPL with full session context."""
    from app_strategist.llm import AnthropicProvider

    llm = AnthropicProvider()
    context = session.to_context_string()
    system = REPL_SYSTEM_PROMPT.format(context=context)

    messages: list[dict] = []

    console.print()
    console.print(Panel("[bold]Follow-up[/bold] Ask questions about the evaluation (or 'quit' to exit)", style="dim"))

    while True:
        try:
            question = console.input("\n[bold cyan]You:[/bold cyan] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Exiting.[/dim]")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            console.print("[dim]Goodbye.[/dim]")
            break

        messages.append({"role": "user", "content": question})

        try:
            response = llm.complete(system, messages)
            console.print(f"\n[bold green]Assistant:[/bold green]\n{response}")
            messages.append({"role": "assistant", "content": response})
        except Exception as e:
            console.print(f"[red]Error:[/red] {e}")
            messages.pop()  # Remove user message so we can retry (only reached if append not yet run)


if __name__ == "__main__":
    app()
