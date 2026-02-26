"""Rich-based console renderer for evaluation output."""

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from app_strategist.models.evaluation import CandidateEvaluation, EmployerEvaluation
from app_strategist.models.session import AnalysisSession

console = Console()


def _render_score_components(components: list, title: str) -> None:
    """Render score components as a table."""
    table = Table(title=title, show_header=True, header_style="bold")
    table.add_column("Component", style="cyan")
    table.add_column("Weight", justify="right")
    table.add_column("Score", justify="right")
    table.add_column("Explanation")
    for c in components:
        table.add_row(
            c.name,
            f"{c.weight * 100:.0f}%",
            f"{c.score:.0f}",
            c.explanation[:80] + "..." if len(c.explanation) > 80 else c.explanation,
        )
    console.print(table)


def render_employer_evaluation(eval_: EmployerEvaluation) -> None:
    """Pretty-print employer-side evaluation."""
    console.print()
    console.print(Panel("[bold]Employer-Side Evaluation[/bold] (Hiring Manager / Recruiter Perspective)", style="blue"))
    console.print()

    # Fit score
    console.print(f"[bold]Fit Score:[/bold] [green]{eval_.fit_score.value:.0f}/100[/green]")
    console.print(f"[dim]{eval_.score_rationale}[/dim]")
    console.print()

    _render_score_components(eval_.fit_score.components, "Score Breakdown")

    if eval_.strengths:
        console.print()
        console.print("[bold green]Strengths[/bold green] (where your experience aligns):")
        for s in eval_.strengths:
            console.print(f"  • {s}")

    if eval_.gaps:
        console.print()
        console.print("[bold yellow]Gaps / Missing Requirements[/bold yellow]:")
        for g in eval_.gaps:
            console.print(f"  • {g}")

    if eval_.suggested_improvements:
        console.print()
        console.print("[bold]Suggested Improvements[/bold]:")
        for i in eval_.suggested_improvements:
            console.print(f"  • {i}")

    if eval_.wording_suggestions:
        console.print()
        console.print("[bold]Wording Suggestions[/bold] (no fabrication):")
        for ws in eval_.wording_suggestions:
            console.print(f"  [dim]Current:[/dim] {ws.current}")
            console.print(f"  [dim]Suggested:[/dim] {ws.suggested}")
            console.print(f"  [dim]Rationale:[/dim] {ws.rationale}")
            console.print(f"  [dim]Status:[/dim] {ws.status}")
            console.print()


def render_candidate_evaluation(eval_: CandidateEvaluation) -> None:
    """Pretty-print candidate-side evaluation."""
    console.print()
    console.print(Panel("[bold]Candidate-Side Evaluation[/bold] (Worker Perspective)", style="green"))
    console.print()

    console.print(f"[bold]Worker Fit Score:[/bold] [green]{eval_.worker_fit_score.value:.0f}/100[/green]")
    console.print(f"[dim]{eval_.score_rationale}[/dim]")
    console.print()

    _render_score_components(eval_.worker_fit_score.components, "Score Breakdown")

    if eval_.positive_alignments:
        console.print()
        console.print("[bold green]Positive Alignments[/bold green]:")
        for a in eval_.positive_alignments:
            console.print(f"  • {a}")

    if eval_.concerns:
        console.print()
        console.print("[bold yellow]Potential Concerns / Red Flags[/bold yellow]:")
        for c in eval_.concerns:
            console.print(f"  • {c}")

    if eval_.questions_to_ask:
        console.print()
        console.print("[bold]Questions to Ask Recruiter/Hiring Manager[/bold]:")
        for q in eval_.questions_to_ask:
            console.print(f"  • {q}")


class ConsoleRenderer:
    """Renders analysis session to console."""

    def render(self, session: AnalysisSession) -> None:
        """Render full analysis to console."""
        render_employer_evaluation(session.employer_eval)
        render_candidate_evaluation(session.candidate_eval)
