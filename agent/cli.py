"""
Command-line interface.

Usage:
  python -m agent.cli "Your research topic"
  python -m agent.cli "Topic" --depth deep --max-iter 20
  python -m agent.cli "Topic" --provider ollama --model qwen2.5:7b
  python -m agent.cli "Topic" --trace  # show full trace at end
"""
from __future__ import annotations

import argparse
import json
import sys

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from .agent import AgentConfig, ResearchAgent
from .llm import LLM
from .prompts import ResearchGoal


console = Console()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Free-tier research agent.")
    parser.add_argument("topic", help="What to research")
    parser.add_argument("--depth", choices=["quick", "standard", "deep"], default="standard")
    parser.add_argument("--max-iter", type=int, default=12)
    parser.add_argument("--words", type=int, default=300, help="Target summary word count")
    parser.add_argument("--provider", default=None, help="groq | gemini | ollama | anthropic | openai")
    parser.add_argument("--model", default=None, help="Override model name")
    parser.add_argument("--no-cache", action="store_true", help="Disable page cache")
    parser.add_argument("--trace", action="store_true", help="Print full execution trace")
    parser.add_argument("--json", action="store_true", help="Output JSON only")
    args = parser.parse_args(argv)

    try:
        goal = ResearchGoal(
            topic=args.topic,
            depth=args.depth,
            max_iterations=args.max_iter,
            target_word_count=args.words,
        )
    except Exception as e:
        console.print(f"[red]Invalid input:[/red] {e}")
        return 1

    if not args.json:
        console.print(Panel.fit(
            f"[bold]Topic:[/bold] {goal.topic}\n"
            f"[bold]Depth:[/bold] {goal.depth}  "
            f"[bold]Max iterations:[/bold] {goal.max_iterations}",
            title="Research Agent",
        ))

    llm = LLM(provider=args.provider, model=args.model)
    config = AgentConfig(
        max_iterations=goal.max_iterations,
        use_cache=not args.no_cache,
    )

    if not args.json:
        console.print(f"[dim]Using {llm.provider} / {llm.model}[/dim]\n")

    agent = ResearchAgent(llm=llm, config=config)
    try:
        with console.status("[bold cyan]Researching...", spinner="dots"):
            brief = agent.run(goal)
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user.[/yellow]")
        return 130
    except Exception as e:
        console.print(f"[red]Agent crashed:[/red] {type(e).__name__}: {e}")
        if args.trace:
            import traceback
            traceback.print_exc()
        return 2

    if args.json:
        payload = brief.model_dump()
        # Drop private trace from JSON output
        print(json.dumps(payload, indent=2))
        return 0

    # Pretty print
    console.print(Panel(Markdown(brief.summary), title="Summary"))

    if brief.key_findings:
        console.print("\n[bold]Key findings[/bold]")
        for f in brief.key_findings:
            console.print(f"  • {f}")

    if brief.source_urls:
        console.print("\n[bold]Sources[/bold]")
        for i, u in enumerate(brief.source_urls, 1):
            console.print(f"  [{i}] {u}")

    if brief.open_questions:
        console.print("\n[bold yellow]Open questions[/bold yellow]")
        for q in brief.open_questions:
            console.print(f"  ? {q}")

    # Stats
    stats = Table(show_header=False, box=None, padding=(0, 1))
    stats.add_row("[dim]Iterations:[/dim]", str(brief.iterations_used))
    stats.add_row("[dim]Notes captured:[/dim]", str(brief.notes_count))
    stats.add_row("[dim]Sources cited:[/dim]", str(len(brief.source_urls)))
    stats.add_row("[dim]Cost (USD):[/dim]", f"${brief.total_cost_usd:.4f}")
    console.print()
    console.print(Panel(stats, title="Run stats", expand=False))

    if args.trace:
        trace = getattr(brief, "_trace", None) or []
        console.print("\n[bold]Trace[/bold]")
        for step in trace:
            console.print(f"  [{step['kind']}] {json.dumps({k: v for k, v in step.items() if k != 'kind'}, default=str)[:200]}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
