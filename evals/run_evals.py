"""
Eval harness (Step 7 of the framework: test and evals).

Runs the golden dataset and reports:
  - Pass/fail per case (must-mention coverage, min source count)
  - Aggregate metrics: pass rate, avg iterations, avg cost, avg notes
  - Cost tracking — most teams skip this and get billed surprises
  - Per-case JSON for diff between runs

This is intentionally simple. Real eval frameworks (Braintrust,
Promptfoo, LangSmith) add:
  - LLM-as-judge for fluency/factuality
  - Variance over multiple runs
  - Statistical significance vs baseline
  - A/B harness for prompt changes

Start with this and graduate when you actually need the extras.

Usage:
  python -m evals.run_evals
  python -m evals.run_evals --provider ollama
  python -m evals.run_evals --limit 3       # quick smoke test
  python -m evals.run_evals --output run.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from agent.agent import AgentConfig, ResearchAgent
from agent.llm import LLM
from agent.prompts import ResearchGoal


GOLDEN_PATH = Path(__file__).parent / "golden.jsonl"


@dataclass
class CaseResult:
    id: str
    topic: str
    passed: bool
    score: float                # 0.0-1.0
    must_mention_hit: list[str]
    must_mention_miss: list[str]
    n_sources: int
    n_notes: int
    iterations: int
    cost_usd: float
    duration_s: float
    summary_preview: str
    failure_reason: str = ""


def load_golden(limit: int | None = None) -> list[dict]:
    cases = []
    with GOLDEN_PATH.open() as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases[:limit] if limit else cases


def score_case(brief, case: dict) -> CaseResult:
    """Apply the case's must-mention and min-source criteria."""
    summary_lower = (brief.summary + " " + " ".join(brief.key_findings)).lower()
    must = case.get("must_mention", [])
    hit = [m for m in must if m.lower() in summary_lower]
    miss = [m for m in must if m.lower() not in summary_lower]

    n_sources = len(brief.source_urls)
    min_sources = case.get("min_sources", 2)

    coverage = len(hit) / len(must) if must else 1.0
    source_ok = n_sources >= min_sources

    # Must hit at least 60% of keywords AND meet source minimum
    passed = coverage >= 0.6 and source_ok

    failure = ""
    if not passed:
        reasons = []
        if coverage < 0.6:
            reasons.append(f"missed keywords: {miss}")
        if not source_ok:
            reasons.append(f"only {n_sources} sources (need {min_sources})")
        failure = "; ".join(reasons)

    return CaseResult(
        id=case["id"],
        topic=case["topic"],
        passed=passed,
        score=coverage * (1.0 if source_ok else 0.5),
        must_mention_hit=hit,
        must_mention_miss=miss,
        n_sources=n_sources,
        n_notes=brief.notes_count,
        iterations=brief.iterations_used,
        cost_usd=brief.total_cost_usd,
        duration_s=0.0,  # filled by caller
        summary_preview=brief.summary[:200],
        failure_reason=failure,
    )


def run_eval(provider: str | None, model: str | None, limit: int | None, max_iter: int) -> list[CaseResult]:
    cases = load_golden(limit)
    print(f"Running {len(cases)} eval cases against {provider or 'default'} provider...\n")

    llm = LLM(provider=provider, model=model)
    config = AgentConfig(max_iterations=max_iter)

    results: list[CaseResult] = []
    for i, case in enumerate(cases, 1):
        print(f"[{i}/{len(cases)}] {case['id']}: {case['topic'][:60]}...")
        agent = ResearchAgent(llm=llm, config=config)
        goal = ResearchGoal(topic=case["topic"], max_iterations=max_iter)

        t0 = time.time()
        try:
            brief = agent.run(goal)
            result = score_case(brief, case)
            result.duration_s = time.time() - t0
        except Exception as e:
            result = CaseResult(
                id=case["id"],
                topic=case["topic"],
                passed=False,
                score=0.0,
                must_mention_hit=[],
                must_mention_miss=case.get("must_mention", []),
                n_sources=0,
                n_notes=0,
                iterations=0,
                cost_usd=0.0,
                duration_s=time.time() - t0,
                summary_preview="",
                failure_reason=f"crash: {type(e).__name__}: {e}",
            )

        marker = "✓" if result.passed else "✗"
        print(f"    {marker} score={result.score:.2f} src={result.n_sources} "
              f"notes={result.n_notes} iter={result.iterations} "
              f"cost=${result.cost_usd:.4f} t={result.duration_s:.1f}s")
        if result.failure_reason:
            print(f"    reason: {result.failure_reason}")
        results.append(result)

    return results


def report(results: list[CaseResult]) -> None:
    n = len(results)
    if n == 0:
        print("No results.")
        return
    n_pass = sum(1 for r in results if r.passed)
    avg_score = sum(r.score for r in results) / n
    avg_iter = sum(r.iterations for r in results) / n
    avg_cost = sum(r.cost_usd for r in results) / n
    avg_dur = sum(r.duration_s for r in results) / n
    total_cost = sum(r.cost_usd for r in results)

    print("\n" + "=" * 60)
    print(f"PASS RATE:    {n_pass}/{n} ({100*n_pass/n:.0f}%)")
    print(f"AVG SCORE:    {avg_score:.2f}")
    print(f"AVG ITER:     {avg_iter:.1f}")
    print(f"AVG DURATION: {avg_dur:.1f}s")
    print(f"AVG COST:     ${avg_cost:.4f}")
    print(f"TOTAL COST:   ${total_cost:.4f}")
    print("=" * 60)

    failures = [r for r in results if not r.passed]
    if failures:
        print("\nFailures:")
        for r in failures:
            print(f"  - {r.id}: {r.failure_reason}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--limit", type=int, default=None, help="Run first N cases only")
    parser.add_argument("--max-iter", type=int, default=10)
    parser.add_argument("--output", default=None, help="Write JSON results to this file")
    args = parser.parse_args(argv)

    results = run_eval(args.provider, args.model, args.limit, args.max_iter)
    report(results)

    if args.output:
        Path(args.output).write_text(json.dumps([asdict(r) for r in results], indent=2))
        print(f"\nWrote results to {args.output}")

    # Exit code reflects pass rate - useful in CI
    pass_rate = sum(1 for r in results if r.passed) / len(results) if results else 0
    return 0 if pass_rate >= 0.7 else 1


if __name__ == "__main__":
    sys.exit(main())
