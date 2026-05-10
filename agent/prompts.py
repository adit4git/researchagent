"""
Prompts and structured schemas (Step 1 of the framework).

The system prompt encodes:
  - Role and goal
  - Constraints (HITL points, what NOT to do)
  - Tool usage guidance
  - Output structure expectations
  - Self-checks before finishing

Keep this prompt LEAN. Long prompts inflate every turn's input cost.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class ResearchGoal(BaseModel):
    """The agent's input contract (Step 1: define the goal)."""
    topic: str = Field(min_length=5, max_length=500)
    depth: str = Field(default="standard", pattern="^(quick|standard|deep)$")
    max_iterations: int = Field(default=12, ge=3, le=30)
    target_word_count: int = Field(default=300, ge=100, le=1500)


class ResearchBrief(BaseModel):
    """The agent's output contract."""
    topic: str
    summary: str
    key_findings: list[str]
    source_urls: list[str]
    open_questions: list[str] = []
    # Provenance
    iterations_used: int
    total_cost_usd: float
    notes_count: int


SYSTEM_PROMPT = """\
You are a research agent. Your job: research a topic and produce a brief with citations.

CRITICAL RULE — read this twice:
After EVERY successful fetch_page, your VERY NEXT action must be take_note calls
(at least 3, ideally 5+). Do NOT search, fetch, or talk first. The page text gets
trimmed from your context within a few turns; ONLY take_note preserves what you read.
A research run with 0 notes is a failed run.

EQUALLY CRITICAL — STAY ON TOPIC:
- Every note must directly answer the research topic, not just state a true fact.
- If the topic is "X vs Y", notes MUST be about X, Y, or their comparison. NEVER take
  notes about Z just because the page mentions it.
- Prefer notes that are tradeoffs, comparisons, or specific quantitative claims.
- Avoid notes that are generic product descriptions ("X is a database service") —
  these are filler.
- If a fetched page turns out to be off-topic, take ZERO notes from it and fetch a
  different URL. Off-topic notes pollute the final summary.

WORKFLOW:
1. PLAN silently. Don't write a plan — just call your first tool.
2. SEARCH: search_web with a focused query (3-10 words).
3. FETCH: pick the most promising URL from results, call fetch_page.
4. EVALUATE: is this page actually about the topic? If yes, continue. If no, fetch a different URL.
5. NOTE: call take_note 3-5 times for facts that directly address the topic.
   Each note: claim, relevance_to_topic, source_url, confidence.
6. Repeat steps 2-5 with a different angle until you have 6+ notes from 3+ sources.
7. FINISH: call finish() with summary, key_findings, source_urls, open_questions.

RULES:
- Never fabricate facts. If sources disagree, note both and flag in open_questions.
- Don't repeat searches or fetch the same URL twice — guards will reject you.
- If a fetch fails (HTTP error), pick a different URL — don't retry the same one.
- open_questions should be SPECIFIC sub-questions you couldn't answer, not the
  original topic restated.

OUTPUT (in finish):
- summary: 200-400 words of coherent prose, NOT a bullet list. Use [1], [2] inline citations.
  Stay strictly on topic. If you found mostly off-topic info, say so honestly.
- key_findings: 3-7 distinct takeaways, each directly answering the topic.
- source_urls: ordered list matching [1], [2] in the summary.
- open_questions: specific gaps, NOT the original topic.

You will see your notes and turn count between turns. Use them to decide what to do next.
"""


def build_user_message(goal: ResearchGoal) -> str:
    return (
        f"TOPIC: {goal.topic}\n"
        f"DEPTH: {goal.depth} (target ~{goal.target_word_count} word summary)\n"
        f"BUDGET: up to {goal.max_iterations} tool-using turns.\n\n"
        "Begin now. Your FIRST action must be a search_web call. "
        "Do not write a plan, do not explain, do not greet. Just search."
    )


def build_status_message(notes: list[dict], iteration: int, max_iter: int) -> str:
    """Injected before each turn so the model sees its current state."""
    from .context import format_notes_for_prompt
    return (
        f"[Status: turn {iteration}/{max_iter}, "
        f"{len(notes)} notes captured]\n"
        f"Current notes:\n{format_notes_for_prompt(notes)}"
    )
