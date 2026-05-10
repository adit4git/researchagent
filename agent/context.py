"""
Context management (Step 6 of the framework).

Search results and fetched pages eat context fast. After ~3-4 fetches, a
naive agent's context window is mostly stale page text. We need to:

  1. Count tokens accurately (using tiktoken as a model-agnostic estimator)
  2. Trim old tool outputs once they're no longer relevant
  3. Summarize when above a threshold
  4. Always preserve: system prompt, original user request, recent turns, all notes

Strategy: rolling window with note preservation. Notes live OUTSIDE the
chat history (in WorkingMemory.notes) so they survive trimming.
"""
from __future__ import annotations

import json

# Token counting strategy:
#   1. Try tiktoken's cl100k_base (used by GPT-4 family, very accurate)
#   2. Fall back to char-based heuristic if tiktoken can't load its data
#      (e.g. offline sandbox, locked-down network)
#
# For Llama/Claude/Gemini the count is approximate but good enough for
# context budgeting. If you need exact counts, use the model's native
# tokenizer.
_ENC = None
try:
    import tiktoken
    _ENC = tiktoken.get_encoding("cl100k_base")
except Exception:
    _ENC = None


def count_tokens(text: str) -> int:
    if not text:
        return 0
    if _ENC is not None:
        return len(_ENC.encode(text))
    # Fallback: ~4 chars per token is a reasonable English-text average
    return max(1, len(text) // 4)


def count_message_tokens(messages: list[dict]) -> int:
    """Approximate total tokens across a message list."""
    total = 0
    for m in messages:
        total += 4  # per-message overhead, OpenAI-ish convention
        for v in m.values():
            if isinstance(v, str):
                total += count_tokens(v)
            elif isinstance(v, list):
                # tool_calls, content blocks, etc.
                total += count_tokens(json.dumps(v, default=str))
    return total


def trim_messages(
    messages: list[dict],
    max_tokens: int,
    keep_first_n: int = 2,   # system prompt + initial user request
    keep_last_n: int = 4,    # recent context the model needs
) -> list[dict]:
    """
    Aggressive trimmer designed for tight context budgets (Groq free tier).

    Strategy, in order:
      1. If under budget, return as-is.
      2. Replace OLDEST tool results with short placeholders (page text
         is the biggest token consumer — and notes have already extracted
         what mattered).
      3. If still over budget, drop middle messages entirely.

    Notes live OUTSIDE the chat history (in WorkingMemory.notes) so they
    survive any amount of trimming.
    """
    if count_message_tokens(messages) <= max_tokens:
        return messages

    if len(messages) <= keep_first_n + keep_last_n:
        return messages

    # Step 2: shrink old tool results to one-line placeholders
    head = messages[:keep_first_n]
    tail_start = len(messages) - keep_last_n
    middle = messages[keep_first_n:tail_start]
    tail = messages[tail_start:]

    shrunk_middle = []
    for m in middle:
        if m.get("role") == "tool":
            content = m.get("content", "")
            if len(content) > 200:
                shrunk_middle.append({
                    **m,
                    "content": f"[truncated tool output, {len(content)} chars; "
                               f"durable info already captured in notes]",
                })
            else:
                shrunk_middle.append(m)
        else:
            shrunk_middle.append(m)

    candidate = head + shrunk_middle + tail
    if count_message_tokens(candidate) <= max_tokens:
        return candidate

    # Step 3: still too big — drop middle entirely
    dropped_tools = sum(
        1 for m in middle
        if m.get("role") == "tool" or (m.get("role") == "assistant" and m.get("tool_calls"))
    )
    placeholder = {
        "role": "user",
        "content": (
            f"[Context trimmed: {len(middle)} earlier messages "
            f"({dropped_tools} tool interactions) were dropped to fit budget. "
            f"Your structured notes are preserved separately. "
            f"Continue from current state.]"
        ),
    }
    return head + [placeholder] + tail


def format_notes_for_prompt(notes: list[dict]) -> str:
    """Render the agent's notes as a compact reference block."""
    if not notes:
        return "(no notes yet)"
    lines = []
    for i, n in enumerate(notes, 1):
        lines.append(f"  {i}. [{n['confidence']}] {n['claim']}  — {n['source_url']}")
    return "\n".join(lines)