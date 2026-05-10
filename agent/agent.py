"""
The agent loop (Step 3 of the framework: choose the right framework).

HARDENED v0.2 — addresses real-world failure modes seen in v0.1:
  1. Strong "you MUST take notes" enforcement after every fetch
  2. Iteration-aware pressure: harder nudges as turns run out
  3. Detect "fetched-but-no-notes" loop and break it
  4. Final-attempt synthesis: when force-finishing, ASK the LLM to
     write a real summary from notes rather than dumping bullets
  5. Tool-name list logged in trace (so debugging is faster)
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from .context import count_message_tokens, trim_messages
from .llm import LLM, ToolCall
from .memory import PersistentCache, WorkingMemory
from .prompts import (
    ResearchBrief,
    ResearchGoal,
    SYSTEM_PROMPT,
    build_status_message,
    build_user_message,
)
from .tools import TOOL_SCHEMAS, execute_tool


MIN_NOTES_TO_FINISH = 5
HARD_FINISH_NOTES = 8


@dataclass
class AgentConfig:
    max_iterations: int = 12
    # Tight default for free-tier providers (Groq gpt-oss-120b ~8k input limit).
    # Bump to 12000+ if using Gemini/Claude/larger paid tiers.
    context_token_budget: int = 6_000
    use_cache: bool = True
    auto_synthesize_on_force_finish: bool = True
    # Emergency trim budget when we hit a 413
    emergency_token_budget: int = 3_000
    # Note quality control: filter notes by topic relevance before synthesis
    filter_notes_by_relevance: bool = True
    relevance_threshold: float = 0.5  # 0-1, drop notes below this


class ResearchAgent:
    """Single-shot research agent. One topic in, one brief out."""

    def __init__(
        self,
        llm: LLM | None = None,
        config: AgentConfig | None = None,
        cache: PersistentCache | None = None,
    ):
        self.llm = llm or LLM()
        self.config = config or AgentConfig()
        self.cache = cache if cache is not None else PersistentCache()

    def run(self, goal: ResearchGoal) -> ResearchBrief:
        mem = WorkingMemory(topic=goal.topic)
        mem.messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_message(goal)},
        ]
        mem.log_step("start", {"topic": goal.topic, "config": self.config.__dict__})

        finish_payload: dict | None = None
        max_iter = min(goal.max_iterations, self.config.max_iterations)
        fetches_without_notes = 0
        consecutive_searches = 0
        consecutive_no_tool_calls = 0  # detect model echoing garbage / stuck
        rate_limit_retries = 0  # cap total backoff time per run
        MAX_RATE_LIMIT_RETRIES = 2  # daily-quota errors are unrecoverable; bail early

        for i in range(1, max_iter + 1):
            mem.iterations = i

            mem.messages = trim_messages(
                mem.messages,
                max_tokens=self.config.context_token_budget,
            )

            # Iteration-aware status
            status_text = build_status_message(mem.notes, i, max_iter)
            turns_left = max_iter - i + 1
            if len(mem.notes) >= HARD_FINISH_NOTES:
                status_text += "\n\n>>> You have plenty of notes. Call finish NOW."
            elif len(mem.notes) >= MIN_NOTES_TO_FINISH and turns_left <= 4:
                status_text += "\n\n>>> Time is running out. Call finish now with what you have."
            elif turns_left <= 3 and len(mem.notes) < MIN_NOTES_TO_FINISH:
                status_text += (
                    "\n\n>>> CRITICAL: Only a few turns left and not enough notes. "
                    "Stop searching. Pick the best URL from your last search results "
                    "and fetch_page it. Then immediately call take_note 3+ times for "
                    "the facts you read."
                )

            status = {"role": "user", "content": status_text}
            turn_messages = mem.messages + [status]

            # ---- Call the model (with 413 emergency-trim and 429 backoff recovery) ----
            resp = None
            for attempt in range(2):
                try:
                    resp = self.llm.chat(turn_messages, tools=TOOL_SCHEMAS)
                    break
                except Exception as e:
                    err_str = str(e)
                    is_too_big = ("413" in err_str
                                  or "too large" in err_str.lower()
                                  or ("context" in err_str.lower() and "exceed" in err_str.lower()))
                    is_rate_limited = (
                        "429" in err_str
                        or "rate limit" in err_str.lower()
                    )

                    if is_too_big and attempt == 0:
                        # Emergency trim — much smaller budget — and retry
                        mem.log_step("emergency_trim", {
                            "iter": i,
                            "before_tokens": count_message_tokens(turn_messages),
                        })
                        mem.messages = trim_messages(
                            mem.messages,
                            max_tokens=self.config.emergency_token_budget,
                            keep_first_n=2,
                            keep_last_n=2,
                        )
                        turn_messages = mem.messages + [status]
                        continue

                    if is_rate_limited and attempt == 0:
                        rate_limit_retries += 1
                        if rate_limit_retries > MAX_RATE_LIMIT_RETRIES:
                            # Persistent rate-limiting almost certainly means
                            # daily quota is exhausted, not per-minute. Bail.
                            mem.log_step("rate_limit_giveup", {
                                "iter": i,
                                "retries_used": rate_limit_retries,
                                "hint": "Likely daily quota exhausted. Switch model or provider.",
                            })
                            mem.messages.append({
                                "role": "user",
                                "content": (
                                    f"[Rate limit hit {rate_limit_retries} times. "
                                    f"Daily quota likely exhausted on this provider/model. "
                                    f"Stopping early.]"
                                ),
                            })
                            # Set a sentinel that the outer loop will check
                            resp = "RATE_LIMIT_GIVEUP"
                            break
                        # Try to extract retry-after seconds from the error
                        import re as _re
                        wait_seconds = 30  # reasonable default for free-tier
                        m = _re.search(r"try again in (\d+(?:\.\d+)?)\s*s", err_str.lower())
                        if m:
                            wait_seconds = min(int(float(m.group(1))) + 1, 60)
                        mem.log_step("rate_limit_backoff", {
                            "iter": i,
                            "wait_seconds": wait_seconds,
                            "retries_used": rate_limit_retries,
                        })
                        import time as _time
                        _time.sleep(wait_seconds)
                        continue

                    # Non-recoverable or already retried
                    mem.log_step("llm_error", {"error": err_str, "type": type(e).__name__})
                    mem.messages.append({
                        "role": "user",
                        "content": f"[LLM error: {err_str[:200]}. Try a different approach.]",
                    })
                    break

            if resp == "RATE_LIMIT_GIVEUP":
                break

            if resp is None:
                continue

            mem.total_input_tokens += resp.input_tokens
            mem.total_output_tokens += resp.output_tokens
            mem.total_cost_usd += resp.cost_usd
            mem.log_step("llm_response", {
                "iter": i,
                "input_tokens": resp.input_tokens,
                "output_tokens": resp.output_tokens,
                "n_tool_calls": len(resp.tool_calls),
                "tool_names": [tc.name for tc in resp.tool_calls],
                "text_preview": (resp.text or "")[:200],
            })

            # ---- No tool calls: model just talked ----
            if not resp.tool_calls:
                consecutive_no_tool_calls += 1

                # If the model has produced 2+ consecutive no-tool-call turns,
                # it's almost certainly stuck (echoing garbage from a bad fetch,
                # or refusing to call tools). Bail to synthesis with what we have.
                if consecutive_no_tool_calls >= 2:
                    mem.log_step("stuck_bailout", {
                        "iter": i,
                        "notes_so_far": len(mem.notes),
                        "reason": "model produced no tool calls 2+ turns in a row",
                    })
                    break

                mem.messages.append({"role": "assistant", "content": resp.text})
                if len(mem.notes) >= MIN_NOTES_TO_FINISH:
                    mem.messages.append({
                        "role": "user",
                        "content": "You have enough notes. Call finish() now. Do not write more prose.",
                    })
                else:
                    mem.messages.append({
                        "role": "user",
                        "content": (
                            "You did not call any tool. You MUST call a tool every turn: "
                            "search_web, fetch_page, take_note, or finish. Try again now."
                        ),
                    })
                continue

            # Reset counter when the model calls tools again
            consecutive_no_tool_calls = 0

            mem.messages.append(self._format_assistant_turn(resp.text, resp.tool_calls))

            this_turn_did_fetch = False
            this_turn_did_note = False
            this_turn_did_search = False

            for tc in resp.tool_calls:
                # Dedup guards
                if tc.name == "search_web":
                    this_turn_did_search = True
                    q = (tc.arguments.get("query") or "").strip().lower()
                    if q in mem.searched_queries:
                        mem.messages.append(self._tool_result(
                            tc.id, tc.name,
                            "You already ran this exact search. Either fetch a URL "
                            "from earlier results or try a meaningfully different query.",
                        ))
                        continue
                    mem.searched_queries.add(q)

                if tc.name == "fetch_page":
                    url = (tc.arguments.get("url") or "").strip()
                    if url in mem.fetched_urls:
                        mem.messages.append(self._tool_result(
                            tc.id, tc.name,
                            "You already fetched that URL. Pick a different one.",
                        ))
                        continue
                    mem.fetched_urls.add(url)
                    this_turn_did_fetch = True

                    if self.config.use_cache:
                        cached = self.cache.get_page(url)
                        if cached:
                            mem.log_step("cache_hit", {"url": url})
                            mem.messages.append(self._tool_result(tc.id, tc.name, cached))
                            continue

                if tc.name == "take_note":
                    this_turn_did_note = True

                # Execute
                result = execute_tool(tc.name, tc.arguments, mem.notes)
                mem.log_step("tool_call", {
                    "iter": i,
                    "name": tc.name,
                    "args_preview": json.dumps(tc.arguments, default=str)[:150],
                    "output_preview": result.output[:200],
                    "is_terminal": result.is_terminal,
                })

                if (tc.name == "fetch_page"
                        and self.config.use_cache
                        and not result.output.startswith(("Fetch failed", "HTTP ", "Invalid URL"))):
                    self.cache.put_page(tc.arguments["url"], result.output)

                mem.messages.append(self._tool_result(tc.id, tc.name, result.output))

                if result.is_terminal:
                    finish_payload = result.finish_payload
                    break

            if finish_payload:
                break

            # FIX 1+2: enforce notes after fetches
            if this_turn_did_fetch and not this_turn_did_note:
                fetches_without_notes += 1
                mem.messages.append({
                    "role": "user",
                    "content": (
                        "You just fetched a page. The page text is in your context "
                        "RIGHT NOW but will be trimmed soon. Before you do anything "
                        "else, call take_note at least 3 times with concrete facts "
                        "from that page. Each note needs claim, source_url, and "
                        "confidence. Do not search or fetch again until you've taken notes."
                    ),
                })
            elif this_turn_did_note:
                fetches_without_notes = 0

            if fetches_without_notes >= 2:
                mem.messages.append({
                    "role": "user",
                    "content": (
                        ">>> You've fetched multiple pages without taking notes. "
                        "STOP fetching. Look at your last successful fetch_page result "
                        "and extract 3-5 facts as take_note calls right now."
                    ),
                })

            # FIX 3 (NEW): break search-loop pattern
            # If the model searches without fetching for 2+ turns in a row,
            # force-pick a URL by injecting a hard instruction.
            if this_turn_did_search and not this_turn_did_fetch:
                consecutive_searches += 1
            elif this_turn_did_fetch:
                consecutive_searches = 0

            if consecutive_searches >= 2:
                mem.messages.append({
                    "role": "user",
                    "content": (
                        ">>> STOP SEARCHING. You've searched multiple times in a row "
                        "without fetching anything. Searches alone produce zero notes. "
                        "Look at your most recent search results, pick the URL that "
                        "looks most authoritative, and call fetch_page on it RIGHT NOW. "
                        "Then immediately take_note 3+ times. No more searches until "
                        "you've fetched and noted."
                    ),
                })
                # Reset so we don't keep injecting the same nudge
                consecutive_searches = 0

        # Filter notes by topic relevance BEFORE synthesis
        if self.config.filter_notes_by_relevance and mem.notes:
            kept, dropped = self._filter_notes_by_relevance(mem.notes, goal.topic)
            if dropped:
                mem.log_step("notes_filtered", {
                    "kept": len(kept),
                    "dropped": len(dropped),
                    "dropped_claims": [n["claim"][:80] for n in dropped],
                })
                mem.notes = kept

        # Loop ended
        if finish_payload is None:
            mem.log_step("force_finish", {"reason": "iteration_cap", "iterations": mem.iterations})
            finish_payload = self._build_finish_from_notes(mem, goal)
        elif self.config.filter_notes_by_relevance and mem.notes:
            # Re-synthesize summary if filtering changed the note set significantly
            # (the model already called finish, but its summary used dropped notes)
            kept_sources = mem.cited_sources()
            old_sources = finish_payload.get("source_urls") or []
            if set(kept_sources) != set(old_sources) and len(mem.notes) >= 2:
                mem.log_step("re_synthesize", {"reason": "notes_filtered_post_finish"})
                try:
                    finish_payload = self._llm_synthesize(
                        mem, goal, kept_sources, reason="notes_filtered"
                    )
                except Exception as e:
                    mem.log_step("re_synthesize_error", {"error": str(e)})

        # Validate finish_payload's source_urls against actually-cited sources.
        # Models occasionally invent URLs at finish() time that they never fetched
        # or noted. We trust the cited_sources() list (built from take_note calls)
        # over the model's source_urls.
        cited = mem.cited_sources()
        proposed_sources = finish_payload.get("source_urls") or []
        invented = [u for u in proposed_sources if u not in cited and u not in mem.fetched_urls]
        if invented:
            mem.log_step("dropped_invented_sources", {
                "count": len(invented),
                "urls": invented[:5],
            })
            # Anchor source_urls to what we actually have evidence for
            finish_payload["source_urls"] = cited

        brief = ResearchBrief(
            topic=goal.topic,
            summary=finish_payload.get("summary", ""),
            key_findings=finish_payload.get("key_findings", []),
            source_urls=finish_payload.get("source_urls") or mem.cited_sources(),
            open_questions=finish_payload.get("open_questions", []),
            iterations_used=mem.iterations,
            total_cost_usd=mem.total_cost_usd,
            notes_count=len(mem.notes),
        )
        self.cache.record_run(mem, brief.summary)
        mem.log_step("done", {
            "iterations": mem.iterations,
            "cost_usd": mem.total_cost_usd,
            "notes": len(mem.notes),
            "sources": len(brief.source_urls),
        })
        brief._trace = mem.trace  # type: ignore[attr-defined]
        return brief

    def _filter_notes_by_relevance(
        self, notes: list[dict], topic: str
    ) -> tuple[list[dict], list[dict]]:
        """
        Score each note's relevance to the topic; return (kept, dropped).

        Uses lexical overlap as a fast, free heuristic:
          - Extract significant words from the topic (3+ chars, non-stopword)
          - For each note, compute overlap fraction with claim + relevance text
          - Keep notes with score >= threshold

        This is intentionally simple. Real systems use embedding similarity
        or LLM-as-judge — both work but cost more. For 6-15 notes, lexical
        overlap catches the obvious off-topic cases (the AWS-in-Azure-vs-GCP
        scenario) at zero cost.
        """
        # Common stopwords to ignore
        STOPWORDS = {
            "the", "a", "an", "and", "or", "but", "for", "of", "to", "in",
            "on", "at", "by", "with", "from", "is", "are", "was", "were",
            "what", "how", "why", "when", "where", "which", "who", "vs",
            "between", "small", "large", "big", "good", "best", "should",
            "would", "could", "can", "may", "might", "do", "does", "did",
            "this", "that", "these", "those", "it", "its", "their", "your",
        }

        def tokens(s: str) -> set[str]:
            words = "".join(c.lower() if c.isalnum() else " " for c in s).split()
            return {w for w in words if len(w) >= 3 and w not in STOPWORDS}

        topic_tokens = tokens(topic)
        if not topic_tokens:
            # Topic is all stopwords? Don't filter.
            return notes, []

        kept, dropped = [], []
        for note in notes:
            note_text = note.get("claim", "") + " " + note.get("relevance_to_topic", "")
            note_tokens = tokens(note_text)
            if not note_tokens:
                dropped.append(note)
                continue
            overlap = len(topic_tokens & note_tokens) / len(topic_tokens)
            if overlap >= self.config.relevance_threshold:
                kept.append(note)
            else:
                dropped.append(note)

        # Safety net: never filter ALL notes — keep the top 3 by overlap if so
        if not kept and notes:
            scored = []
            for note in notes:
                note_text = note.get("claim", "") + " " + note.get("relevance_to_topic", "")
                note_tokens = tokens(note_text)
                overlap = len(topic_tokens & note_tokens) / len(topic_tokens) if note_tokens else 0
                scored.append((overlap, note))
            scored.sort(key=lambda x: -x[0])
            kept = [n for _, n in scored[:3]]
            dropped = [n for _, n in scored[3:]]

        return kept, dropped

    # ---------- helpers ----------

    @staticmethod
    def _format_assistant_turn(text: str, tool_calls: list[ToolCall]) -> dict:
        return {
            "role": "assistant",
            "content": text or None,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments),
                    },
                }
                for tc in tool_calls
            ],
        }

    @staticmethod
    def _tool_result(call_id: str, name: str, content: str) -> dict:
        return {
            "role": "tool",
            "tool_call_id": call_id,
            "name": name,
            "content": content,
        }

    def _build_finish_from_notes(self, mem: WorkingMemory, goal: ResearchGoal) -> dict:
        """When force-finishing, synthesize via LLM if possible."""
        sources = mem.cited_sources()
        if not mem.notes:
            return {
                "summary": (
                    f"Could not gather sufficient information on '{mem.topic}'. "
                    f"The agent ran for {mem.iterations} iterations without "
                    f"capturing any structured notes. Common causes: search backend "
                    f"rate-limited, model failed to call tools correctly, or all "
                    f"target sites blocked the fetcher. Run with --trace to see why."
                ),
                "key_findings": [],
                "source_urls": [],
                "open_questions": ["All of them — research did not complete."],
            }

        if self.config.auto_synthesize_on_force_finish:
            try:
                return self._llm_synthesize(mem, goal, sources)
            except Exception as e:
                mem.log_step("synthesize_error", {"error": str(e)})

        # Fallback
        bullets = []
        for n in mem.notes[:10]:
            src_idx = sources.index(n["source_url"]) + 1 if n["source_url"] in sources else 0
            bullets.append(f"- {n['claim']} [{src_idx}]")
        return {
            "summary": (
                f"Research on '{mem.topic}' did not complete cleanly. Captured findings:\n\n"
                + "\n".join(bullets)
            ),
            "key_findings": [n["claim"] for n in mem.notes[:5]],
            "source_urls": sources,
            "open_questions": ["Agent hit iteration cap before producing a clean synthesis."],
        }

    def _llm_synthesize(
        self,
        mem: WorkingMemory,
        goal: ResearchGoal,
        sources: list[str],
        reason: str = "iteration_cap",
    ) -> dict:
        """
        One final LLM call that ONLY synthesizes, no tools.

        `reason` controls the trailing open-question note:
          - 'iteration_cap': agent ran out of turns
          - 'notes_filtered': filter dropped sources, summary was rewritten
          - None: don't append anything
        """
        notes_block = "\n".join(
            f"  [{sources.index(n['source_url'])+1}] ({n['confidence']}) {n['claim']}"
            for n in mem.notes
        )
        sources_block = "\n".join(f"  [{i+1}] {u}" for i, u in enumerate(sources))
        prompt = f"""You are synthesizing research notes into a final brief. The agent collected these notes but ran out of iterations before calling finish. Produce the final brief now.

TOPIC: {goal.topic}

NOTES (numbered citations match sources below):
{notes_block}

SOURCES:
{sources_block}

Respond with a JSON object exactly matching this shape (no other text, no markdown fences):
{{
  "summary": "200-400 words of coherent prose with [1], [2] inline citations",
  "key_findings": ["3-7 bullet strings"],
  "source_urls": ["full URLs in the same order as citations"],
  "open_questions": ["honest gaps"]
}}
"""
        resp = self.llm.chat(
            messages=[
                {"role": "system", "content": "You are a research synthesizer. Output valid JSON only."},
                {"role": "user", "content": prompt},
            ],
            tools=None,
            temperature=0.2,
            max_tokens=1500,
        )
        mem.total_input_tokens += resp.input_tokens
        mem.total_output_tokens += resp.output_tokens
        mem.total_cost_usd += resp.cost_usd

        text = resp.text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text
            if text.endswith("```"):
                text = text.rsplit("```", 1)[0]
            text = text.strip()
        if text.startswith("json"):
            text = text[4:].strip()

        payload = json.loads(text)
        payload["source_urls"] = sources
        payload.setdefault("open_questions", [])
        if reason == "iteration_cap":
            payload["open_questions"].append(
                "Agent hit iteration cap; brief was synthesized from captured notes."
            )
        elif reason == "notes_filtered":
            # The agent finished cleanly. Some notes were filtered for
            # relevance and the summary was rewritten. No need to surface
            # this internal cleanup as an "open question" to the user.
            pass
        return payload
