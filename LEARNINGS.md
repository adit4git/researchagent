# Building a Production-Grade Research Agent: Eight Versions of Lessons Learned

A retrospective on building a free-tier research and summarization agent from v0.1
to v0.8. Each version exposed a real failure mode that no demo or tutorial would
have caught, and each fix that worked turned out to do double duty against later,
unrelated failures.

This document is organized around the 7-step agent framework we started from:

1. Start with a Goal
2. Pick the Right Model
3. Choose the Right Framework
4. Connect Tools
5. Divide Memory
6. Manage Context
7. Test and Evals

For each step we cover: what the framework promises, what actually happened, what
we ended up shipping, and the rule of thumb that emerged.

---

## Meta-lesson: failures cluster, fixes generalize

Before the per-step lessons, the most important meta-observation across eight
iterations: **defensive layers don't just fix the specific failure that prompted
them. Each one ends up rescuing later, unrelated failures.**

| Layer added in version | Why it was added | What else it later saved us from |
|---|---|---|
| Provider abstraction (v0.1) | Wanted multi-LLM support | Recovered from Llama-3.3 broken tool calls (v0.2), Gemini 2.0 deprecation (v0.7), Gemini 2.5 thinking bug (v0.8) |
| Force-finish from notes (v0.2) | Models forget to call finish() | Salvaged runs that hit 413/429/quota walls in v0.3-v0.7 |
| Stuck-bailout (v0.5) | Models echo garbage forever | Caught Gemini 2.5 Flash returning empty output in v0.7 |
| 413 emergency-trim (v0.5) | gpt-oss-120b context limit | Mirrored to handle 429 backoff (v0.5) and quota giveup (v0.7) |
| Trace logging (v0.1) | Future debugging | Was load-bearing in EVERY version's diagnosis |

The real takeaway: **production agents aren't a clean architecture you design
upfront. They're a layered defensive system you accumulate by hitting walls.**
The 7-step framework gives you the skeleton; the failure modes give you the muscle.

---

## Step 1: Start with a Goal

### What the framework says
Define the problem clearly. Set measurable goals. Identify HITL points.
Define agent constraints.

### What we actually learned

**Lesson 1.1: A "topic" string is not a goal — wrap it in a typed schema.**

Even for a one-line user input, formalizing it through Pydantic catches issues
upfront and forces the agent prompt to commit to specific deliverables:

```python
class ResearchGoal(BaseModel):
    topic: str = Field(min_length=5, max_length=500)
    depth: str = Field(default="standard", pattern="^(quick|standard|deep)$")
    max_iterations: int = Field(default=12, ge=3, le=30)
    target_word_count: int = Field(default=300, ge=100, le=1500)
```

The `max_iterations` field turned out to be the single most important constraint —
it's the budget the agent must work within, and it's what forced us to think hard
about what happens when the budget is reached without `finish()` being called.

**Lesson 1.2: The "stay on topic" constraint is non-trivial.**

Smaller models will dutifully take notes about whatever the page mentions, even
when irrelevant to the topic. We had a run where the topic was "Azure vs GCP" and
the agent took 3 notes about AWS because the source page was a 3-way comparison.
The fix had three layers:

1. Schema-level: `take_note` requires a `relevance_to_topic` field.
2. Prompt-level: explicit "if the topic is X vs Y, NEVER take notes about Z."
3. Post-hoc: lexical relevance filter drops off-topic notes before synthesis.

Adding the `relevance_to_topic` field alone got us most of the way — forcing the
model to articulate *why* a note matters before recording it tends to suppress the
generic "X is a database service" filler notes.

**Rule of thumb:** if a smaller model can technically satisfy your tool's schema
with garbage, the schema is too loose. Add a "justify it" field.

---

## Step 2: Pick the Right Model

### What the framework says
LRM (large reasoning model) for complex reasoning. LLM for average use. SLM (small
language model) for routing and rewriting. Pick based on cost vs capability.

### What we actually learned

This is where we hit the most failure modes. Six different models, four different
classes of failure:

**Lesson 2.1: Llama 3.3 70B on Groq is broken for tool use.**

Despite being a flagship model, Llama 3.3 emits tool calls wrapped in XML-like
`<function=name>...</function>` tags instead of the proper `tool_calls` channel.
Groq's strict validator rejects these as malformed. Result: HTTP 400 on every call,
agent never gets past iter 1. This was version v0.1's failure.

We considered building a recovery layer (parse the malformed format, synthesize
a valid tool call), but decided against it. The root issue is in Meta's training
data and Groq's gateway interaction; a client-side workaround for a server-side
bug is technical debt waiting to bite. We switched models instead.

**Lesson 2.2: Gemini 2.5 Flash has hidden "thinking" enabled by default.**

This was the most obscure bug we hit. Gemini 2.5 models burn output tokens on
internal reasoning before producing visible content. With `max_output_tokens=2048`,
thinking consumed 1400+ tokens leaving nothing for the actual response. The model
returned empty content with `output_tokens: 0` and zero tool calls. No error.

Fix:
```python
config = gtypes.GenerateContentConfig(
    thinking_config=gtypes.ThinkingConfig(thinking_budget=0),
    # ... rest of config
)
```

We don't need internal reasoning for tool-calling — the agent loop IS the
reasoning. This was version v0.8's last fix and unblocked a clean run.

**Lesson 2.3: Free-tier models change frequently and silently.**

`gemini-2.0-flash` was our default in v0.1. By v0.7 it was deprecated and
soft-throttled. The model name still resolved but every request returned 429
RESOURCE_EXHAUSTED. The fix was a one-line model-name change to `gemini-2.5-flash`,
but we burned a debug session figuring this out. Free tiers churn fast.

**Lesson 2.4: Schema validators differ wildly between providers.**

Groq tolerates `default` fields in JSON Schema. Anthropic does too. Gemini's
old SDK rejects them with `"Unknown field for Schema: default"`. Other fields
that bite: `additionalProperties`, `$schema`, `examples`, `title`, `$ref`.

We added a recursive sanitizer that strips Gemini-incompatible fields:

```python
_GEMINI_SCHEMA_DROP_KEYS = {
    "default", "additionalProperties", "$schema", "examples", "title",
    "$ref", "definitions", "$defs", "patternProperties", "const",
    # ...
}
```

This is defense-in-depth — different model versions have different validators,
and what's tolerated today may be rejected tomorrow.

**Lesson 2.5: "Free" has tight per-minute and per-day caps.**

Even when everything else works, free-tier rate limits are a real constraint:

- **Groq free tier (Jan 2026):** ~30 requests/minute, harsh after sustained use
- **Gemini 2.5 Flash:** 10 RPM, 250 RPD — fine for development
- **Gemini 2.5 Flash-Lite:** 15 RPM, 1000 RPD — best for high-volume
- **Gemini 2.5 Pro:** 5 RPM, 100 RPD — too tight for agent workflows

Picking the right tier matters. We landed on Flash for the research agent because
the topic-research task isn't reasoning-bound and Flash is forgiving enough.

**Rule of thumb:** for tool-using agents, prioritize **tool-call reliability**
over **raw intelligence**. A 70B model that fails 30% of tool calls is worse than
a 7B model that succeeds 95% of the time. Tool-call success rate is the metric
that matters.

---

## Step 3: Choose the Right Framework

### What the framework says
Simple workflows: Gumloop, Langflow, Dify, n8n, Flowise, Smolagents. Production:
Anthropic Agent SDK, LangGraph, Google ADK, CrewAI, LlamaIndex, OpenAI Agent SDK,
Microsoft Agent Framework.

### What we actually learned

**Lesson 3.1: The agent loop is 50 lines. Everything else is opinions.**

Our entire "framework" was a while-loop with tool calls:

```python
while not done and iterations < max:
    1. Trim messages to fit budget
    2. Inject status (current notes, turn number)
    3. Call LLM with tools
    4. If text only -> append, increment, continue
    5. If tool calls -> execute each, append results
    6. If finish() -> validate, return
```

We never reached for LangChain, CrewAI, or anything else. By the time we'd
shipped v0.8, we'd added emergency-trim, rate-limit backoff, garbage detection,
search-loop breaking, fetched-without-noted enforcement, post-finish resynthesis,
and source validation — features no off-the-shelf framework would have given us
out of the box.

The frameworks add value when you need:
- Multi-agent orchestration (agent A delegates to agent B)
- Persistent state across long-running workflows
- Visual debugging of agent state machines
- A team of non-Python users who need to author agents

For a single-shot research agent, plain Python was the right call. Adding a
framework would have been hiding the loop instead of mastering it.

**Lesson 3.2: The status message before each turn is the most important prompt.**

We injected a per-turn status message showing the agent its current notes and
turn budget:

```python
[Status: turn 5/12, 4 notes captured]
Current notes:
  1. [high] Azure SQL Database is a managed PaaS engine. — https://...
  2. [high] GCP Cloud SQL supports automatic backups. — https://...
  ...
```

Plus iteration-aware pressure:
- Late in the run with not enough notes: "STOP SEARCHING. Pick the best URL and fetch_page NOW."
- Late with enough notes: "Call finish() now. Do not write more prose."
- Plenty of notes: "You have plenty of notes. Call finish NOW."

This single feature did more for note quality than any prompt-engineering on the
system prompt. The agent needs constant feedback about where it is in the budget.

**Rule of thumb:** start with raw Python. Switch to a framework only when you
have a specific feature need the framework handles natively.

---

## Step 4: Connect Tools

### What the framework says
MCP, agents-as-tools, function calling, file system access, A2A protocol.

### What we actually learned

**Lesson 4.1: Four tools is plenty.** Most agent tutorials over-complicate this.
Our final toolset:

- `search_web(query)` — DuckDuckGo, no API key
- `fetch_page(url)` — httpx + readability
- `take_note(claim, source_url, confidence, relevance_to_topic)` — structured capture
- `finish(summary, key_findings, source_urls, open_questions)` — terminate

Adding more (separate `query_database`, `summarize_text`, `cite_source` tools)
would just give the model more ways to misuse the toolset.

**Lesson 4.2: Tools should never raise. Always return a string.**

A tool error becomes input the model can react to. Every tool wrapper:

```python
try:
    result = do_thing(...)
except SomeError as e:
    return f"Tool failed: {type(e).__name__}: {e}. Try a different approach."
```

This was crucial in v0.5 when DuckDuckGo started rate-limiting. The model would
see `"Search failed: rate limit"` and try a different query, instead of the agent
loop crashing.

**Lesson 4.3: Add validation guards as separate logic, not in the tool.**

Examples from our final agent:

- Dedup: if the model tries to search the same query twice, intercept and tell it
- Dedup: if the model tries to fetch the same URL twice, intercept and tell it
- Cache: if the URL is already in our SQLite cache, serve from cache
- Budget: cap individual fetched pages to 4500 chars

Putting these in the agent loop instead of the tool means the model sees a clear
error message and can adapt, rather than getting confused by silent behavior.

**Lesson 4.4: User-Agent matters. Use a real one.**

Default httpx User-Agent gets blocked by many sites with 403/429. Switching to a
Chrome UA cut 403 errors by ~80% in our testing:

```python
headers={
    "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/121.0.0.0 Safari/537.36"),
    "Accept": "text/html,application/xhtml+xml,...",
    "Accept-Language": "en-US,en;q=0.9",
}
```

**Lesson 4.5: Detect and reject garbage content at fetch time.**

In v0.5 we hit a page that returned mojibake (replacement-character flood). The
model echoed the garbage back at maximum tokens for two iterations, no tool calls,
agent stuck. Fix: detect garbage at fetch and return a clean error string instead.

```python
def _looks_like_garbage(text: str) -> bool:
    sample = text[:3000]
    if sample.count("\ufffd") / max(len(sample), 1) > 0.05:
        return True
    if non_printable_chars / len(sample) > 0.05:
        return True
    if whitespace_chars / len(sample) < 0.05:
        return True
    return False
```

Also reject non-HTML content-types upfront (PDFs, JSON, images). Saves the model
from being confused by raw binary.

**Rule of thumb:** every tool needs to handle three failure modes — its own
errors, the data being unusable, and the model misusing it. Address all three at
the tool layer; don't push it to the model.

---

## Step 5: Divide Memory

### What the framework says
Graph memory, cache memory, procedural memory, episodic memory, file system
memory.

### What we actually learned

**Lesson 5.1: Two layers is enough for a single-shot agent.**

We mapped the framework's five memory types to two practical layers:

- **Working memory (in-RAM):** current chat history, current notes, fetched URLs,
  search queries, token counters, trace log. Lives one run, dies with the run.
- **Persistent cache (SQLite):** URL → cleaned page text, with timestamp. Lives
  across runs, makes re-runs and evals fast.

We deliberately *did not* use a vector database. The full research session is
short-lived (8-12 turns) and fits in the LLM's context. Vector search becomes
relevant in conversational agents over a corpus (use case #5), not here.

**Lesson 5.2: Notes are a separate memory layer, deliberately.**

The most important memory design decision: notes live OUTSIDE the chat history
(in `WorkingMemory.notes`, a list of dicts). When we trim the conversation to
fit context budget, notes survive. When the model "forgets" the page text we
fetched, the notes preserve what mattered.

```python
@dataclass
class WorkingMemory:
    messages: list[dict] = field(default_factory=list)  # gets trimmed
    notes: list[dict] = field(default_factory=list)     # never trimmed
```

This is the unsung hero of the design. Without it, every context-trim would
also wipe out the model's research progress.

**Lesson 5.3: SQLite is enough. You don't need Redis or Postgres.**

For URL caching, run history, and trace storage, SQLite (Python stdlib) handles
everything. Single-file. No server. No connection pooling. No serialization
boilerplate. Switch to Postgres only when you have multiple writers or need
real concurrency.

**Rule of thumb:** start with one memory layer (in-RAM). Add a second (persistent)
when you need cross-session persistence. Add a third (vector) only when you have
a corpus larger than the context window.

---

## Step 6: Manage Context

### What the framework says
Compress old context through summarization. Monitor context effectiveness with
metrics. Add context intelligently. Use agentic context engineering loop.

### What we actually learned

This was the source of most of our failures. Six versions of fixes here.

**Lesson 6.1: Context budgets are tighter than you think.**

We started with 12000 tokens. Hit 413 errors on Groq's 8k-input-limit free tier.
Cut to 6000. Hit them again on smaller models. The actual usable budget for a
free-tier model is roughly:

```
budget = model_context - tool_schemas - system_prompt - max_output_tokens - safety_margin
       = 8000 - 500 - 500 - 2048 - 1000
       = 3952 tokens
```

For a free-tier 8k model, you have ~4k tokens of *conversation history* to play
with. That's surprisingly little.

**Lesson 6.2: Single-page caps matter more than total budget.**

A 8000-char fetched page is ~2000 tokens — half your effective budget on one
fetch. Cutting `MAX_PAGE_CHARS` from 8000 to 4500 was the single biggest reduction
in 413 errors. Every page should be small enough that 2-3 of them fit comfortably.

**Lesson 6.3: Trim aggressively. Replace, don't drop.**

Naive trim drops the middle messages. Better: replace old tool-result messages
with one-line placeholders. That preserves conversation structure (the assistant's
tool calls still pair with their results) while reclaiming most of the tokens.

```python
# Before:
{"role": "tool", "content": "<8000 chars of page text>"}
# After:
{"role": "tool", "content": "[truncated, 8000 chars; info captured in notes]"}
```

This was the trim improvement in v0.5.

**Lesson 6.4: Emergency-trim on 413, retry within the same turn.**

When the budget calculation fails (page bigger than expected, system prompt grew,
etc.), the model returns 413. The fix: catch the 413, run a much more aggressive
trim (down to 3000 tokens), retry the same turn. Don't burn an iteration.

```python
try:
    resp = self.llm.chat(turn_messages, tools=TOOL_SCHEMAS)
except Exception as e:
    if is_too_big(e):
        mem.messages = trim_messages(mem.messages, max_tokens=EMERGENCY_BUDGET)
        resp = self.llm.chat(mem.messages + [status], tools=TOOL_SCHEMAS)
```

**Lesson 6.5: Token counting is approximate but free.**

We use `tiktoken` for token estimates, with a fallback to `len(text) // 4` when
tiktoken can't load (offline environments). The 4-chars-per-token heuristic is
within 20% accuracy for English. Good enough for budget enforcement, free for
char counting, no API calls needed.

**Rule of thumb:** assume the budget is half what the docs say. Plan for
emergency trim. Always preserve the system prompt and the most recent turns.

---

## Step 7: Test and Evals

### What the framework says
Unit tests for specific functions. Edge case discovery. Cost per task.
Observability and tracing. Prompt versioning and A/B testing.

### What we actually learned

**Lesson 7.1: Trace logging is the single most useful feature.**

Every defensive fix from v0.2 onward was prompted by reading a trace. The
`mem.log_step(kind, payload)` calls scattered through the agent let us reconstruct
what happened on every turn:

```python
mem.log_step("tool_call", {
    "iter": i, "name": tc.name,
    "args_preview": str(tc.arguments)[:150],
    "output_preview": result.output[:200],
})
```

Run with `--trace` and you see:
- LLM input/output token counts per turn
- Which tool got called and with what args
- Tool output (truncated for readability)
- Cache hits vs fresh fetches
- Emergency-trim events
- Rate-limit backoffs
- Final iteration count and cost

This is what separates a debuggable agent from an opaque one. **Add tracing
before you add anything else.**

**Lesson 7.2: A 12-case golden set catches most regressions.**

Our `evals/golden.jsonl` has 12 hand-curated topics with must-mention keywords
and minimum source counts. After any change:

```bash
python -m evals.run_evals --limit 5   # smoke test
python -m evals.run_evals             # full suite
```

The eval harness reports pass rate, average iterations, average cost per task,
and which keywords were missed. Failing cases get a `reason` string.

**Lesson 7.3: Cost-per-task is a critical metric.**

Even on free tiers ($0 cost), tracking iteration counts is the proxy for cost.
A run that takes 12 iterations to do what should take 6 is twice as expensive.
The eval harness tracks this:

```
PASS RATE:    9/12 (75%)
AVG SCORE:    0.81
AVG ITER:     7.3
AVG COST:     $0.0000
TOTAL COST:   $0.0001
```

This metric set is what tells you whether a "fix" actually improved things or
just shifted failures around.

**Lesson 7.4: Keyword overlap is a weak but free quality signal.**

Our eval scorer uses lexical overlap: does the summary mention the must-mention
keywords from the case? It's a coarse metric — a summary can hit all keywords
and still be wrong, or miss them and still be useful. But it's free, it's
deterministic, and it catches gross regressions.

For real factuality, you'd add LLM-as-judge with a *different* model than the
one being evaluated. We didn't add this — it doubles cost and adds another model
dependency. Worth it for production; overkill for the build phase.

**Rule of thumb:** logging beats metrics, metrics beat opinions. Trace
everything; measure what you care about; don't trust feelings about whether
something is "better."

---

## The full version-by-version journey

| Version | Trigger | Fix | Notes |
|---|---|---|---|
| **v0.1** | Initial build | Provider abstraction, basic agent loop, golden eval set | Llama-3.3 broken on Groq, 0 notes captured |
| **v0.2** | Tool-call format errors | Switched to gpt-oss-120b. Added stronger prompts, search-loop breaker | Search loop, then 413 cascades |
| **v0.3** | Persistent 413s | Tightened context budget 12k → 6k | Still ate iterations on 413 |
| **v0.4** | Off-topic notes | Added `relevance_to_topic` field, lexical filter | 6 notes but unbalanced summary |
| **v0.5** | Mojibake page broke model | Garbage detector, content-type filter, stuck-bailout | Hit Groq rate limits |
| **v0.6** | Switched to Gemini | Schema sanitizer, migrated to google-genai SDK | Hit `Unknown field: default` |
| **v0.7** | Daily quota exhausted | Updated default model 2.0 → 2.5 flash, retry cap | Empty responses with 0 output tokens |
| **v0.8** | Gemini 2.5 thinking ate output | `thinking_budget=0` config | **Clean run: 8 iter, 9 notes, 2 sources, balanced summary** |

---

## What "production-grade" actually means

After eight iterations, here's what we ended up shipping:

**Eight defensive layers, in order of when they fire:**

1. **Schema sanitization** — strip provider-incompatible JSON Schema fields
2. **Tool execution wrapper** — every tool returns a string, never raises
3. **Garbage detection** — reject mojibake/binary at fetch time
4. **Cache** — SQLite hits before re-fetching
5. **Dedup guards** — no repeated searches or fetches
6. **Trim before each turn** — keep context under budget
7. **Emergency trim on 413** — recover within same iteration
8. **Rate-limit backoff with cap** — sleep on 429, give up after 2 retries
9. **Note enforcement** — force note-taking after fetches
10. **Search-loop breaker** — force a fetch if the model just searches
11. **Stuck-bailout** — exit if 2 consecutive empty-tool-call turns
12. **Topic relevance filter** — drop off-topic notes pre-synthesis
13. **Source validation** — drop URLs the model invented at finish() time
14. **LLM-synthesized force-finish** — build a brief from notes if cap is hit

**Plus the always-on observability:**

- Full trace log of every turn
- Token counters for input/output
- Cost tracker per provider
- Iteration counter
- Persistent run history in SQLite

---

## Three takeaways for anyone building one of these

**1. The agent loop is the easy part.** It fits on one page. The 90% of work
that determines whether the agent runs in production is in the boundary
conditions — what happens when a tool fails, when a model returns garbage,
when a budget overflows, when a free tier runs out.

**2. Defense layers compound.** No single fix made the agent production-grade.
But the cumulative weight of 14 small layers, each addressing a real failure
we hit, turns the loop from "it works on the demo" to "it almost always
finishes successfully even when things go wrong."

**3. Trace before you build.** The trace logging from v0.1 was invoked in
every subsequent debug session. If we'd added it later, we'd have wasted hours
debugging blind. If you're building any kind of agent, **add structured trace
logging before anything else** — even before the agent loop is fully working.

---

## What's next

This agent is good enough to ship for a personal/internal use case. For
production deployment to external users you'd still want:

- LLM-as-judge for factuality scoring (evals/run_evals.py extension)
- Provider fallback chain (try Gemini → Groq → Ollama on errors)
- Async tool execution (parallel search + fetch when topic has independent angles)
- Hybrid search (combine DuckDuckGo with a keyword index for known sources)
- Per-user quota tracking (when serving multiple users from one API key)

But the foundation — the loop, the memory layers, the eval harness, the
provider abstraction, and the dozen-plus defensive layers — generalizes
cleanly to other agent types. Use case #5 (Document Q&A with RAG) reuses
~70% of this codebase verbatim.

That's the real win. Not that we built a research agent, but that we built
the muscle to recognize, diagnose, and fix the failure modes that all agents
share.
