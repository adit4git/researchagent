# Research & Summarization Agent

A production-style research agent that takes a topic, searches the web, reads sources, and produces a structured brief with citations. Built to run **for $0** using free tools, with optional paid upgrades.

This implements the 7-step framework end-to-end:

| Step | What it does | Where in code |
|---|---|---|
| 1. Goal | Validates topic, sets depth/length constraints | `agent/agent.py` (`ResearchGoal`) |
| 2. Model | Provider-agnostic — Groq, Gemini, Ollama, Anthropic | `agent/llm.py` |
| 3. Framework | Pure Python tool-loop (no LangChain/CrewAI needed) | `agent/agent.py` |
| 4. Tools | Web search + page fetch + note-taking | `agent/tools.py` |
| 5. Memory | Working scratchpad + persistent JSON cache | `agent/memory.py` |
| 6. Context | Token budget, summarization, source dedup | `agent/context.py` |
| 7. Evals | Golden dataset + factuality + cost tracking | `evals/run_evals.py` |

## Why this stack

The framework infographic is solid, but you don't need eight frameworks to start. A tool-calling loop in plain Python (~200 lines) teaches you what frameworks are *hiding* from you. Once you understand the loop, switching to LangGraph or CrewAI takes an afternoon.

## Free-first stack

| Component | Free option (default) | Paid upgrade |
|---|---|---|
| LLM | **Groq** (Llama 3.3 70B, free tier, very fast) | Anthropic Claude / OpenAI |
| LLM (local) | **Ollama** (qwen2.5, llama3.2) | — |
| LLM (cloud free) | **Google Gemini** (free tier) | Gemini Pro |
| Web search | **DuckDuckGo** (no key needed) | Tavily / Brave / Serper |
| Page fetch | **httpx + readability-lxml** | Firecrawl |
| Tracing | **Local JSONL logs** | Langfuse / LangSmith |
| Storage | **SQLite** (built into Python) | Postgres / Redis |

You can run everything with **zero API keys** using Ollama + DuckDuckGo. The default cloud path uses Groq, which has a generous free tier and is fast enough for interactive use.

## Quick start

```bash
# 1. Install
pip install -r requirements.txt

# 2a. Easiest free path: Groq (sign up at console.groq.com, free tier)
export GROQ_API_KEY=gsk_...

# 2b. Fully local (no key, no internet for the LLM):
#     Install ollama from ollama.com, then:
ollama pull qwen2.5:7b
export LLM_PROVIDER=ollama

# 2c. Google Gemini free tier:
export GEMINI_API_KEY=...
export LLM_PROVIDER=gemini

# 3. Run
python -m agent.cli "What are the tradeoffs of using SQLite vs Postgres for a small SaaS app?"
```

## Web UI (Streamlit)

The same agent loop is available in a browser UI (query input, summary, citations, run stats, and trace).

```bash
streamlit run streamlit_app.py
```

In the UI, the default provider is **Gemini**. Set:

```bash
export GEMINI_API_KEY=...
```

## Project structure

```
research_agent/
├── agent/
│   ├── __init__.py
│   ├── agent.py        # The main tool-loop
│   ├── llm.py          # Provider-agnostic LLM wrapper
│   ├── tools.py        # search_web, fetch_page, take_note, finish
│   ├── memory.py       # Working memory + persistent cache
│   ├── context.py      # Token counting, summarization, trimming
│   ├── prompts.py      # System prompt + output schema
│   └── cli.py          # Command-line entry point
├── evals/
│   ├── golden.jsonl    # 20 hand-curated test queries
│   └── run_evals.py    # Runs the golden set, scores results
├── examples/
│   └── sample_run.md   # Example output
├── streamlit_app.py    # Streamlit web UI
├── requirements.txt
└── README.md
```

## What you'll learn by building this

1. **The agent loop is just a while-loop** with tool calls. Frameworks add structure but no magic.
2. **Context management is most of the work.** Search results blow up fast; you'll need summarization within ~3 iterations.
3. **Cost is dominated by failed paths.** Tracing every step is non-negotiable.
4. **Evals catch regressions you'll never see manually.** The golden set is the difference between a demo and a product.

See `evals/run_evals.py` for the full eval harness.
