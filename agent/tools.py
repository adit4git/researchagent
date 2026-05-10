"""
Agent tools (Step 4 of the framework).

We use a minimal but realistic tool set:
  - search_web(query)      : DuckDuckGo, no API key
  - fetch_page(url)        : httpx + readability for clean text
  - take_note(claim, ...)  : structured fact capture (the agent's notepad)
  - finish(summary, ...)   : terminate the loop with a final answer

Each tool has:
  - A JSON schema (sent to the model)
  - A Python implementation
  - Error handling that returns a string the model can read

Design principle: tools should be hard to misuse. We validate inputs,
truncate big outputs, and always return a string (never raise to the loop).
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Callable
from urllib.parse import urlparse

import httpx
from ddgs import DDGS
from readability import Document
from bs4 import BeautifulSoup


# Hard caps prevent context blow-up (Step 6: manage context)
# Tuned for free-tier providers (Groq gpt-oss-120b ~8k context per request).
# Single page fetches that exceed this get truncated.
MAX_PAGE_CHARS = 4500
MAX_SEARCH_RESULTS = 5
FETCH_TIMEOUT = 15.0


# ---------- Tool schemas (sent to the LLM) ----------

TOOL_SCHEMAS: list[dict] = [
    {
        "name": "search_web",
        "description": (
            "Search the web for sources on a topic. Returns a list of "
            "results with title, url, and snippet. Call this 1-3 times "
            "with different queries to get good coverage. Don't search "
            "for the same thing twice."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query, 3-10 words. Be specific.",
                },
                "max_results": {
                    "type": "integer",
                    "description": f"How many results (1-{MAX_SEARCH_RESULTS}). Default 5.",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "fetch_page",
        "description": (
            "Fetch and clean the main text content of a web page. Use "
            "this on URLs from search_web that look promising. Returns "
            "up to ~8000 chars of cleaned article text. Don't fetch the "
            "same URL twice."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "Full URL including https://"},
            },
            "required": ["url"],
        },
    },
    {
        "name": "take_note",
        "description": (
            "Record a factual claim that DIRECTLY answers the research topic. "
            "Use this AS YOU READ pages. Each note must be: (a) a single "
            "self-contained fact, (b) relevant to the topic — not just true, "
            "(c) ideally a comparison, tradeoff, or specific recommendation "
            "rather than a generic description. The final summary is built "
            "from these notes, so noise here = noise there."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "claim": {
                    "type": "string",
                    "description": (
                        "A single factual claim, in your own words, under 40 words. "
                        "MUST be directly relevant to the research topic. "
                        "If the page is about something else (e.g. AWS when topic "
                        "is Azure vs GCP), DO NOT take a note — fetch a different page instead."
                    ),
                },
                "relevance_to_topic": {
                    "type": "string",
                    "description": (
                        "One sentence: how does this claim help answer the research topic? "
                        "If you can't answer this clearly, the note isn't worth taking."
                    ),
                },
                "source_url": {
                    "type": "string",
                    "description": "URL of the page that supports this claim.",
                },
                "confidence": {
                    "type": "string",
                    "enum": ["high", "medium", "low"],
                    "description": "How confident you are in this claim based on the source.",
                },
            },
            "required": ["claim", "relevance_to_topic", "source_url", "confidence"],
        },
    },
    {
        "name": "finish",
        "description": (
            "Terminate research and produce the final brief. Only call this "
            "when you have enough notes (at least 5 high/medium-confidence "
            "claims from at least 3 distinct sources)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "A 200-400 word structured brief synthesizing the notes. Use [1], [2] inline citations matching the order of source_urls.",
                },
                "key_findings": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "3-7 bullet-point key findings.",
                },
                "source_urls": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Ordered list of source URLs used (matches [1], [2] in summary).",
                },
                "open_questions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Things you couldn't confidently answer. Be honest.",
                },
            },
            "required": ["summary", "key_findings", "source_urls"],
        },
    },
]


# ---------- Tool implementations ----------

def search_web(query: str, max_results: int = 5, **_) -> str:
    """DuckDuckGo search. Free, no API key."""
    max_results = min(max_results, MAX_SEARCH_RESULTS)
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        if not results:
            return "No results found. Try a different query."
        out = []
        for i, r in enumerate(results, 1):
            out.append(f"[{i}] {r.get('title', 'Untitled')}\n    URL: {r.get('href', '')}\n    {r.get('body', '')[:200]}")
        return "\n\n".join(out)
    except Exception as e:
        return f"Search failed: {type(e).__name__}: {e}"


def _looks_like_garbage(text: str) -> bool:
    """
    Detect mojibake / binary / replacement-char-flooded text.

    Why: some pages return content in encodings that survive httpx/readability
    parsing as a stream of U+FFFD replacement characters or non-printable bytes.
    Feeding this to an LLM causes it to echo the garbage and burn output tokens
    without making tool calls (we saw this in the wild).
    """
    if not text or len(text) < 50:
        return True

    # Sample to keep this fast on large pages
    sample = text[:3000]

    # Replacement char flood
    repl_count = sample.count("\ufffd")
    if repl_count > 50 or repl_count / max(len(sample), 1) > 0.05:
        return True

    # Non-printable / control char flood (excluding newlines/tabs)
    weird = sum(
        1 for c in sample
        if not c.isprintable() and c not in "\n\r\t "
    )
    if weird / max(len(sample), 1) > 0.05:
        return True

    # Very low whitespace fraction = probably not natural text
    spaces = sample.count(" ") + sample.count("\n")
    if spaces / max(len(sample), 1) < 0.05:
        return True

    return False


def fetch_page(url: str, **_) -> str:
    """Fetch a page and extract main article text."""
    # Validate URL
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return f"Invalid URL scheme: {url}"
    if not parsed.netloc:
        return f"Invalid URL: {url}"

    try:
        with httpx.Client(
            timeout=FETCH_TIMEOUT,
            follow_redirects=True,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/121.0.0.0 Safari/537.36"
                ),
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
                "Accept-Encoding": "gzip, deflate, br",
            },
        ) as client:
            resp = client.get(url)
            resp.raise_for_status()
    except httpx.HTTPStatusError as e:
        return f"HTTP {e.response.status_code} fetching {url}"
    except Exception as e:
        return f"Fetch failed: {type(e).__name__}: {e}"

    # Reject non-HTML content types early
    content_type = resp.headers.get("content-type", "").lower()
    if content_type and not any(t in content_type for t in ("html", "xml", "text/plain")):
        return (
            f"Skipped {url}: content-type is {content_type!r}, not HTML. "
            f"Pick a different URL."
        )

    # Try readability first - it strips nav, ads, footers
    try:
        doc = Document(resp.text)
        title = doc.title()
        cleaned_html = doc.summary()
        text = BeautifulSoup(cleaned_html, "html.parser").get_text(separator="\n", strip=True)
    except Exception:
        # Fallback: dumb text extraction
        text = BeautifulSoup(resp.text, "html.parser").get_text(separator="\n", strip=True)
        title = url

    # Reject mojibake / binary garbage
    if _looks_like_garbage(text):
        return (
            f"Skipped {url}: extracted content appears to be binary, "
            f"non-text, or has encoding errors. Pick a different URL."
        )

    # Truncate to keep context manageable
    if len(text) > MAX_PAGE_CHARS:
        text = text[:MAX_PAGE_CHARS] + f"\n\n[... truncated, full page was {len(text)} chars ...]"

    return f"# {title}\nURL: {url}\n\n{text}"


def take_note(claim: str, source_url: str, confidence: str, notes_store: list,
              relevance_to_topic: str = "", **_) -> str:
    """Record a structured note. notes_store is injected by the agent."""
    if confidence not in ("high", "medium", "low"):
        confidence = "medium"
    note = {
        "claim": claim.strip(),
        "source_url": source_url.strip(),
        "confidence": confidence,
        "relevance_to_topic": relevance_to_topic.strip(),
        "ts": time.time(),
    }
    notes_store.append(note)
    return f"Note #{len(notes_store)} recorded ({confidence}). You have {len(notes_store)} notes total."


def finish(**kwargs) -> str:
    """Marker - the agent loop intercepts this and stops."""
    return "Finishing."


# ---------- Dispatcher ----------

@dataclass
class ToolResult:
    name: str
    output: str
    is_terminal: bool = False
    finish_payload: dict | None = None


def execute_tool(name: str, args: dict, notes_store: list) -> ToolResult:
    """Dispatch a tool call. Always returns a string output for the model."""
    try:
        if name == "search_web":
            return ToolResult(name, search_web(**args))
        if name == "fetch_page":
            return ToolResult(name, fetch_page(**args))
        if name == "take_note":
            return ToolResult(name, take_note(notes_store=notes_store, **args))
        if name == "finish":
            return ToolResult(name, "Finished.", is_terminal=True, finish_payload=args)
        return ToolResult(name, f"Unknown tool: {name}")
    except TypeError as e:
        # Bad arguments - tell the model so it can retry
        return ToolResult(name, f"Bad arguments for {name}: {e}")
    except Exception as e:
        return ToolResult(name, f"Tool {name} crashed: {type(e).__name__}: {e}")
