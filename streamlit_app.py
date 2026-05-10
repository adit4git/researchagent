"""
Streamlit UI for the research agent.

Run:
  streamlit run streamlit_app.py
"""
from __future__ import annotations

import json
import os
import traceback
from typing import Any

import streamlit as st

from agent.agent import AgentConfig, ResearchAgent
from agent.llm import LLM
from agent.prompts import ResearchGoal


PROVIDERS = ["gemini", "groq", "ollama", "anthropic", "openai"]
DEPTH_OPTIONS = ["quick", "standard", "deep"]


def _init_state() -> None:
    if "runs" not in st.session_state:
        st.session_state.runs = []


def _render_trace(trace: list[dict[str, Any]]) -> None:
    if not trace:
        st.write("No trace captured.")
        return
    for step in trace:
        kind = step.get("kind", "unknown")
        payload = {k: v for k, v in step.items() if k != "kind"}
        preview = json.dumps(payload, default=str)[:200]
        st.code(f"[{kind}] {preview}", language="text")


def _render_run(run: dict[str, Any]) -> None:
    with st.chat_message("user"):
        st.write(run["topic"])

    with st.chat_message("assistant"):
        if run.get("error"):
            st.error(run["error"])
            if run.get("traceback"):
                with st.expander("Error traceback"):
                    st.code(run["traceback"], language="text")
            return

        brief = run["brief"]
        st.markdown(brief["summary"])

        if brief["key_findings"]:
            st.subheader("Key findings")
            for finding in brief["key_findings"]:
                st.write(f"- {finding}")

        if brief["source_urls"]:
            st.subheader("Sources")
            for i, url in enumerate(brief["source_urls"], 1):
                st.markdown(f"{i}. [{url}]({url})")

        if brief["open_questions"]:
            st.subheader("Open questions")
            for question in brief["open_questions"]:
                st.write(f"- {question}")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Iterations", brief["iterations_used"])
        c2.metric("Notes", brief["notes_count"])
        c3.metric("Sources", len(brief["source_urls"]))
        c4.metric("Cost (USD)", f"{brief['total_cost_usd']:.4f}")

        with st.expander("Trace", expanded=False):
            _render_trace(run.get("trace", []))

        st.download_button(
            label="Download JSON",
            data=json.dumps(
                {
                    "topic": run["topic"],
                    "provider": run["provider"],
                    "model": run["model"],
                    "brief": brief,
                    "trace": run.get("trace", []),
                },
                indent=2,
                default=str,
            ),
            file_name="research_run.json",
            mime="application/json",
            key=f"download_{run['id']}",
        )


def _run_agent(
    topic: str,
    depth: str,
    max_iter: int,
    words: int,
    provider: str,
    model_override: str,
    use_cache: bool,
) -> dict[str, Any]:
    llm = None
    try:
        llm = LLM(provider=provider, model=model_override or None)
        goal = ResearchGoal(
            topic=topic,
            depth=depth,
            max_iterations=max_iter,
            target_word_count=words,
        )
        config = AgentConfig(max_iterations=max_iter, use_cache=use_cache)
    except Exception as exc:
        return {
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "provider": provider,
            "model": llm.model if llm else (model_override or "(default)"),
            "brief": None,
            "trace": [],
        }

    status = st.status(
        f"Researching with {llm.provider}/{llm.model}...",
        expanded=True,
    )
    try:
        with status:
            agent = ResearchAgent(llm=llm, config=config)
            brief = agent.run(goal)
        status.update(label="Research complete.", state="complete", expanded=False)
        return {
            "error": None,
            "traceback": None,
            "provider": llm.provider,
            "model": llm.model,
            "brief": brief.model_dump(),
            "trace": getattr(brief, "_trace", []) or [],
        }
    except Exception as exc:
        status.update(label="Research failed.", state="error", expanded=True)
        return {
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "provider": llm.provider,
            "model": llm.model,
            "brief": None,
            "trace": [],
        }


def main() -> None:
    st.set_page_config(page_title="Research Agent", layout="wide")
    _init_state()

    st.title("Research Agent Web UI")
    st.caption("Same agent loop as CLI, with summary output and full execution trace.")

    with st.sidebar:
        st.header("Settings")
        provider = st.selectbox("Provider", PROVIDERS, index=0)
        model_override = st.text_input(
            "Model override (optional)",
            placeholder="Leave blank for provider default",
        ).strip()
        depth = st.selectbox("Depth", DEPTH_OPTIONS, index=1)
        max_iter = st.slider("Max iterations", min_value=3, max_value=30, value=12, step=1)
        words = st.slider("Target summary words", min_value=100, max_value=1500, value=300, step=50)
        use_cache = st.checkbox("Use page cache", value=True)

        if provider == "gemini" and not (
            os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        ):
            st.warning("Set GEMINI_API_KEY (or GOOGLE_API_KEY) before running Gemini.")

        if st.button("Clear run history"):
            st.session_state.runs = []
            st.rerun()

    for run in st.session_state.runs:
        _render_run(run)

    prompt = st.chat_input("Type a research topic/question")
    if prompt:
        result = _run_agent(
            topic=prompt,
            depth=depth,
            max_iter=max_iter,
            words=words,
            provider=provider,
            model_override=model_override,
            use_cache=use_cache,
        )
        run = {
            "id": len(st.session_state.runs),
            "topic": prompt,
            "provider": result["provider"],
            "model": result["model"],
            "error": result["error"],
            "traceback": result["traceback"],
            "brief": result["brief"],
            "trace": result["trace"],
        }
        st.session_state.runs.append(run)
        st.rerun()


if __name__ == "__main__":
    main()
