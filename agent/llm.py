"""
Provider-agnostic LLM wrapper.

The agent loop only needs ONE method: chat(messages, tools) -> response.
Each provider's SDK is wrapped to expose the same interface, using a
near-OpenAI-style tool-calling format internally.

Free providers (default order):
  1. Groq      - free tier, very fast, Llama 3.3 70B
  2. Gemini    - free tier, Google
  3. Ollama    - fully local, no key, no rate limits

Paid providers (optional):
  4. Anthropic - Claude (best tool use)
  5. OpenAI    - GPT
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class ToolCall:
    """A tool the model wants to invoke."""
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMResponse:
    """Normalized response from any provider."""
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    # Token accounting - critical for Step 7 (cost-per-task)
    input_tokens: int = 0
    output_tokens: int = 0
    # Free providers report no cost; paid providers compute it
    cost_usd: float = 0.0
    raw: Any = None  # provider-native response, for debugging


# Approximate per-million-token prices (USD). Free tiers = 0.
# Update these when providers change pricing.
PRICING = {
    "groq": {"in": 0.0, "out": 0.0},          # Free tier
    "gemini": {"in": 0.0, "out": 0.0},        # Free tier (under quota)
    "ollama": {"in": 0.0, "out": 0.0},        # Local
    "anthropic-sonnet": {"in": 3.0, "out": 15.0},
    "openai-gpt4o-mini": {"in": 0.15, "out": 0.60},
}


def _calc_cost(provider: str, in_tok: int, out_tok: int) -> float:
    p = PRICING.get(provider, {"in": 0.0, "out": 0.0})
    return (in_tok * p["in"] + out_tok * p["out"]) / 1_000_000


# Gemini's schema validator is stricter than OpenAI's. It rejects JSON Schema
# fields that don't map to its internal Schema proto, including: default,
# additionalProperties, $schema, examples, title, and others. We strip them
# recursively before sending tool definitions to Gemini.
_GEMINI_SCHEMA_DROP_KEYS = {
    "default", "additionalProperties", "$schema", "examples", "title",
    "$ref", "definitions", "$defs", "patternProperties", "const",
    "exclusiveMinimum", "exclusiveMaximum", "multipleOf", "minProperties",
    "maxProperties", "uniqueItems",
}


def _sanitize_schema_for_gemini(schema):
    """Recursively strip JSON Schema fields Gemini rejects."""
    if isinstance(schema, dict):
        return {
            k: _sanitize_schema_for_gemini(v)
            for k, v in schema.items()
            if k not in _GEMINI_SCHEMA_DROP_KEYS
        }
    if isinstance(schema, list):
        return [_sanitize_schema_for_gemini(item) for item in schema]
    return schema


class LLM:
    """Unified LLM interface. Pick provider from env or constructor."""

    def __init__(
        self,
        provider: str | None = None,
        model: str | None = None,
    ):
        self.provider = provider or os.getenv("LLM_PROVIDER", "groq")
        self.model = model or self._default_model()
        self._client = self._build_client()

    def _default_model(self) -> str:
        defaults = {
            "groq": "llama-3.3-70b-versatile",
            # gemini-2.0-flash was deprecated Feb 2026, retired March 2026.
            # Current free-tier models: 2.5-pro (5 RPM), 2.5-flash (10 RPM),
            # 2.5-flash-lite (15 RPM, 1000 RPD). 2.5-flash is the best default.
            "gemini": "gemini-2.5-flash",
            "ollama": "qwen2.5:7b",
            "anthropic": "claude-sonnet-4-5",
            "openai": "gpt-4o-mini",
        }
        return os.getenv("LLM_MODEL") or defaults[self.provider]

    def _build_client(self):
        if self.provider == "groq":
            from groq import Groq
            return Groq(api_key=os.environ["GROQ_API_KEY"])
        if self.provider == "gemini":
            # New SDK: google-genai (note the dash). Replaces deprecated google-generativeai.
            from google import genai
            return genai.Client(api_key=os.environ.get("GEMINI_API_KEY") or os.environ["GOOGLE_API_KEY"])
        if self.provider == "ollama":
            import ollama
            return ollama.Client(host=os.getenv("OLLAMA_HOST", "http://localhost:11434"))
        if self.provider == "anthropic":
            from anthropic import Anthropic
            return Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        if self.provider == "openai":
            from openai import OpenAI
            return OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        raise ValueError(f"Unknown provider: {self.provider}")

    # ---------- public API ----------

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        """Send messages, optionally with tools. Returns normalized response."""
        if self.provider in ("groq", "openai"):
            return self._chat_openai_style(messages, tools, temperature, max_tokens)
        if self.provider == "anthropic":
            return self._chat_anthropic(messages, tools, temperature, max_tokens)
        if self.provider == "gemini":
            return self._chat_gemini(messages, tools, temperature, max_tokens)
        if self.provider == "ollama":
            return self._chat_ollama(messages, tools, temperature, max_tokens)
        raise ValueError(self.provider)

    # ---------- provider implementations ----------

    def _chat_openai_style(self, messages, tools, temperature, max_tokens) -> LLMResponse:
        """Groq and OpenAI share the same chat-completions format."""
        kwargs = dict(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if tools:
            kwargs["tools"] = [{"type": "function", "function": t} for t in tools]
            kwargs["tool_choice"] = "auto"

        resp = self._client.chat.completions.create(**kwargs)
        msg = resp.choices[0].message

        tool_calls = []
        for tc in (msg.tool_calls or []):
            tool_calls.append(ToolCall(
                id=tc.id,
                name=tc.function.name,
                arguments=json.loads(tc.function.arguments or "{}"),
            ))

        provider_key = "groq" if self.provider == "groq" else "openai-gpt4o-mini"
        return LLMResponse(
            text=msg.content or "",
            tool_calls=tool_calls,
            input_tokens=resp.usage.prompt_tokens,
            output_tokens=resp.usage.completion_tokens,
            cost_usd=_calc_cost(provider_key, resp.usage.prompt_tokens, resp.usage.completion_tokens),
            raw=resp,
        )

    def _chat_anthropic(self, messages, tools, temperature, max_tokens) -> LLMResponse:
        """Anthropic has a slightly different format - system separate, content blocks."""
        # Pull system message out
        system_msg = ""
        clean_messages = []
        for m in messages:
            if m["role"] == "system":
                system_msg = m["content"]
            else:
                clean_messages.append(self._convert_to_anthropic(m))

        kwargs = dict(
            model=self.model,
            system=system_msg,
            messages=clean_messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if tools:
            kwargs["tools"] = [
                {
                    "name": t["name"],
                    "description": t["description"],
                    "input_schema": t["parameters"],
                }
                for t in tools
            ]

        resp = self._client.messages.create(**kwargs)

        text_parts = []
        tool_calls = []
        for block in resp.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(
                    id=block.id,
                    name=block.name,
                    arguments=block.input,
                ))

        return LLMResponse(
            text="".join(text_parts),
            tool_calls=tool_calls,
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
            cost_usd=_calc_cost("anthropic-sonnet", resp.usage.input_tokens, resp.usage.output_tokens),
            raw=resp,
        )

    def _convert_to_anthropic(self, msg: dict) -> dict:
        """Translate OpenAI-style tool result messages to Anthropic format."""
        if msg["role"] == "tool":
            return {
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": msg["tool_call_id"],
                    "content": msg["content"],
                }],
            }
        if msg["role"] == "assistant" and msg.get("tool_calls"):
            content = []
            if msg.get("content"):
                content.append({"type": "text", "text": msg["content"]})
            for tc in msg["tool_calls"]:
                content.append({
                    "type": "tool_use",
                    "id": tc["id"],
                    "name": tc["function"]["name"],
                    "input": json.loads(tc["function"]["arguments"]),
                })
            return {"role": "assistant", "content": content}
        return msg

    def _chat_gemini(self, messages, tools, temperature, max_tokens) -> LLMResponse:
        """
        Gemini via the new google-genai SDK.

        Key API differences from the deprecated google-generativeai:
          - Single Client object, not a configured module
          - generate_content takes contents=, config=, model= explicitly
          - Tools and system instruction go inside types.GenerateContentConfig
          - FunctionDeclaration uses parameters_json_schema= not parameters=
          - Roles: user / model (not user / assistant)
        """
        from google.genai import types as gtypes

        # Pull out system instruction; Gemini handles it separately
        system_text = ""
        history = []
        for m in messages:
            if m["role"] == "system":
                system_text = m["content"]
            elif m["role"] == "user":
                history.append(gtypes.Content(role="user", parts=[gtypes.Part.from_text(text=m["content"])]))
            elif m["role"] == "assistant":
                # If there are tool_calls, those become function_call parts; text is a text part
                parts = []
                if m.get("content"):
                    parts.append(gtypes.Part.from_text(text=m["content"]))
                for tc in (m.get("tool_calls") or []):
                    args = tc["function"]["arguments"]
                    if isinstance(args, str):
                        args = json.loads(args) if args else {}
                    parts.append(gtypes.Part.from_function_call(
                        name=tc["function"]["name"],
                        args=args,
                    ))
                if parts:
                    history.append(gtypes.Content(role="model", parts=parts))
            elif m["role"] == "tool":
                # Tool result -> function_response part with role=user
                history.append(gtypes.Content(
                    role="user",
                    parts=[gtypes.Part.from_function_response(
                        name=m.get("name", "tool"),
                        response={"result": m["content"]},
                    )],
                ))

        # Build tool declarations using sanitized schemas
        gemini_tools = None
        if tools:
            declarations = [
                gtypes.FunctionDeclaration(
                    name=t["name"],
                    description=t["description"],
                    parameters_json_schema=_sanitize_schema_for_gemini(t["parameters"]),
                )
                for t in tools
            ]
            gemini_tools = [gtypes.Tool(function_declarations=declarations)]

        # Disable "thinking" for 2.5 Flash/Flash-Lite. Thinking tokens are
        # counted against max_output_tokens, so an enabled thinking budget can
        # eat the entire output, leaving the model with nothing left to emit
        # (empty response, 0 tool calls). 2.5 models default thinking to ON.
        # We don't need internal reasoning for tool-calling — the agent loop
        # IS the reasoning. thinking_budget=0 applies to 2.5 Flash/Flash-Lite;
        # 2.5 Pro doesn't allow disabling thinking but accepts the field.
        thinking_config = None
        if "2.5-flash" in self.model or "2.5-pro" in self.model or "3-flash" in self.model:
            try:
                thinking_config = gtypes.ThinkingConfig(thinking_budget=0)
            except Exception:
                # Older SDK versions may not have ThinkingConfig
                thinking_config = None

        config = gtypes.GenerateContentConfig(
            system_instruction=system_text or None,
            temperature=temperature,
            max_output_tokens=max_tokens,
            tools=gemini_tools,
            thinking_config=thinking_config,
            # Disable automatic function calling - we run the loop ourselves
            automatic_function_calling=gtypes.AutomaticFunctionCallingConfig(disable=True),
        )

        resp = self._client.models.generate_content(
            model=self.model,
            contents=history,
            config=config,
        )

        text_parts = []
        tool_calls = []
        # Walk content parts to extract text and function_call blocks
        for cand in (resp.candidates or []):
            if not cand.content or not cand.content.parts:
                continue
            for part in cand.content.parts:
                if getattr(part, "text", None):
                    text_parts.append(part.text)
                if getattr(part, "function_call", None) and part.function_call.name:
                    fc = part.function_call
                    args = dict(fc.args) if fc.args else {}
                    tool_calls.append(ToolCall(
                        id=f"call_{fc.name}_{len(tool_calls)}",
                        name=fc.name,
                        arguments=args,
                    ))

        usage = resp.usage_metadata
        in_tok = getattr(usage, "prompt_token_count", 0) or 0
        out_tok = getattr(usage, "candidates_token_count", 0) or 0

        return LLMResponse(
            text="".join(text_parts),
            tool_calls=tool_calls,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cost_usd=0.0,
            raw=resp,
        )

    def _chat_ollama(self, messages, tools, temperature, max_tokens) -> LLMResponse:
        """Ollama supports OpenAI-style tool calls in recent versions."""
        kwargs = dict(
            model=self.model,
            messages=messages,
            options={"temperature": temperature, "num_predict": max_tokens},
        )
        if tools:
            kwargs["tools"] = [{"type": "function", "function": t} for t in tools]

        resp = self._client.chat(**kwargs)
        msg = resp["message"]

        tool_calls = []
        for tc in msg.get("tool_calls", []) or []:
            args = tc["function"]["arguments"]
            if isinstance(args, str):
                args = json.loads(args)
            tool_calls.append(ToolCall(
                id=f"call_{tc['function']['name']}",
                name=tc["function"]["name"],
                arguments=args,
            ))

        return LLMResponse(
            text=msg.get("content", ""),
            tool_calls=tool_calls,
            input_tokens=resp.get("prompt_eval_count", 0),
            output_tokens=resp.get("eval_count", 0),
            cost_usd=0.0,
            raw=resp,
        )
