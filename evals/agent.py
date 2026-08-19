"""The agent under test: drives one scenario against a tool surface.

`Agent` is a protocol so the harness can run with a real model or with a scripted
stand-in. The scripted one exists so the runner, metrics and report can be tested
without credentials and without spending tokens — it is a test double, never a
substitute for a real measurement.

Two OpenAI loops live here. `ResponsesAgent` is the default, because
`/v1/responses` is the only endpoint that takes function tools together with
reasoning. `ChatCompletionsAgent` stays for OpenAI-compatible gateways, which
implement Chat Completions almost universally and `/v1/responses` rarely — one
`OPENAI_BASE_URL` plus `--api chat` points the same harness at them.
"""

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from .config import DEFAULT_MODEL
from .transcript import ToolCallRecord, Transcript

# The eval harness measures how well an agent uses the tool surface, so the system
# prompt stays minimal: everything about HOW to use the tools must come from the
# tool descriptions and the server's own instructions, which is exactly what the
# baseline-vs-proxy comparison is about. Putting guidance here would mask the
# difference we are trying to measure.
SYSTEM_PROMPT = (
    "Ты — ассистент по поиску путешествий. Отвечай пользователю по-русски, опираясь "
    "ТОЛЬКО на данные, которые вернули инструменты. Если нужного поля в ответе нет — "
    "прямо скажи, что сервис его не вернул, и не подставляй общие знания."
)

MAX_TURNS = 12


class ToolExecutor(Protocol):
    async def __call__(self, name: str, arguments: dict[str, Any]) -> ToolCallRecord: ...


class Agent(Protocol):
    @property
    def label(self) -> str: ...

    async def run(
        self,
        *,
        scenario_id: str,
        variant: str,
        request: str,
        tools: list[dict[str, Any]],
        server_instructions: str,
        execute: ToolExecutor,
    ) -> Transcript: ...


def to_openai_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Map our upstream-shaped tool dicts onto the Chat Completions tool schema."""
    converted = []
    for tool in tools:
        converted.append(
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get("inputSchema") or {"type": "object", "properties": {}},
                },
            }
        )
    return converted


def to_responses_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Same tools, Responses API shape — flat, with no nested `function` object."""
    return [
        {
            "type": "function",
            "name": tool["name"],
            "description": tool.get("description", ""),
            "parameters": tool.get("inputSchema") or {"type": "object", "properties": {}},
        }
        for tool in tools
    ]


def build_system_prompt(server_instructions: str) -> str:
    if not server_instructions:
        return SYSTEM_PROMPT
    return f"{SYSTEM_PROMPT}\n\n--- Инструкции MCP-сервера ---\n{server_instructions}"


def make_client(api_key: str | None = None):
    try:
        from openai import AsyncOpenAI
    except ImportError as exc:  # pragma: no cover - depends on optional install
        raise RuntimeError("пакет `openai` не установлен — добавьте его: `uv add openai`") from exc
    # Zero-arg construction reads OPENAI_API_KEY / OPENAI_BASE_URL from the env,
    # which app.config has already populated from .env.
    return AsyncOpenAI(api_key=api_key) if api_key else AsyncOpenAI()


@dataclass
class ChatCompletionsAgent:
    """Agent loop over Chat Completions.

    Kept for OpenAI-compatible gateways, which almost all implement this endpoint
    and often not `/v1/responses`. On OpenAI itself prefer `ResponsesAgent`:
    reasoning models reject function tools here unless reasoning is switched off,
    and running a reasoning model with reasoning off is not the model you meant
    to measure.

    `reasoning_effort` is sent only when set, because non-reasoning models reject
    it. Sampling params (temperature/top_p) are never sent: reasoning models reject
    them, and for an eval we want the model's own default behavior anyway.
    """

    model: str = DEFAULT_MODEL
    effort: str | None = None
    max_turns: int = MAX_TURNS
    max_completion_tokens: int = 16000

    @property
    def label(self) -> str:
        return f"{self.model}/{self.effort}" if self.effort else self.model

    def _request_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"max_completion_tokens": self.max_completion_tokens}
        if self.effort:
            kwargs["reasoning_effort"] = self.effort
        return kwargs

    async def run(
        self,
        *,
        scenario_id: str,
        variant: str,
        request: str,
        tools: list[dict[str, Any]],
        server_instructions: str,
        execute: ToolExecutor,
    ) -> Transcript:
        client = make_client()
        transcript = Transcript(scenario_id=scenario_id, variant=variant)
        started = time.monotonic()

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": build_system_prompt(server_instructions)},
            {"role": "user", "content": request},
        ]
        api_tools = to_openai_tools(tools)

        try:
            for _ in range(self.max_turns):
                response = await client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=api_tools,
                    **self._request_kwargs(),
                )
                transcript.turns += 1
                if response.usage:
                    transcript.input_tokens += response.usage.prompt_tokens
                    transcript.output_tokens += response.usage.completion_tokens

                choice = response.choices[0]
                message = choice.message

                if getattr(message, "refusal", None):
                    transcript.failure = f"refusal:{message.refusal}"
                    break

                if not message.tool_calls:
                    transcript.answer_text = message.content or ""
                    if choice.finish_reason == "length":
                        transcript.failure = "max_completion_tokens"
                    break

                messages.append(message.model_dump(exclude_none=True))

                for call in message.tool_calls:
                    # Arguments arrive as a JSON *string*; never string-match on it.
                    try:
                        arguments = json.loads(call.function.arguments or "{}")
                    except json.JSONDecodeError:
                        arguments = {}
                    record = await execute(call.function.name, arguments)
                    transcript.tool_calls.append(record)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call.id,
                            "content": record.result_text,
                        }
                    )
            else:
                transcript.failure = f"turn_cap:{self.max_turns}"
        except Exception as exc:  # one bad scenario must not kill the whole run
            transcript.failure = f"{type(exc).__name__}: {exc}"

        transcript.duration_s = time.monotonic() - started
        return transcript


@dataclass
class ResponsesAgent:
    """Agent loop over the Responses API — the default on OpenAI itself.

    Chat Completions refuses function tools together with reasoning on the current
    reasoning models ("use /v1/responses or set reasoning_effort to 'none'"), and
    the second option would quietly measure a reasoning model with its reasoning
    switched off. This endpoint takes both, so the model under test is the model
    the numbers get attributed to.

    History is threaded explicitly instead of via `previous_response_id`, and
    `store=False`, so an eval run leaves nothing behind server-side and each
    scenario is reproducible from its own transcript.
    """

    model: str = DEFAULT_MODEL
    effort: str | None = None
    max_turns: int = MAX_TURNS
    max_output_tokens: int = 16000

    @property
    def label(self) -> str:
        return f"{self.model}/{self.effort}" if self.effort else self.model

    def _request_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "max_output_tokens": self.max_output_tokens,
            "store": False,
        }
        if self.effort:
            kwargs["reasoning"] = {"effort": self.effort}
        return kwargs

    async def run(
        self,
        *,
        scenario_id: str,
        variant: str,
        request: str,
        tools: list[dict[str, Any]],
        server_instructions: str,
        execute: ToolExecutor,
    ) -> Transcript:
        client = make_client()
        transcript = Transcript(scenario_id=scenario_id, variant=variant)
        started = time.monotonic()

        conversation: list[Any] = [{"role": "user", "content": request}]
        api_tools = to_responses_tools(tools)

        try:
            for _ in range(self.max_turns):
                response = await client.responses.create(
                    model=self.model,
                    instructions=build_system_prompt(server_instructions),
                    input=conversation,
                    tools=api_tools,
                    **self._request_kwargs(),
                )
                transcript.turns += 1
                if response.usage:
                    transcript.input_tokens += response.usage.input_tokens
                    transcript.output_tokens += response.usage.output_tokens

                calls = [item for item in response.output if item.type == "function_call"]

                if not calls:
                    transcript.answer_text = response.output_text or ""
                    if response.status == "incomplete":
                        reason = getattr(response.incomplete_details, "reason", "incomplete")
                        transcript.failure = f"incomplete:{reason}"
                    break

                # Reasoning items must be echoed back alongside the calls, or the
                # model loses the thread it was in the middle of.
                conversation += [item.model_dump(exclude_none=True) for item in response.output]

                for call in calls:
                    try:
                        arguments = json.loads(call.arguments or "{}")
                    except json.JSONDecodeError:
                        arguments = {}
                    record = await execute(call.name, arguments)
                    transcript.tool_calls.append(record)
                    conversation.append(
                        {
                            "type": "function_call_output",
                            "call_id": call.call_id,
                            "output": record.result_text,
                        }
                    )
            else:
                transcript.failure = f"turn_cap:{self.max_turns}"
        except Exception as exc:  # one bad scenario must not kill the whole run
            transcript.failure = f"{type(exc).__name__}: {exc}"

        transcript.duration_s = time.monotonic() - started
        return transcript


Plan = tuple[list[tuple[str, dict[str, Any]]], str]


def by_scenario(scenario_id: str, variant: str) -> Any:
    return scenario_id


def by_scenario_and_variant(scenario_id: str, variant: str) -> Any:
    return (scenario_id, variant)


@dataclass
class ScriptedAgent:
    """Test double: replays a fixed plan so the harness itself can be tested offline.

    `plan` maps a key -> (tool calls to make, answer text). A test can build an
    answer from real fixture values to produce a genuinely grounded answer, or from
    invented ones to produce a genuinely fabricated one.

    `key` picks what the plan is keyed on. `by_scenario` is the test default;
    `evals/demo.py` passes `by_scenario_and_variant`, so one scenario can show a
    good answer on one surface and a bad one on the other — which is the whole
    point of a side-by-side trace viewer. That single difference used to justify a
    second class with a byte-for-byte copy of the loop below.

    `tokens` supplies synthetic (input, output) counts per variant for hand-written
    traces, where no model ran and the report still needs a plausible column. Left
    unset, the counts stay zero — a scripted run must not look like a measurement.
    """

    plan: dict[Any, Plan]
    label_: str = "scripted"
    key: Callable[[str, str], Any] = by_scenario
    tokens: Callable[[str], tuple[int, int]] | None = None

    @property
    def label(self) -> str:
        return self.label_

    async def run(
        self,
        *,
        scenario_id: str,
        variant: str,
        request: str,
        tools: list[dict[str, Any]],
        server_instructions: str,
        execute: ToolExecutor,
    ) -> Transcript:
        transcript = Transcript(scenario_id=scenario_id, variant=variant)
        started = time.monotonic()

        calls, answer = self.plan.get(self.key(scenario_id, variant), ([], ""))
        for name, arguments in calls:
            record = await execute(name, dict(arguments))
            transcript.tool_calls.append(record)
        transcript.turns = len(calls) + 1
        transcript.answer_text = answer
        if self.tokens is not None:
            transcript.input_tokens, transcript.output_tokens = self.tokens(variant)
        transcript.duration_s = time.monotonic() - started
        return transcript
