"""Cost of the tool surface itself — the headline number.

This is what an agent pays on every single session before it has searched
anything: the `tools/list` catalog plus the server's always-on instructions,
rendered into the request. Tutu's article measured *response* compression; this
measures the part that is paid whether or not any tool is ever called.

OpenAI has no token-counting endpoint, so "exact" here means: issue one real
minimal request carrying the full tool surface and read `usage.prompt_tokens`
back. That is the only way to capture the provider's own tool-serialization
overhead rather than guessing at it. It costs one prompt's worth of input tokens
per variant and nothing else — the completion is capped at a single token.

The offline counter uses tiktoken over the same payload. Better than a
chars-per-token guess because it is the real BPE for OpenAI models, but still an
estimate: it cannot know how the API frames tool definitions internally. Every
estimated figure is labelled `~` wherever it surfaces, because a precise-looking
number nobody can back up is worse than an honest approximation.
"""

import json
from dataclasses import dataclass
from typing import Any, Protocol

from .agent import (
    build_system_prompt,
    make_client,
    to_openai_tools,
    to_responses_tools,
)
from .config import DEFAULT_MODEL
from .options import Api

PROBE_MESSAGE = "привет"

# Fallback for when tiktoken cannot load its BPE data (it fetches on first use and
# caches; an offline machine with a cold cache gets nothing). Mixed Russian/JSON
# sits between Cyrillic's ~2-3 chars/token and JSON/English's ~4.
HEURISTIC_CHARS_PER_TOKEN = 3.2
FALLBACK_ENCODING = "o200k_base"


@dataclass(frozen=True)
class SurfaceCost:
    variant: str
    tokens: int
    bytes_: int
    exact: bool

    @property
    def label(self) -> str:
        return f"{self.tokens}" if self.exact else f"~{self.tokens}"


class TokenCounter(Protocol):
    @property
    def exact(self) -> bool: ...

    @property
    def estimate_reason(self) -> str | None:
        """Why this run is estimating, for the report to say out loud. `None` on an
        exact counter. A bare "неточно" sent readers hunting for a missing key even
        when the key was there and a flag had asked for the estimate."""
        ...

    async def count(self, *, tools: list[dict[str, Any]], system: str) -> int: ...


def render_tools(tools: list[dict[str, Any]], api: Api) -> list[dict[str, Any]]:
    """Tool definitions exactly as the chosen endpoint receives them. The two
    endpoints nest them differently, so measuring against the wrong one reports a
    surface cost the run never paid.

    Takes the enum rather than its string: as `api == "responses"` this branch fell
    through to the Chat serialization on any value that was not that exact literal,
    so a typo anywhere upstream would have quietly re-measured the headline saving
    against the wrong endpoint — with no error to notice. A string still works and
    is validated here rather than silently falling through, so the failure mode is
    a loud `ValueError` instead of a wrong number."""
    return to_responses_tools(tools) if Api(api) is Api.RESPONSES else to_openai_tools(tools)


def surface_bytes(tools: list[dict[str, Any]], system: str, api: Api = Api.RESPONSES) -> int:
    """Raw size of exactly what the agent sends, so bytes and tokens describe the
    same payload — measuring one against the rendered prompt and the other against
    the bare instructions would make the two columns silently incomparable."""
    return len(json.dumps(render_tools(tools, api), ensure_ascii=False).encode()) + len(
        build_system_prompt(system).encode()
    )


def _payload_text(tools: list[dict[str, Any]], system: str, api: Api) -> str:
    return (
        json.dumps(render_tools(tools, api), ensure_ascii=False)
        + build_system_prompt(system)
        + PROBE_MESSAGE
    )


@dataclass
class OfflineTokenCounter:
    """tiktoken where available, chars-per-token where it is not."""

    model: str = DEFAULT_MODEL
    api: Api = Api.RESPONSES
    chars_per_token: float = HEURISTIC_CHARS_PER_TOKEN
    reason: str = "точный замер отключён"

    @property
    def exact(self) -> bool:
        return False

    @property
    def estimate_reason(self) -> str | None:
        return self.reason

    def _encoding(self):
        try:
            import tiktoken
        except ImportError:
            return None
        try:
            return tiktoken.encoding_for_model(self.model)
        except (KeyError, ValueError):
            pass
        try:
            return tiktoken.get_encoding(FALLBACK_ENCODING)
        except Exception:
            # Cold tiktoken cache with no network — degrade rather than crash.
            return None

    async def count(self, *, tools: list[dict[str, Any]], system: str) -> int:
        text = _payload_text(tools, system, self.api)
        encoding = self._encoding()
        if encoding is None:
            return int(len(text) / self.chars_per_token)
        return len(encoding.encode(text))


@dataclass
class ChatApiTokenCounter:
    """One real Chat Completions request per variant; reads back what it charged."""

    model: str = DEFAULT_MODEL

    @property
    def exact(self) -> bool:
        return True

    @property
    def estimate_reason(self) -> str | None:
        return None

    async def count(self, *, tools: list[dict[str, Any]], system: str) -> int:
        client = make_client()
        response = await client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": build_system_prompt(system)},
                {"role": "user", "content": PROBE_MESSAGE},
            ],
            tools=to_openai_tools(tools),
            max_completion_tokens=1,
        )
        if response.usage is None:
            raise RuntimeError("API не вернул usage — точный подсчёт токенов невозможен")
        return response.usage.prompt_tokens


@dataclass
class ResponsesApiTokenCounter:
    """Same idea over the Responses API.

    Must match the endpoint the agent actually uses: the two serialize tool
    definitions differently, so counting on one and running on the other would
    report a surface cost the run never paid.

    `max_output_tokens` is 16 rather than 1: a reasoning model spends its first
    output tokens on reasoning, and a cap of 1 makes it fail before it produces
    anything. Input tokens — the number we want — are already fixed by then, so
    the extra allowance costs a rounding error and buys reliability.
    """

    model: str = DEFAULT_MODEL

    @property
    def exact(self) -> bool:
        return True

    @property
    def estimate_reason(self) -> str | None:
        return None

    async def count(self, *, tools: list[dict[str, Any]], system: str) -> int:
        client = make_client()
        response = await client.responses.create(
            model=self.model,
            instructions=build_system_prompt(system),
            input=[{"role": "user", "content": PROBE_MESSAGE}],
            tools=to_responses_tools(tools),
            max_output_tokens=16,
            store=False,
        )
        if response.usage is None:
            raise RuntimeError("API не вернул usage — точный подсчёт токенов невозможен")
        return response.usage.input_tokens


async def measure_surface(
    counter: TokenCounter,
    variant_name: str,
    tools: list[dict[str, Any]],
    system: str,
    api: Api = Api.RESPONSES,
) -> SurfaceCost:
    return SurfaceCost(
        variant=variant_name,
        tokens=await counter.count(tools=tools, system=system),
        bytes_=surface_bytes(tools, system, api),
        exact=counter.exact,
    )
