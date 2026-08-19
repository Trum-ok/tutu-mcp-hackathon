"""The two things being compared: the raw upstream tool surface vs. our proxy's.

Both variants read the SAME backend, so tool *results* are byte-identical between
them and the only difference is the tool surface the agent sees. That isolation is
the whole experiment — if the data differed too, a win could just mean better data.

The proxy variant calls the same `apply_compact_overrides` / `apply_result_appendix`
/ `run_check_groundedness_tool` functions the live MCP server calls, so this
measures the real proxy behavior rather than a re-implementation of it. It skips
only the HTTP/MCP transport hop, which is identical for both variants anyway.
"""

import time
from dataclasses import dataclass, field, replace
from typing import Any

from tutu_mcp.backend import ToolBackend
from tutu_mcp.groundedness import run_check_groundedness_tool
from tutu_mcp.premises import (
    ASSESS_REQUEST_TOOL,
    SessionPremises,
    run_assess_request_tool,
    strip_control_fields,
)
from tutu_mcp.proxy.compact_tools import apply_compact_overrides, apply_result_appendix
from tutu_mcp.proxy.server import CHECK_GROUNDEDNESS_TOOL, PROXY_INSTRUCTIONS
from tutu_mcp.replay.store import FixtureNotFoundError

from .transcript import ToolCallRecord

BASELINE = "baseline"
PROXY = "proxy"


@dataclass
class Variant:
    name: str
    tools: list[dict[str, Any]]
    server_instructions: str
    # tool name -> pre-trim description, used for the instructions-tool appendix
    trimmed_originals: dict[str, str] = field(default_factory=dict)
    backend: ToolBackend | None = None
    # None on baseline: the premise gate is part of what the proxy variant IS, so
    # switching it on for baseline too would erase the difference being measured
    premises: SessionPremises | None = None

    def session_scope(self) -> "Variant":
        """A copy with fresh premise state, one per scenario.

        Scenarios run through `asyncio.gather`, so a shared session would let one
        scenario's assumptions and one-shot gate leak into another's — the eval
        equivalent of the cross-client leak the server guards against.
        """
        return replace(self, premises=None if self.premises is None else SessionPremises())

    def _record(self, name, arguments, text, is_error, started, **extra) -> ToolCallRecord:
        return ToolCallRecord(
            name=name,
            arguments=arguments,
            result_text=text,
            is_error=is_error,
            duration_s=time.monotonic() - started,
            **extra,
        )

    async def execute(self, name: str, arguments: dict[str, Any]) -> ToolCallRecord:
        started = time.monotonic()
        session = self.premises

        if name == "assess_request":
            if session is None:  # not on this surface — same shape as an unknown tool
                return self._record(name, arguments, f"unknown tool: {name}", True, started)
            text, is_error = run_assess_request_tool(arguments, session)
            return self._record(name, arguments, text, is_error, started)

        if name == "check_groundedness":
            text, is_error = run_check_groundedness_tool(
                arguments,
                assumed_values=session.assumed_values() if session else None,
                assumptions=session.assumption_lines() if session else None,
            )
            return self._record(name, arguments, text, is_error, started)

        clean, sources, assume = strip_control_fields(arguments)
        if session is not None:
            decision = session.evaluate(name, clean, sources, assume)
            if decision is not None:
                return self._record(name, arguments, decision.to_json(), False, started)
        arguments = clean

        assert self.backend is not None
        try:
            result = await self.backend.call_tool(name, arguments)
        except FixtureNotFoundError as exc:
            return ToolCallRecord(
                name=name,
                arguments=arguments,
                result_text=str(exc),
                is_error=True,
                fixture_miss=True,
                duration_s=time.monotonic() - started,
            )

        text = result.text
        if self.trimmed_originals:
            text = apply_result_appendix(name, text, self.trimmed_originals)
        if session is not None:
            session.record_result(result.text)
            preamble = session.preamble()
            if preamble:
                text = f"{text}\n\n## Обязательная преамбула ответа\n{preamble}"

        return self._record(name, arguments, text, result.is_error, started)


async def build_variants(
    backend: ToolBackend, upstream_instructions: str, *, names: list[str] | None = None
) -> list[Variant]:
    raw_tools = await backend.list_tools()
    compacted, trimmed_originals = apply_compact_overrides(raw_tools)

    all_variants = {
        BASELINE: Variant(
            name=BASELINE,
            tools=raw_tools,
            server_instructions=upstream_instructions,
            backend=backend,
        ),
        PROXY: Variant(
            name=PROXY,
            tools=[*compacted, ASSESS_REQUEST_TOOL, CHECK_GROUNDEDNESS_TOOL],
            server_instructions=PROXY_INSTRUCTIONS,
            trimmed_originals=trimmed_originals,
            backend=backend,
            premises=SessionPremises(),
        ),
    }

    wanted = names or [BASELINE, PROXY]
    unknown = set(wanted) - set(all_variants)
    if unknown:
        raise KeyError(f"unknown variants: {sorted(unknown)}")
    return [all_variants[n] for n in wanted]
