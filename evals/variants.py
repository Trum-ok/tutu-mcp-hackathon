"""The two things being compared: the raw upstream tool surface vs. our proxy's.

Both variants read the SAME backend, so tool *results* are byte-identical between
them and the only difference is the tool surface the agent sees. That isolation is
the whole experiment — if the data differed too, a win could just mean better data.

The proxy variant runs every call through `tutu_mcp.proxy.dispatch.dispatch` — the
exact same pipeline the live MCP server calls — so this measures the real proxy
behavior rather than a re-implementation of it. It skips only the HTTP/MCP
transport hop, which is identical for both variants anyway.
"""

import time
from dataclasses import dataclass, field, replace
from typing import Any

# Defined in `evals.options` (a stdlib-only leaf) and re-exported here, where
# they belong: `EvalOptions` needs them for its default, and importing this
# module to get two strings would drag the whole proxy into `tutu.py --help`.
from evals.options import BASELINE as BASELINE
from evals.options import PROXY as PROXY
from evals.options import VARIANT_NAMES, check_variants
from tutu_mcp.backend import ToolBackend
from tutu_mcp.premises import SessionPremises
from tutu_mcp.proxy.compact_tools import apply_compact_overrides
from tutu_mcp.proxy.dispatch import dispatch
from tutu_mcp.proxy.surface import PROXY_INSTRUCTIONS, proxy_catalog

from .transcript import ToolCallRecord


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

    def _record(
        self,
        name: str,
        arguments: dict[str, Any],
        text: str,
        is_error: bool,
        started: float,
        *,
        fixture_miss: bool = False,
    ) -> ToolCallRecord:
        return ToolCallRecord(
            name=name,
            arguments=arguments,
            result_text=text,
            is_error=is_error,
            fixture_miss=fixture_miss,
            duration_s=time.monotonic() - started,
        )

    async def execute(self, name: str, arguments: dict[str, Any]) -> ToolCallRecord:
        started = time.monotonic()
        assert self.backend is not None

        result = await dispatch(
            self.premises, self.backend, name, arguments, self.trimmed_originals
        )
        return self._record(
            name, arguments, result.text, result.is_error, started, fixture_miss=result.fixture_miss
        )


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
            tools=proxy_catalog(compacted),
            server_instructions=PROXY_INSTRUCTIONS,
            trimmed_originals=trimmed_originals,
            backend=backend,
            premises=SessionPremises(),
        ),
    }

    wanted = names or list(VARIANT_NAMES)
    check_variants(wanted)
    return [all_variants[n] for n in wanted]
