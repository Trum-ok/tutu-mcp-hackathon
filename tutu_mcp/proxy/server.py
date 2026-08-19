"""Builds the proxy MCP server: same tool surface as `mcp.tutu.ru`, with
compacted always-on descriptions and one extra tool for groundedness checks.

Tool listing/dispatch is wired directly on the low-level `Server` (reached
via `MCPServer._lowlevel_server`, the same private seam `MCPServer` itself
uses to bolt on extensions — see its `_install_extension_interceptor`)
instead of the `@server.tool()` decorator, because that decorator derives
its JSON schema from a Python function's signature. A proxy needs to pass
upstream's own arbitrary schema through untouched (or lightly trimmed by
`compact_tools`), not redeclare ~50 parameters as a Python function. Because
`on_list_tools`/`on_call_tool` replace MCPServer's own handlers wholesale,
`check_groundedness` is dispatched here too rather than via `@server.tool()`
(whose registration the replaced handlers would never see).
"""

from typing import Any

import mcp_types as types
from mcp.server.mcpserver import MCPServer

from tutu_mcp.backend import ToolBackend
from tutu_mcp.groundedness import CHECK_GROUNDEDNESS_TOOL, run_check_groundedness_tool
from tutu_mcp.premises import (
    ASSESS_REQUEST_TOOL,
    SessionPremises,
    run_assess_request_tool,
    strip_control_fields,
)
from tutu_mcp.replay.store import FixtureNotFoundError

from .compact_tools import apply_compact_overrides, apply_result_appendix

# Premise state is per MCP session and must never be global: two clients on one
# proxy would otherwise see each other's assumptions — a data leak on a shared
# deployment, and on stage it means the judge's session inherits ours.
MCP_SESSION_HEADER = "mcp-session-id"
STDIO_SESSION_KEY = "stdio"  # one process, one conversation
MAX_SESSIONS = 256

PROXY_INSTRUCTIONS = (
    "Compacted proxy in front of mcp.tutu.ru: identical tools and behavior, but the "
    "always-on tools/list catalog is trimmed for the biggest offenders (see each "
    "trimmed tool's description for which get_<domain>_instructions tool absorbed the "
    "detail — that tool's CALL RESULT carries it, not its tools/list entry). Also "
    "exposes `check_groundedness` — before answering, pass your drafted answer text "
    "plus the raw JSON text of every tool_result you used this turn, and it flags any "
    "price/time/train-or-flight-code/URL in your answer that isn't actually present in "
    "those results.\n\n"
    "PREMISE GATE — read before your first search. Every value that NARROWS a search "
    "must come from the user or from an earlier tool_result; there is no third source. "
    "Call `assess_request` FIRST with the user's request verbatim (it is local, free and "
    "instant) — it surfaces which parameters are blocking, flags a date that contradicts "
    "the weekday the user gave, and lets values the user actually typed pass the gate "
    "without a retry. When a search argument has no such source, the call returns "
    "`clarification_required` INSTEAD of data, and you resolve it one of three ways: ask "
    'the user (preferred), repeat the call with `_sources={"<field>": "user"}` if the '
    'user did supply it, or repeat with `_assume={"<field>": "<rationale>"}` to proceed '
    "on an openly declared assumption. In that last case the result carries a preamble your "
    "answer MUST OPEN with — `check_groundedness` fails an answer that discloses an "
    "assumption only at the end, or not at all. Both `_sources` and `_assume` are stripped "
    "before the call reaches Tutu, so no upstream schema changes."
)

# `CHECK_GROUNDEDNESS_TOOL` is re-exported from `app.groundedness`, where its
# Pydantic argument model lives next to the function that consumes it.


def _run_check_groundedness(
    arguments: dict[str, Any], session: SessionPremises
) -> types.CallToolResult:
    # assumptions come from the session, never from the agent's arguments — an
    # agent able to omit them could hide them from the check built to expose them
    text, is_error = run_check_groundedness_tool(
        arguments,
        assumed_values=session.assumed_values(),
        assumptions=session.assumption_lines(),
    )
    return types.CallToolResult(
        content=[types.TextContent(type="text", text=text)], is_error=is_error
    )


def _session_key(ctx: Any) -> str:
    """Identify the MCP session this request belongs to.

    Streamable HTTP carries `mcp-session-id`; stdio has no such header because it
    has no second conversation to confuse ours with.
    """
    headers = getattr(getattr(ctx, "transport", None), "headers", None) or {}
    return headers.get(MCP_SESSION_HEADER) or STDIO_SESSION_KEY


def build_server(backend: ToolBackend, *, name: str = "tutu-mcp-proxy") -> MCPServer:
    server = MCPServer(name=name, instructions=PROXY_INSTRUCTIONS)

    catalog: list[dict[str, Any]] | None = None
    trimmed_originals: dict[str, str] = {}
    sessions: dict[str, SessionPremises] = {}

    def session_for(ctx: Any) -> SessionPremises:
        key = _session_key(ctx)
        if key not in sessions:
            # insertion-ordered: the oldest conversation is the first to go
            while len(sessions) >= MAX_SESSIONS:
                sessions.pop(next(iter(sessions)))
            sessions[key] = SessionPremises()
        return sessions[key]

    async def load_catalog() -> list[dict[str, Any]]:
        nonlocal catalog, trimmed_originals
        if catalog is None:
            compacted, trimmed_originals = apply_compact_overrides(await backend.list_tools())
            catalog = [*compacted, ASSESS_REQUEST_TOOL, CHECK_GROUNDEDNESS_TOOL]
        return catalog

    async def on_list_tools(
        ctx: Any, params: types.PaginatedRequestParams | None
    ) -> types.ListToolsResult:
        tools = [types.Tool.model_validate(t) for t in await load_catalog()]
        return types.ListToolsResult(tools=tools)

    async def on_call_tool(ctx: Any, params: types.CallToolRequestParams) -> types.CallToolResult:
        await load_catalog()  # ensures trimmed_originals is populated before any appendix lookup
        session = session_for(ctx)
        arguments = params.arguments or {}

        if params.name == "assess_request":
            text, is_error = run_assess_request_tool(arguments, session)
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=text)], is_error=is_error
            )

        if params.name == "check_groundedness":
            return _run_check_groundedness(arguments, session)

        # Control fields are ours, not Tutu's: strip them before the call goes out,
        # or the fixture store misses on every scenario and upstream is handed a
        # field its schema never declared.
        clean, sources, assume = strip_control_fields(arguments)

        decision = session.evaluate(params.name, clean, sources, assume)
        if decision is not None:
            # Deliberately NOT is_error: clients surface a tool error to the user as
            # a red failure and agents retry it blindly. This is a successful call
            # whose payload happens to be a question.
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=decision.to_json())], is_error=False
            )

        try:
            result = await backend.call_tool(params.name, clean)
        except FixtureNotFoundError as exc:
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=str(exc))], is_error=True
            )

        session.record_result(result.text)
        text = apply_result_appendix(params.name, result.text, trimmed_originals)
        preamble = session.preamble()
        if preamble:
            text = f"{text}\n\n## Обязательная преамбула ответа\n{preamble}"
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=text)], is_error=result.is_error
        )

    # accesses the private `_lowlevel_server` seam — see module docstring
    server._lowlevel_server.add_request_handler(
        "tools/list", types.PaginatedRequestParams, on_list_tools
    )
    server._lowlevel_server.add_request_handler(
        "tools/call", types.CallToolRequestParams, on_call_tool
    )

    return server
