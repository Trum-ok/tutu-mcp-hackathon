"""Builds the proxy MCP server: same tool surface as `mcp.tutu.ru`, with
compacted always-on descriptions and one extra tool for groundedness checks.

Neither handler can be a plain `@server.tool()`: that decorator derives its
JSON schema from a Python function's signature, and a proxy needs to pass
upstream's own arbitrary schema through untouched (or lightly trimmed by
`compact_tools`), not redeclare ~50 parameters as a Python function. Because
`on_list_tools`/`on_call_tool` replace MCPServer's own handlers wholesale,
`check_groundedness` is dispatched here too rather than via `@server.tool()`
(whose registration the replaced handlers would never see).

`tools/call` is wired through `Extension.intercept_tool_call` — the SDK's
sanctioned hook for exactly this (`on_call_tool` always short-circuits, so
the wrapped default handler, which has no tools registered, is never
reached). `tools/list` has no such public hook: `Extension.tools()` still
derives its schema from a function signature, and `MethodBinding` explicitly
refuses to bind spec methods like `tools/list` (see its docstring, which
points back at this exact seam for that case). So `on_list_tools` is wired
directly on the low-level `Server`, reached via `MCPServer._lowlevel_server`
— the same private attribute `MCPServer` itself uses to bolt on extensions.
"""

from collections.abc import Awaitable, Callable
from typing import Any

import anyio
import mcp_types as types
from mcp.server.context import CallNext, ServerRequestContext
from mcp.server.extension import Extension
from mcp.server.mcpserver import MCPServer
from mcp.shared.exceptions import MCPError

from tutu_mcp.backend import ToolBackend, call_with_timeout_retry
from tutu_mcp.premises import SessionPremises

from .compact_tools import apply_compact_overrides
from .dispatch import backend_error, dispatch
from .surface import PROXY_INSTRUCTIONS, proxy_catalog

# Premise state is per MCP session and must never be global: two clients on one
# proxy would otherwise see each other's assumptions — a data leak on a shared
# deployment, and on stage it means the judge's session inherits ours.
MCP_SESSION_HEADER = "mcp-session-id"
STDIO_SESSION_KEY = "stdio"  # one process, one conversation
MAX_SESSIONS = 256

_OnCallTool = Callable[
    [ServerRequestContext[Any, Any], types.CallToolRequestParams], Awaitable[types.CallToolResult]
]


class _ToolCallDispatch(Extension):
    """Routes `tools/call` to `build_server`'s `on_call_tool` closure via the
    SDK's public interception point, instead of another `_lowlevel_server` write.
    """

    identifier = "ru.tutu/mcp-proxy"

    def __init__(self, on_call_tool: _OnCallTool) -> None:
        self._on_call_tool = on_call_tool

    async def intercept_tool_call(
        self,
        params: types.CallToolRequestParams,
        ctx: ServerRequestContext[Any, Any],
        call_next: CallNext,
    ) -> types.CallToolResult:
        return await self._on_call_tool(ctx, params)


def _session_key(ctx: Any) -> str:
    """Identify the MCP session this request belongs to.

    Streamable HTTP carries `mcp-session-id`; stdio has no such header because it
    has no second conversation to confuse ours with.
    """
    headers = getattr(getattr(ctx, "request", None), "headers", None) or {}
    return headers.get(MCP_SESSION_HEADER) or STDIO_SESSION_KEY


def build_server(backend: ToolBackend, *, name: str = "tutu-mcp-proxy") -> MCPServer:
    catalog: list[dict[str, Any]] | None = None
    trimmed_originals: dict[str, str] = {}
    sessions: dict[str, SessionPremises] = {}
    # Guards the fetch-and-fill below: without it, two `tools/list`/`tools/call`
    # requests arriving before the catalog has ever loaded would both see `None`
    # and both hit `backend.list_tools()` — redundant against a shared rate limit,
    # though not incorrect on its own.
    catalog_lock = anyio.Lock()

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
            async with catalog_lock:
                # re-check: another task may have filled it while we waited for the lock
                if catalog is None:
                    raw = await call_with_timeout_retry(backend.list_tools)
                    compacted, trimmed_originals = apply_compact_overrides(raw)
                    catalog = proxy_catalog(compacted)
        return catalog

    async def on_list_tools(
        ctx: Any, params: types.PaginatedRequestParams | None
    ) -> types.ListToolsResult:
        try:
            tools = [types.Tool.model_validate(t) for t in await load_catalog()]
        except Exception as exc:
            # `tools/list` has no per-item error slot like `CallToolResult.is_error`,
            # so a broken catalog is a protocol-level failure either way — but this
            # gives it the same fixture/timeout/upstream classification `dispatch`
            # uses for `tools/call`, instead of the SDK's generic "Internal server error".
            raise MCPError(
                code=types.INTERNAL_ERROR, message=backend_error("tools/list", exc).text
            ) from exc
        return types.ListToolsResult(tools=tools)

    async def on_call_tool(ctx: Any, params: types.CallToolRequestParams) -> types.CallToolResult:
        try:
            # also populates trimmed_originals, needed below for the appendix lookup
            await load_catalog()
        except Exception as exc:
            error = backend_error(params.name, exc)
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=error.text)], is_error=True
            )
        session = session_for(ctx)
        arguments = params.arguments or {}

        result = await dispatch(session, backend, params.name, arguments, trimmed_originals)
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=result.text)], is_error=result.is_error
        )

    server = MCPServer(
        name=name,
        instructions=PROXY_INSTRUCTIONS,
        extensions=[_ToolCallDispatch(on_call_tool)],
    )

    # No public hook replaces `tools/list` wholesale — see module docstring.
    server._lowlevel_server.add_request_handler(
        "tools/list", types.PaginatedRequestParams, on_list_tools
    )

    return server
