"""Live `ToolBackend` — talks to the real mcp.tutu.ru over Streamable HTTP."""

import contextlib
from collections.abc import Awaitable, Callable
from typing import Any

import anyio
import mcp_types as types
from mcp.client import Client as MCPClient
from mcp.shared.exceptions import MCPError

from tutu_mcp.backend import BackendTimeoutError, BackendUnavailableError, ToolCallResult


def _translate(exc: Exception) -> Exception:
    """Turns whatever the MCP client raises into a `BackendError`, at the one
    place that knows this backend's failure shapes.

    The client raises two different shapes for a timeout: a bare
    `TimeoutError` from the transport's own bounded send (before the
    request's own timeout arms), and an `MCPError(code=REQUEST_TIMEOUT)`
    once armed — which is the shape `read_timeout_seconds` actually elapsing
    takes. Missing the second one would leave the whole point of configuring
    a timeout unreachable: it would always classify as a generic failure.
    """
    if isinstance(exc, TimeoutError):
        return BackendTimeoutError(str(exc))
    if isinstance(exc, MCPError) and exc.code == types.REQUEST_TIMEOUT:
        return BackendTimeoutError(str(exc))
    return BackendUnavailableError(str(exc))


class UpstreamClient:
    """Thin wrapper around the official MCP client, scoped to one upstream server.

    One instance opens one long-lived connection (`connect`/`aclose`) so repeated
    calls reuse the same session instead of re-running the handshake every time.
    """

    def __init__(self, url: str, *, timeout_s: float | None = None) -> None:
        self._url = url
        self._timeout_s = timeout_s
        self._client: MCPClient | None = None
        # The proxy serves concurrent `tools/call`s off one client, so two of them
        # can meet a broken connection at the same moment. Without this, both would
        # tear the client down and hand `self._client` to a handshake of their own,
        # and the loser's fresh connection would be dropped on the floor still open.
        self._reconnect_lock = anyio.Lock()

    async def connect(self) -> None:
        self._client = MCPClient(self._url, read_timeout_seconds=self._timeout_s)
        await self._client.__aenter__()

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.__aexit__(None, None, None)
            self._client = None

    async def __aenter__(self) -> "UpstreamClient":
        await self.connect()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    def _require_client(self) -> MCPClient:
        if self._client is None:
            raise RuntimeError(
                "UpstreamClient is not connected — use it as an async context manager"
            )
        return self._client

    async def _reconnect(self, stale: MCPClient | None) -> None:
        """Best-effort close of the dead connection, then a fresh handshake.

        `stale` is the client the caller actually failed on. Whoever takes the lock
        second finds a different client already in place and returns without
        touching it — its call then simply retries on the connection the first one
        opened, which is the outcome both were after anyway.
        """
        async with self._reconnect_lock:
            if self._client is not stale:
                return
            if self._client is not None:
                # already broken — nothing left to salvage from it
                with contextlib.suppress(Exception):
                    await self._client.__aexit__(None, None, None)
                self._client = None
            await self.connect()

    async def _call[T](self, call: Callable[[MCPClient], Awaitable[T]]) -> T:
        """Runs `call` against the live client; on any non-timeout failure,
        reconnects once and retries — a network blip or an upstream restart
        would otherwise wedge every later call into `BackendUnavailableError`
        until the whole process restarts. A timeout is excluded: the server
        was reachable and just slow, so tearing down a live connection over
        it would only make the next call worse, not better.
        """
        client = self._require_client()
        try:
            return await call(client)
        except Exception as exc:
            translated = _translate(exc)
            if isinstance(translated, BackendTimeoutError):
                raise translated from exc
            await self._reconnect(client)
            try:
                return await call(self._require_client())
            except Exception as retry_exc:
                raise _translate(retry_exc) from retry_exc

    async def list_tools(self) -> list[dict[str, Any]]:
        """Every page of upstream's catalog, joined.

        Tutu answers today's 16 tools in one page, so dropping `nextCursor` looked
        free — until the 17th tool, where a silently truncated catalog is worse
        than a loud failure: the agent would simply not know a tool exists. Pages
        are joined here rather than forwarded outward because the proxy caches and
        rewrites the catalog as a whole anyway (see `proxy/server.py`).
        """
        tools: list[dict[str, Any]] = []
        cursor: str | None = None
        seen_cursors: set[str] = set()
        while True:
            result = await self._call(lambda client, c=cursor: client.list_tools(cursor=c))
            tools.extend(
                t.model_dump(mode="json", by_alias=True, exclude_none=True) for t in result.tools
            )
            cursor = result.next_cursor
            # A server that keeps handing back a cursor it already gave would spin
            # this loop forever; stopping loses nothing a repeat page would add.
            if cursor is None or cursor in seen_cursors:
                return tools
            seen_cursors.add(cursor)

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> ToolCallResult:
        result = await self._call(lambda client: client.call_tool(name, arguments))
        text = "\n".join(block.text for block in result.content if block.type == "text")
        return ToolCallResult(text=text, is_error=result.is_error)

    def server_info(self) -> dict[str, Any]:
        """Upstream's `initialize` result: the always-on instructions plus identity.

        The instructions block is loaded by every client on every session, so it
        counts toward the tool surface the eval measures — not just `tools/list`.
        """
        client = self._require_client()
        result = client.session.initialize_result
        if result is None:
            return {"instructions": "", "name": None, "version": None}
        return {
            "instructions": result.instructions or "",
            "name": result.server_info.name if result.server_info else None,
            "version": result.server_info.version if result.server_info else None,
        }
