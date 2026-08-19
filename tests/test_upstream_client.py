"""`UpstreamClient` wires its timeout through to the underlying MCP client
instead of leaving a hung `mcp.tutu.ru` connection to block `serve` forever.

It's also the one place that knows the MCP SDK's failure shapes, so it's the
one responsible for translating them into `BackendError` subclasses
(`tutu_mcp.backend`) — that's what lets `tutu_mcp.proxy.dispatch` classify a
failure without needing to know which `ToolBackend` implementation raised it.
"""

from types import SimpleNamespace
from typing import ClassVar

import anyio
import mcp_types as types
import pytest
from mcp.shared.exceptions import MCPError

from tutu_mcp import upstream as upstream_pkg
from tutu_mcp.backend import (
    BackendTimeoutError,
    BackendUnavailableError,
    call_with_timeout_retry,
)
from tutu_mcp.upstream.client import UpstreamClient


class _FakeMCPClient:
    last_kwargs: dict | None = None

    def __init__(self, url, **kwargs):
        self.url = url
        _FakeMCPClient.last_kwargs = kwargs

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return None


async def test_connect_passes_the_configured_timeout_to_the_mcp_client(monkeypatch):
    monkeypatch.setattr(upstream_pkg.client, "MCPClient", _FakeMCPClient)

    async with UpstreamClient("https://example.invalid/mcp", timeout_s=5.0):
        pass

    assert _FakeMCPClient.last_kwargs == {"read_timeout_seconds": 5.0}


async def test_connect_defaults_to_no_timeout(monkeypatch):
    monkeypatch.setattr(upstream_pkg.client, "MCPClient", _FakeMCPClient)

    async with UpstreamClient("https://example.invalid/mcp"):
        pass

    assert _FakeMCPClient.last_kwargs == {"read_timeout_seconds": None}


class _RaisingMCPClient(_FakeMCPClient):
    """Raises a fixed exception from every `call_tool`/`list_tools`, to
    exercise `UpstreamClient`'s translation into `BackendError`."""

    def __init__(self, url, exc: Exception, **kwargs):
        super().__init__(url, **kwargs)
        self._exc = exc

    async def call_tool(self, name, arguments):
        raise self._exc

    async def list_tools(self, *, cursor=None):
        raise self._exc


async def test_bare_timeout_error_translates_to_backend_timeout(monkeypatch):
    """The transport's own bounded send raises a plain `TimeoutError` before
    the request's own `read_timeout_seconds` ever arms."""
    monkeypatch.setattr(
        upstream_pkg.client,
        "MCPClient",
        lambda url, **kw: _RaisingMCPClient(url, TimeoutError(), **kw),
    )

    async with UpstreamClient("https://example.invalid/mcp") as client:
        with pytest.raises(BackendTimeoutError):
            await client.call_tool("search_rail", {})
        with pytest.raises(BackendTimeoutError):
            await client.list_tools()


async def test_mcp_request_timeout_translates_to_backend_timeout(monkeypatch):
    """`read_timeout_seconds` actually elapsing raises `MCPError(code=REQUEST_TIMEOUT)`,
    not a bare `TimeoutError` — missing this shape would leave a live timeout
    unreachable as `BackendTimeoutError` in practice."""
    timeout_error = MCPError(code=types.REQUEST_TIMEOUT, message="Request timed out")
    monkeypatch.setattr(
        upstream_pkg.client,
        "MCPClient",
        lambda url, **kw: _RaisingMCPClient(url, timeout_error, **kw),
    )

    async with UpstreamClient("https://example.invalid/mcp") as client:
        with pytest.raises(BackendTimeoutError):
            await client.call_tool("search_rail", {})


async def test_other_failures_translate_to_backend_unavailable(monkeypatch):
    monkeypatch.setattr(
        upstream_pkg.client,
        "MCPClient",
        lambda url, **kw: _RaisingMCPClient(url, ConnectionError("boom"), **kw),
    )

    async with UpstreamClient("https://example.invalid/mcp") as client:
        with pytest.raises(BackendUnavailableError):
            await client.call_tool("search_rail", {})


async def test_a_connection_failure_reconnects_once_and_the_retry_succeeds(monkeypatch):
    """A broken connection must not wedge every later call into
    `BackendUnavailableError` until the process restarts — one reconnect
    should recover it."""
    constructions: list[None] = []

    class _ConnectionResetThenOk(_FakeMCPClient):
        async def call_tool(self, name, arguments):
            if len(constructions) == 1:  # the original connection — not reconnected yet
                raise ConnectionError("connection reset by peer")
            return SimpleNamespace(
                content=[SimpleNamespace(type="text", text="ok")], is_error=False
            )

    def factory(url, **kw):
        constructions.append(None)
        return _ConnectionResetThenOk(url, **kw)

    monkeypatch.setattr(upstream_pkg.client, "MCPClient", factory)

    async with UpstreamClient("https://example.invalid/mcp") as client:
        result = await client.call_tool("search_rail", {})

    assert result.text == "ok"
    assert len(constructions) == 2  # the original connection, plus one reconnect


async def test_a_timeout_does_not_trigger_a_reconnect(monkeypatch):
    """The server was reachable and just slow — tearing down a live connection
    over that would make the next call worse, not better."""
    constructions: list[None] = []
    timeout_error = MCPError(code=types.REQUEST_TIMEOUT, message="Request timed out")

    def factory(url, **kw):
        constructions.append(None)
        return _RaisingMCPClient(url, timeout_error, **kw)

    monkeypatch.setattr(upstream_pkg.client, "MCPClient", factory)

    async with UpstreamClient("https://example.invalid/mcp") as client:
        with pytest.raises(BackendTimeoutError):
            await client.call_tool("search_rail", {})

    assert len(constructions) == 1


class _PagedMCPClient(_FakeMCPClient):
    """Serves the catalog in two pages, the way a server with more tools would."""

    PAGES: ClassVar[dict] = {
        None: (["search_rail", "search_avia"], "page-2"),
        "page-2": (["search_hotels"], None),
    }

    def __init__(self, url, **kw):
        super().__init__(url, **kw)
        self.cursors: list[str | None] = []

    async def list_tools(self, *, cursor=None):
        self.cursors.append(cursor)
        names, next_cursor = self.PAGES[cursor]
        tools = [
            SimpleNamespace(
                model_dump=lambda mode=None, by_alias=None, exclude_none=None, n=name: {"name": n}
            )
            for name in names
        ]
        return SimpleNamespace(tools=tools, next_cursor=next_cursor)


async def test_a_paginated_catalog_is_joined_rather_than_truncated(monkeypatch):
    """Tutu answers today's 16 tools in one page. On the 17th, dropping
    `nextCursor` would hand the agent a short catalog with no error at all —
    it simply would not know the missing tools exist."""
    clients: list[_PagedMCPClient] = []

    def factory(url, **kw):
        clients.append(_PagedMCPClient(url, **kw))
        return clients[-1]

    monkeypatch.setattr(upstream_pkg.client, "MCPClient", factory)

    async with UpstreamClient("https://example.invalid/mcp") as client:
        tools = await client.list_tools()

    assert [t["name"] for t in tools] == ["search_rail", "search_avia", "search_hotels"]
    assert clients[0].cursors == [None, "page-2"]


async def test_a_repeated_cursor_stops_the_loop(monkeypatch):
    """A server that keeps handing back the same cursor must not spin `serve`
    forever — the catalog load happens on the request path."""

    class _StuckClient(_FakeMCPClient):
        async def list_tools(self, *, cursor=None):
            return SimpleNamespace(tools=[], next_cursor="always-the-same")

    monkeypatch.setattr(upstream_pkg.client, "MCPClient", lambda url, **kw: _StuckClient(url, **kw))

    async with UpstreamClient("https://example.invalid/mcp") as client:
        assert await client.list_tools() == []


async def test_concurrent_failures_reconnect_once_between_them(monkeypatch):
    """Two in-flight calls meeting the same broken connection must not each open
    a client: the loser would leave its own connection dangling, still open and
    owned by a request task that is about to finish."""
    constructions: list[None] = []

    class _FirstConnectionIsDead(_FakeMCPClient):
        async def call_tool(self, name, arguments):
            if len(constructions) == 1:
                await anyio.sleep(0)  # let the sibling call reach the same failure
                raise ConnectionError("connection reset by peer")
            return SimpleNamespace(
                content=[SimpleNamespace(type="text", text="ok")], is_error=False
            )

    def factory(url, **kw):
        constructions.append(None)
        return _FirstConnectionIsDead(url, **kw)

    monkeypatch.setattr(upstream_pkg.client, "MCPClient", factory)

    async with (
        UpstreamClient("https://example.invalid/mcp") as client,
        anyio.create_task_group() as tg,
    ):
        tg.start_soon(client.call_tool, "search_rail", {})
        tg.start_soon(client.call_tool, "search_avia", {})

    assert len(constructions) == 2, "the original connection plus exactly one reconnect"


async def test_one_logical_call_never_costs_more_than_three_upstream_requests(monkeypatch):
    """Two retry layers stack: `call_with_timeout_retry` repeats a timeout, and
    `_call` reconnects on a connection failure. The worst mix is timeout →
    failure → retry, and it must stay bounded — every request here is spent
    against the shared hackathon rate limit."""
    attempts: list[str] = []
    timeout_error = MCPError(code=types.REQUEST_TIMEOUT, message="Request timed out")

    class _TimeoutThenBroken(_FakeMCPClient):
        async def call_tool(self, name, arguments):
            attempts.append(name)
            raise timeout_error if len(attempts) == 1 else ConnectionError("boom")

    monkeypatch.setattr(
        upstream_pkg.client, "MCPClient", lambda url, **kw: _TimeoutThenBroken(url, **kw)
    )

    async with UpstreamClient("https://example.invalid/mcp") as client:
        with pytest.raises(BackendUnavailableError):
            await call_with_timeout_retry(lambda: client.call_tool("search_rail", {}))

    assert len(attempts) == 3
