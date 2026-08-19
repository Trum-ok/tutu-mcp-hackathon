"""`UpstreamClient` wires its timeout through to the underlying MCP client
instead of leaving a hung `mcp.tutu.ru` connection to block `serve` forever.

It's also the one place that knows the MCP SDK's failure shapes, so it's the
one responsible for translating them into `BackendError` subclasses
(`tutu_mcp.backend`) — that's what lets `tutu_mcp.proxy.dispatch` classify a
failure without needing to know which `ToolBackend` implementation raised it.
"""

from types import SimpleNamespace

import mcp_types as types
import pytest
from mcp.shared.exceptions import MCPError

from tutu_mcp import upstream as upstream_pkg
from tutu_mcp.backend import BackendTimeoutError, BackendUnavailableError
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

    async def list_tools(self):
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
