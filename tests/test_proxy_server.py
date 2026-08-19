"""The assembled MCP server: catalog contents, gate wiring, session isolation.

Reaches the registered handlers through `_lowlevel_server._request_handlers`
so these exercise the real dispatch path rather than re-calling the helpers
it happens to be built from. `build_server` itself only writes to that seam
for `tools/list`; `tools/call` is registered by the SDK's own extension
machinery (see `tutu_mcp.proxy.server` module docstring), but ends up in the
same dict, so the fetch here works for both.
"""

import asyncio
import json

import anyio
import mcp_types as types
import pytest
from mcp.shared.exceptions import MCPError

from tutu_mcp.backend import BackendTimeoutError
from tutu_mcp.proxy.dispatch import PREAMBLE_KEY
from tutu_mcp.proxy.server import MAX_SESSIONS, PROXY_INSTRUCTIONS, build_server
from tutu_mcp.replay.mock_client import MockUpstreamClient

RAIL_ARGS = {
    "origin": "Санкт-Петербург",
    "destination": "Москва",
    "departure_date": "2026-08-25",
}


class FakeCtx:
    """Stands in for a `ServerRequestContext`: headers live on `.request` (the
    transport's HTTP request object), matching what `_make_context` actually
    populates — not a `.transport` attribute, which the real dataclass has
    no such field for."""

    def __init__(self, session_id: str | None = None):
        headers = {"mcp-session-id": session_id} if session_id else {}
        self.request = type("R", (), {"headers": headers})()


@pytest.fixture
def server(repo_fixtures):
    return build_server(MockUpstreamClient(repo_fixtures))


def _handler(server, method):
    return server._lowlevel_server._request_handlers[method].handler


async def _call(server, name, arguments, ctx=None):
    return await _handler(server, "tools/call")(
        ctx or FakeCtx(), types.CallToolRequestParams(name=name, arguments=arguments)
    )


def _text(result) -> str:
    return "".join(b.text for b in result.content if b.type == "text")


async def test_catalog_carries_upstream_tools_plus_ours(server):
    result = await _handler(server, "tools/list")(FakeCtx(), None)
    names = {t.name for t in result.tools}

    assert {"search_rail", "search_hotels", "create_checkout_link"} <= names
    assert {"assess_request", "check_groundedness"} <= names


async def test_concurrent_first_calls_load_the_catalog_only_once(repo_fixtures):
    """Two requests arriving before the catalog has ever loaded must not both
    hit the backend — the lock in `build_server` exists precisely to prevent
    that redundant (and, against a shared rate limit, costly) double fetch."""
    calls = 0
    inner = MockUpstreamClient(repo_fixtures)

    class _SlowBackend:
        async def list_tools(self):
            nonlocal calls
            calls += 1
            await anyio.sleep(0.01)  # give the second caller a chance to race in
            return await inner.list_tools()

        async def call_tool(self, name, arguments):
            return await inner.call_tool(name, arguments)

    server = build_server(_SlowBackend())

    await asyncio.gather(
        _handler(server, "tools/list")(FakeCtx(), None),
        _handler(server, "tools/list")(FakeCtx(), None),
    )

    assert calls == 1


async def test_instructions_document_the_control_fields(server):
    """The two keys are never added to any proxied `inputSchema`, so the server
    instructions are the only place an agent can learn they exist."""
    assert "_sources" in PROXY_INSTRUCTIONS
    assert "_assume" in PROXY_INSTRUCTIONS


async def test_gate_returns_a_question_not_an_error(server):
    """`is_error` would render as a red failure in the client and invite a blind retry."""
    result = await _call(server, "search_rail", {**RAIL_ARGS, "price_max": 4000})
    payload = json.loads(_text(result))

    assert result.is_error is False
    assert payload["status"] == "clarification_required"


async def test_control_fields_do_not_reach_the_fixture_lookup(server):
    """The strip has to happen server-side: fixtures match on exact arguments."""
    result = await _call(
        server, "search_rail", {**RAIL_ARGS, "_sources": {"price_max": "user"}, "price_max": 3000}
    )

    assert result.is_error is False
    assert "offers" in _text(result)


async def test_declared_assumption_carries_a_mandatory_preamble(server):
    result = await _call(
        server, "search_rail", {**RAIL_ARGS, "price_max": 3000, "_assume": {"price_max": "бюджет"}}
    )
    payload = json.loads(_text(result))

    # inside the JSON, not appended after it — see `attach_preamble`
    assert "бюджет" in payload[PREAMBLE_KEY]
    assert payload["offers"], "данные должны прийти вместе с преамбулой, а не вместо неё"


async def test_sessions_do_not_share_premise_state(server):
    """Two clients on one proxy must not inherit each other's assumptions."""
    first = await _call(server, "search_rail", {**RAIL_ARGS, "price_max": 4000}, FakeCtx("aaa"))
    second = await _call(server, "search_rail", {**RAIL_ARGS, "price_max": 4000}, FakeCtx("bbb"))

    # each session gets its own one-shot gate; without isolation the second would pass
    for result in (first, second):
        assert "clarification_required" in _text(result)


async def test_repeating_an_unchanged_call_releases_instead_of_looping(server):
    ctx = FakeCtx("same")
    first = await _call(server, "search_rail", {**RAIL_ARGS, "price_max": 3000}, ctx)
    second = await _call(server, "search_rail", {**RAIL_ARGS, "price_max": 3000}, ctx)

    assert "clarification_required" in _text(first)
    released = json.loads(_text(second))
    assert released["offers"]
    assert released[PREAMBLE_KEY]


async def test_session_table_is_bounded(server):
    for i in range(MAX_SESSIONS + 20):
        await _call(server, "get_rail_instructions", {}, FakeCtx(f"s{i}"))

    # the table lives in build_server's closure, so assert on behavior: session "s0"
    # was evicted long ago and must therefore gate again rather than be remembered
    result = await _call(server, "search_rail", {**RAIL_ARGS, "price_max": 3000}, FakeCtx("s0"))
    assert "clarification_required" in _text(result)


async def test_unknown_arguments_still_surface_a_fixture_miss_as_an_error(server):
    result = await _call(server, "search_rail", {"origin": "Нигде", "departure_date": "2026-08-25"})

    assert result.is_error is True
    assert "No fixture" in _text(result)


class _FailingBackend:
    """Wraps a real backend but makes every `call_tool` raise, to exercise the
    proxy's handling of upstream failures that are not a fixture miss.

    Raises whatever a `ToolBackend` implementation is expected to raise at
    its own boundary — a `BackendError` subclass for a classified failure
    (`tutu_mcp.backend`), or any other exception for one no implementation
    recognized. Classifying an SDK-specific shape like `MCPError` is the
    live `UpstreamClient`'s job (see `tests/test_upstream_client.py`), not
    something a generic fake should need to know about."""

    def __init__(self, inner, exc: Exception):
        self._inner = inner
        self._exc = exc

    async def list_tools(self):
        return await self._inner.list_tools()

    async def call_tool(self, name, arguments):
        raise self._exc


async def test_upstream_timeout_surfaces_as_a_structured_error_not_a_crash(repo_fixtures):
    server = build_server(
        _FailingBackend(MockUpstreamClient(repo_fixtures), BackendTimeoutError("timed out"))
    )

    result = await _call(server, "search_rail", RAIL_ARGS)

    assert result.is_error is True
    assert json.loads(_text(result))["status"] == "upstream_timeout"


class _TimeoutThenOkBackend:
    """Times out on the first `call_tool`, then serves normally — the shape a
    transient network hiccup or a slow queued request actually takes."""

    def __init__(self, inner):
        self._inner = inner
        self.attempts = 0

    async def list_tools(self):
        return await self._inner.list_tools()

    async def call_tool(self, name, arguments):
        self.attempts += 1
        if self.attempts == 1:
            raise BackendTimeoutError("timed out")
        return await self._inner.call_tool(name, arguments)


async def test_a_transient_timeout_is_retried_and_recovers(repo_fixtures):
    backend = _TimeoutThenOkBackend(MockUpstreamClient(repo_fixtures))
    server = build_server(backend)

    result = await _call(server, "search_rail", RAIL_ARGS)

    assert result.is_error is False
    assert "offers" in _text(result)
    assert backend.attempts == 2


async def test_unexpected_backend_failure_surfaces_as_a_structured_error(repo_fixtures):
    server = build_server(
        _FailingBackend(MockUpstreamClient(repo_fixtures), ConnectionError("boom"))
    )

    result = await _call(server, "search_rail", RAIL_ARGS)
    payload = json.loads(_text(result))

    assert result.is_error is True
    assert payload["status"] == "upstream_unavailable"
    assert payload["tool"] == "search_rail"
    assert "boom" in payload["error"]


class _FailingCatalogBackend:
    """Wraps a real backend but makes `list_tools` raise, to exercise proxy
    handling of a broken catalog — as opposed to a failed `call_tool`."""

    def __init__(self, inner, exc: Exception):
        self._inner = inner
        self._exc = exc

    async def list_tools(self):
        raise self._exc

    async def call_tool(self, name, arguments):
        return await self._inner.call_tool(name, arguments)


async def test_broken_catalog_surfaces_as_a_classified_mcp_error_on_tools_list(repo_fixtures):
    server = build_server(
        _FailingCatalogBackend(MockUpstreamClient(repo_fixtures), BackendTimeoutError("timed out"))
    )

    with pytest.raises(MCPError) as excinfo:
        await _handler(server, "tools/list")(FakeCtx(), None)

    assert excinfo.value.code == types.INTERNAL_ERROR
    assert json.loads(excinfo.value.message)["status"] == "upstream_timeout"


async def test_broken_catalog_surfaces_as_a_structured_error_on_tools_call(repo_fixtures):
    server = build_server(
        _FailingCatalogBackend(MockUpstreamClient(repo_fixtures), ConnectionError("boom"))
    )

    result = await _call(server, "search_rail", RAIL_ARGS)
    payload = json.loads(_text(result))

    assert result.is_error is True
    assert payload["status"] == "upstream_unavailable"
    assert payload["tool"] == "search_rail"


async def test_instructions_result_carries_the_trimmed_prose(server):
    result = await _call(server, "get_rail_instructions", {})
    text = _text(result)

    assert "Field reference (trimmed from inputSchema)" in text
    assert "Full reference for `search_rail`" in text


async def test_an_http_client_without_a_session_id_gets_no_shared_state(server):
    """Falling back to the stdio key put every unidentified HTTP client into one
    bucket, where one client's assumptions and spent one-shot gate became the
    next client's starting position."""
    anonymous = FakeCtx()  # HTTP request, no mcp-session-id header
    args = {**RAIL_ARGS, "price_max": 3000}

    first = await _call(server, "search_rail", args, anonymous)
    second = await _call(server, "search_rail", args, FakeCtx())

    # each gets its own one-shot gate; a shared bucket would let the second through
    assert "clarification_required" in _text(first)
    assert "clarification_required" in _text(second)


async def test_stdio_still_keeps_one_conversation(server):
    """No `.request` at all means stdio, where a single shared session is correct —
    it is the same conversation, and forgetting it between calls would re-ask
    everything the user already answered."""
    stdio = type("Ctx", (), {})()
    args = {**RAIL_ARGS, "price_max": 3000}

    first = await _call(server, "search_rail", args, stdio)
    second = await _call(server, "search_rail", args, stdio)

    assert "clarification_required" in _text(first)
    assert "offers" in _text(second), "повтор того же вызова должен разблокироваться"


class _CountingCatalogBackend:
    """Counts `list_tools` calls; serves everything else from the real mock."""

    def __init__(self, inner, fail_after: int | None = None):
        self._inner = inner
        self._fail_after = fail_after
        self.loads = 0

    async def list_tools(self):
        self.loads += 1
        if self._fail_after is not None and self.loads > self._fail_after:
            raise BackendTimeoutError("timed out")
        return await self._inner.list_tools()

    async def call_tool(self, name, arguments):
        return await self._inner.call_tool(name, arguments)


async def test_an_expired_catalog_is_refetched(repo_fixtures):
    """Cached for the life of the process, the catalog kept serving tools upstream
    may have renamed or dropped hours earlier — and a stale catalog looks exactly
    like a fresh one from the outside."""
    backend = _CountingCatalogBackend(MockUpstreamClient(repo_fixtures))
    server = build_server(backend, catalog_ttl_s=0)

    await _handler(server, "tools/list")(FakeCtx(), None)
    await _handler(server, "tools/list")(FakeCtx(), None)

    assert backend.loads == 2


async def test_a_failed_refresh_keeps_serving_the_last_good_catalog(repo_fixtures):
    """Expiry must not turn a working proxy into a broken one the moment upstream
    blinks: an outdated catalog is still a usable catalog."""
    backend = _CountingCatalogBackend(MockUpstreamClient(repo_fixtures), fail_after=1)
    server = build_server(backend, catalog_ttl_s=0)

    first = await _handler(server, "tools/list")(FakeCtx(), None)
    second = await _handler(server, "tools/list")(FakeCtx(), None)

    assert {t.name for t in second.tools} == {t.name for t in first.tools}
