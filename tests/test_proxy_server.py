"""The assembled MCP server: catalog contents, gate wiring, session isolation.

Reaches the registered handlers through `_lowlevel_server._request_handlers`,
the same private seam `build_server` writes to — so these exercise the real
dispatch path rather than re-calling the helpers it happens to be built from.
"""

import json

import mcp_types as types
import pytest

from tutu_mcp.proxy.server import MAX_SESSIONS, PROXY_INSTRUCTIONS, build_server
from tutu_mcp.replay.mock_client import MockUpstreamClient

RAIL_ARGS = {
    "origin": "Санкт-Петербург",
    "destination": "Москва",
    "departure_date": "2026-08-25",
}


class FakeCtx:
    """Stands in for a request context carrying one session's headers."""

    def __init__(self, session_id: str | None = None):
        headers = {"mcp-session-id": session_id} if session_id else {}
        self.transport = type("T", (), {"headers": headers})()


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


async def test_declared_assumption_appends_a_mandatory_preamble(server):
    result = await _call(
        server, "search_rail", {**RAIL_ARGS, "price_max": 3000, "_assume": {"price_max": "бюджет"}}
    )
    text = _text(result)

    assert "Обязательная преамбула" in text
    assert "бюджет" in text


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
    assert "offers" in _text(second)
    assert "Обязательная преамбула" in _text(second)


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


async def test_instructions_result_carries_the_trimmed_prose(server):
    result = await _call(server, "get_rail_instructions", {})
    text = _text(result)

    assert "Field reference (trimmed from inputSchema)" in text
    assert "Full reference for `search_rail`" in text
