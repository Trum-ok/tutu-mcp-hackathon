import json

import pytest

from tutu_mcp.backend import BackendMissError
from tutu_mcp.proxy.dispatch import backend_error
from tutu_mcp.replay.mock_client import MockUpstreamClient, UnknownToolError
from tutu_mcp.replay.store import FixtureNotFoundError, FixtureStore


@pytest.fixture
def mock_client(repo_fixtures) -> MockUpstreamClient:
    return MockUpstreamClient(repo_fixtures)


async def test_list_tools_returns_the_recorded_catalog(mock_client):
    tools = await mock_client.list_tools()

    names = {t["name"] for t in tools}
    assert "search_rail" in names
    assert "create_checkout_link" in names
    assert len(tools) == 16


async def test_call_tool_replays_a_recorded_scenario(mock_client):
    result = await mock_client.call_tool(
        "search_rail",
        {"origin": "Санкт-Петербург", "destination": "Москва", "departure_date": "2026-08-25"},
    )

    assert result.is_error is False
    assert "offers" in result.text


async def test_call_tool_replays_the_recorded_error_scenario(mock_client):
    result = await mock_client.call_tool(
        "search_rail",
        {"origin": "Санкт-Петербург", "destination": "Москва", "departure_date": "not-a-date"},
    )

    assert result.is_error is True


async def test_call_tool_raises_for_an_unrecorded_argument_combination(mock_client):
    with pytest.raises(FixtureNotFoundError):
        await mock_client.call_tool("search_rail", {"origin": "nowhere-recorded"})


async def test_an_invented_tool_name_is_not_reported_as_a_missing_fixture(mock_client):
    """The old message sent the caller to `tutu.py record` for a tool that does
    not exist — recording it is impossible, so the advice could only mislead."""
    with pytest.raises(UnknownToolError) as excinfo:
        await mock_client.call_tool("no_such_tool", {})

    message = str(excinfo.value)
    assert "search_rail" in message
    assert "16" in message
    assert "record" not in message


async def test_an_unknown_tool_reaches_the_agent_as_its_own_status(mock_client):
    with pytest.raises(UnknownToolError) as excinfo:
        await mock_client.call_tool("serch_rail", {})

    result = backend_error("serch_rail", excinfo.value)
    payload = json.loads(result.text)

    assert payload["status"] == "unknown_tool"
    # not a hole in what we recorded, so it must not inflate the fixture-miss count
    assert result.fixture_miss is False


async def test_a_fresh_clone_still_reports_the_missing_recording(tmp_path):
    """No catalog on disk is a normal state of a fresh clone, not a wrong name —
    the gap worth naming there is the missing recording."""
    client = MockUpstreamClient(FixtureStore(tmp_path))

    with pytest.raises(BackendMissError) as excinfo:
        await client.call_tool("search_rail", {})

    assert not isinstance(excinfo.value, UnknownToolError)
