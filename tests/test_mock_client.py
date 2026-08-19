import pytest

from tutu_mcp.replay.mock_client import MockUpstreamClient
from tutu_mcp.replay.store import FixtureNotFoundError


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
