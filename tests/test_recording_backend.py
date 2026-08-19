"""Record-on-miss: what gets written to disk, and what deliberately does not.

The fixture set is the only thing standing between an offline run and a made-up
number, so the interesting question here is not "does it record" but "what does
it refuse to record".
"""

from typing import Any

from tutu_mcp.backend import ToolCallResult
from tutu_mcp.replay.recording import RecordingBackend
from tutu_mcp.replay.store import FixtureStore, scenario_slug

ARGS = {"origin": "Санкт-Петербург", "destination": "Москва"}


class StubUpstream:
    """Stands in for the live client: records what it was asked, answers as told."""

    def __init__(self, result: ToolCallResult, tools: list[dict[str, Any]] | None = None) -> None:
        self._result = result
        self._tools = tools or []
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def list_tools(self) -> list[dict[str, Any]]:
        return self._tools

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> ToolCallResult:
        self.calls.append((name, arguments))
        return self._result


def _backend(tmp_path, result: ToolCallResult) -> tuple[RecordingBackend, StubUpstream]:
    upstream = StubUpstream(result)
    return RecordingBackend(FixtureStore(tmp_path), upstream), upstream


async def test_a_miss_is_fetched_once_and_written_to_disk(tmp_path):
    ok = ToolCallResult(text='{"offers": []}', is_error=False)
    backend, upstream = _backend(tmp_path, ok)

    assert await backend.call_tool("search_rail", ARGS) == ok
    assert len(upstream.calls) == 1

    scenario = scenario_slug("search_rail", ARGS)
    assert (tmp_path / "search_rail" / f"{scenario}.json").is_file()
    assert backend.recorded == [("search_rail", scenario)]
    # and the second call is served from what was just written
    assert await backend.call_tool("search_rail", ARGS) == ok
    assert len(upstream.calls) == 1


async def test_an_upstream_error_is_never_written_as_a_fixture(tmp_path):
    """A 429 or a timeout is an outage, not Tutu's answer for these arguments.

    Recorded once, it would replay as data on every later offline run — the whole
    fixture set silently describing an upstream that was down that afternoon.
    """
    failure = ToolCallResult(text='{"status": "rate_limited"}', is_error=True)
    backend, _ = _backend(tmp_path, failure)

    assert await backend.call_tool("search_rail", ARGS) == failure

    scenario = scenario_slug("search_rail", ARGS)
    assert not (tmp_path / "search_rail" / f"{scenario}.json").exists()
    assert backend.recorded == []
    # named, so the run can report the miss as still open
    assert backend.skipped_errors == [("search_rail", scenario)]


async def test_a_refused_recording_leaves_the_miss_open(tmp_path):
    """The next attempt must go upstream again, not read a hole that was never dug."""
    backend, upstream = _backend(tmp_path, ToolCallResult(text="{}", is_error=True))

    await backend.call_tool("search_rail", ARGS)
    await backend.call_tool("search_rail", ARGS)

    assert len(upstream.calls) == 2


async def test_the_catalog_is_recorded_on_first_use(tmp_path):
    tools = [{"name": "search_rail", "description": "", "inputSchema": {}}]
    store = FixtureStore(tmp_path)
    upstream = StubUpstream(ToolCallResult(text="{}", is_error=False), tools=tools)
    backend = RecordingBackend(store, upstream)

    assert await backend.list_tools() == tools
    assert store.load_tools_list() == tools
