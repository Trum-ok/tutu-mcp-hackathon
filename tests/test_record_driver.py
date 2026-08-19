"""The recording driver, exercised without touching the network.

The live path itself (`tutu.py record`) can only be run by hand against
mcp.tutu.ru; what is pinned here is everything around it — ordering, dependency
handoff, and the fact that a missing dependency skips one spec instead of
failing the run.
"""

import json

import pytest

from evals.fixtures_recipe import FIXTURE_CALLS
from evals.scenarios import HOTEL_CHECK_IN, RAIL_DATE
from tutu_mcp.backend import ToolCallResult
from tutu_mcp.replay.bootstrap import RecordSpec, record_calls
from tutu_mcp.replay.store import FixtureStore


class _FakeClient:
    """Answers every call with `payload`, remembering what it was asked."""

    def __init__(self, payload: dict | None = None) -> None:
        self.payload = payload if payload is not None else {}
        self.calls: list[tuple[str, dict]] = []

    async def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        return ToolCallResult(text=json.dumps(self.payload, ensure_ascii=False), is_error=False)


@pytest.fixture
def store(tmp_path) -> FixtureStore:
    return FixtureStore(tmp_path)


async def test_a_later_spec_reads_an_earlier_ones_result(store):
    client = _FakeClient({"offers": [{"details_ref": "REF-1"}]})
    calls = [
        RecordSpec("search_rail", "basic", {"origin": "Мск"}),
        RecordSpec(
            "get_rail_seatmap",
            "basic",
            lambda rec: {"details_ref": rec["search_rail/basic"]["offers"][0]["details_ref"]},
        ),
    ]

    await record_calls(store, client, calls)

    assert client.calls[1] == ("get_rail_seatmap", {"details_ref": "REF-1"})
    assert store.find_result("get_rail_seatmap", {"details_ref": "REF-1"}).is_error is False


async def test_a_missing_dependency_skips_one_spec_and_keeps_going(store):
    """A partial fixture set is worth having: one empty search must not cost
    every unrelated scenario after it."""
    client = _FakeClient({"offers": []})
    calls = [
        RecordSpec("search_rail", "basic", {"origin": "Мск"}),
        RecordSpec("get_rail_seatmap", "basic", lambda rec: None),
        RecordSpec("search_bus", "basic", {"origin": "Мск"}),
    ]

    recorded = await record_calls(store, client, calls)

    assert [name for name, _ in client.calls] == ["search_rail", "search_bus"]
    assert "get_rail_seatmap/basic" not in recorded


def test_the_recipe_records_what_the_scenarios_replay():
    """The reason this list lives in `evals/`: its dates and tools are dictated by
    the scenarios, and were previously copied into the proxy package by hand."""
    static = {
        (spec.tool, spec.scenario): spec.arguments
        for spec in FIXTURE_CALLS
        if isinstance(spec.arguments, dict)
    }

    assert static[("search_rail", "spb_msk_basic")]["departure_date"] == RAIL_DATE
    assert static[("search_hotels", "spb_basic")]["check_in"] == HOTEL_CHECK_IN
    # Every tool an eval scenario can reach has at least one recorded spec.
    assert {"search_rail", "search_avia", "search_bus", "search_etrain", "search_hotels"} <= {
        spec.tool for spec in FIXTURE_CALLS
    }
