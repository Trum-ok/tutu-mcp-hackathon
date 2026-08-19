"""A broken fixture on OUR disk must never be reported as a broken Tutu.

`dispatch.backend_error` classifies by `.status`, so anything the store lets
escape untranslated lands in the agent's lap as `upstream_unavailable` — the
one classification that is certainly wrong when the fault is a file we
recorded ourselves.
"""

import json
from pathlib import Path

import pytest

from tutu_mcp.proxy.dispatch import backend_error, dispatch
from tutu_mcp.replay.mock_client import MockUpstreamClient
from tutu_mcp.replay.store import FixtureCorruptError, FixtureMissingError, FixtureStore

RAIL_ARGS = {"origin": "Питер", "destination": "Москва"}


@pytest.fixture
def empty_store(tmp_path: Path) -> FixtureStore:
    return FixtureStore(tmp_path)


def write_scenario(store: FixtureStore, body: str) -> None:
    tool_dir = store.fixtures_dir / "search_rail"
    tool_dir.mkdir(parents=True, exist_ok=True)
    (tool_dir / "a.json").write_text(body, encoding="utf-8")


async def test_missing_catalog_is_a_fixture_gap_not_an_upstream_outage(empty_store):
    client = MockUpstreamClient(empty_store)

    with pytest.raises(FixtureMissingError) as excinfo:
        await client.list_tools()

    payload = json.loads(backend_error("tools/list", excinfo.value).text)
    assert payload["status"] == "fixture_not_found"


async def test_malformed_catalog_reports_a_corrupt_fixture(empty_store):
    (empty_store.fixtures_dir / "_meta").mkdir()
    (empty_store.fixtures_dir / "_meta/tools_list.json").write_text("{ nope", encoding="utf-8")
    client = MockUpstreamClient(empty_store)

    with pytest.raises(FixtureCorruptError):
        await client.list_tools()


@pytest.mark.parametrize(
    "body",
    [
        "{ not json",
        json.dumps({"scenario": "a", "arguments": RAIL_ARGS, "result": {"txt": "wrong key"}}),
        json.dumps({"scenario": "a", "arguments": RAIL_ARGS}),
    ],
    ids=["invalid-json", "bad-result-shape", "missing-result"],
)
async def test_corrupt_scenario_is_counted_as_our_gap(empty_store, body):
    write_scenario(empty_store, body)

    result = await dispatch(None, MockUpstreamClient(empty_store), "search_rail", RAIL_ARGS, {})

    assert json.loads(result.text)["status"] == "fixture_corrupt"
    # The eval report separates "Tutu failed" from "we never recorded this";
    # a file we wrote badly belongs on our side of that line.
    assert result.fixture_miss is True


async def test_error_text_names_the_file_without_leaking_the_host_path(empty_store):
    write_scenario(empty_store, "{ not json")

    result = await dispatch(None, MockUpstreamClient(empty_store), "search_rail", RAIL_ARGS, {})

    message = json.loads(result.text)["error"]
    assert "search_rail/a.json" in message
    assert str(empty_store.fixtures_dir) not in message


async def test_demo_traces_name_the_unreadable_fixture_instead_of_tracebacking(
    tmp_path, monkeypatch, capsys
):
    """`make demo-traces` строит планы по фикстурам ещё до того, как что-то
    подключено. На свежем клоне без записей это нормальное состояние, а не сбой
    харнесса, — и выглядеть оно должно как остальные проверки предусловий."""
    from evals.demo import run_demo

    monkeypatch.setenv("TUTU_FIXTURES_DIR", str(tmp_path))

    assert await run_demo() == 2

    err = capsys.readouterr().err
    assert "search_rail/spb_msk_basic.json" in err
    assert "tutu.py record" in err


async def test_the_self_check_reports_a_corrupt_fixture_the_same_way(tmp_path, monkeypatch, capsys):
    from evals.options import AgentKind, EvalOptions
    from evals.run import run_evals

    write_scenario(FixtureStore(tmp_path), "{ not json")
    (tmp_path / "search_rail" / "spb_msk_basic.json").write_text("{ not json", encoding="utf-8")
    monkeypatch.setenv("TUTU_FIXTURES_DIR", str(tmp_path))

    assert await run_evals(EvalOptions(agent=AgentKind.SCRIPTED)) == 2

    assert "spb_msk_basic.json" in capsys.readouterr().err
