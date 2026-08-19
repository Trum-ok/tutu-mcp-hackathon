import pytest

from tutu_mcp.backend import ToolCallResult
from tutu_mcp.replay.store import (
    FixtureCorruptError,
    FixtureMissingError,
    FixtureNotFoundError,
    FixtureStore,
    normalize_arguments,
)


def test_save_and_find_result_roundtrips_on_exact_arguments(tmp_path):
    store = FixtureStore(tmp_path)
    args = {"origin": "Москва", "destination": "Питер", "departure_date": "2026-08-25"}
    store.save_tool_result("search_rail", "basic", args, ToolCallResult(text="{}", is_error=False))

    result = store.find_result("search_rail", dict(args))  # a fresh dict, same content

    assert result == ToolCallResult(text="{}", is_error=False)


def test_argument_order_does_not_affect_matching(tmp_path):
    store = FixtureStore(tmp_path)
    store.save_tool_result(
        "search_rail",
        "basic",
        {"a": 1, "b": 2},
        ToolCallResult(text="ok", is_error=False),
    )

    assert store.find_result("search_rail", {"b": 2, "a": 1}).text == "ok"


def test_missing_fixture_raises_with_available_scenarios_listed(tmp_path):
    store = FixtureStore(tmp_path)
    store.save_tool_result(
        "search_rail", "known", {"origin": "A"}, ToolCallResult(text="ok", is_error=False)
    )

    with pytest.raises(FixtureNotFoundError) as excinfo:
        store.find_result("search_rail", {"origin": "B"})

    assert "known" in str(excinfo.value)


def test_missing_tool_directory_raises_cleanly(tmp_path):
    store = FixtureStore(tmp_path)

    with pytest.raises(FixtureNotFoundError):
        store.find_result("search_avia", {"origin": "A"})


def _store_with_schema(tmp_path, defaults: dict) -> FixtureStore:
    store = FixtureStore(tmp_path)
    store.save_tools_list(
        [
            {
                "name": "search_rail",
                "description": "",
                "inputSchema": {
                    "type": "object",
                    "properties": {k: {"default": v} for k, v in defaults.items()},
                },
            }
        ]
    )
    return store


def test_arguments_spelled_out_at_their_schema_default_still_match(tmp_path):
    """A model writes every optional argument explicitly; a human recording
    fixtures writes none of them. Both are the same request upstream, so both must
    reach the same fixture — otherwise mock-mode evals are almost all misses."""
    store = _store_with_schema(tmp_path, {"page": 1, "page_size": 10, "view": "compact"})
    store.save_tool_result(
        "search_rail", "basic", {"origin": "Мск"}, ToolCallResult(text="ok", is_error=False)
    )

    result = store.find_result(
        "search_rail",
        {"origin": "Мск", "page": 1, "page_size": 10, "view": "compact", "carriers": None},
    )

    assert result.text == "ok"


def test_a_non_default_value_is_a_different_request_and_still_misses(tmp_path):
    store = _store_with_schema(tmp_path, {"page_size": 10})
    store.save_tool_result(
        "search_rail", "basic", {"origin": "Мск"}, ToolCallResult(text="ok", is_error=False)
    )

    with pytest.raises(FixtureNotFoundError):
        store.find_result("search_rail", {"origin": "Мск", "page_size": 30})


def test_matching_falls_back_to_exact_when_no_catalog_is_recorded(tmp_path):
    store = FixtureStore(tmp_path)  # no _meta/tools_list.json
    store.save_tool_result(
        "search_rail", "basic", {"origin": "Мск"}, ToolCallResult(text="ok", is_error=False)
    )

    assert store.find_result("search_rail", {"origin": "Мск"}).text == "ok"


@pytest.mark.parametrize("empty", [None, "", []])
def test_an_emptied_out_filter_still_finds_its_fixture(repo_fixtures, empty):
    """An agent told to drop an argument often sends it as empty instead of
    omitting it. All three spellings mean the same request, so all three must hit
    the recording made without the field."""
    args = {
        "origin": "Москва",
        "destination": "Санкт-Петербург",
        "departure_date": "2026-08-25",
        "service_class": empty,
    }
    result = repo_fixtures.find_result("search_avia", args)
    assert not result.is_error


def test_false_is_not_treated_as_empty():
    """`direct_only=False` is a real answer, not an unset field — conflating them
    would make a filtered search play back an unfiltered recording."""
    assert normalize_arguments({"direct_only": False}) == '{"direct_only": false}'


def test_load_payload_returns_the_tool_body_by_scenario_name(tmp_path):
    store = FixtureStore(tmp_path)
    store.save_tool_result(
        "search_rail",
        "basic",
        {"origin": "A"},
        ToolCallResult(text='{"offers": [{"price": 1}]}', is_error=False),
    )

    assert store.load_payload("search_rail", "basic") == {"offers": [{"price": 1}]}


@pytest.mark.parametrize(
    "body",
    [
        "{ not json",
        '{"result": {"txt": "wrong key"}}',
        '{"result": {"text": "не JSON, а просто текст"}}',
    ],
    ids=["invalid-json", "missing-text", "text-is-not-json"],
)
def test_load_payload_reports_a_corrupt_fixture_instead_of_raising_json_errors(tmp_path, body):
    """Тот же класс ошибки, что и на пути воспроизведения: битую запись чинят
    перезаписью, и сообщение должно называть файл, а не ронять трейсбек в
    середину `make demo-traces`."""
    store = FixtureStore(tmp_path)
    (tmp_path / "search_rail").mkdir()
    (tmp_path / "search_rail" / "basic.json").write_text(body, encoding="utf-8")

    with pytest.raises(FixtureCorruptError) as excinfo:
        store.load_payload("search_rail", "basic")

    assert "search_rail/basic.json" in str(excinfo.value)


def test_load_payload_of_an_unrecorded_scenario_is_a_gap_not_a_crash(tmp_path):
    with pytest.raises(FixtureMissingError):
        FixtureStore(tmp_path).load_payload("search_rail", "never_recorded")
