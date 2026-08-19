"""An empty search result is the article's most-cited failure: the agent reads it
as "поезда нет" when it only ever meant "нет предложений под эти фильтры"."""

import json

import pytest

from tutu_mcp.groundedness import check_groundedness
from tutu_mcp.proxy.empty_results import (
    EMPTY_RESULT_KEY,
    annotate_empty_result,
    empty_result_note,
)

from .conftest import load_result_payload


def test_a_filtered_empty_result_names_the_filter_that_emptied_it():
    """The real recorded case: train_numbers=[999999] over a route with 48 trains.
    The server counts what it dropped, so the agent can state a fact instead of
    guessing at a timetable it was never given."""
    payload = load_result_payload("search_rail", "spb_msk_no_such_train")
    note = empty_result_note(payload)

    assert note is not None
    assert "48" in note and "номеру поезда" in note
    assert "не ходит" in note, "формулировка, которой агент должен избежать, названа прямо"


def test_an_empty_result_with_no_filters_still_warns_off_the_timetable_claim():
    note = empty_result_note({"offers": [], "meta": {"total_matched": 0}})

    assert note is not None
    assert "расписание" in note


def test_a_non_empty_result_is_left_alone():
    payload = load_result_payload("search_rail", "spb_msk_basic")

    assert empty_result_note(payload) is None
    assert annotate_empty_result(json.dumps(payload)) == json.dumps(payload)


@pytest.mark.parametrize("text", ["not json at all", "[1, 2, 3]", '"a string"'])
def test_anything_that_is_not_a_result_object_passes_through(text):
    assert annotate_empty_result(text) == text


def test_the_annotated_result_is_still_valid_json():
    payload = load_result_payload("search_rail", "spb_msk_no_such_train")
    annotated = json.loads(annotate_empty_result(json.dumps(payload)))

    assert annotated["offers"] == []
    assert annotated["meta"]["post_filter_dropped_wrong_train_number"] == 48
    assert EMPTY_RESULT_KEY in annotated


def test_the_note_can_never_ground_a_claim():
    """It carries numbers ("48"), and an `_`-prefixed key keeps them out of the
    evidence index — otherwise our own advice could confirm an invented figure."""
    annotated = json.loads(
        annotate_empty_result('{"offers": [], "meta": {"post_filter_dropped_over_cap": 48}}')
    )
    report = check_groundedness("Нашлось 48 вариантов по 48 ₽", [annotated])

    assert [c.status for c in report.checks] == ["unavailable"]
