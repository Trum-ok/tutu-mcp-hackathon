"""Deterministic checks, tested in isolation from the runner they plug into."""

from evals.checks import (
    ABSENCE_CLAIM_PHRASES,
    AnswerAvoids,
    ClarifiedBeforeSearch,
    DidNotAsk,
    GateFired,
)
from evals.transcript import ToolCallRecord, Transcript
from tutu_mcp.groundedness import GroundednessReport

EMPTY_GROUNDING = GroundednessReport(checks=[])


def test_url_query_string_is_not_mistaken_for_a_clarifying_question():
    """A checkout/search link's `?tn=749&date=...` must not read as a question in
    either direction: it should neither trip the over-ask guard nor satisfy the
    "clarified before search" check on an answer that never actually asked anything."""
    answer = (
        "Самый дешёвый — 749У за 3168 ₽. Оформить: "
        "https://example.tutu.ru/order/?tn=749&date=2026-08-25"
    )
    transcript = Transcript(scenario_id="s", variant="proxy", answer_text=answer)

    assert DidNotAsk().run(transcript, EMPTY_GROUNDING).passed is True
    assert ClarifiedBeforeSearch().run(transcript, EMPTY_GROUNDING).passed is False


def test_a_real_clarifying_question_outside_a_url_still_counts():
    transcript = Transcript(
        scenario_id="s", variant="proxy", answer_text="На какую дату ищем поезд?"
    )

    assert DidNotAsk().run(transcript, EMPTY_GROUNDING).passed is False
    assert ClarifiedBeforeSearch().run(transcript, EMPTY_GROUNDING).passed is True


def _tool_call(result_text: str) -> ToolCallRecord:
    return ToolCallRecord(name="search_rail", arguments={}, result_text=result_text, is_error=False)


def test_gate_fired_detects_a_real_gate_decision():
    gate_payload = '{"status": "clarification_required", "tool": "search_rail", "reason": "..."}'
    transcript = Transcript(scenario_id="s", variant="proxy", tool_calls=[_tool_call(gate_payload)])

    assert GateFired().run(transcript, EMPTY_GROUNDING).passed is True


def test_gate_fired_ignores_a_coincidental_substring_match():
    """A result that merely quotes the phrase (an echoed upstream message, a
    translated error) must not be mistaken for the gate's own structured
    payload — only the `status` field counts, not the raw text."""
    echoed = (
        '{"status": "upstream_unavailable", "tool": "search_rail", '
        '"error": "upstream said: clarification_required for this route"}'
    )
    transcript = Transcript(scenario_id="s", variant="proxy", tool_calls=[_tool_call(echoed)])

    assert GateFired().run(transcript, EMPTY_GROUNDING).passed is False


REAL_DISCLAIMED_ANSWER = (
    "На 25 августа 2026 года поезд №999999 из Санкт-Петербурга в Москву не найден в продаже: "
    "0 предложений, 48 вариантов по маршруту исключены по фильтру номера поезда. Это означает, "
    "что на сервисе нет предложения под этим номером; по этому поиску нельзя утверждать, "
    "что поезд вообще не ходит по расписанию."
)


def test_naming_a_claim_in_order_to_refuse_it_is_not_making_it():
    """Verbatim from a real proxy run. The answer is exactly what the check exists
    to reward, and substring matching failed it for quoting the wording it refused."""
    transcript = Transcript(scenario_id="s", variant="proxy", answer_text=REAL_DISCLAIMED_ANSWER)
    result = AnswerAvoids(ABSENCE_CLAIM_PHRASES, label="avoids_timetable_claim").run(
        transcript, GroundednessReport(checks=[])
    )

    assert result.passed, result.detail


def test_the_bare_claim_still_fails():
    transcript = Transcript(
        scenario_id="s", variant="baseline", answer_text="Такой поезд не ходит по этому маршруту."
    )
    result = AnswerAvoids(ABSENCE_CLAIM_PHRASES, label="avoids_timetable_claim").run(
        transcript, GroundednessReport(checks=[])
    )

    assert not result.passed


def test_a_disclaimer_far_away_does_not_license_a_later_claim():
    """One clause of reach, not the whole answer — otherwise a single "это не так"
    at the top would license every fabrication below it."""
    text = (
        "Это не окончательный ответ. "
        + "Дальше идёт длинное описание маршрута и вагонов. " * 3
        + "Поезд не ходит."
    )
    transcript = Transcript(scenario_id="s", variant="baseline", answer_text=text)
    result = AnswerAvoids(ABSENCE_CLAIM_PHRASES, label="avoids_timetable_claim").run(
        transcript, GroundednessReport(checks=[])
    )

    assert not result.passed


def test_the_missing_field_scenario_asks_about_something_actually_missing():
    """Guards the scenario against the fixture set drifting under it: the whole
    point is that the server did not return the field, so if a recording ever
    starts carrying it, this scenario silently stops testing anything."""
    import json
    from pathlib import Path

    from evals.scenarios import SCENARIOS_BY_ID

    request = SCENARIOS_BY_ID["hotels_missing_field"].request.lower()
    asked = [word for word in ("заезд", "парковк") if word in request]
    assert asked, "сценарий должен спрашивать про поля, перечисленные здесь"

    fixtures = Path("fixtures/search_hotels").glob("*.json")
    for path in fixtures:
        text = json.loads(path.read_text(encoding="utf-8"))["result"]["text"].lower()
        for word in asked:
            assert word not in text, (
                f"{path.name} содержит «{word}» — сценарий больше не про пробел"
            )
