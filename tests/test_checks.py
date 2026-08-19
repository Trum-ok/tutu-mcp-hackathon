"""Deterministic checks, tested in isolation from the runner they plug into."""

from evals.checks import ClarifiedBeforeSearch, DidNotAsk, GateFired
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
