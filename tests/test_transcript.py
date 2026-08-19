"""`Transcript.result_payloads()` feeds `check_groundedness`'s evidence set —
what it excludes matters as much as what it includes.
"""

import json

from evals.transcript import ToolCallRecord, Transcript


def test_result_payloads_excludes_error_calls():
    """A failed call's JSON body must never "confirm" a fabricated claim: our own
    error payloads (`tutu_mcp.proxy.dispatch.backend_error`) can legitimately
    contain the same numbers or codes an agent invented, e.g. inside a fixture
    miss's echoed arguments."""
    transcript = Transcript(
        scenario_id="s",
        variant="proxy",
        tool_calls=[
            ToolCallRecord(
                name="search_rail",
                arguments={},
                result_text=(
                    '{"status": "fixture_not_found", "tool": "search_rail", '
                    '"error": "No fixture for search_rail(train_numbers=[\'777A\'])"}'
                ),
                is_error=True,
            ),
            ToolCallRecord(
                name="search_rail",
                arguments={},
                result_text='{"offers": [{"price": {"amount": 1500}}]}',
                is_error=False,
            ),
        ],
    )

    payloads = transcript.result_payloads()

    assert len(payloads) == 1
    assert payloads[0]["offers"][0]["price"]["amount"] == 1500


def _call(name, payload):
    return ToolCallRecord(name=name, arguments={}, result_text=json.dumps(payload), is_error=False)


def test_a_preflight_verdict_counts_as_the_gate_firing():
    """The ideal run: `assess_request` says ask, the agent asks, and no expensive
    search ever happens. Counting only a gated `tools/call` scored that as the
    mechanism failing."""
    t = Transcript(scenario_id="s", variant="proxy")
    t.tool_calls.append(_call("assess_request", {"verdict": "ask_user_first"}))

    assert t.gate_fired()


def test_a_clean_preflight_is_not_a_firing():
    t = Transcript(scenario_id="s", variant="proxy")
    t.tool_calls.append(_call("assess_request", {"verdict": "proceed"}))

    assert not t.gate_fired()


def test_a_gated_call_still_counts():
    t = Transcript(scenario_id="s", variant="proxy")
    t.tool_calls.append(_call("search_hotels", {"status": "clarification_required"}))

    assert t.gate_fired()
