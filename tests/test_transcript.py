"""`Transcript.result_payloads()` feeds `check_groundedness`'s evidence set —
what it excludes matters as much as what it includes.
"""

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
