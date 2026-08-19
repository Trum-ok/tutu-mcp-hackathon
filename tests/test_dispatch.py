"""`dispatch()`'s own contracts, as opposed to the transport wrapping around it
(covered by `test_proxy_server.py`/`test_premise_evals.py`).
"""

import json

from tutu_mcp.backend import ToolCallResult
from tutu_mcp.premises import SessionPremises
from tutu_mcp.proxy.dispatch import dispatch

RAIL_ARGS = {
    "origin": "Санкт-Петербург",
    "destination": "Москва",
    "departure_date": "2026-08-25",
}


class _EchoingErrorBackend:
    """A backend whose call returns a normal (non-exception) `is_error` result
    that happens to echo a value back — the shape a real Tutu business error
    (not a fixture miss) can take."""

    async def list_tools(self):
        return []

    async def call_tool(self, name, arguments):
        return ToolCallResult(
            text='{"error": "unknown details_ref", "echo": {"price_max": 7777}}',
            is_error=True,
        )


async def test_error_result_does_not_seed_the_premise_gates_seen_values():
    """A failed call's echoed value must not count as "seen" — otherwise an
    agent could launder an invented number through any tool that echoes its
    arguments back on failure, then reuse it past the premise gate."""
    session = SessionPremises()
    backend = _EchoingErrorBackend()

    # get_offer_details carries no premise policy, so this reaches the backend
    # and returns unchanged — the failure itself is not what's under test.
    first = await dispatch(session, backend, "get_offer_details", {"details_ref": "bogus"}, {})
    assert first.is_error is True

    second = await dispatch(session, backend, "search_rail", {**RAIL_ARGS, "price_max": 7777}, {})

    assert json.loads(second.text)["status"] == "clarification_required"
