"""The one `tools/call` pipeline, shared by the live MCP server and the eval
harness's proxy variant.

Before this module existed, `proxy/server.py` and `evals/variants.py` each
hand-wrote the same routing → premise gate → backend call → appendix/preamble
sequence. The two copies drifted (the eval harness's `check_groundedness`
branch was missing the baseline guard `assess_request` already had, and it
caught a narrower set of backend failures than the server did) because a fix
to one was never mechanically forced onto the other. Routing through one
function makes that drift impossible: both callers see the exact same
decisions, just wrapped for their own transport (`types.CallToolResult` for
MCP, `ToolCallRecord` for the eval transcript).
"""

import json
from dataclasses import dataclass
from typing import Any

from tutu_mcp.backend import BackendError, ToolBackend, call_with_timeout_retry
from tutu_mcp.premises import SessionPremises, strip_control_fields

from .compact_tools import apply_result_appendix
from .surface import SYNTHETIC


@dataclass(frozen=True)
class DispatchResult:
    text: str
    is_error: bool
    # True when the mock backend had no fixture for these arguments — a gap in
    # OUR eval recording, not Tutu misbehaving. Kept separate from `is_error`
    # so the two never get conflated by a caller that only checks one of them.
    fixture_miss: bool = False


def backend_error(name: str, exc: Exception) -> DispatchResult:
    """Turns any backend failure — a tool call or a catalog load — into a
    domain-shaped error result.

    Without this, only a fixture miss was handled — a live upstream timeout or
    a broken connection would escape as a bare SDK exception, indistinguishable
    to the agent from a bug in the proxy itself. `status` lets it branch (retry
    a timeout, give up on a hard failure) instead of just seeing a generic error.
    Exported so `proxy/server.py` can apply the same classification to a failed
    `backend.list_tools()`, which `dispatch()` itself never calls.

    Classification itself is not this function's job: each `ToolBackend`
    translates its own failures into a `BackendError` subclass at its own
    boundary (see `tutu_mcp.backend`), so this only has to read `.status` —
    it never needs to know which backend implementation raised.
    """
    status = exc.status if isinstance(exc, BackendError) else "upstream_unavailable"
    payload = {"status": status, "tool": name, "error": str(exc)}
    return DispatchResult(
        json.dumps(payload, ensure_ascii=False),
        is_error=True,
        fixture_miss=status == "fixture_not_found",
    )


async def dispatch(
    session: SessionPremises | None,
    backend: ToolBackend,
    name: str,
    arguments: dict[str, Any],
    trimmed_originals: dict[str, str],
) -> DispatchResult:
    """Runs one `tools/call` through the full pipeline: the two synthetic
    tools, the premise gate, the backend call, and the result appendix/preamble.

    `session=None` models a client that was never given the premise gate at
    all — the eval harness's baseline variant, measuring the untouched
    upstream surface. On that surface the tools in `SYNTHETIC` (`assess_request`,
    `check_groundedness`) are simply absent from `tools/list`, so both report
    themselves as an unknown tool rather than leaking proxy-only behavior to a
    client that was never handed them.
    """
    if name in SYNTHETIC:
        if session is None:
            return DispatchResult(f"unknown tool: {name}", is_error=True)
        _, run = SYNTHETIC[name]
        text, is_error = run(arguments, session)
        return DispatchResult(text, is_error)

    # Control fields are ours, not Tutu's: strip them before the call goes out,
    # or the fixture store misses on every scenario and upstream is handed a
    # field its schema never declared.
    clean, sources, assume = strip_control_fields(arguments)

    if session is not None:
        decision = session.evaluate(name, clean, sources, assume)
        if decision is not None:
            # Deliberately NOT is_error: clients surface a tool error to the user
            # as a red failure and agents retry it blindly. This is a successful
            # call whose payload happens to be a question.
            return DispatchResult(decision.to_json(), is_error=False)

    try:
        result = await call_with_timeout_retry(lambda: backend.call_tool(name, clean))
    except Exception as exc:
        return backend_error(name, exc)

    text = apply_result_appendix(name, result.text, trimmed_originals)
    if session is not None:
        if not result.is_error:
            # An error payload's own echoed arguments (e.g. a fixture miss quoting
            # the invented value back) must never become a "seen" value — that
            # would let the premise gate wave the same invention through next call.
            session.record_result(result.text)
        preamble = session.preamble()
        if preamble:
            text = f"{text}\n\n## Обязательная преамбула ответа\n{preamble}"

    return DispatchResult(text, result.is_error)
