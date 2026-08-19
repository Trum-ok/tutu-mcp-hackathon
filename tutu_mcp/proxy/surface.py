"""The proxy's own tool surface: the two synthetic tools it adds on top of
whatever Tutu exposes (`assess_request`, `check_groundedness`), plus the
catalog assembly and always-on instructions both the live server and the eval
harness build on.

Before this module existed, the catalog list `[*compacted, ASSESS_REQUEST_TOOL,
CHECK_GROUNDEDNESS_TOOL]` was hand-written twice (`proxy/server.py` and
`evals/variants.py`), and `proxy/dispatch.py` named the same two tools a third
time as plain strings, unconnected to either list. None of the three copies
were derived from each other, so adding a third synthetic tool meant editing
three files in three layers and hoping none were missed. `SYNTHETIC` is now
the one place a synthetic tool is declared; `proxy_catalog` and `dispatch`
both read it instead of naming tools by hand.
"""

from collections.abc import Callable
from typing import Any

from tutu_mcp.groundedness import CHECK_GROUNDEDNESS_TOOL, run_check_groundedness_tool
from tutu_mcp.premises import (
    ASSESS_REQUEST_TOOL,
    GATE_STATUS,
    SessionPremises,
    run_assess_request_tool,
)

_ToolRunner = Callable[[dict[str, Any], SessionPremises], tuple[str, bool]]


def _run_assess_request(arguments: dict[str, Any], session: SessionPremises) -> tuple[str, bool]:
    return run_assess_request_tool(arguments, session)


def _run_check_groundedness(
    arguments: dict[str, Any], session: SessionPremises
) -> tuple[str, bool]:
    return run_check_groundedness_tool(
        arguments,
        assumed_values=session.assumed_values(),
        assumptions=session.assumption_lines(),
        # Evidence comes from what this proxy actually delivered, not from what the
        # agent chose to hand back — see the tool's docstring.
        session_payloads=session.result_payloads(),
    )


SYNTHETIC: dict[str, tuple[dict[str, Any], _ToolRunner]] = {
    ASSESS_REQUEST_TOOL["name"]: (ASSESS_REQUEST_TOOL, _run_assess_request),
    CHECK_GROUNDEDNESS_TOOL["name"]: (CHECK_GROUNDEDNESS_TOOL, _run_check_groundedness),
}
"""name -> (its `tools/list` spec, the function that runs it). Every tool the
proxy adds that Tutu itself doesn't have."""


def proxy_catalog(compacted: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The full `tools/list` payload: Tutu's own (compacted) tools plus ours."""
    return [*compacted, *(spec for spec, _ in SYNTHETIC.values())]


PROXY_INSTRUCTIONS = (
    "Compacted proxy in front of mcp.tutu.ru: identical tools and behavior, but the "
    "always-on tools/list catalog is trimmed for the biggest offenders (see each "
    "trimmed tool's description for which get_<domain>_instructions tool absorbed the "
    "detail — that tool's CALL RESULT carries it, not its tools/list entry). Also "
    f"exposes `{CHECK_GROUNDEDNESS_TOOL['name']}` — before answering, pass your drafted "
    "answer text plus the raw JSON text of every tool_result you used this turn, and it "
    "flags any price/time/train-or-flight-code/URL in your answer that isn't actually "
    "present in those results.\n\n"
    "PREMISE GATE — read before your first search. Every value that NARROWS a search "
    "must come from the user or from an earlier tool_result; there is no third source. "
    f"Call `{ASSESS_REQUEST_TOOL['name']}` FIRST with the user's request verbatim (it is "
    "local, free and instant) — it surfaces which parameters are blocking, flags a date "
    "that contradicts the weekday the user gave, and lets values the user actually typed "
    "pass the gate without a retry. When a search argument has no such source, the call "
    f"returns `{GATE_STATUS}` INSTEAD of data. Every blocked slot carries a `do` saying "
    "how to close it and a `resolution` naming its kind: `drop_filter` means YOU invented "
    "that filter — remove the argument and repeat, and do NOT ask the user about a "
    "constraint they never mentioned; `call_tool` means another tool closes the gap; "
    "`ask_user` means only the user can, and that is the one worth interrupting them for. "
    'You may also repeat the call with `_sources={"<field>": "user"}` if the user did '
    'supply the value, or with `_assume={"<field>": "<rationale>"}` to proceed on an '
    "openly declared assumption. In that last case the result carries `_answer_preamble` "
    "— a field INSIDE the returned JSON — holding a preamble your "
    f"answer MUST OPEN with — `{CHECK_GROUNDEDNESS_TOOL['name']}` fails an answer that "
    "discloses an assumption only at the end, or not at all. Both `_sources` and `_assume` "
    "are stripped before the call reaches Tutu, so no upstream schema changes."
)
