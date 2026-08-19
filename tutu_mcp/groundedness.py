"""Deterministic groundedness check: does every factual claim in an agent's
answer actually appear in the MCP `tool_result` payload(s) it was given?

This is the thing the Tutu MCP article's own instructions flag as "the
single largest failure mode in agent transcripts" — an agent inventing a
price, time, train/flight number, name or URL that the server never
returned. Their evals score this with an LLM-as-judge (a second model);
this module instead extracts typed claims from the answer with regex and
checks set-membership against the values actually present in the JSON
payload — no model call, no judge variance, fully reproducible.

A claim that is absent from the payload is not automatically a fabrication:
it may be a value the agent openly ASSUMED and declared through the premise gate
(`tutu_mcp/premises.py`). Those two deserve different colors in a trace, so a check
carries a `status` — `confirmed` (in the payload), `assumed` (declared upfront),
`user_stated` (the user's own threshold quoted back, which no payload owes us),
or `unavailable` (in none of those, i.e. invented) — and the report additionally
verifies that any assumption was disclosed in the OPENING of the answer rather
than in a closing caveat nobody reads.

Not a substitute for an LLM judge on fuzzy claims ("this hotel is great for
families") — it only catches the mechanically checkable ones: prices,
times, URLs, and train/flight/order-style codes. That's a deliberate scope
cut, not an oversight.
"""

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from pydantic import BaseModel, Field

from tutu_mcp.backend import ERROR_STATUSES
from tutu_mcp.toolspec import parse_args, tool_spec

# Our own domain-shaped error results (tutu_mcp.proxy.dispatch.backend_error).
# A failed call's payload can echo the very value an agent invented (a fixture
# miss quotes back its unmatched arguments) — treating it as evidence would let
# that invention "confirm" itself.
_ERROR_STATUSES = ERROR_STATUSES

_PRICE_RE = re.compile(r"(?<!\d)(\d[\d\s]{0,9}(?:[.,]\d{1,2})?)\s?(₽|руб\.?|RUB)", re.IGNORECASE)
_URL_RE = re.compile(r"https?://\S+")
_TIME_RE = re.compile(r"\b([01]\d|2[0-3]):([0-5]\d)\b")
# train-style "749У" / "12А" and flight-style "SU 1200" / "SU1200" codes
_CODE_RE = re.compile(r"\b\d{1,4}[А-ЯЁA-Z]{1,3}\b|\b[А-ЯЁA-Z]{2}[- ]?\d{2,4}\b", re.UNICODE)
_WHITESPACE_RE = re.compile(r"\s")
_URL_TRAILING_PUNCT = ".,;:)]}»\"'"


@dataclass(frozen=True)
class Claim:
    kind: str  # "price" | "url" | "time" | "code"
    text: str  # verbatim substring pulled from the answer
    value: float | str  # normalized form used for the grounding lookup


@dataclass(frozen=True)
class ClaimCheck:
    claim: Claim
    status: str  # "confirmed" | "assumed" | "user_stated" | "unavailable"

    @property
    def grounded(self) -> bool:
        """Kept so every existing caller reads the same thing it always did:
        only a value actually present in a tool_result counts as grounded. An
        assumption is disclosed, not proven."""
        return self.status == "confirmed"

    @property
    def fabricated(self) -> bool:
        """The failure this module exists to catch: a value that came from nowhere.
        Distinct from `not grounded`, which also covers a disclosed assumption and
        the user's own words quoted back — neither is the model making things up."""
        return self.status == "unavailable"


# how far into an answer a disclosure still counts as "upfront". Roughly the
# first paragraph: past that the reader has already started trusting the tables.
DISCLOSURE_WINDOW = 300

_DISCLOSURE_MARKERS = (
    "допущен",
    "предполож",
    "не подтвержд",
    "не найден",
    "не известн",
    "неизвестн",
    "уточните",
    "внимание",
)


@dataclass(frozen=True)
class GroundednessReport:
    checks: list[ClaimCheck]
    # assumptions the premise gate recorded this session, if any
    assumptions: list[str] = field(default_factory=list)
    disclosure_position: int | None = None
    # tool_results excluded from the grounding index because they were one of
    # OUR error payloads, not real Tutu data — see `_ERROR_STATUSES`
    ignored_error_payloads: int = 0

    @property
    def assumption_disclosed(self) -> bool | None:
        """None when there was nothing to disclose; otherwise whether the answer
        opened with it. A trailing caveat is read only by someone who already
        distrusts the answer — precisely the reader who did not need it."""
        if not self.assumptions:
            return None
        return self.disclosure_position is not None

    @property
    def checkable(self) -> list[ClaimCheck]:
        """Claims a payload could, in principle, confirm.

        A threshold the user themselves named ("дешевле 3000 ₽") is not one: it is
        the request restated, and no tool_result is obliged to contain it. Scoring
        it as ungrounded punished a correct answer for repeating the question.
        """
        return [c for c in self.checks if c.status != "user_stated"]

    @property
    def rate(self) -> float | None:
        if not self.checkable:
            return None
        return sum(1 for c in self.checkable if c.grounded) / len(self.checkable)

    @property
    def ungrounded(self) -> list[ClaimCheck]:
        return [c for c in self.checks if not c.grounded]

    @property
    def fabricated(self) -> list[ClaimCheck]:
        return [c for c in self.checks if c.fabricated]


def extract_claims(answer_text: str) -> list[Claim]:
    claims: list[Claim] = []

    for match in _PRICE_RE.finditer(answer_text):
        # \s in the price regex also swallows Cyrillic-text thousands separators
        # (plain, non-breaking U+00A0, narrow no-break U+202F space)
        digits = _WHITESPACE_RE.sub("", match.group(1)).replace(",", ".")
        try:
            value = float(digits)
        except ValueError:
            continue
        claims.append(Claim(kind="price", text=match.group(0), value=value))

    for match in _URL_RE.finditer(answer_text):
        url = match.group(0).rstrip(_URL_TRAILING_PUNCT)
        claims.append(Claim(kind="url", text=url, value=url))

    for match in _TIME_RE.finditer(answer_text):
        claims.append(Claim(kind="time", text=match.group(0), value=match.group(0)))

    for match in _CODE_RE.finditer(answer_text):
        claims.append(Claim(kind="code", text=match.group(0), value=match.group(0).upper()))

    return claims


@lru_cache(maxsize=512)
def _WORD_BOUNDED(value: str) -> re.Pattern[str]:  # noqa: N802 - reads as a constructor
    return re.compile(rf"(?<!\w){re.escape(value)}(?!\w)", re.IGNORECASE)


_PRICE_KEY_HINTS = ("price", "amount")


@dataclass(frozen=True)
class GroundingIndex:
    # numbers found under a price-ish key (amount, price, price_from, ...) only — NOT every
    # number in the payload, since ids/counts/page numbers/ratings are common small integers
    # that would trivially (and wrongly) "ground" a fabricated price otherwise
    price_numbers: set[float]
    strings: list[str]  # kept as a list (not lowercased/deduped) for substring checks

    def has_number(self, value: float, *, tolerance: float = 0.01) -> bool:
        if any(abs(value - n) <= tolerance for n in self.price_numbers):
            return True
        # an agent commonly rounds a price to whole currency units
        return any(abs(round(value) - round(n)) == 0 for n in self.price_numbers)

    def has_exact_string(self, value: str) -> bool:
        return value in self.strings

    def has_substring(self, value: str) -> bool:
        needle = value.lower()
        return any(needle in s.lower() for s in self.strings)


def _flatten(
    payload: Any, price_numbers: set[float], strings: list[str], *, key: str | None = None
) -> None:
    if isinstance(payload, dict):
        for k, v in payload.items():
            # Our own annotations (`_answer_preamble`) are not Tutu's data, and a
            # preamble quotes the very assumed value the check is meant to expose —
            # indexing it would let an assumption confirm itself as fact.
            if k.startswith("_"):
                continue
            _flatten(v, price_numbers, strings, key=k)
    elif isinstance(payload, list):
        for v in payload:
            _flatten(v, price_numbers, strings, key=key)
    elif isinstance(payload, bool):
        return
    elif isinstance(payload, int | float):
        if key is not None and any(hint in key.lower() for hint in _PRICE_KEY_HINTS):
            price_numbers.add(float(payload))
    elif isinstance(payload, str):
        strings.append(payload)


def _is_error_payload(payload: Any) -> bool:
    return isinstance(payload, dict) and payload.get("status") in _ERROR_STATUSES


def build_grounding_index(payloads: list[Any]) -> GroundingIndex:
    price_numbers: set[float] = set()
    strings: list[str] = []
    for payload in payloads:
        if _is_error_payload(payload):
            continue
        _flatten(payload, price_numbers, strings)
    return GroundingIndex(price_numbers=price_numbers, strings=strings)


def find_disclosure_position(answer_text: str) -> int | None:
    """Where the answer first admits it is standing on an assumption, or None."""
    head = answer_text[:DISCLOSURE_WINDOW].lower()
    hits = [head.find(marker) for marker in _DISCLOSURE_MARKERS if marker in head]
    return min(hits) if hits else None


def _claim_spellings(claim: Claim) -> set[str]:
    """Every form a claim's value could take in a declared assumption.

    A price arrives here as a float, while the assumption recorded it from the
    tool arguments as an int — so "4000.0" never matched "4000" and an openly
    declared price came out `unavailable` (invented) instead of `assumed`. Same
    number, two spellings; both have to be tried.
    """
    forms = {str(claim.value).strip().lower(), claim.text.strip().lower()}
    if isinstance(claim.value, float) and claim.value.is_integer():
        forms.add(str(int(claim.value)))
    return forms


def _stated_by_user(claim: Claim, user_request: str) -> bool:
    """Whether this exact value appears in the user's own request.

    Word-bounded, like the premise gate's own provenance check: a substring match
    would find "3000" inside an offer id and clear a fabricated price.
    """
    if not user_request:
        return False
    return any(_WORD_BOUNDED(form).search(user_request) for form in _claim_spellings(claim) if form)


def check_groundedness(
    answer_text: str,
    tool_results: list[Any],
    *,
    assumed_values: set[str] | None = None,
    assumptions: list[str] | None = None,
    user_request: str = "",
) -> GroundednessReport:
    """`tool_results` are the parsed JSON payloads of every `tool_result` the agent
    saw this turn (i.e. `json.loads(content[0].text)` for each `tools/call` response).

    `assumed_values` / `assumptions` come from the session's premise gate: values
    the agent invented but declared. They downgrade a claim from `unavailable`
    (invented silently) to `assumed` (invented openly) — different failures that
    must not share a color."""
    index = build_grounding_index(tool_results)
    ignored_error_payloads = sum(1 for p in tool_results if _is_error_payload(p))
    assumed = {v.strip().lower() for v in (assumed_values or set())}
    checks: list[ClaimCheck] = []
    for claim in extract_claims(answer_text):
        if claim.kind == "price":
            assert isinstance(claim.value, float)
            grounded = index.has_number(claim.value)
        elif claim.kind == "url":
            assert isinstance(claim.value, str)
            grounded = index.has_exact_string(claim.value)
        else:  # "time" / "code" — short tokens are meaningful only as substrings of a longer field
            assert isinstance(claim.value, str)
            grounded = index.has_substring(claim.value)

        if grounded:
            status = "confirmed"
        elif assumed & _claim_spellings(claim):
            status = "assumed"
        elif _stated_by_user(claim, user_request):
            # The user's own constraint, restated. Checked AFTER the payload so a
            # value that is both stays `confirmed` — evidence beats a citation.
            status = "user_stated"
        else:
            status = "unavailable"
        checks.append(ClaimCheck(claim=claim, status=status))

    return GroundednessReport(
        checks=checks,
        assumptions=list(assumptions or []),
        disclosure_position=find_disclosure_position(answer_text),
        ignored_error_payloads=ignored_error_payloads,
    )


def report_to_json(report: GroundednessReport) -> str:
    return json.dumps(
        {
            "groundedness_rate": report.rate,
            "claims": [
                {
                    "kind": c.claim.kind,
                    "text": c.claim.text,
                    "grounded": c.grounded,
                    "status": c.status,
                }
                for c in report.checks
            ],
            "assumptions": report.assumptions,
            "assumption_disclosed": report.assumption_disclosed,
            "disclosure_position": report.disclosure_position,
            "ignored_error_payloads": report.ignored_error_payloads,
        },
        ensure_ascii=False,
    )


class CheckGroundednessArgs(BaseModel):
    answer_text: str = Field(description="The drafted answer text to check.")
    tool_result_json: list[str] | None = Field(
        default=None,
        description=(
            "Optional. Leave it out — the proxy checks against the tool_results it "
            "already delivered you this session. Pass raw JSON only for evidence the "
            "proxy never saw."
        ),
    )


CHECK_GROUNDEDNESS_TOOL = tool_spec(
    "check_groundedness",
    (
        "Deterministically check a drafted answer against the tool_result payload(s) it was "
        "based on. Normally you pass ONLY `answer_text`: the proxy already holds every "
        "tool_result it delivered you this session and checks against those, so do not "
        "copy payloads back — that costs you thousands of tokens and truncates. Pass "
        "`tool_result_json` only for evidence that did not come through this proxy. "
        "Returns which price/time/code/URL claims are grounded vs unsupported by that "
        "data, plus a groundedness rate."
    ),
    CheckGroundednessArgs,
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True},
)


def run_check_groundedness_tool(
    arguments: dict[str, Any],
    *,
    assumed_values: set[str] | None = None,
    assumptions: list[str] | None = None,
    session_payloads: list[Any] | None = None,
    user_request: str = "",
) -> tuple[str, bool]:
    """The `check_groundedness` tool body, as `(result_text, is_error)`.

    Kept transport-free so both the MCP server and the eval harness invoke the
    exact same code — the eval measures the real tool, not a re-implementation.

    `assumed_values` / `assumptions` are supplied by the CALLER from the session's
    premise state, never by the agent: an agent that could omit them would be able
    to hide its own assumptions from the check meant to expose them.

    `session_payloads` is the same idea applied to the evidence itself — the
    tool_results the proxy delivered this session. It is the default source, and
    `tool_result_json` only supplements it, because evidence the agent selects is
    evidence the agent can select FAVOURABLY. Copying payloads back also cost it
    thousands of output tokens and truncated often enough to fail the check on
    transport rather than on substance.
    """
    args = parse_args(CheckGroundednessArgs, arguments)
    if isinstance(args, str):
        return args, True

    answer_text = args.answer_text
    payloads: list[Any] = list(session_payloads or [])
    try:
        payloads += [json.loads(raw) for raw in args.tool_result_json or []]
    except json.JSONDecodeError as exc:
        # a shape pydantic cannot catch: valid list[str], invalid JSON inside
        return f"invalid arguments: tool_result_json содержит не-JSON: {exc}", True

    if not payloads:
        # Nothing to check against: claiming an answer is grounded on no evidence
        # would be a worse answer than admitting the check cannot run.
        return (
            "не с чем сверять: прокси не видел ни одного tool_result в этой сессии, "
            "а tool_result_json не передан. Сначала выполните поиск.",
            True,
        )

    report = check_groundedness(
        answer_text,
        payloads,
        assumed_values=assumed_values,
        assumptions=assumptions,
        user_request=user_request,
    )
    # An undisclosed assumption is a failed check, not a footnote on a passing one.
    return report_to_json(report), report.assumption_disclosed is False
