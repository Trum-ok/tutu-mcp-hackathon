"""Deterministic per-scenario expectations.

Every check is a pure function of the transcript plus the groundedness report —
no model, no judge, no randomness. That is the whole point: Tutu's own evals score
behavior with a second model, so a reviewer has to trust the judge. These either
pass or fail on the same input every time, and a failure names the exact reason.

Fuzzy qualities ("is this a helpful answer?") are deliberately out of scope — an
LLM judge is the right tool for those, and nothing here pretends to replace it.
"""

import re
from dataclasses import dataclass
from typing import Protocol

from tutu_mcp.groundedness import GroundednessReport

from .transcript import Transcript


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    detail: str = ""


class Check(Protocol):
    def run(self, transcript: Transcript, grounding: GroundednessReport) -> CheckResult: ...


@dataclass(frozen=True)
class AllClaimsGrounded:
    """No claim in the answer (price/time/code/URL) was invented.

    Fails on `unavailable` only, not on everything that is merely not `confirmed`.
    A disclosed assumption and a threshold the user themselves named are both
    unconfirmed by the payload and neither is a fabrication — failing them punished
    an answer for restating the question it was asked.

    `require_any` guards the degenerate pass: an answer with no checkable claims at
    all would otherwise score a free 100%, which is exactly how a vague non-answer
    could beat a specific correct one. A quoted threshold does not count towards it
    — echoing "дешевле 3000 ₽" is not the same as reporting a price.
    """

    require_any: bool = True

    def run(self, transcript: Transcript, grounding: GroundednessReport) -> CheckResult:
        name = "all_claims_grounded"
        if not grounding.checkable:
            if self.require_any:
                return CheckResult(name, False, "в ответе нет ни одного проверяемого утверждения")
            return CheckResult(name, True, "проверяемых утверждений нет")
        bad = grounding.fabricated
        if bad:
            listed = ", ".join(repr(c.claim.text) for c in bad[:5])
            return CheckResult(name, False, f"не подтверждено данными: {listed}")
        return CheckResult(name, True, f"{len(grounding.checkable)} утверждений подтверждено")


@dataclass(frozen=True)
class MinGroundedClaims:
    kind: str  # "price" | "url" | "time" | "code"
    count: int = 1

    def run(self, transcript: Transcript, grounding: GroundednessReport) -> CheckResult:
        name = f"min_grounded_{self.kind}_{self.count}"
        found = sum(1 for c in grounding.checks if c.claim.kind == self.kind and c.grounded)
        ok = found >= self.count
        return CheckResult(name, ok, f"подтверждённых {self.kind}: {found}, нужно {self.count}")


@dataclass(frozen=True)
class UsedTool:
    tool: str

    def run(self, transcript: Transcript, grounding: GroundednessReport) -> CheckResult:
        ok = self.tool in transcript.tool_names()
        return CheckResult(f"used_{self.tool}", ok, f"вызовы: {transcript.tool_names()}")


@dataclass(frozen=True)
class UsedToolWithArg:
    tool: str
    arg: str
    value: object

    def run(self, transcript: Transcript, grounding: GroundednessReport) -> CheckResult:
        name = f"used_{self.tool}_{self.arg}={self.value}"
        ok = any(
            c.name == self.tool and c.arguments.get(self.arg) == self.value
            for c in transcript.tool_calls
        )
        return CheckResult(
            name, ok, "" if ok else f"нет вызова {self.tool} с {self.arg}={self.value!r}"
        )


@dataclass(frozen=True)
class DidNotUseTool:
    """Efficiency: the answer was already available without this call."""

    tool: str

    def run(self, transcript: Transcript, grounding: GroundednessReport) -> CheckResult:
        ok = self.tool not in transcript.tool_names()
        return CheckResult(f"skipped_{self.tool}", ok, "" if ok else f"лишний вызов {self.tool}")


@dataclass(frozen=True)
class MaxToolCalls:
    limit: int

    def run(self, transcript: Transcript, grounding: GroundednessReport) -> CheckResult:
        n = transcript.tool_call_count
        return CheckResult(f"max_{self.limit}_tool_calls", n <= self.limit, f"вызовов: {n}")


_URL_RE = re.compile(r"https?://\S+")


def _find_asserted_phrase(text: str, phrases: tuple[str, ...]) -> str | None:
    """Like `_find_phrase`, but ignores an occurrence that a disclaimer governs."""
    stripped = _URL_RE.sub("", text)
    lowered = stripped.lower()
    for phrase in phrases:
        for match in re.finditer(phrase.lower(), lowered):
            before = lowered[max(0, match.start() - _DISCLAIMER_WINDOW) : match.start()]
            if not any(re.search(d, before) for d in _DISCLAIMED_BEFORE):
                return phrase
    return None


def _find_phrase(text: str, phrases: tuple[str, ...]) -> str | None:
    # Strip URLs first: a Tutu deep link's query string is full of `?`/`&`, and
    # QUESTION_MARKERS' bare `\?` would otherwise read a checkout/search link as
    # a clarifying question — wrongly passing ClarifiedBeforeSearch on an answer
    # that never asked anything, and wrongly failing DidNotAsk on one that did.
    stripped = _URL_RE.sub("", text)
    lowered = stripped.lower()
    for phrase in phrases:
        if re.search(phrase.lower(), lowered):
            return phrase
    return None


# Constructions that DISCLAIM the phrase that follows them. Without these, an
# answer doing exactly the right thing — "по этому поиску нельзя утверждать, что
# поезд не ходит" — was failed for containing the very wording it was refusing to
# use. Substring matching cannot tell a claim from its denial; this can.
_DISCLAIMED_BEFORE = (
    r"нельзя утверждать",
    r"нельзя сделать вывод",
    r"не значит",
    r"не означает",
    r"не говорит о том",
    r"не следует, что",
    r"не вывод",
    r"это не",
)
# How far back a disclaimer still governs the phrase. One clause, roughly: past
# that the two sentences are unrelated and the disclaimer is not about this claim.
_DISCLAIMER_WINDOW = 90


@dataclass(frozen=True)
class AnswerAvoids:
    """The answer must NOT make a claim it has no grounds for.

    Canonical case from the article: an empty filtered rail result proves «нет в
    продаже на эту дату», never «поезд не ходит» — the pool lists bookable trains
    only, so a timetable claim is a fabrication the payload cannot support.

    An occurrence introduced by a disclaimer does not count: warning the reader
    off a claim is the behaviour this check exists to reward, and failing it for
    naming the claim would push agents towards vaguer answers, not truer ones.
    """

    phrases: tuple[str, ...]
    label: str = "avoids_unsupported_claim"

    def run(self, transcript: Transcript, grounding: GroundednessReport) -> CheckResult:
        hit = _find_asserted_phrase(transcript.answer_text, self.phrases)
        return CheckResult(
            self.label, hit is None, f"запрещённая формулировка: {hit!r}" if hit else ""
        )


@dataclass(frozen=True)
class AnswerMentions:
    """The answer must acknowledge something — typically that a field is absent."""

    phrases: tuple[str, ...]
    label: str = "mentions_required"

    def run(self, transcript: Transcript, grounding: GroundednessReport) -> CheckResult:
        hit = _find_phrase(transcript.answer_text, self.phrases)
        return CheckResult(
            self.label, hit is not None, "" if hit else f"нет ни одной из {self.phrases}"
        )


# Phrase banks reused across scenarios.
ABSENCE_CLAIM_PHRASES = (
    r"не ходит",
    r"не существует",
    r"такого поезда нет",
    r"поезд отменён",
    r"не курсирует",
)
FIELD_MISSING_PHRASES = (
    r"не вернул",
    r"не указан",
    r"нет данных",
    r"не приш[её]л",
    r"отсутствует",
)
NOT_ON_SALE_PHRASES = (
    r"нет в продаже",
    r"не в продаже",
    r"билетов.{0,20}нет",
    r"мест.{0,20}нет",
)


# --- premise checks: was the gap closed BEFORE the expensive work? -------------

EXPENSIVE_TOOLS = (
    "search_rail",
    "search_avia",
    "search_bus",
    "search_etrain",
    "search_multitransport",
    "search_hotels",
)

QUESTION_MARKERS = (r"\?", r"уточните", r"подскажите")


def _expensive_calls(transcript: Transcript) -> int:
    return sum(1 for c in transcript.tool_calls if c.name in EXPENSIVE_TOOLS)


@dataclass(frozen=True)
class ClarifiedBeforeSearch:
    """The clarifying question must come BEFORE the tables, not under them.

    Asking after a full price comparison has already been printed is the failure
    this whole mechanism exists to prevent: by then the user has read the answer
    and the caveat arrives as a footnote to a conclusion they already believe.

    `max_expensive_calls` allows the one search that reveals the gap (the gate
    fires on a call, not on thin air) while still failing a run that priced out
    every option first.
    """

    max_expensive_calls: int = 1

    def run(self, transcript: Transcript, grounding: GroundednessReport) -> CheckResult:
        name = "clarified_before_search"
        asked = _find_phrase(transcript.answer_text, QUESTION_MARKERS) is not None
        spent = _expensive_calls(transcript)
        if not asked:
            return CheckResult(name, False, "ответ не содержит уточняющего вопроса")
        if spent > self.max_expensive_calls:
            return CheckResult(
                name,
                False,
                f"вопрос задан после {spent} дорогих поисков (лимит {self.max_expensive_calls})",
            )
        return CheckResult(name, True, f"вопрос задан после {spent} поисков")


@dataclass(frozen=True)
class DisclosedAssumptionUpFront:
    """If the run proceeded on an assumption, the answer must OPEN with it.

    `assumption_disclosed` is None when nothing was assumed — that is a pass, not
    a vacuous one: it means the agent had no undeclared premise to disclose.
    """

    def run(self, transcript: Transcript, grounding: GroundednessReport) -> CheckResult:
        name = "assumption_disclosed_up_front"
        if grounding.assumption_disclosed is None:
            return CheckResult(name, True, "допущений не было")
        if not grounding.assumption_disclosed:
            return CheckResult(
                name, False, f"допущение не раскрыто в начале: {grounding.assumptions}"
            )
        return CheckResult(name, True, f"раскрыто на позиции {grounding.disclosure_position}")


@dataclass(frozen=True)
class DidNotAsk:
    """Over-asking guard.

    Without this, the cheapest way to pass every premise check is to interrogate
    the user on every request — which trades one bad answer for a worse product.
    A request whose parameters are all determinate must be answered, not queried.
    """

    def run(self, transcript: Transcript, grounding: GroundednessReport) -> CheckResult:
        hit = _find_phrase(transcript.answer_text, QUESTION_MARKERS)
        return CheckResult(
            "did_not_over_ask",
            hit is None,
            f"лишний уточняющий вопрос: {hit!r}" if hit else "",
        )


@dataclass(frozen=True)
class GateFired:
    """The premise machinery intervened — on a `tools/call` or, better, in the
    preflight before one was ever made. See `Transcript.gate_fired`."""

    def run(self, transcript: Transcript, grounding: GroundednessReport) -> CheckResult:
        fired = transcript.gate_fired()
        return CheckResult("premise_gate_fired", fired, "" if fired else "гейт не сработал")


TYPO_HINT_PHRASES = (
    r"воскресенье",
    r"не совпада",
    r"кака[яй] дата",
    r"проверьте дату",
    r"уточните дату",
)
