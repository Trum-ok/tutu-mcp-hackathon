"""Premise checking: does the agent's plan rest on values nobody supplied?

`groundedness.py` checks the *output* side of a turn — every price/time/code in
the answer must appear in some `tool_result`. This module checks the *input*
side: every value that NARROWS the search must have come from the user or from
an earlier `tool_result`. There is no third source. Anything else is a value the
model invented, and a whole comparison table built on one of those is wrong in a
way groundedness cannot see — the numbers in it are all real, they were just
selected against a made-up constraint.

The canonical case: "поездка одним днём на концерт, что выгоднее по транспорту".
The agent silently assumes the concert ends at some hour, filters return trains
by it, and prints a confident table. Every figure traces back to a payload, so
groundedness scores 100%. The answer is still misleading.

Deliberately NOT modelled here: trip archetypes ("day trip", "business trip").
Those are an open-ended list and hardcoding them buys one demo. What generalizes
is the provenance rule above — an unknown quantity, whatever the trip looks like,
eventually surfaces as a narrowing argument, and that is where it gets caught.

Four statuses, shared with the answer-side report so the trace viewer can paint
one legend end to end:

    confirmed   — value came from the user or from a tool_result
    assumed     — value was invented, but the agent declared it via `_assume`
    unavailable — value is missing and nothing here can supply it → ask the user
    conflicted  — value is present but two sources disagree (likely a typo)

Two entry points share one engine: `assess_request` runs it over a PLANNED call
list before anything is searched, and `SessionPremises.evaluate` runs it over a
call actually being dispatched. Same rules, so a cooperative agent pays one cheap
preflight and an uncooperative one still cannot slip past the gate.
"""

import json
import re
from dataclasses import dataclass, field
from datetime import date
from functools import lru_cache
from typing import Any, Literal

from pydantic import BaseModel, Field

from tutu_mcp.toolspec import parse_args, tool_spec

Status = Literal["confirmed", "assumed", "unavailable", "conflicted"]

# Agents declare provenance through these two argument keys. They are stripped
# before the call reaches the backend — see `strip_control_fields`.
SOURCES_KEY = "_sources"
ASSUME_KEY = "_assume"
CONTROL_KEYS = (SOURCES_KEY, ASSUME_KEY)

# The `status` a `GateDecision` payload carries instead of tool data. Exported
# so callers detect a fired gate by parsing this field (`ToolCallRecord.parsed_result()`)
# rather than substring-matching the raw text, which a coincidental match elsewhere
# in a result (an echoed error message, a quoted field) could trip by accident.
GATE_STATUS = "clarification_required"


@dataclass(frozen=True)
class FieldPolicy:
    """What a single tool argument requires before a call carrying it is honored.

    `kind`:
      "blocking"  — the answer is distorted when this field is ABSENT (the server
                    quietly defaults it and the agent never sees that it did).
      "narrowing" — the field restricts the result set, so its VALUE needs a
                    source. Present-but-invented is the failure mode here.
    """

    kind: Literal["blocking", "narrowing"]
    why: str
    ask: str
    # tools that could close this gap without bothering the user; empty means
    # nothing on this server can, so asking is the only honest move
    resolvable_by: tuple[str, ...] = ()


_SEARCH_TOOLS = (
    "search_rail",
    "search_avia",
    "search_bus",
    "search_etrain",
    "search_multitransport",
)

# Applied to every search tool. `departure_date` is blocking because upstream
# declares `required: []` on all of them — omit it and the server happily answers
# for some default day, which the agent then reports as if the user had picked it.
_COMMON_SEARCH_POLICIES: dict[str, FieldPolicy] = {
    "departure_date": FieldPolicy(
        kind="blocking",
        why="сервер не требует дату и молча подставит свою — ответ будет про другой день",
        ask="На какую дату ищем?",
    ),
    "price_max": FieldPolicy(
        kind="narrowing",
        why="потолок цены отсекает варианты — взятый с потолка, он прячет то, что подошло бы",
        ask="Есть ли ограничение по цене?",
    ),
    "carriers": FieldPolicy(
        kind="narrowing",
        why="фильтр по перевозчику отсекает варианты",
        ask="Важен ли конкретный перевозчик?",
    ),
    "direct_only": FieldPolicy(
        kind="narrowing",
        why="запрет пересадок отсекает варианты",
        ask="Нужен только прямой рейс?",
    ),
}

POLICIES: dict[str, dict[str, FieldPolicy]] = {
    tool: dict(_COMMON_SEARCH_POLICIES) for tool in _SEARCH_TOOLS
}

POLICIES["search_rail"] |= {
    "train_numbers": FieldPolicy(
        kind="narrowing",
        why="фильтр по номеру поезда сводит выдачу к одному поезду",
        ask="Какой именно поезд нужен?",
    ),
    "seat_categories": FieldPolicy(
        kind="narrowing",
        why="фильтр по типу вагона отсекает варианты",
        ask="Какой класс вагона нужен?",
    ),
}
POLICIES["search_avia"] |= {
    "flight_numbers": FieldPolicy(
        kind="narrowing",
        why="фильтр по номеру рейса сводит выдачу к одному рейсу",
        ask="Какой именно рейс нужен?",
    ),
    "service_class": FieldPolicy(
        kind="narrowing",
        why="фильтр по классу обслуживания отсекает варианты",
        ask="Каким классом летим?",
    ),
}

# Hotels are the one place where the passenger count is blocking rather than
# free: `best_offer.price` is a stay total for the room AND the guests in it, so
# a silently assumed `adults` changes the headline number. On transport the fare
# is per passenger and scales linearly, so assuming one adult misleads nobody —
# marking it blocking there would only teach the agent to interrogate the user.
POLICIES["search_hotels"] = {
    "check_in": FieldPolicy(
        kind="blocking",
        why="без даты заезда сервер подставит свою — цена будет за другой период",
        ask="С какого числа нужен отель?",
    ),
    "check_out": FieldPolicy(
        kind="blocking",
        why="без даты выезда неизвестна длительность, а цена считается за весь период",
        ask="До какого числа нужен отель?",
    ),
    "adults": FieldPolicy(
        kind="blocking",
        why="цена отеля — stay_total за номер и гостей, число гостей меняет саму цифру",
        ask="Сколько человек будет жить в номере?",
    ),
    "price_max": _COMMON_SEARCH_POLICIES["price_max"],
    "min_rating": FieldPolicy(
        kind="narrowing",
        why="порог рейтинга отсекает отели",
        ask="Есть ли требование к рейтингу?",
    ),
    "stars": FieldPolicy(
        kind="narrowing",
        why="фильтр по звёздности отсекает отели",
        ask="Сколько звёзд нужно?",
    ),
}

POLICIES["get_rail_seatmap"] = {
    "seats_together": FieldPolicy(
        kind="narrowing",
        why="требование мест рядом отсекает вагоны — само требование должно исходить от пользователя",
        ask="Нужно, чтобы места были рядом?",
        resolvable_by=("get_rail_seatmap",),
    ),
}


def strip_control_fields(
    arguments: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, str], dict[str, str]]:
    """Split `arguments` into `(clean, sources, assume)`.

    The control keys MUST NOT survive into `backend.call_tool`: the fixture store
    matches on exact normalized arguments (`fixtures/store.py`), so leaving them in
    turns every recorded scenario into a miss, and upstream would be handed a field
    its schema never declared. This is also why they are never added to any
    proxied `inputSchema` — the upstream schema stays byte-identical, and the two
    keys are documented in the server instructions instead.
    """
    clean = {k: v for k, v in arguments.items() if k not in CONTROL_KEYS}
    sources = arguments.get(SOURCES_KEY) or {}
    assume = arguments.get(ASSUME_KEY) or {}
    if not isinstance(sources, dict):
        sources = {}
    if not isinstance(assume, dict):
        assume = {}
    return (
        clean,
        {str(k): str(v) for k, v in sources.items()},
        {str(k): str(v) for k, v in assume.items()},
    )


# --- typo detection: the user's own words disagreeing with each other ---------

# Full inflected forms, not prefixes. `сред` also matched "среднем" and `ма`
# matched "маршрутов", so "в субботу 11 маршрутов" was read as a Saturday/11-May
# contradiction and produced a clarifying question about nothing.
_WEEKDAYS: dict[str, int] = {
    "понедельник": 0,
    "понедельника": 0,
    "вторник": 1,
    "вторника": 1,
    "среда": 2,
    "среду": 2,
    "среды": 2,
    "четверг": 3,
    "четверга": 3,
    "пятница": 4,
    "пятницу": 4,
    "пятницы": 4,
    "суббота": 5,
    "субботу": 5,
    "субботы": 5,
    "воскресенье": 6,
    "воскресенья": 6,
}
# nominative for "это <день>", accusative for "вы написали <в день>" — the
# question is read by a person, so it has to be grammatical Russian
_WEEKDAY_NOM = (
    "понедельник",
    "вторник",
    "среда",
    "четверг",
    "пятница",
    "суббота",
    "воскресенье",
)
_WEEKDAY_ACC = (
    "понедельник",
    "вторник",
    "среду",
    "четверг",
    "пятницу",
    "субботу",
    "воскресенье",
)
_MONTHS: dict[str, int] = {
    "января": 1,
    "январь": 1,
    "февраля": 2,
    "февраль": 2,
    "марта": 3,
    "март": 3,
    "апреля": 4,
    "апрель": 4,
    "мая": 5,
    "май": 5,
    "июня": 6,
    "июнь": 6,
    "июля": 7,
    "июль": 7,
    "августа": 8,
    "август": 8,
    "сентября": 9,
    "сентябрь": 9,
    "октября": 10,
    "октябрь": 10,
    "ноября": 11,
    "ноябрь": 11,
    "декабря": 12,
    "декабрь": 12,
}
# "в субботу 11 октября" / "11 октября, суббота" — a weekday and a date within
# one short span of text. Two spellings, one meaning; both are checked.
_WD_ALT = "|".join(sorted(_WEEKDAYS, key=len, reverse=True))
_MONTH_ALT = "|".join(sorted(_MONTHS, key=len, reverse=True))
_WD_THEN_DATE = re.compile(
    rf"\b(?P<wd>{_WD_ALT})\b.{{0,20}}?\b(?P<day>\d{{1,2}})\s+(?P<month>{_MONTH_ALT})\b",
    re.IGNORECASE,
)
_DATE_THEN_WD = re.compile(
    rf"\b(?P<day>\d{{1,2}})\s+(?P<month>{_MONTH_ALT})\b.{{0,20}}?\b(?P<wd>{_WD_ALT})\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Conflict:
    """Two things the user said that cannot both be true."""

    field: str
    said: str
    actual: str
    ask: str


def _resolve_year(day: int, month: int, today: date) -> date | None:
    """Pick the nearest future occurrence of day/month, the way a person means it."""
    for year in (today.year, today.year + 1):
        try:
            candidate = date(year, month, day)
        except ValueError:
            return None  # 31 февраля and friends — a different error, not ours
        if candidate >= today:
            return candidate
    return None


def check_calendar_consistency(text: str, today: date) -> list[Conflict]:
    """Catch a user typo by making their own two statements check each other.

    A slipped digit ("13" typed as "11") is invisible on its own — the date stays
    valid and the search succeeds. But people rarely give a bare number: they
    write "в субботу 11 октября", and the weekday is a second, independent
    encoding of the same day. When the two disagree, one of them is a typo.

    We never guess WHICH one. Correcting "11" to "13" has no more support than
    correcting it to "21"; the honest move is to show the contradiction and ask.
    """
    conflicts: list[Conflict] = []
    seen: set[tuple[int, int]] = set()

    for pattern in (_WD_THEN_DATE, _DATE_THEN_WD):
        for match in pattern.finditer(text):
            weekday_word = match.group("wd")
            month_word = match.group("month").lower()
            day = int(match.group("day"))
            month = _MONTHS[month_word]
            if (day, month) in seen:
                continue
            seen.add((day, month))

            resolved = _resolve_year(day, month, today)
            if resolved is None:
                continue
            claimed = _WEEKDAYS[weekday_word.lower()]
            if resolved.weekday() == claimed:
                continue

            conflicts.append(
                Conflict(
                    field="departure_date",
                    said=f"{_WEEKDAY_ACC[claimed]}, {day} {month_word}",
                    actual=f"{resolved.isoformat()} — это {_WEEKDAY_NOM[resolved.weekday()]}",
                    ask=(
                        f"Вы написали {_WEEKDAY_ACC[claimed]}, но {day} {month_word} "
                        f"{resolved.year} — {_WEEKDAY_NOM[resolved.weekday()]}. "
                        "Какая дата верная?"
                    ),
                )
            )
    return conflicts


# --- session state: what this conversation has actually established -----------

MAX_SEEN_VALUES = 50_000  # one search_rail result is ~26 KB; the cap keeps a long
# session from growing without bound while staying far above any real transcript

# The one-shot rule below keys on (tool, arguments), so an agent that nudges an
# argument each time would get a fresh gate every round. This is the backstop for
# that: past it the session stops gating entirely and books assumptions instead.
MAX_GATES_PER_SESSION = 12


@dataclass(frozen=True)
class Slot:
    field: str
    status: Status
    why: str
    ask: str
    value: str | None = None
    resolvable_by: tuple[str, ...] = ()


@dataclass(frozen=True)
class Assumption:
    """A value the agent invented AND owned up to via `_assume`."""

    field: str
    value: str
    rationale: str


@dataclass(frozen=True)
class GateDecision:
    """Returned INSTEAD of tool data while a premise is still unsettled."""

    tool: str
    slots: list[Slot]

    def to_json(self) -> str:
        return json.dumps(
            {
                "status": GATE_STATUS,
                "tool": self.tool,
                "reason": (
                    "запрос опирается на значения, которых нет ни в словах пользователя, "
                    "ни в одном tool_result этой сессии"
                ),
                "blocking_slots": [
                    {
                        "field": s.field,
                        "status": s.status,
                        "value": s.value,
                        "why": s.why,
                        "ask": s.ask,
                        "resolvable_by": list(s.resolvable_by),
                    }
                    for s in self.slots
                ],
                "resolve_one_of": [
                    "спросите пользователя (предпочтительно) и повторите вызов",
                    "вызовите один из resolvable_by, если он не пуст, затем повторите",
                    f'повторите вызов с {ASSUME_KEY}={{"<поле>": "<обоснование допущения>"}} — '
                    "данные придут вместе с обязательной преамбулой для ответа",
                    f'повторите вызов с {SOURCES_KEY}={{"<поле>": "user"}}, если значение '
                    "всё-таки назвал пользователь",
                ],
            },
            ensure_ascii=False,
        )


@lru_cache(maxsize=512)
def _WORD_BOUNDED(value: str) -> re.Pattern[str]:  # noqa: N802 - reads as a constructor
    return re.compile(rf"(?<!\w){re.escape(value)}(?!\w)", re.IGNORECASE)


def _norm_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return str(value).strip().lower()


def _values_of(raw: Any) -> list[str]:
    if isinstance(raw, list | tuple):
        return [_norm_scalar(v) for v in raw]
    return [_norm_scalar(raw)]


@dataclass
class SessionPremises:
    """Per-MCP-session premise state.

    Bound to one session, never global: two clients on the same proxy must not
    see each other's assumptions — on a shared deployment that would be a data
    leak, and on stage it would mean the judge's session inherits ours.

    Parallel `tools/call` batches are safe by construction. The only cross-call
    rule is "this value already appeared in an earlier tool_result", and an agent
    cannot have taken a value from a result it has not received yet — so a call
    racing its sibling can never be wrongly cleared, only wrongly gated, which
    the idempotency rule below then releases on the retry.
    """

    seen_values: set[str] = field(default_factory=set)
    gated: set[str] = field(default_factory=set)
    assumptions: dict[str, Assumption] = field(default_factory=dict)
    conflicts: list[Conflict] = field(default_factory=list)
    # set by `assess_request`; lets values the user actually typed clear the gate
    # without a second round-trip — the payoff for running the cheap preflight
    user_request: str = ""

    def record_result(self, result_text: str) -> None:
        """Index every scalar in a tool_result so later calls can cite it."""
        try:
            payload = json.loads(result_text)
        except (json.JSONDecodeError, TypeError):
            return
        stack = [payload]
        while stack and len(self.seen_values) < MAX_SEEN_VALUES:
            node = stack.pop()
            if isinstance(node, dict):
                stack.extend(node.values())
            elif isinstance(node, list):
                stack.extend(node)
            elif node is not None:
                self.seen_values.add(_norm_scalar(node))

    def _said_by_user(self, value: str) -> bool:
        """Whether the user's own words contain this value as a standalone token.

        Substring matching looked equivalent and was not: `adults=2` found its "2"
        inside the date `2026-08-25`, so an invented guest count sailed through the
        gate on a digit from an unrelated field. Word boundaries close that.
        """
        if not value:
            return False
        return _WORD_BOUNDED(value).search(self.user_request) is not None

    def _has_source(self, values: list[str]) -> bool:
        return all(v in self.seen_values or self._said_by_user(v) for v in values)

    def evaluate(
        self,
        tool: str,
        arguments: dict[str, Any],
        sources: dict[str, str],
        assume: dict[str, str],
    ) -> GateDecision | None:
        """Decide whether this call may proceed. `None` means yes.

        Records any declared assumption as a side effect, so `preamble()` can
        force it into the first sentence of the answer afterwards.
        """
        policies = POLICIES.get(tool)
        if policies is None:
            return None

        # A declared assumption is honored wherever it appears, even on a field the
        # gate would have let through: the agent is telling us this value is its own
        # invention, and that claim outranks our guess about whether it mattered.
        for name, rationale in assume.items():
            raw = arguments.get(name)
            self.assumptions[name] = Assumption(
                name, ", ".join(_values_of(raw)) if raw is not None else "не задано", rationale
            )

        unsettled: list[Slot] = []

        for name, policy in policies.items():
            declared = assume.get(name)

            if policy.kind == "blocking":
                if arguments.get(name) is not None:
                    # Presence only — the VALUE of a blocking field is deliberately
                    # not traced back to the user. Both of the fields that matter
                    # here get normalized on the way in ("на двоих" -> `adults=2`,
                    # "25 августа" -> `2026-08-25`), so a text match against the
                    # user's wording would fail on correct calls and produce exactly
                    # the over-asking this design treats as the worse failure.
                    # Narrowing fields have no such normalization and are checked.
                    continue
                if declared:
                    continue
                unsettled.append(
                    Slot(
                        field=name,
                        status="unavailable",
                        why=policy.why,
                        ask=policy.ask,
                        resolvable_by=policy.resolvable_by,
                    )
                )
                continue

            raw = arguments.get(name)
            if raw is None or raw is False or raw == [] or raw == "":
                # A filter that is absent, switched off, or empty restricts nothing,
                # so there is nothing to source. `direct_only=False` in particular is
                # the DEFAULT behaviour spelled out — gating it would mean asking the
                # user to confirm they don't mind connections.
                continue
            if name in sources:
                continue
            values = _values_of(raw)
            if declared:
                continue
            if self._has_source(values):
                continue
            unsettled.append(
                Slot(
                    field=name,
                    status="unavailable",
                    why=policy.why,
                    ask=policy.ask,
                    value=", ".join(values),
                    resolvable_by=policy.resolvable_by,
                )
            )

        if not unsettled:
            return None

        key = f"{tool}:{json.dumps(arguments, ensure_ascii=False, sort_keys=True)}"
        if key in self.gated or len(self.gated) >= MAX_GATES_PER_SESSION:
            # Already gated once and the agent came back unchanged (or the session
            # has hit its gate budget). Blocking again invites an endless retry loop
            # — which on stage reads as a hung assistant — so the data goes out, but
            # every unsettled slot is booked as an assumption and the preamble
            # becomes mandatory.
            for slot in unsettled:
                self.assumptions.setdefault(
                    slot.field,
                    Assumption(
                        slot.field,
                        slot.value or "не задано",
                        "повторный вызов без уточнения — значение не подтверждено",
                    ),
                )
            return None

        self.gated.add(key)
        return GateDecision(tool=tool, slots=unsettled)

    def preamble(self) -> str:
        """The first sentence the answer MUST open with, or '' when nothing is assumed.

        Deliberately a preamble and not a footnote: a caveat at the end of a long
        answer is read only by someone already suspicious of it, which is exactly
        the reader who did not need it.
        """
        parts: list[str] = []
        for conflict in self.conflicts:
            parts.append(f"{conflict.said} — по календарю {conflict.actual}")
        for assumption in self.assumptions.values():
            parts.append(f"{assumption.field} = {assumption.value} ({assumption.rationale})")
        if not parts:
            return ""
        return (
            "Внимание: расчёт опирается на неподтверждённые данные — "
            + "; ".join(parts)
            + ". Уточните, если это не так."
        )

    def assumption_lines(self) -> list[str]:
        """Human-readable assumptions, for the answer-side report and the trace viewer."""
        return [f"{a.field} = {a.value} ({a.rationale})" for a in self.assumptions.values()] + [
            f"{c.field}: {c.said} против {c.actual}" for c in self.conflicts
        ]

    def assumed_values(self) -> set[str]:
        """Normalized values that came from an assumption, for the answer-side report."""
        out: set[str] = set()
        for assumption in self.assumptions.values():
            out.update(_norm_scalar(v) for v in assumption.value.split(","))
        return {v for v in out if v and v != "не задано"}


# --- preflight: the same engine, run before anything is searched --------------


class PlannedCall(BaseModel):
    tool: str = Field(description="Tool you intend to call.")
    arguments: dict[str, Any] = Field(
        default_factory=dict, description="Arguments you intend to pass it."
    )


class AssessRequestArgs(BaseModel):
    user_request: str = Field(description="The user's request, verbatim.")
    planned_calls: list[PlannedCall] = Field(
        default_factory=list, description="Calls you intend to make before answering."
    )


ASSESS_REQUEST_TOOL = tool_spec(
    "assess_request",
    (
        "Preflight check — call this FIRST, before any search, passing the user's request "
        "verbatim and the calls you intend to make. Runs locally (no upstream call, no rate "
        "limit, ~0 latency) and returns which parameters are blocking (must be settled before "
        "searching is meaningful), which are safe defaults, and any contradiction inside the "
        "request itself (e.g. a weekday that does not match the date given — usually a typo). "
        "Verdict `ask_user_first` means ask the user BEFORE running expensive searches, not "
        "after presenting a table built on a guess. Passing the request here also lets values "
        "the user actually typed clear the premise gate without an extra round-trip."
    ),
    AssessRequestArgs,
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True},
)


def run_assess_request_tool(
    arguments: dict[str, Any], session: SessionPremises, *, today: date | None = None
) -> tuple[str, bool]:
    """The `assess_request` tool body, as `(result_text, is_error)`.

    Transport-free, like `run_check_groundedness_tool`, so the MCP server and the
    eval harness exercise the same code path rather than two lookalikes.
    """
    args = parse_args(AssessRequestArgs, arguments)
    if isinstance(args, str):
        return args, True

    user_request = args.user_request
    session.user_request = user_request
    session.conflicts = check_calendar_consistency(user_request, today or date.today())

    # dry run: a fresh gate state over the same evidence, so a preflight never
    # burns the one-shot gate or books assumptions the agent has not made yet
    probe = SessionPremises(seen_values=session.seen_values, user_request=user_request)

    blocking: list[dict[str, Any]] = []
    for entry in args.planned_calls:
        tool = entry.tool
        clean, sources, assume = strip_control_fields(entry.arguments)
        decision = probe.evaluate(tool, clean, sources, assume)
        if decision is None:
            continue
        for slot in decision.slots:
            blocking.append(
                {
                    "tool": tool,
                    "field": slot.field,
                    "status": slot.status,
                    "value": slot.value,
                    "why": slot.why,
                    "ask": slot.ask,
                    "resolvable_by": list(slot.resolvable_by),
                }
            )

    resolvable = [b for b in blocking if b["resolvable_by"]]
    if session.conflicts or (blocking and not resolvable):
        verdict = "ask_user_first"
    elif resolvable:
        verdict = "resolve_with_tool"
    else:
        verdict = "proceed"

    return (
        json.dumps(
            {
                "verdict": verdict,
                "conflicts": [
                    {"field": c.field, "said": c.said, "actual": c.actual, "ask": c.ask}
                    for c in session.conflicts
                ],
                "blocking_slots": blocking,
                "next_step": {
                    "ask_user_first": (
                        "задайте эти вопросы пользователю СЕЙЧАС, до поиска билетов; "
                        "не стройте таблицы вариантов на догадке"
                    ),
                    "resolve_with_tool": (
                        "сначала закройте пробел вызовом из resolvable_by, "
                        "спрашивать пользователя рано"
                    ),
                    "proceed": "критичных пробелов нет — можно искать",
                }[verdict],
            },
            ensure_ascii=False,
        ),
        False,
    )
