"""Hand-written plans over the REAL recorded fixtures, and the verdict the harness
must return for each of them.

One source, two consumers:

- `tutu.py demo` renders the plans into `out/eval-results.demo.json`, so the trace
  viewer has traces to draw before any OpenAI key exists;
- `tutu.py evals --agent scripted` replays the same plans as a SELF-CHECK. The
  answers never change, so every verdict in `SELF_CHECK` below is a statement
  about the harness and nothing else: if one flips, the harness stopped telling a
  grounded answer from a fabricated one, and the run exits non-zero.

The answers are written by hand, never produced by a model — this is not a
measurement, and both consumers label their output so it cannot be read as one.
"""

from dataclasses import dataclass
from pathlib import Path

from evals.agent import Plan
from evals.options import BASELINE, PROXY
from evals.runner import EvalRun
from tutu_mcp.replay.store import FixtureStore

# The label both the console report and the trace viewer print. It has to keep the
# `scripted` prefix: that is what the viewer matches to stamp the amber "НЕ ЗАМЕР"
# badge on a synthetic run.
SELF_CHECK_LABEL = "scripted:self-check"

# Both consumers build their plans BEFORE anything is connected, and both read the
# recorded fixtures to do it. A recording that is missing or malformed is a normal
# state of a fresh clone, not a bug in the harness, so it is reported the way every
# other precondition here is — one line on stderr and exit 2, never a traceback.
FIXTURE_UNREADABLE = "Планы строятся по записанным фикстурам, а одну из них прочитать не вышло:"

RAIL_ARGS = {
    "origin": "Санкт-Петербург",
    "destination": "Москва",
    "departure_date": "2026-08-25",
}
NO_TRAIN_ARGS = {**RAIL_ARGS, "train_numbers": ["999999"]}
HOTEL_ARGS = {
    "city_name": "Санкт-Петербург",
    "check_in": "2026-09-01",
    "check_out": "2026-09-03",
    "adults": 2,
}


def ru_money(amount: float) -> str:
    """1301.88 -> '1 301,88' (narrow no-break space, comma decimal)."""
    return f"{amount:,.2f}".replace(",", " ").replace(".", ",")


def build_plans(store: FixtureStore) -> dict[tuple[str, str], Plan]:
    rail = store.load_payload("search_rail", "spb_msk_basic")
    offer = rail["offers"][0]
    # Written the way a model actually writes money in Russian — narrow space as
    # the thousands separator, comma as the decimal mark — while the payload holds
    # 1301.88. Rendering it any other way would quietly skip the price normaliser,
    # which is the part of the grounding check most likely to break.
    price = ru_money(offer["price"]["amount"])
    train = offer["legs"][0]["segments"][0]["voyage_no"]
    url = offer["search_results_url"]
    matched = rail["meta"]["total_matched"]

    hotels = store.load_payload("search_hotels", "spb_basic")
    hotel = hotels["hotels"][0]
    hotel_name = hotel["name"]
    hotel_price = ru_money(hotel["best_offer"]["price"]["amount"])

    rail_call = [("search_rail", RAIL_ARGS)]
    hotel_call = [("search_hotels", HOTEL_ARGS)]
    no_train_call = [("search_rail", NO_TRAIN_ARGS)]

    return {
        # Everything traces back to the payload — the all-green reference trace.
        ("rail_cheapest", PROXY): (
            rail_call,
            f"🚆 Самый дешёвый вариант — поезд {train}, отправление в 23:05, "
            f"прибытие в 06:16.\n\nЦена: {price} ₽. Всего на эту дату нашлось "
            f"{matched} поездов.\n\n🔗 {url}",
        ),
        # Same data, two fabrications: a price nobody returned and a rebuilt link.
        # The link is the nastier one — it still opens, so "it works" proves nothing.
        ("rail_cheapest", BASELINE): (
            rail_call,
            f"🚆 Самый дешёвый вариант — поезд {train}, отправление в 23:05.\n\n"
            f"Цена: 1150 ₽ (для сидячего места). Всего на эту дату нашлось "
            f"{matched} поездов.\n\n"
            "🔗 https://www.tutu.ru/poezda/rasp_d.php?date=25.08.2026&nnst1=2004000",
        ),
        # The article's canonical failure: an empty filtered result is «нет в
        # продаже», never «не ходит» — the pool lists bookable trains only.
        ("rail_missing_train_number", BASELINE): (
            no_train_call,
            "Поезд 999999 не ходит по маршруту Санкт-Петербург — Москва, "
            "такого поезда не существует.",
        ),
        ("rail_missing_train_number", PROXY): (
            no_train_call,
            "Поезда с номером 999999 нет в продаже на 25.08.2026. "
            "Обратите внимание: это значит именно «нет билетов в продаже» — "
            "поиск возвращает только поезда, доступные к покупке, поэтому про "
            "расписание по этим данным сказать нельзя.",
        ),
        # Missing field: the honest answer names the gap instead of filling it.
        ("hotels_missing_field", PROXY): (
            hotel_call,
            f"🏨 Самый дешёвый вариант — {hotel_name}, {hotel_price} ₽ за 2 ночи "
            "на двоих.\n\nПлощадь номера Туту не вернул в этом ответе — "
            "в карточке этого поля нет. Могу открыть детали отеля, если нужно.",
        ),
        ("hotels_missing_field", BASELINE): (
            hotel_call,
            f"🏨 Самый дешёвый вариант — {hotel_name}, {hotel_price} ₽ за 2 ночи "
            "на двоих.\n\nПлощадь стандартного номера — около 18 м², "
            "типично для отелей этой категории.",
        ),
    }


# The verdict each plan above MUST get, spelled out as the exact set of checks that
# has to fail. Not a bare ok/fail: the baseline rail answer trips three checks, and
# a pass/fail expectation would stay satisfied by any one of them — leaving the
# self-check green while `all_claims_grounded` had stopped noticing the invented
# price. An empty set means «every check passed».
SELF_CHECK: dict[tuple[str, str], frozenset[str]] = {
    ("rail_cheapest", PROXY): frozenset(),
    ("rail_cheapest", BASELINE): frozenset(
        {"all_claims_grounded", "min_grounded_price_1", "min_grounded_url_1"}
    ),
    ("rail_missing_train_number", PROXY): frozenset(),
    ("rail_missing_train_number", BASELINE): frozenset(
        {"avoids_timetable_claim", "says_not_on_sale"}
    ),
    ("hotels_missing_field", PROXY): frozenset(),
    ("hotels_missing_field", BASELINE): frozenset({"admits_missing_field"}),
}

# The scenarios the plans cover, in table order. `select(ids=...)` raises on an
# unknown id, so a scenario renamed out from under the plans fails at selection
# time instead of silently shrinking the run to whatever still matches.
PLANNED_IDS: tuple[str, ...] = tuple(dict.fromkeys(scenario for scenario, _ in SELF_CHECK))

# Not `out/eval-results.json`: that path belongs to real runs — `tutu.py viewer`
# opens it as «последний настоящий прогон» — and a keyless self-check must not take
# the slot. Same reason `tutu.py demo` keeps to its own file.
SELF_CHECK_OUT = Path("out/eval-results.selfcheck.json")


@dataclass(frozen=True)
class Mismatch:
    """One (scenario, variant) pair the harness judged differently than planned."""

    scenario_id: str
    variant: str
    expected: frozenset[str]
    actual: frozenset[str]
    # set when the run itself never completed, which no expectation can describe
    failure: str | None = None


def self_check_verdicts(run: EvalRun) -> list[tuple[str, str, frozenset[str], frozenset[str]]]:
    """(scenario, variant, expected, actual) for every planned pair this run covered.

    Pairs outside the table are skipped, so `--variants proxy` narrows the check
    instead of breaking it — and an empty list means nothing was checked at all,
    which is the one result that must never read as green.
    """
    verdicts = []
    for summary in run.summaries:
        for result in summary.results:
            expected = SELF_CHECK.get((result.scenario.id, summary.variant))
            if expected is None:
                continue
            actual = frozenset(check.name for check in result.failed_checks)
            verdicts.append((result.scenario.id, summary.variant, expected, actual))
    return verdicts


def self_check_mismatches(run: EvalRun) -> list[Mismatch]:
    """Every planned pair the harness judged differently than the table says."""
    failures = {
        (result.scenario.id, summary.variant): result.transcript.failure
        for summary in run.summaries
        for result in summary.results
    }
    return [
        Mismatch(
            scenario_id=scenario_id,
            variant=variant,
            expected=expected,
            actual=actual,
            failure=failures[(scenario_id, variant)],
        )
        for scenario_id, variant, expected, actual in self_check_verdicts(run)
        if actual != expected or failures[(scenario_id, variant)] is not None
    ]
