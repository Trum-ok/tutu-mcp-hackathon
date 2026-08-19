"""Premise gate: input-side provenance, typo detection, and the strip that keeps
fixtures matching.
"""

import json
from datetime import date

import pytest

from tutu_mcp.premises import (
    ASSUME_KEY,
    MAX_GATES_PER_SESSION,
    SOURCES_KEY,
    SessionPremises,
    check_calendar_consistency,
    run_assess_request_tool,
    strip_control_fields,
)

# --- the strip: fixtures break loudly if control fields leak downstream --------


def test_control_fields_never_reach_the_backend():
    clean, sources, assume = strip_control_fields(
        {
            "from_city": "Москва",
            "departure_date": "2026-08-25",
            SOURCES_KEY: {"price_max": "user"},
            ASSUME_KEY: {"departure_date": "ближайшие выходные"},
        }
    )

    assert clean == {"from_city": "Москва", "departure_date": "2026-08-25"}
    assert sources == {"price_max": "user"}
    assert assume == {"departure_date": "ближайшие выходные"}


def test_stripped_arguments_still_match_a_recorded_fixture(repo_fixtures):
    """The exact failure this strip exists to prevent.

    `FixtureStore` matches on the full normalized argument dict, so a leaked
    `_sources` key turns every recorded scenario into a miss — the whole mock
    demo would go dark at once.
    """
    recorded = repo_fixtures.load_scenarios("search_rail")
    arguments = next(e["arguments"] for e in recorded if e["scenario"] == "spb_msk_basic")

    clean, _, _ = strip_control_fields({**arguments, SOURCES_KEY: {"price_max": "user"}})

    assert repo_fixtures.find_result("search_rail", clean).text


def test_malformed_control_fields_are_ignored_not_crashed():
    clean, sources, assume = strip_control_fields({"a": 1, SOURCES_KEY: "нет", ASSUME_KEY: []})
    assert (clean, sources, assume) == ({"a": 1}, {}, {})


# --- the gate ------------------------------------------------------------------


def test_missing_blocking_field_gates_the_call():
    session = SessionPremises()
    decision = session.evaluate("search_rail", {"from_city": "Москва"}, {}, {})

    assert decision is not None
    assert [s.field for s in decision.slots] == ["departure_date"]
    assert json.loads(decision.to_json())["status"] == "clarification_required"


def test_invented_narrowing_value_gates_the_call():
    """The canonical failure: a filter nobody asked for, silently applied."""
    session = SessionPremises()
    decision = session.evaluate(
        "search_rail",
        {"departure_date": "2026-10-11", "price_max": 4000},
        {},
        {},
    )

    assert decision is not None
    assert [s.field for s in decision.slots] == ["price_max"]


def test_value_the_user_typed_clears_the_gate():
    """The payoff for running the preflight: no second round-trip."""
    session = SessionPremises(user_request="поезда дешевле 3000 рублей на 2026-08-25")
    assert (
        session.evaluate("search_rail", {"departure_date": "2026-08-25", "price_max": 3000}, {}, {})
        is None
    )


def test_value_taken_from_an_earlier_tool_result_clears_the_gate():
    session = SessionPremises()
    session.record_result(json.dumps({"offers": [{"train_number": "749А"}]}))

    assert (
        session.evaluate(
            "search_rail",
            {"departure_date": "2026-08-25", "train_numbers": ["749А"]},
            {},
            {},
        )
        is None
    )


def test_declared_source_clears_the_gate():
    session = SessionPremises()
    assert (
        session.evaluate(
            "search_rail",
            {"departure_date": "2026-08-25", "price_max": 4000},
            {"price_max": "user"},
            {},
        )
        is None
    )


def test_declared_assumption_lets_the_call_through_but_forces_a_preamble():
    session = SessionPremises()
    decision = session.evaluate(
        "search_rail",
        {"departure_date": "2026-10-11", "price_max": 4000},
        {},
        {"price_max": "типовой бюджет поездки"},
    )

    assert decision is None
    assert "price_max" in session.preamble()
    assert "типовой бюджет поездки" in session.preamble()


def test_declared_assumption_survives_on_a_tool_with_no_policy():
    """Most of the catalog has no premise policy, and that used to be the branch
    that dropped `_assume` on the floor: `strip_control_fields` had already taken
    the key out of the call, so the declaration vanished with it. The answer lost
    its preamble and `check_groundedness` scored the value `unavailable` —
    "invented silently" — for an agent that had just said the opposite out loud.
    """
    session = SessionPremises()

    decision = session.evaluate(
        "get_offer_details", {"details_ref": "R-77"}, {}, {"details_ref": "выбрал сам"}
    )

    assert decision is None
    assert "details_ref" in session.preamble()
    assert "выбрал сам" in session.preamble()
    assert session.assumed_values() == {"r-77"}


def test_gate_fires_once_then_releases_as_an_assumption():
    """Without this the agent can loop forever, which on stage looks like a hang."""
    session = SessionPremises()
    arguments = {"departure_date": "2026-10-11", "price_max": 4000}

    assert session.evaluate("search_rail", arguments, {}, {}) is not None
    assert session.evaluate("search_rail", arguments, {}, {}) is None
    assert "не подтверждено" in session.preamble()


def test_no_preamble_when_nothing_was_assumed():
    session = SessionPremises(user_request="на 2026-08-25")
    session.evaluate("search_rail", {"departure_date": "2026-08-25"}, {}, {})
    assert session.preamble() == ""


def test_unknown_tools_pass_through_untouched():
    session = SessionPremises()
    assert session.evaluate("create_checkout_link", {"offer_hash": "x"}, {}, {}) is None


def test_hotel_guest_count_is_blocking_but_transport_passenger_count_is_not():
    """Not everything unknown is worth a question — over-asking is its own failure.

    A hotel price is a stay total for the room and its guests, so a silently
    assumed `adults` changes the headline number. A rail fare is per passenger, so
    it does not.
    """
    session = SessionPremises(user_request="отель в Питере с 2026-09-01 по 2026-09-03")
    hotel = session.evaluate(
        "search_hotels",
        {"city_name": "Санкт-Петербург", "check_in": "2026-09-01", "check_out": "2026-09-03"},
        {},
        {},
    )
    assert hotel is not None and [s.field for s in hotel.slots] == ["adults"]

    rail = SessionPremises().evaluate(
        "search_rail", {"departure_date": "2026-08-25", "passengers": 2}, {}, {}
    )
    assert rail is None


def test_seen_values_are_capped():
    session = SessionPremises()
    session.record_result(json.dumps({"v": list(range(100_000))}))
    assert len(session.seen_values) <= 50_000


def test_real_fixture_payload_indexes_without_error(repo_fixtures):
    session = SessionPremises()
    session.record_result(
        repo_fixtures.find_result(
            "search_rail",
            {
                **next(
                    e["arguments"]
                    for e in repo_fixtures.load_scenarios("search_rail")
                    if e["scenario"] == "spb_msk_basic"
                )
            },
        ).text
    )
    assert session.seen_values


# --- typo detection ------------------------------------------------------------


TODAY = date(2026, 8, 19)


@pytest.mark.parametrize(
    "text",
    [
        "поездка одним днём на концерт в субботу 11 октября",
        "11 октября, суббота — концерт",
    ],
)
def test_weekday_contradicting_the_date_is_flagged(text):
    conflicts = check_calendar_consistency(text, TODAY)
    assert len(conflicts) == 1
    assert "воскресенье" in conflicts[0].ask


@pytest.mark.parametrize(
    "text",
    [
        "концерт 13 октября, во вторник",  # consistent
        "едем 11 октября",  # no weekday to cross-check
        "поедем в субботу",  # no date to cross-check
        "билеты на 31 февраля",  # an invalid date is a different error
    ],
)
def test_consistent_or_uncheckable_text_is_not_flagged(text):
    assert check_calendar_consistency(text, TODAY) == []


def test_the_proxy_never_guesses_which_half_of_the_contradiction_is_wrong():
    """Correcting "11" to "13" has no more support than correcting it to "21"."""
    conflict = check_calendar_consistency("в субботу 11 октября", TODAY)[0]
    assert "11 октября" in conflict.said
    assert conflict.ask.endswith("Какая дата верная?")


# --- preflight -----------------------------------------------------------------


def test_preflight_asks_before_searching_on_the_canonical_case():
    session = SessionPremises()
    text, is_error = run_assess_request_tool(
        {
            "user_request": (
                "поездка одним днём в Питер на концерт в субботу 11 октября, "
                "что выгоднее по транспорту"
            ),
            "planned_calls": [
                {"tool": "search_rail", "arguments": {"from_city": "Москва", "to_city": "СПб"}}
            ],
        },
        session,
        today=TODAY,
    )
    report = json.loads(text)

    assert not is_error
    assert report["verdict"] == "ask_user_first"
    assert report["conflicts"], "опечатка в дне недели должна быть поймана"
    assert any(s["field"] == "departure_date" for s in report["blocking_slots"])


def test_preflight_does_not_burn_the_one_shot_gate():
    """A dry run must not consume the real gate, or the agent's actual call sails through."""
    session = SessionPremises()
    run_assess_request_tool(
        {
            "user_request": "поезда в Питер",
            "planned_calls": [{"tool": "search_rail", "arguments": {"from_city": "Москва"}}],
        },
        session,
        today=TODAY,
    )

    assert session.evaluate("search_rail", {"from_city": "Москва"}, {}, {}) is not None
    assert session.assumptions == {}


def test_preflight_says_proceed_when_nothing_is_missing():
    session = SessionPremises()
    text, _ = run_assess_request_tool(
        {
            "user_request": "самый дешёвый поезд Санкт-Петербург — Москва на 2026-08-25",
            "planned_calls": [
                {
                    "tool": "search_rail",
                    "arguments": {
                        "from_city": "Санкт-Петербург",
                        "to_city": "Москва",
                        "departure_date": "2026-08-25",
                    },
                }
            ],
        },
        session,
        today=TODAY,
    )

    assert json.loads(text)["verdict"] == "proceed"


def test_preflight_prefers_a_tool_over_a_question_when_one_can_close_the_gap():
    session = SessionPremises(user_request="места рядом")
    text, _ = run_assess_request_tool(
        {
            "user_request": "подбери два места рядом",
            "planned_calls": [
                {
                    "tool": "get_rail_seatmap",
                    "arguments": {"details_ref": "ref", "seats_together": 2},
                }
            ],
        },
        session,
        today=TODAY,
    )
    report = json.loads(text)

    assert report["verdict"] == "resolve_with_tool"
    assert report["blocking_slots"][0]["resolvable_by"] == ["get_rail_seatmap"]


def test_preflight_records_the_request_so_later_calls_clear_the_gate():
    session = SessionPremises()
    run_assess_request_tool(
        {"user_request": "поезда дешевле 3000 рублей на 2026-08-25"}, session, today=TODAY
    )

    assert (
        session.evaluate("search_rail", {"departure_date": "2026-08-25", "price_max": 3000}, {}, {})
        is None
    )


def test_invalid_preflight_arguments_are_an_error_not_a_crash():
    _, is_error = run_assess_request_tool({}, SessionPremises(), today=TODAY)
    assert is_error


def test_preamble_leads_with_the_conflict_not_a_footnote():
    session = SessionPremises()
    run_assess_request_tool({"user_request": "концерт в субботу 11 октября"}, session, today=TODAY)
    assert session.preamble().startswith("Внимание:")


def test_declared_assumption_counts_even_on_a_field_the_gate_would_allow():
    """The agent owning up to a guess outranks our own view of whether it mattered."""
    session = SessionPremises()
    decision = session.evaluate(
        "search_rail",
        {"departure_date": "2026-08-29"},
        {},
        {"departure_date": "«на выходных» — взял субботу"},
    )

    assert decision is None
    assert "2026-08-29" in session.preamble()
    assert "взял субботу" in session.preamble()


# --- regressions found reviewing this module -----------------------------------


def test_short_value_does_not_ride_on_digits_from_an_unrelated_field():
    """`price_max=5` once passed because "5" appears inside "2026-08-25"."""
    session = SessionPremises(user_request="поезд Москва — Питер на 2026-08-25")
    decision = session.evaluate(
        "search_rail", {"departure_date": "2026-08-25", "price_max": 5}, {}, {}
    )

    assert decision is not None
    assert [s.field for s in decision.slots] == ["price_max"]


def test_a_value_the_user_really_typed_still_clears():
    session = SessionPremises(user_request="поезда дешевле 3000 рублей")
    assert (
        session.evaluate("search_rail", {"departure_date": "x", "price_max": 3000}, {}, {}) is None
    )


@pytest.mark.parametrize(
    "text",
    [
        "в субботу 11 маршрутов до центра",  # `ма` used to match "маршрутов"
        "поедем в среднем 15 мартовских туров",  # `сред` used to match "среднем"
    ],
)
def test_ordinary_words_are_not_read_as_a_date_contradiction(text):
    """A false contradiction costs a clarifying question about nothing."""
    assert check_calendar_consistency(text, TODAY) == []


@pytest.mark.parametrize(
    ("text", "expected_weekday"),
    [
        ("концерт в субботу 11 октября", "воскресенье"),
        ("встреча в пятницу 1 мая", "суббота"),
        ("едем в среду 15 марта", "понедельник"),
    ],
)
def test_real_contradictions_still_caught_after_tightening(text, expected_weekday):
    conflicts = check_calendar_consistency(text, TODAY)
    assert len(conflicts) == 1
    assert expected_weekday in conflicts[0].ask


@pytest.mark.parametrize("value", [False, [], ""])
def test_a_filter_that_narrows_nothing_does_not_gate(value):
    """`direct_only=False` is the default spelled out, not a restriction."""
    session = SessionPremises(user_request="поезда в Питер")
    assert (
        session.evaluate(
            "search_rail", {"departure_date": "2026-10-11", "direct_only": value}, {}, {}
        )
        is None
    )


def test_a_session_stops_gating_after_its_budget_is_spent():
    """An agent nudging arguments each round must not be gated forever."""
    session = SessionPremises()
    for i in range(MAX_GATES_PER_SESSION):
        assert session.evaluate("search_rail", {"departure_date": "d", "price_max": i}, {}, {})

    assert (
        session.evaluate("search_rail", {"departure_date": "d", "price_max": 999}, {}, {}) is None
    )
    assert session.preamble()


def test_blocking_field_value_is_not_traced_to_the_user_by_design():
    """Documented limit: "на двоих" normalizes to `adults=2`, so a text match
    would fail on a CORRECT call. Presence is checked, provenance is not."""
    session = SessionPremises(user_request="отель в Питере")
    assert (
        session.evaluate(
            "search_hotels",
            {"check_in": "2026-09-01", "check_out": "2026-09-03", "adults": 2},
            {},
            {},
        )
        is None
    )


def test_an_invented_filter_is_dropped_not_asked_about():
    """The regression this branch exists for: the agent added `direct_only=True`
    on a request that never mentioned connections, and the gate turned that into
    a question to the user — who then got interrogated about their own request."""
    session = SessionPremises(user_request="Найди самый дешёвый поезд Санкт-Петербург — Москва")
    decision = session.evaluate(
        "search_rail",
        {"departure_date": "2026-08-25", "direct_only": True},
        {},
        {},
    )

    assert decision is not None
    slot = next(s for s in decision.slots if s.field == "direct_only")
    assert slot.resolution == "drop_filter"
    # the wording has to rule out "send it as empty" — that is what the model did
    assert "не передавайте" in slot.instruction.lower()
    assert "не пустую строку" in slot.instruction


def test_a_missing_blocking_field_still_goes_to_the_user():
    """Dropping is only right when there IS something to drop. An absent blocking
    field cannot be dropped — nobody but the user can supply it."""
    session = SessionPremises(user_request="Найди отель в Санкт-Петербурге")
    decision = session.evaluate("search_hotels", {"check_in": "2026-09-01"}, {}, {})

    assert decision is not None
    slot = next(s for s in decision.slots if s.field == "adults")
    assert slot.resolution == "ask_user"
    assert slot.instruction == slot.ask


def test_preflight_tells_the_agent_to_drop_its_own_filter():
    session = SessionPremises()
    text, is_error = run_assess_request_tool(
        {
            "user_request": "Найди самый дешёвый поезд Санкт-Петербург — Москва на 2026-08-25.",
            "planned_calls": [
                {
                    "tool": "search_rail",
                    "arguments": {"departure_date": "2026-08-25", "direct_only": True},
                }
            ],
        },
        session,
        today=TODAY,
    )
    report = json.loads(text)

    assert not is_error
    assert report["verdict"] == "drop_invented_filters"
    assert "уберите" in report["next_step"].lower()
    assert "не передавайте" in report["blocking_slots"][0]["do"].lower()
    assert report["blocking_slots"][0]["resolution"] == "drop_filter"


def test_a_user_stated_filter_is_neither_dropped_nor_asked_about():
    """Dropping a filter the user DID ask for would silently widen their search —
    the mirror image of the bug, and just as wrong."""
    session = SessionPremises(user_request="Нужен только прямой поезд, без пересадок")
    decision = session.evaluate(
        "search_rail",
        {"departure_date": "2026-08-25", "direct_only": True},
        {"direct_only": "user"},
        {},
    )

    assert decision is None


def test_a_leap_day_resolves_to_the_next_leap_year_instead_of_vanishing():
    """`_resolve_year` used to give up on the first `ValueError`, so 29 February
    resolved to nothing and the weekday check silently skipped the date most
    likely to carry a typo. 2028-02-29 is a Tuesday."""
    conflicts = check_calendar_consistency("поеду в понедельник 29 февраля", date(2026, 8, 19))

    assert len(conflicts) == 1
    assert "2028-02-29" in conflicts[0].actual


def test_a_correct_leap_day_weekday_is_not_flagged():
    assert check_calendar_consistency("поеду во вторник 29 февраля", date(2026, 8, 19)) == []


def test_a_date_invalid_in_every_year_still_resolves_to_nothing():
    assert check_calendar_consistency("поеду в среду 31 февраля", date(2026, 8, 19)) == []
