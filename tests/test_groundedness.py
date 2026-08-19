import json

from tutu_mcp.groundedness import check_groundedness, run_check_groundedness_tool

from .conftest import load_result_payload


def test_real_claims_from_the_cheapest_offer_are_grounded():
    payload = load_result_payload("search_rail", "spb_msk_basic")
    offer = payload["offers"][0]
    price = offer["price"]["amount"]
    url = offer["search_results_url"]
    train_number = offer["legs"][0]["segments"][0]["voyage_no"]

    answer = f"Самый дешёвый вариант — поезд {train_number}, цена {price} ₽. Подробнее: {url}"
    report = check_groundedness(answer, [payload])

    assert report.rate == 1.0
    assert all(check.grounded for check in report.checks)


def test_fabricated_price_and_url_are_flagged():
    payload = load_result_payload("search_rail", "spb_msk_basic")

    answer = (
        "Есть отличный вариант всего за 4567,89 ₽! Подробности тут: https://not-tutu.example/scam"
    )
    report = check_groundedness(answer, [payload])

    kinds = {check.claim.kind: check.grounded for check in report.checks}
    assert kinds["price"] is False
    assert kinds["url"] is False


def test_rounded_price_still_counts_as_grounded():
    payload = load_result_payload("search_rail", "spb_msk_basic")
    exact_price = payload["offers"][0]["price"]["amount"]
    rounded = round(exact_price)

    report = check_groundedness(f"Билет за {rounded} ₽.", [payload])

    assert report.checks[0].grounded is True


def test_no_claims_in_answer_gives_no_rate():
    report = check_groundedness("Отличный выбор для поездки!", [{"irrelevant": True}])

    assert report.checks == []
    assert report.rate is None


def test_price_claims_only_match_price_tagged_fields_not_unrelated_counts():
    # regression: a payload full of small integers (page numbers, counts, ids) must not
    # ground a fabricated "1 ₽" just because some unrelated `count: 1` field exists
    payload = {"meta": {"page": 1, "total_matched": 1}, "offers": [{"segments_count": 1}]}

    report = check_groundedness("Билет всего за 1 ₽!", [payload])

    assert report.checks[0].grounded is False


def test_our_own_error_payload_does_not_ground_the_value_it_echoes():
    # regression: a fixture miss (or any backend_error payload) quotes the
    # unmatched arguments back — an invented "777A" must not "confirm" itself
    # just because it also appears in our own error text
    error_payload = {
        "status": "fixture_not_found",
        "tool": "search_rail",
        "error": "No fixture for search_rail(train_numbers=['777A'], price_max=4000)",
    }

    report = check_groundedness("Поезд 777A, цена 4000 руб.", [error_payload])

    assert all(not check.grounded for check in report.checks)
    assert report.ignored_error_payloads == 1


def test_the_check_defaults_to_what_the_proxy_delivered():
    """No `tool_result_json` at all: the proxy checks against the results it
    already handed the agent, so the agent never copies payloads back."""
    text, is_error = run_check_groundedness_tool(
        {"answer_text": "Цена 1 301,88 ₽"},
        session_payloads=[{"offers": [{"price": {"amount": 1301.88, "currency": "RUB"}}]}],
    )
    report = json.loads(text)

    assert not is_error
    assert report["groundedness_rate"] == 1.0


def test_agent_supplied_evidence_only_adds_to_the_session_evidence():
    """A payload the agent passes in supplements what the proxy saw — it cannot
    replace it, so an agent cannot swap in a friendlier set of facts."""
    text, _ = run_check_groundedness_tool(
        {
            "answer_text": "Цена 1 301,88 ₽ и цена 999 ₽",
            "tool_result_json": ['{"offers": [{"price": {"amount": 999}}]}'],
        },
        session_payloads=[{"offers": [{"price": {"amount": 1301.88}}]}],
    )
    report = json.loads(text)

    assert {c["text"] for c in report["claims"]} == {"1 301,88 ₽", "999 ₽"}
    assert all(c["grounded"] for c in report["claims"])


def test_a_check_with_no_evidence_at_all_is_an_error_not_a_pass():
    """Zero payloads used to mean zero claims to contradict, i.e. a free pass."""
    text, is_error = run_check_groundedness_tool({"answer_text": "Цена 1 301,88 ₽"})

    assert is_error
    assert "не с чем сверять" in text


def test_a_threshold_the_user_named_is_not_a_fabrication():
    """ "поезда дешевле 3000 ₽" restated in the answer is the request quoted back.
    No payload owes us that number, and scoring it as invented failed a correct
    answer on both variants."""
    report = check_groundedness(
        "Нашлось 4 поезда дешевле 3000 ₽, самый дешёвый 1 301,88 ₽",
        [{"offers": [{"price": {"amount": 1301.88}}]}],
        user_request="Поезда Санкт-Петербург — Москва дешевле 3000 рублей",
    )
    by_text = {c.claim.text: c for c in report.checks}

    assert by_text["3000 ₽"].status == "user_stated"
    assert not by_text["3000 ₽"].fabricated
    assert by_text["1 301,88 ₽"].status == "confirmed"
    # the quoted threshold is out of the denominator, so the rate stays honest
    assert report.rate == 1.0


def test_a_price_in_both_the_request_and_the_payload_stays_confirmed():
    """Evidence beats a citation — otherwise quoting a number would downgrade it."""
    report = check_groundedness(
        "Цена 1 301,88 ₽",
        [{"offers": [{"price": {"amount": 1301.88}}]}],
        user_request="есть вариант за 1 301,88 ₽?",
    )

    assert [c.status for c in report.checks] == ["confirmed"]


def test_a_number_the_user_never_said_is_still_a_fabrication():
    report = check_groundedness(
        "Цена 9 999 ₽",
        [{"offers": [{"price": {"amount": 1301.88}}]}],
        user_request="самый дешёвый поезд",
    )

    assert report.fabricated
    assert report.rate == 0.0
