from tutu_mcp.groundedness import check_groundedness

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
