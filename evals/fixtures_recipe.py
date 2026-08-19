"""WHICH calls `tutu.py record` records — the eval harness's own shopping list.

Lives in `evals/` rather than next to the driver in `tutu_mcp/replay/` because
that is where its source of truth is: every date and city here has to match the
scenarios in `evals/scenarios.py` exactly, or a mock run misses on every fixture
it was supposed to replay. Those constants used to be copied into the driver by
hand, which meant the proxy package silently owned a list dictated by the
harness — and two literals that had to stay byte-identical across a package
boundary nothing checks.

The direction of the dependency is the point: `evals` may import `tutu_mcp`,
never the other way round.
"""

from collections.abc import Callable
from typing import Any

from evals.scenarios import HOTEL_CHECK_IN, HOTEL_CHECK_OUT, RAIL_DATE
from tutu_mcp.replay.bootstrap import Recorded, RecordSpec

SPB = "Санкт-Петербург"
MSK = "Москва"

Build = Callable[[dict[str, Any]], dict[str, Any]]


def _from_rail_offer(build: Build) -> Callable[[Recorded], dict[str, Any] | None]:
    """Arguments derived from the first offer of the basic rail search.

    Seatmaps, offer details and checkout links all need a `details_ref` that only
    a real search can mint, so they can only be recorded after it — and skipped
    cleanly when it returned nothing.
    """

    def arguments(recorded: Recorded) -> dict[str, Any] | None:
        offers = (recorded.get("search_rail/spb_msk_basic") or {}).get("offers") or []
        return build(offers[0]) if offers else None

    return arguments


def _from_hotel(build: Build) -> Callable[[Recorded], dict[str, Any] | None]:
    def arguments(recorded: Recorded) -> dict[str, Any] | None:
        hotels = (recorded.get("search_hotels/spb_basic") or {}).get("hotels") or []
        if not hotels:
            return None
        return build(hotels[0])

    return arguments


def _hotel_checkout(hotel: dict[str, Any]) -> dict[str, Any]:
    ref = hotel.get("checkout_ref") or hotel.get("best_offer", {}).get("checkout_ref")
    # An empty dict rather than None: the spec still records, and the resulting
    # upstream error is itself a useful fixture.
    return {"product_type": "hotels", **(ref or {})}


def _route(origin: str, destination: str, date: str = RAIL_DATE) -> dict[str, Any]:
    return {"origin": origin, "destination": destination, "departure_date": date}


FIXTURE_CALLS: list[RecordSpec] = [
    *[
        RecordSpec(f"get_{domain}_instructions", "default", {}, group="instructions")
        for domain in ("avia", "rail", "bus", "etrain", "hotels", "multitransport")
    ],
    *[
        RecordSpec(
            "fetch_resource",
            uri.replace("tutu://", "").replace("/", "_"),
            {"uri": uri},
            group="resources",
        )
        for uri in ("tutu://help/overview", "tutu://status", "tutu://amenities/dictionary")
    ],
    RecordSpec("search_rail", "spb_msk_basic", _route(SPB, MSK), group="search_rail"),
    RecordSpec("search_rail", "spb_msk_city_alias", _route("Питер", "Мск"), group="search_rail"),
    RecordSpec(
        "search_rail",
        "spb_msk_price_max",
        {**_route(SPB, MSK), "price_max": 3000},
        group="search_rail",
    ),
    RecordSpec(
        "search_rail",
        "spb_msk_no_such_train",
        {**_route(SPB, MSK), "train_numbers": ["999999"]},
        group="search_rail",
    ),
    RecordSpec(
        "search_rail",
        "invalid_date",
        _route(SPB, MSK, "not-a-date"),
        group="search_rail",
    ),
    RecordSpec(
        "get_offer_details",
        "rail_basic",
        _from_rail_offer(lambda o: {"product_type": "rail", "details_ref": o["details_ref"]}),
        group="get_offer_details (rail)",
    ),
    RecordSpec(
        "get_rail_seatmap",
        "basic",
        _from_rail_offer(lambda o: {"details_ref": o["details_ref"]}),
        group="get_rail_seatmap",
    ),
    RecordSpec(
        "get_rail_seatmap",
        "together_2",
        _from_rail_offer(
            lambda o: {"details_ref": o["details_ref"], "task": "together", "seats_together": 2}
        ),
        group="get_rail_seatmap",
    ),
    RecordSpec(
        "create_checkout_link",
        "rail_seat_page",
        _from_rail_offer(lambda o: dict(o["checkout_ref"])),
        group="create_checkout_link (rail)",
    ),
    RecordSpec("search_avia", "msk_spb_basic", _route(MSK, SPB), group="search_avia"),
    RecordSpec("search_bus", "msk_spb_basic", _route(MSK, SPB), group="search_bus"),
    RecordSpec(
        "search_etrain", "msk_mytishchi_basic", _route(MSK, "Мытищи"), group="search_etrain"
    ),
    RecordSpec(
        "search_multitransport", "msk_spb_basic", _route(MSK, SPB), group="search_multitransport"
    ),
    RecordSpec(
        "search_hotels",
        "spb_basic",
        {
            "city_name": SPB,
            "check_in": HOTEL_CHECK_IN,
            "check_out": HOTEL_CHECK_OUT,
            "adults": 2,
        },
        group="search_hotels",
    ),
    RecordSpec(
        "get_offer_details",
        "hotel_basic",
        _from_hotel(
            lambda h: {
                "product_type": "hotels",
                "hotel_id": h.get("hotel_id") or h.get("hotel_geo_id"),
                "check_in": HOTEL_CHECK_IN,
                "check_out": HOTEL_CHECK_OUT,
                "adults": 2,
            }
        ),
        group="get_offer_details (hotel)",
    ),
    RecordSpec(
        "create_checkout_link",
        "hotel_page",
        _from_hotel(_hotel_checkout),
        group="create_checkout_link (hotel)",
    ),
]
