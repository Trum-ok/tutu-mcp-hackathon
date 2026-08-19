"""Records real mcp.tutu.ru responses as fixtures for `TUTU_PROXY_MODE=mock`.

Run once (or after upstream changes) against the live server:

    uv run python tutu.py record

Deliberately sequential with a short delay between calls — the hackathon's
rate limit is shared across every team, so this script is polite by design
rather than fast. A handful of scenarios chain off an earlier call's
`details_ref` / `checkout_ref` (seatmap, offer details, checkout links), so
recording order matters; keep new scenarios in dependency order.

Known gap: no 429 fixture. Provoking one on purpose would burn the shared
rate limit for every other team on-site, so that fixture has to be
hand-authored (or captured incidentally) instead of recorded here.
"""

import asyncio
import json
from typing import Any

from tutu_mcp.config import load_settings
from tutu_mcp.replay.store import FixtureStore
from tutu_mcp.upstream.client import UpstreamClient

DELAY_BETWEEN_CALLS_SECONDS = 0.4
RAIL_DATE = "2026-08-25"
HOTEL_CHECK_IN = "2026-09-01"
HOTEL_CHECK_OUT = "2026-09-03"


async def record(
    store: FixtureStore, client: UpstreamClient, tool: str, scenario: str, arguments: dict[str, Any]
) -> dict[str, Any] | None:
    try:
        result = await client.call_tool(tool, arguments)
    except Exception as exc:  # this is a recording script — log and keep going
        print(f"  ! {tool}/{scenario} raised {type(exc).__name__}: {exc}")
        return None

    store.save_tool_result(tool, scenario, arguments, result)
    status = "ERROR" if result.is_error else "ok"
    print(f"  - {tool}/{scenario}: {status}, {len(result.text)} bytes")
    await asyncio.sleep(DELAY_BETWEEN_CALLS_SECONDS)

    if result.is_error:
        return None
    try:
        return json.loads(result.text)
    except json.JSONDecodeError:
        return None


async def record_fixtures() -> None:
    settings = load_settings()
    store = FixtureStore(settings.fixtures_dir)

    async with UpstreamClient(settings.upstream_url) as client:
        print(f"Recording from {settings.upstream_url} into {settings.fixtures_dir}")

        print("initialize")
        info = client.server_info()
        store.save_server_info(info)
        print(
            f"  - {info['name']} {info['version']}, instructions {len(info['instructions'])} bytes"
        )

        print("tools/list")
        tools = await client.list_tools()
        store.save_tools_list(tools)
        print(f"  - {len(tools)} tools, {len(json.dumps(tools, ensure_ascii=False))} bytes")
        await asyncio.sleep(DELAY_BETWEEN_CALLS_SECONDS)

        print("instructions")
        for domain in ("avia", "rail", "bus", "etrain", "hotels", "multitransport"):
            await record(store, client, f"get_{domain}_instructions", "default", {})

        print("resources")
        for uri in ("tutu://help/overview", "tutu://status", "tutu://amenities/dictionary"):
            scenario = uri.replace("tutu://", "").replace("/", "_")
            await record(store, client, "fetch_resource", scenario, {"uri": uri})

        print("search_rail")
        rail_basic = await record(
            store,
            client,
            "search_rail",
            "spb_msk_basic",
            {"origin": "Санкт-Петербург", "destination": "Москва", "departure_date": RAIL_DATE},
        )
        await record(
            store,
            client,
            "search_rail",
            "spb_msk_city_alias",
            {"origin": "Питер", "destination": "Мск", "departure_date": RAIL_DATE},
        )
        await record(
            store,
            client,
            "search_rail",
            "spb_msk_price_max",
            {
                "origin": "Санкт-Петербург",
                "destination": "Москва",
                "departure_date": RAIL_DATE,
                "price_max": 3000,
            },
        )
        await record(
            store,
            client,
            "search_rail",
            "spb_msk_no_such_train",
            {
                "origin": "Санкт-Петербург",
                "destination": "Москва",
                "departure_date": RAIL_DATE,
                "train_numbers": ["999999"],
            },
        )
        await record(
            store,
            client,
            "search_rail",
            "invalid_date",
            {"origin": "Санкт-Петербург", "destination": "Москва", "departure_date": "not-a-date"},
        )

        if rail_basic and rail_basic.get("offers"):
            rail_offer = rail_basic["offers"][0]
            details_ref = rail_offer["details_ref"]
            checkout_ref = rail_offer["checkout_ref"]

            print("get_offer_details (rail)")
            await record(
                store,
                client,
                "get_offer_details",
                "rail_basic",
                {"product_type": "rail", "details_ref": details_ref},
            )

            print("get_rail_seatmap")
            await record(store, client, "get_rail_seatmap", "basic", {"details_ref": details_ref})
            await record(
                store,
                client,
                "get_rail_seatmap",
                "together_2",
                {"details_ref": details_ref, "task": "together", "seats_together": 2},
            )

            print("create_checkout_link (rail)")
            await record(
                store, client, "create_checkout_link", "rail_seat_page", dict(checkout_ref)
            )
        else:
            print("  ! no rail offers recorded, skipping dependent rail fixtures")

        print("search_avia")
        await record(
            store,
            client,
            "search_avia",
            "msk_spb_basic",
            {"origin": "Москва", "destination": "Санкт-Петербург", "departure_date": RAIL_DATE},
        )

        print("search_bus")
        await record(
            store,
            client,
            "search_bus",
            "msk_spb_basic",
            {"origin": "Москва", "destination": "Санкт-Петербург", "departure_date": RAIL_DATE},
        )

        print("search_etrain")
        await record(
            store,
            client,
            "search_etrain",
            "msk_mytishchi_basic",
            {"origin": "Москва", "destination": "Мытищи", "departure_date": RAIL_DATE},
        )

        print("search_multitransport")
        await record(
            store,
            client,
            "search_multitransport",
            "msk_spb_basic",
            {"origin": "Москва", "destination": "Санкт-Петербург", "departure_date": RAIL_DATE},
        )

        print("search_hotels")
        hotels_basic = await record(
            store,
            client,
            "search_hotels",
            "spb_basic",
            {
                "city_name": "Санкт-Петербург",
                "check_in": HOTEL_CHECK_IN,
                "check_out": HOTEL_CHECK_OUT,
                "adults": 2,
            },
        )

        if hotels_basic and hotels_basic.get("hotels"):
            hotel = hotels_basic["hotels"][0]
            hotel_id = hotel.get("hotel_id") or hotel.get("hotel_geo_id")

            print("get_offer_details (hotel)")
            await record(
                store,
                client,
                "get_offer_details",
                "hotel_basic",
                {
                    "product_type": "hotels",
                    "hotel_id": hotel_id,
                    "check_in": HOTEL_CHECK_IN,
                    "check_out": HOTEL_CHECK_OUT,
                    "adults": 2,
                },
            )

            checkout_ref = hotel.get("checkout_ref") or hotel.get("best_offer", {}).get(
                "checkout_ref"
            )
            if checkout_ref:
                print("create_checkout_link (hotel)")
                await record(
                    store,
                    client,
                    "create_checkout_link",
                    "hotel_page",
                    {"product_type": "hotels", **checkout_ref},
                )
        else:
            print("  ! no hotel offers recorded, skipping dependent hotel fixtures")

        print("Done.")
