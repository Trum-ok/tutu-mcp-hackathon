"""Generates `out/eval-results.demo.json` without calling any model.

The trace viewer needs traces to render, and a real run needs an OpenAI key plus
several minutes. This builds a small set of hand-authored answers over the REAL
recorded fixtures, so every "confirmed" claim is genuinely confirmed against real
Tutu data and every fabricated one is genuinely absent from it — the viewer's
highlighting is exercised for real, not faked with hardcoded colors.

IMPORTANT: this is not a measurement. The answers are written by hand to cover
the interesting states (grounded / fabricated / unsupported-absence-claim), not
produced by a model. The agent label in the output says `demo:hand-written`, and
the viewer prints it in the header, so a demo dataset can never be mistaken on
screen for a real eval run.

    uv run python tutu.py demo
"""

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evals import report as report_mod
from evals.agent import ToolExecutor
from evals.runner import run_eval
from evals.scenarios import select
from evals.tokens import OfflineTokenCounter
from evals.transcript import Transcript
from evals.variants import BASELINE, PROXY, build_variants
from tutu_mcp.backends import backend_for
from tutu_mcp.config import load_settings
from tutu_mcp.replay.store import FixtureStore

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

Plan = tuple[list[tuple[str, dict[str, Any]]], str]


@dataclass
class DemoAgent:
    """Like ScriptedAgent but keyed by (scenario, variant), so one scenario can
    show a good answer on one surface and a bad one on the other — which is the
    whole point of a side-by-side trace viewer."""

    plans: dict[tuple[str, str], Plan]

    @property
    def label(self) -> str:
        return "demo:hand-written"

    async def run(
        self,
        *,
        scenario_id: str,
        variant: str,
        request: str,
        tools: list[dict[str, Any]],
        server_instructions: str,
        execute: ToolExecutor,
    ) -> Transcript:
        transcript = Transcript(scenario_id=scenario_id, variant=variant)
        started = time.monotonic()
        calls, answer = self.plans.get((scenario_id, variant), ([], ""))
        for name, arguments in calls:
            transcript.tool_calls.append(await execute(name, dict(arguments)))
        transcript.turns = len(calls) + 1
        transcript.answer_text = answer
        # plausible-looking but clearly synthetic; a real run overwrites these
        transcript.input_tokens = 12000 if variant == BASELINE else 8000
        transcript.output_tokens = 320
        transcript.duration_s = time.monotonic() - started
        return transcript


def ru_money(amount: float) -> str:
    """1301.88 -> '1 301,88' (narrow no-break space, comma decimal)."""
    return f"{amount:,.2f}".replace(",", " ").replace(".", ",")


def load_fixture(store: FixtureStore, tool: str, scenario: str) -> dict[str, Any]:
    path = store.fixtures_dir / tool / f"{scenario}.json"
    return json.loads(json.loads(path.read_text(encoding="utf-8"))["result"]["text"])


def build_plans(store: FixtureStore) -> dict[tuple[str, str], Plan]:
    rail = load_fixture(store, "search_rail", "spb_msk_basic")
    offer = rail["offers"][0]
    # Written the way a model actually writes money in Russian — narrow space as
    # the thousands separator, comma as the decimal mark — while the payload holds
    # 1301.88. Rendering it any other way would quietly skip the price normaliser,
    # which is the part of the grounding check most likely to break.
    price = ru_money(offer["price"]["amount"])
    train = offer["legs"][0]["segments"][0]["voyage_no"]
    url = offer["search_results_url"]
    matched = rail["meta"]["total_matched"]

    hotels = load_fixture(store, "search_hotels", "spb_basic")
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


async def run_demo() -> int:
    settings = load_settings()
    store = FixtureStore(settings.fixtures_dir)

    scenario_ids = ["rail_cheapest", "rail_missing_train_number", "hotels_missing_field"]
    scenarios = select(ids=scenario_ids)
    agent = DemoAgent(plans=build_plans(store))

    # Hard-wired to the mock: these traces exist to be reproducible on a laptop
    # with no key and no network, so `TUTU_PROXY_MODE=live` must not reach them.
    async with backend_for(settings, live=False) as wiring:
        variants = await build_variants(wiring.backend, wiring.instructions())

        run = await run_eval(
            agent=agent,
            scenarios=scenarios,
            variants=variants,
            token_counter=OfflineTokenCounter(),
        )

    # Deliberately NOT out/eval-results.json: that path belongs to real runs, and two
    # people generating different things into one file overwrite each other's work.
    out = Path("out/eval-results.demo.json")
    report_mod.write_json(run, out)
    print(report_mod.render_console(run))
    print(f"\nДемо-трейсы записаны в {out} (агент: {agent.label} — это НЕ замер)")
    return 0
