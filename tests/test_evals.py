"""The harness must distinguish a grounded answer from a fabricated one.

These run the real runner over the real fixtures with a scripted agent, so they
exercise the same code path a model-driven run takes — only the model is replaced.
"""

import pytest

from evals.agent import ScriptedAgent
from evals.runner import evaluate, run_eval, run_scenario
from evals.scenarios import Scenario, select
from evals.tokens import OfflineTokenCounter
from evals.transcript import ToolCallRecord, Transcript
from evals.variants import BASELINE, PROXY, build_variants
from tutu_mcp.replay.mock_client import MockUpstreamClient

from .conftest import load_result_payload

RAIL_ARGS = {
    "origin": "Санкт-Петербург",
    "destination": "Москва",
    "departure_date": "2026-08-25",
}


@pytest.fixture
def rail_offer():
    payload = load_result_payload("search_rail", "spb_msk_basic")
    return payload["offers"][0]


@pytest.fixture
def backend(repo_fixtures) -> MockUpstreamClient:
    return MockUpstreamClient(repo_fixtures)


async def _variants(backend, repo_fixtures, names=None):
    return await build_variants(backend, repo_fixtures.instructions(), names=names)


async def test_grounded_answer_passes_every_check(backend, repo_fixtures, rail_offer):
    answer = (
        f"Самый дешёвый — поезд {rail_offer['legs'][0]['segments'][0]['voyage_no']} "
        f"за {rail_offer['price']['amount']} ₽. Ссылка: {rail_offer['search_results_url']}"
    )
    agent = ScriptedAgent(plan={"rail_cheapest": ([("search_rail", RAIL_ARGS)], answer)})
    scenario = select(ids=["rail_cheapest"])[0]
    variant = (await _variants(backend, repo_fixtures, [BASELINE]))[0]

    result = await run_scenario(agent, scenario, variant)

    assert result.success, result.failed_checks
    assert result.grounding.rate == 1.0


async def test_fabricated_price_and_url_fail_the_run(backend, repo_fixtures):
    answer = "Нашёл поезд за 100 ₽. Ссылка: https://not-tutu.example/deal"
    agent = ScriptedAgent(plan={"rail_cheapest": ([("search_rail", RAIL_ARGS)], answer)})
    scenario = select(ids=["rail_cheapest"])[0]
    variant = (await _variants(backend, repo_fixtures, [BASELINE]))[0]

    result = await run_scenario(agent, scenario, variant)

    assert not result.success
    failed = {c.name for c in result.failed_checks}
    assert "all_claims_grounded" in failed


async def test_unsupported_absence_claim_is_caught():
    """The article's canonical failure: an empty result is «нет в продаже»,
    never «поезд не ходит» — the pool only lists bookable trains."""
    scenario = select(ids=["rail_missing_train_number"])[0]
    transcript = Transcript(
        scenario_id=scenario.id,
        variant=BASELINE,
        answer_text="Такого поезда нет, он не ходит по этому маршруту.",
        tool_calls=[
            ToolCallRecord(
                name="search_rail", arguments={}, result_text='{"offers": []}', is_error=False
            )
        ],
    )

    result = evaluate(scenario, transcript)

    assert not result.success
    assert "avoids_timetable_claim" in {c.name for c in result.failed_checks}


async def test_honest_not_on_sale_wording_passes():
    scenario = select(ids=["rail_missing_train_number"])[0]
    transcript = Transcript(
        scenario_id=scenario.id,
        variant=BASELINE,
        answer_text="Поезда 999999 нет в продаже на эту дату.",
        tool_calls=[
            ToolCallRecord(
                name="search_rail", arguments={}, result_text='{"offers": []}', is_error=False
            )
        ],
    )

    result = evaluate(scenario, transcript)

    assert result.success, result.failed_checks


async def test_fixture_miss_is_not_counted_as_a_tool_error(backend, repo_fixtures):
    """A gap in our recording must never read as upstream misbehaving."""
    agent = ScriptedAgent(
        plan={"rail_cheapest": ([("search_rail", {"origin": "Ниоткуда"})], "ответ")}
    )
    scenario = select(ids=["rail_cheapest"])[0]
    variant = (await _variants(backend, repo_fixtures, [BASELINE]))[0]

    result = await run_scenario(agent, scenario, variant)

    assert result.transcript.fixture_miss_count == 1
    assert result.transcript.tool_error_count == 0


async def test_both_variants_see_identical_search_results(backend, repo_fixtures):
    """The experiment's isolation property: only the tool SURFACE differs between
    variants, never the data — otherwise a win could just mean better data."""
    baseline, proxy = await _variants(backend, repo_fixtures, [BASELINE, PROXY])

    base_result = await baseline.execute("search_rail", RAIL_ARGS)
    proxy_result = await proxy.execute("search_rail", RAIL_ARGS)

    assert base_result.result_text == proxy_result.result_text


async def test_proxy_moves_trimmed_prose_into_the_instructions_result(backend, repo_fixtures):
    baseline, proxy = await _variants(backend, repo_fixtures, [BASELINE, PROXY])

    base_tools = {t["name"]: t for t in baseline.tools}
    proxy_tools = {t["name"]: t for t in proxy.tools}
    base_instr = await baseline.execute("get_rail_instructions", {})
    proxy_instr = await proxy.execute("get_rail_instructions", {})

    # surface shrinks...
    assert len(proxy_tools["search_rail"]["description"]) < len(
        base_tools["search_rail"]["description"]
    )
    # ...and the detail is still reachable, just on demand
    assert len(proxy_instr.result_text) > len(base_instr.result_text)
    assert "search_rail" in proxy_instr.result_text


async def test_proxy_surface_is_cheaper_than_baseline(backend, repo_fixtures):
    agent = ScriptedAgent(plan={})
    scenarios = select(ids=["rail_cheapest"])
    variants = await _variants(backend, repo_fixtures, [BASELINE, PROXY])

    run = await run_eval(
        agent=agent,
        scenarios=scenarios,
        variants=variants,
        token_counter=OfflineTokenCounter(),
    )

    base = run.by_variant(BASELINE)
    proxy = run.by_variant(PROXY)
    assert base is not None and proxy is not None
    assert base.surface is not None and proxy.surface is not None
    assert proxy.surface.tokens < base.surface.tokens
    assert run.surface_exact is False


async def test_scenario_with_no_checkable_claims_does_not_score_a_free_pass():
    """A vague non-answer must not beat a specific one just by making no claims."""
    scenario = Scenario(
        id="vague",
        domain="rail",
        request="—",
        checks=select(ids=["rail_cheapest"])[0].checks,
    )
    transcript = Transcript(
        scenario_id="vague",
        variant=BASELINE,
        answer_text="Есть подходящие варианты, посмотрите на сайте.",
        tool_calls=[
            ToolCallRecord(
                name="search_rail", arguments={}, result_text='{"offers": []}', is_error=False
            )
        ],
    )

    result = evaluate(scenario, transcript)

    assert not result.success
    assert "all_claims_grounded" in {c.name for c in result.failed_checks}


class ExplodingCounter:
    """An exact counter reaches the provider — so it can fail where nothing else in
    an offline run can."""

    exact = True
    estimate_reason = None

    async def count(self, *, tools, system):
        raise RuntimeError("connection reset")


async def test_a_failed_surface_measurement_does_not_take_the_run_down(repo_fixtures, capsys):
    """The surface probe is one call; the scenarios are the measurement. Losing the
    probe must cost the surface row and nothing else — `surface` is optional end to
    end precisely so this can degrade."""
    variants = await build_variants(
        MockUpstreamClient(repo_fixtures), repo_fixtures.instructions(), names=[PROXY]
    )
    agent = ScriptedAgent(plan={"rail_cheapest": ([("search_rail", RAIL_ARGS)], "ответ")})

    run = await run_eval(
        agent=agent,
        scenarios=select(ids=["rail_cheapest"]),
        variants=variants,
        token_counter=ExplodingCounter(),
    )

    summary = run.by_variant(PROXY)
    assert summary is not None
    assert summary.surface is None
    assert summary.total == 1
    assert "не измерена" in capsys.readouterr().err
