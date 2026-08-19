"""End-to-end premise-gate behavior through the real runner and real fixtures.

The unit tests in `test_premises.py` pin the engine. These pin the thing that
actually matters: that an agent driving the proxy variant cannot produce a
confident table built on a value nobody supplied, and that an agent given a
determinate request is not interrogated for no reason.
"""

import json

import pytest

from evals.agent import ScriptedAgent
from evals.runner import run_scenario
from evals.scenarios import select
from evals.variants import BASELINE, PROXY, build_variants
from tutu_mcp.premises import ASSUME_KEY
from tutu_mcp.replay.mock_client import MockUpstreamClient

RAIL_ARGS = {
    "origin": "Санкт-Петербург",
    "destination": "Москва",
    "departure_date": "2026-08-25",
}


@pytest.fixture
def backend(repo_fixtures) -> MockUpstreamClient:
    return MockUpstreamClient(repo_fixtures)


async def _variant(backend, repo_fixtures, name):
    return (await build_variants(backend, repo_fixtures.instructions(), names=[name]))[0]


async def test_baseline_lets_the_invented_premise_through(backend, repo_fixtures):
    """The failure being fixed, reproduced on the untouched upstream surface.

    Every number here is real — the agent invents only the RETURN DEADLINE it
    filtered by, and mentions it nowhere. Groundedness scores it perfect, which
    is precisely why the input side needed its own check.
    """
    scenario = select(ids=["concert_day_trip_unknown_time"])[0]
    agent = ScriptedAgent(
        plan={
            scenario.id: (
                [("search_rail", {**RAIL_ARGS, "price_max": 4000})],
                "Вот сравнение вариантов: поезд, самолёт и автобус. Успеваете на обратный.",
            )
        }
    )
    variant = await _variant(backend, repo_fixtures, BASELINE)

    result = await run_scenario(agent, scenario, variant)

    assert not result.success
    assert "clarified_before_search" in {c.name for c in result.failed_checks}
    # nothing was flagged as assumed, because baseline has no premise state at all
    assert result.grounding.assumptions == []


async def test_proxy_returns_a_question_instead_of_data(backend, repo_fixtures):
    scenario = select(ids=["concert_day_trip_unknown_time"])[0]
    agent = ScriptedAgent(
        plan={scenario.id: ([("search_rail", {**RAIL_ARGS, "price_max": 4000})], "")}
    )
    variant = await _variant(backend, repo_fixtures, PROXY)

    result = await run_scenario(agent, scenario, variant)
    gate = json.loads(result.transcript.tool_calls[0].result_text)

    assert gate["status"] == "clarification_required"
    assert [s["field"] for s in gate["blocking_slots"]] == ["price_max"]
    # a question, not an error: an error would surface to the user as a red failure
    assert result.transcript.tool_calls[0].is_error is False


async def test_proxy_run_passes_when_the_agent_asks_before_searching(backend, repo_fixtures):
    scenario = select(ids=["concert_day_trip_unknown_time"])[0]
    agent = ScriptedAgent(
        plan={
            scenario.id: (
                [("search_rail", {**RAIL_ARGS, "price_max": 4000})],
                "Во сколько заканчивается концерт? От этого зависит, успеете ли вы на "
                "обратный рейс — посчитаю сразу после вашего ответа.",
            )
        }
    )
    variant = await _variant(backend, repo_fixtures, PROXY)

    result = await run_scenario(agent, scenario, variant)

    assert result.success, result.failed_checks


async def test_declared_assumption_must_lead_the_answer(backend, repo_fixtures):
    """Same data, same call — the only difference is where the caveat sits."""
    scenario = select(ids=["relative_date"])[0]
    call = ("search_rail", {**RAIL_ARGS, "_assume": {"departure_date": "взял субботу"}})
    tables = " Поезд в 12:00, автобус в 14:00. " * 12  # pushes any footnote past the window

    buried = ScriptedAgent(
        plan={scenario.id: ([call], tables + "Предположил, что вы едете в субботу.")}
    )
    upfront = ScriptedAgent(
        plan={
            scenario.id: (
                [call],
                "Уточните: «на выходных» — это суббота или воскресенье? Пока считаю "
                "по субботе. " + tables,
            )
        }
    )

    variant = await _variant(backend, repo_fixtures, PROXY)
    buried_result = await run_scenario(buried, scenario, variant)
    upfront_result = await run_scenario(upfront, scenario, variant)

    assert "assumption_disclosed_up_front" in {c.name for c in buried_result.failed_checks}
    assert upfront_result.success, upfront_result.failed_checks


async def test_determinate_request_is_answered_not_interrogated(backend, repo_fixtures):
    """Over-asking guard: the gate must stay silent when nothing is missing."""
    scenario = select(ids=["no_overask"])[0]
    payload = json.loads((await backend.call_tool("search_rail", RAIL_ARGS)).text)
    offer = payload["offers"][0]
    agent = ScriptedAgent(
        plan={
            scenario.id: (
                [("search_rail", RAIL_ARGS)],
                f"Самый дешёвый — за {offer['price']['amount']} ₽.",
            )
        }
    )
    variant = await _variant(backend, repo_fixtures, PROXY)

    result = await run_scenario(agent, scenario, variant)

    assert result.success, result.failed_checks
    assert all("clarification_required" not in c.result_text for c in result.transcript.tool_calls)


async def test_premise_state_does_not_leak_between_scenarios(backend, repo_fixtures):
    """Scenarios run concurrently off one Variant; a shared session would let an
    assumption made in one decide the verdict of another."""
    variant = await _variant(backend, repo_fixtures, PROXY)
    scenario = select(ids=["concert_day_trip_unknown_time"])[0]
    agent = ScriptedAgent(
        plan={scenario.id: ([("search_rail", {**RAIL_ARGS, "price_max": 4000})], "")}
    )

    first = await run_scenario(agent, scenario, variant)
    second = await run_scenario(agent, scenario, variant)

    # the one-shot gate must fire again on a fresh session, not stay spent
    for result in (first, second):
        assert "clarification_required" in result.transcript.tool_calls[0].result_text
    assert variant.premises is not None and not variant.premises.gated


async def test_preflight_clears_the_gate_for_a_value_the_user_typed(backend, repo_fixtures):
    scenario = select(ids=["rail_price_filter"])[0]
    agent = ScriptedAgent(
        plan={
            scenario.id: (
                [
                    ("assess_request", {"user_request": scenario.request}),
                    ("search_rail", {**RAIL_ARGS, "price_max": 3000}),
                ],
                "Нашлось несколько поездов дешевле 3000 ₽.",
            )
        }
    )
    variant = await _variant(backend, repo_fixtures, PROXY)

    result = await run_scenario(agent, scenario, variant)
    search = result.transcript.tool_calls[1]

    assert "clarification_required" not in search.result_text
    assert "offers" in search.result_text


async def test_assess_request_is_absent_from_the_baseline_surface(backend, repo_fixtures):
    baseline = await _variant(backend, repo_fixtures, BASELINE)
    proxy = await _variant(backend, repo_fixtures, PROXY)

    assert "assess_request" not in {t["name"] for t in baseline.tools}
    assert "assess_request" in {t["name"] for t in proxy.tools}


@pytest.mark.parametrize("tool", ["assess_request", "check_groundedness"])
async def test_control_tools_refuse_calls_on_the_baseline_surface(backend, repo_fixtures, tool):
    """Neither control tool is on baseline's `tools/list` — a baseline agent that
    calls one anyway (or a judge probing the surface) must not get a real answer
    out of it, or the comparison would secretly hand baseline proxy-only behavior."""
    baseline = await _variant(backend, repo_fixtures, BASELINE)

    result = await baseline.execute(tool, {})

    assert result.is_error is True
    assert result.result_text == f"unknown tool: {tool}"


async def test_declared_assumption_downgrades_the_claim_instead_of_failing_it(
    backend, repo_fixtures
):
    """The whole point of `_assume`: openly invented is not the same failure as
    silently invented, and the two must not share a color in the report. Runs
    through the proxy variant end to end, because the value has to survive
    `strip_control_fields`, the gate, and the session's `assumed_values()` to
    reach `check_groundedness` — a chain that was broken for every tool without
    a premise policy.
    """
    variant = (await _variant(backend, repo_fixtures, PROXY)).session_scope()

    # A real search first: `check_groundedness` refuses to score an answer when the
    # session holds no evidence at all, and the assumed value must be judged against
    # evidence that genuinely does not contain it.
    await variant.execute("search_rail", RAIL_ARGS)
    await variant.execute(
        "get_offer_details",
        {"details_ref": "нет-такого", "price_max": 7712, ASSUME_KEY: {"price_max": "мой потолок"}},
    )
    record = await variant.execute(
        "check_groundedness", {"answer_text": "Дороже 7712 ₽ ничего не рассматривал."}
    )

    statuses = {c["text"]: c["status"] for c in json.loads(record.result_text)["claims"]}
    assert statuses["7712 \u20bd"] == "assumed"
