"""`make evals-dry` has to go red when the harness stops judging the fixed plans
the way `evals/plans.py` says it must.

The plans never change, so their verdicts are a statement about the harness alone.
These tests hold both ends of that: the verdicts still match, and a deliberately
swapped answer really does break them — a self-check that cannot fail is a green
light wired to nothing.
"""

from evals.agent import ScriptedAgent, by_scenario_and_variant
from evals.options import AgentKind, EvalOptions
from evals.plans import (
    PLANNED_IDS,
    SELF_CHECK,
    SELF_CHECK_LABEL,
    build_plans,
    self_check_mismatches,
    self_check_verdicts,
)
from evals.run import run_evals
from evals.runner import run_eval
from evals.scenarios import select
from evals.tokens import OfflineTokenCounter
from evals.variants import BASELINE, PROXY, build_variants
from tutu_mcp.replay.mock_client import MockUpstreamClient


async def _run(repo_fixtures, plans):
    agent = ScriptedAgent(plan=plans, label_=SELF_CHECK_LABEL, key=by_scenario_and_variant)
    backend = MockUpstreamClient(repo_fixtures)
    variants = await build_variants(backend, repo_fixtures.instructions(), names=[BASELINE, PROXY])
    return await run_eval(
        agent=agent,
        scenarios=select(ids=list(PLANNED_IDS)),
        variants=variants,
        token_counter=OfflineTokenCounter(),
    )


def test_the_plans_and_the_expectations_cover_the_same_pairs(repo_fixtures):
    """Otherwise a plan could quietly lose its expectation and stop being checked."""
    assert set(build_plans(repo_fixtures)) == set(SELF_CHECK)


async def test_every_planned_verdict_still_holds(repo_fixtures):
    run = await _run(repo_fixtures, build_plans(repo_fixtures))

    assert self_check_mismatches(run) == []
    assert len(self_check_verdicts(run)) == len(SELF_CHECK)


async def test_a_fabricated_answer_on_the_grounded_surface_breaks_the_check(repo_fixtures):
    plans = build_plans(repo_fixtures)
    plans[("rail_cheapest", PROXY)] = plans[("rail_cheapest", BASELINE)]

    mismatches = self_check_mismatches(await _run(repo_fixtures, plans))

    assert [(m.scenario_id, m.variant) for m in mismatches] == [("rail_cheapest", PROXY)]
    assert mismatches[0].expected == frozenset()
    assert mismatches[0].actual == SELF_CHECK[("rail_cheapest", BASELINE)]


async def test_the_self_check_refuses_a_scenario_selection():
    """Its scenario set is fixed by the plans; narrowing it would silently shrink
    what the run proves, so the flag is rejected instead of ignored."""
    code = await run_evals(EvalOptions(agent=AgentKind.SCRIPTED, scenarios=("rail_cheapest",)))

    assert code == 2
