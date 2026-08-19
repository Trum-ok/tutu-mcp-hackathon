"""One scripted agent covers both offline uses: harness self-check and demo traces.

`DemoAgent` used to be a second class whose `run` was a byte-for-byte copy of
`ScriptedAgent.run`, differing only in what the plan was keyed on and in two
synthetic token counts.
"""

from evals.agent import ScriptedAgent, by_scenario_and_variant
from evals.demo import demo_agent
from evals.transcript import ToolCallRecord


async def _execute(name, arguments):
    return ToolCallRecord(name=name, arguments=arguments, result_text="{}", is_error=False)


async def _run(agent, scenario_id="rail_cheapest", variant="proxy"):
    return await agent.run(
        scenario_id=scenario_id,
        variant=variant,
        request="",
        tools=[],
        server_instructions="",
        execute=_execute,
    )


async def test_the_default_plan_is_keyed_by_scenario_alone():
    agent = ScriptedAgent(plan={"rail_cheapest": ([("search_rail", {})], "ответ")})

    for variant in ("baseline", "proxy"):
        transcript = await _run(agent, variant=variant)
        assert transcript.answer_text == "ответ"
        assert transcript.tool_names() == ["search_rail"]


async def test_a_variant_keyed_plan_can_differ_per_surface():
    """The reason the demo needs its own key: the same scenario has to show a good
    answer on one surface and a bad one on the other."""
    agent = ScriptedAgent(
        plan={("s", "baseline"): ([], "выдуманный"), ("s", "proxy"): ([], "обоснованный")},
        key=by_scenario_and_variant,
    )

    assert (await _run(agent, "s", "baseline")).answer_text == "выдуманный"
    assert (await _run(agent, "s", "proxy")).answer_text == "обоснованный"


async def test_a_scripted_run_reports_no_tokens_unless_asked():
    """A run with no model behind it must not look like a measurement."""
    transcript = await _run(ScriptedAgent(plan={}))

    assert (transcript.input_tokens, transcript.output_tokens) == (0, 0)


async def test_the_demo_agent_keeps_its_synthetic_counts():
    agent = demo_agent({("s", "baseline"): ([], ""), ("s", "proxy"): ([], "")})

    baseline = await _run(agent, "s", "baseline")
    proxy = await _run(agent, "s", "proxy")

    assert baseline.input_tokens > proxy.input_tokens
    assert agent.label == "demo:hand-written"


async def test_the_plans_arguments_are_not_handed_out_for_mutation():
    """A plan is reused across variants; a backend that mutated the dict it was
    given would change what the other variant runs."""
    arguments = {"origin": "Мск"}
    seen = []

    async def mutating_execute(name, args):
        seen.append(args)
        args["origin"] = "изменено"
        return ToolCallRecord(name=name, arguments=args, result_text="{}", is_error=False)

    agent = ScriptedAgent(plan={"s": ([("search_rail", arguments)], "")})
    await agent.run(
        scenario_id="s",
        variant="proxy",
        request="",
        tools=[],
        server_instructions="",
        execute=mutating_execute,
    )

    assert arguments == {"origin": "Мск"}
    assert seen[0] == {"origin": "изменено"}
