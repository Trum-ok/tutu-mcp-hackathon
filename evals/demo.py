"""Generates `out/eval-results.demo.json` without calling any model.

The trace viewer needs traces to render, and a real run needs an OpenAI key plus
several minutes. This builds a small set of hand-authored answers over the REAL
recorded fixtures, so every "confirmed" claim is genuinely confirmed against real
Tutu data and every fabricated one is genuinely absent from it — the viewer's
highlighting is exercised for real, not faked with hardcoded colors. The answers
themselves live in `evals/plans.py`, shared with the `--agent scripted` self-check.

IMPORTANT: this is not a measurement. The answers are written by hand to cover
the interesting states (grounded / fabricated / unsupported-absence-claim), not
produced by a model. The agent label in the output says `demo:hand-written`, and
the viewer prints it in the header, so a demo dataset can never be mistaken on
screen for a real eval run.

    uv run python tutu.py demo
"""

import sys
from pathlib import Path
from typing import Any

from evals import report as report_mod
from evals.agent import Plan, ScriptedAgent, by_scenario_and_variant
from evals.plans import FIXTURE_UNREADABLE, PLANNED_IDS, build_plans
from evals.runner import run_eval
from evals.scenarios import select
from evals.tokens import OfflineTokenCounter
from evals.variants import BASELINE, build_variants
from tutu_mcp.backend import BackendError
from tutu_mcp.backends import backend_for
from tutu_mcp.config import load_settings
from tutu_mcp.replay.store import FixtureStore


def demo_agent(plans: dict[Any, Plan]) -> ScriptedAgent:
    """`ScriptedAgent` keyed by (scenario, variant), with synthetic token counts.

    Plausible-looking but clearly synthetic numbers: a real run overwrites them,
    and the report labels the agent so nobody reads these traces as a measurement.
    """
    return ScriptedAgent(
        plan=plans,
        label_="demo:hand-written",
        key=by_scenario_and_variant,
        tokens=lambda variant: (12000 if variant == BASELINE else 8000, 320),
    )


async def run_demo() -> int:
    settings = load_settings()
    store = FixtureStore(settings.fixtures_dir)

    scenarios = select(ids=list(PLANNED_IDS))
    try:
        agent = demo_agent(build_plans(store))
    except BackendError as exc:
        print(f"{FIXTURE_UNREADABLE}\n{exc}", file=sys.stderr)
        return 2

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
