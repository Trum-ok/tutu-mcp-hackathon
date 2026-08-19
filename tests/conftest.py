import json
from pathlib import Path

import pytest

from evals.agent import ScriptedAgent, by_scenario_and_variant
from evals.plans import PLANNED_IDS, SELF_CHECK_LABEL, build_plans
from evals.runner import EvalRun, run_eval
from evals.scenarios import select
from evals.tokens import OfflineTokenCounter
from evals.variants import BASELINE, PROXY, build_variants
from tutu_mcp.replay.mock_client import MockUpstreamClient
from tutu_mcp.replay.store import FixtureStore

REPO_FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"


@pytest.fixture
def repo_fixtures() -> FixtureStore:
    """The real fixtures recorded from mcp.tutu.ru (`uv run python tutu.py record`)."""
    return FixtureStore(REPO_FIXTURES_DIR)


def load_result_payload(tool: str, scenario: str) -> dict:
    raw = json.loads((REPO_FIXTURES_DIR / tool / f"{scenario}.json").read_text(encoding="utf-8"))
    return json.loads(raw["result"]["text"])


@pytest.fixture
async def planned_run() -> EvalRun:
    """Прогон рукописных планов (`evals/plans.py`) по реальным фикстурам — ровно то,
    что `tutu.py demo` кладёт в `out/eval-results.demo.json` и что потом печёт вьювер.

    Артефактным тестам нужен настоящий `EvalRun`, а не собранный вручную: половина
    полей отчёта — вычисляемые свойства сводки, и заглушка проверяла бы разметку
    заглушки, а не то, что уедет на страницу.
    """
    store = FixtureStore(REPO_FIXTURES_DIR)
    agent = ScriptedAgent(
        plan=build_plans(store), label_=SELF_CHECK_LABEL, key=by_scenario_and_variant
    )
    variants = await build_variants(
        MockUpstreamClient(store), store.instructions(), names=[BASELINE, PROXY]
    )
    return await run_eval(
        agent=agent,
        scenarios=select(ids=list(PLANNED_IDS)),
        variants=variants,
        token_counter=OfflineTokenCounter(),
    )
