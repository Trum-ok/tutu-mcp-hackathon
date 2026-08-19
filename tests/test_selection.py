"""Bad `--scenarios` / `--domains` / `--variants` / `--concurrency` values fail loudly.

Two of them used to raise `KeyError` (a traceback in the user's face), the
third — an unknown domain — filtered every scenario out and exited 0, which is
indistinguishable from a run where everything passed, and `--concurrency 0`
hung the run instead of failing at all.
"""

import pytest

from evals.options import EvalOptions, SelectionError, check_variants
from evals.run import run_evals
from evals.runner import run_eval
from evals.scenarios import DOMAINS, SCENARIOS_BY_ID, select


def test_unknown_scenario_id_lists_what_is_available():
    with pytest.raises(SelectionError) as excinfo:
        select(ids=["nope_bad_id"])

    assert "nope_bad_id" in str(excinfo.value)
    assert "rail_cheapest" in str(excinfo.value)


def test_unknown_domain_is_rejected_rather_than_silently_matching_nothing():
    with pytest.raises(SelectionError) as excinfo:
        select(domains=["nonexistent"])

    assert ", ".join(DOMAINS) in str(excinfo.value)


def test_valid_but_disjoint_filters_are_an_error_too():
    with pytest.raises(SelectionError):
        select(ids=["rail_cheapest"], domains=["hotels"])


def test_valid_selection_still_narrows():
    chosen = select(ids=["rail_cheapest"])

    assert [s.id for s in chosen] == ["rail_cheapest"]
    assert len(select()) == len(SCENARIOS_BY_ID)


def test_unknown_variant_is_rejected_before_any_backend_work():
    with pytest.raises(SelectionError):
        check_variants(["baseline", "foo"])

    assert check_variants(["baseline", "proxy"]) is None


async def test_zero_concurrency_is_rejected_instead_of_hanging_the_run(capsys):
    """`asyncio.Semaphore(0)` admits nobody, so the run would sit forever after
    printing its plan — the failure mode hardest to tell from a broken harness."""
    code = await run_evals(EvalOptions(concurrency=0))

    assert code == 2
    assert "--concurrency" in capsys.readouterr().err


async def test_the_runner_refuses_it_too_for_a_caller_that_bypasses_the_cli():
    with pytest.raises(ValueError, match="concurrency"):
        await run_eval(agent=None, scenarios=[], variants=[], token_counter=None, concurrency=0)
