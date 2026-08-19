"""Runs every (scenario × variant) pair and aggregates the numbers."""

import asyncio
import math
from collections.abc import Callable
from dataclasses import dataclass, field

from tutu_mcp.groundedness import GroundednessReport, check_groundedness
from tutu_mcp.premises import SessionPremises

from .agent import Agent
from .checks import CheckResult
from .options import Api
from .scenarios import Scenario
from .tokens import SurfaceCost, TokenCounter, measure_surface
from .transcript import Transcript
from .variants import Variant


@dataclass
class ScenarioResult:
    scenario: Scenario
    variant: str
    transcript: Transcript
    grounding: GroundednessReport
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return self.transcript.failure is None and all(c.passed for c in self.checks)

    @property
    def failed_checks(self) -> list[CheckResult]:
        return [c for c in self.checks if not c.passed]


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    # nearest-rank; on a handful of scenarios anything fancier is false precision
    index = math.ceil(pct / 100 * len(ordered)) - 1
    return ordered[min(max(index, 0), len(ordered) - 1)]


@dataclass
class VariantSummary:
    variant: str
    results: list[ScenarioResult]
    surface: SurfaceCost | None = None

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def successes(self) -> int:
        return sum(1 for r in self.results if r.success)

    @property
    def success_rate(self) -> float:
        return self.successes / self.total if self.total else 0.0

    @property
    def total_claims(self) -> int:
        return sum(len(r.grounding.checks) for r in self.results)

    @property
    def grounded_claims(self) -> int:
        return sum(sum(1 for c in r.grounding.checks if c.grounded) for r in self.results)

    @property
    def groundedness_rate(self) -> float | None:
        """Pooled over all claims, not averaged over scenarios — a scenario making
        twenty claims should weigh more than one making a single claim."""
        return self.grounded_claims / self.total_claims if self.total_claims else None

    @property
    def tool_calls(self) -> int:
        return sum(r.transcript.tool_call_count for r in self.results)

    @property
    def tool_errors(self) -> int:
        return sum(r.transcript.tool_error_count for r in self.results)

    @property
    def fixture_misses(self) -> int:
        return sum(r.transcript.fixture_miss_count for r in self.results)

    @property
    def input_tokens(self) -> int:
        return sum(r.transcript.input_tokens for r in self.results)

    @property
    def output_tokens(self) -> int:
        return sum(r.transcript.output_tokens for r in self.results)

    @property
    def gate_fires(self) -> int:
        """Scenarios where the proxy answered a question instead of data at least once."""
        return sum(1 for r in self.results if r.transcript.gate_fired())

    @property
    def runs_with_assumptions(self) -> int:
        return sum(1 for r in self.results if r.grounding.assumptions)

    @property
    def disclosed_assumptions(self) -> int:
        return sum(1 for r in self.results if r.grounding.assumption_disclosed)

    @property
    def disclosure_rate(self) -> float | None:
        """Of the runs that proceeded on an assumption, how many said so UP FRONT.

        None when nothing was assumed — which is a better outcome than a perfect
        disclosure rate, not a missing measurement.
        """
        total = self.runs_with_assumptions
        return self.disclosed_assumptions / total if total else None

    @property
    def over_asks(self) -> int:
        """Clarifying questions asked on scenarios that had nothing to clarify.

        Tracked next to the under-asking numbers on purpose: a gate that asks about
        everything would score perfectly on every other premise metric here while
        making the product worse.
        """
        return sum(
            1
            for r in self.results
            for c in r.checks
            if c.name == "did_not_over_ask" and not c.passed
        )

    @property
    def run_failures(self) -> list[ScenarioResult]:
        return [r for r in self.results if r.transcript.failure is not None]

    def latency_p50(self) -> float:
        return _percentile([r.transcript.duration_s for r in self.results], 50)

    def latency_p95(self) -> float:
        return _percentile([r.transcript.duration_s for r in self.results], 95)


@dataclass
class EvalRun:
    agent_label: str
    summaries: list[VariantSummary]
    surface_exact: bool
    # why the surface figures are estimates, when they are; `None` when exact
    surface_estimate_reason: str | None = None

    def by_variant(self, name: str) -> VariantSummary | None:
        return next((s for s in self.summaries if s.variant == name), None)


def evaluate(
    scenario: Scenario, transcript: Transcript, premises: SessionPremises | None = None
) -> ScenarioResult:
    grounding = check_groundedness(
        transcript.answer_text,
        transcript.result_payloads(),
        assumed_values=premises.assumed_values() if premises else None,
        assumptions=premises.assumption_lines() if premises else None,
        # Taken from the scenario, not from the session: baseline has no session,
        # and sourcing this differently per variant would hand one of them an
        # advantage that has nothing to do with what is being measured.
        user_request=scenario.request,
    )
    checks = [c.run(transcript, grounding) for c in scenario.checks]
    return ScenarioResult(
        scenario=scenario,
        variant=transcript.variant,
        transcript=transcript,
        grounding=grounding,
        checks=checks,
    )


async def run_scenario(agent: Agent, scenario: Scenario, variant: Variant) -> ScenarioResult:
    # one premise session per scenario: scenarios run concurrently, and a shared
    # one would let an assumption made in one leak into another's verdict
    scoped = variant.session_scope()
    transcript = await agent.run(
        scenario_id=scenario.id,
        variant=scoped.name,
        request=scenario.request,
        tools=scoped.tools,
        server_instructions=scoped.server_instructions,
        execute=scoped.execute,
    )
    return evaluate(scenario, transcript, scoped.premises)


async def run_eval(
    *,
    agent: Agent,
    scenarios: list[Scenario],
    variants: list[Variant],
    token_counter: TokenCounter,
    concurrency: int = 1,
    on_result: Callable[[ScenarioResult], None] | None = None,
    api: Api = Api.RESPONSES,
) -> EvalRun:
    """`concurrency` stays at 1 by default: against the live upstream the rate limit
    is shared with every other hackathon team, and against fixtures the run is fast
    enough that parallelism buys nothing but noisier latency numbers.

    `api` only tells the surface measurement which endpoint's tool serialization to
    size against; the agent already knows which one it calls."""
    semaphore = asyncio.Semaphore(concurrency)
    summaries = []

    for variant in variants:
        surface = await measure_surface(
            token_counter, variant.name, variant.tools, variant.server_instructions, api
        )

        async def guarded(scenario: Scenario, v: Variant = variant) -> ScenarioResult:
            async with semaphore:
                result = await run_scenario(agent, scenario, v)
                if on_result is not None:
                    on_result(result)
                return result

        results = await asyncio.gather(*(guarded(s) for s in scenarios))
        summaries.append(
            VariantSummary(variant=variant.name, results=list(results), surface=surface)
        )

    return EvalRun(
        agent_label=agent.label,
        summaries=summaries,
        surface_exact=token_counter.exact,
        surface_estimate_reason=token_counter.estimate_reason,
    )
