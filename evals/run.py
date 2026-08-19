"""Orchestrates one baseline-vs-proxy run: builds the agent, the token counter
and the tool surfaces, then hands them to `run_eval`.

Split from the CLI on purpose. Every knob arrives here as a typed parameter, so
what the harness accepts is stated in one signature that ruff and ty both check
— it used to be twelve `add_argument` strings read back off an
`argparse.Namespace`, where a mistyped attribute surfaced only mid-run, after
the run had already spent tokens and (with `--live`) upstream rate limit.
"""

import sys

from evals import report as report_mod
from evals.agent import (
    Agent,
    ChatCompletionsAgent,
    Plan,
    ResponsesAgent,
    ScriptedAgent,
    by_scenario_and_variant,
    make_client,
)
from evals.config import (
    MISSING_CREDENTIALS_HELP,
    InvalidEffortError,
    openai_credentials_source,
    openai_effort_default,
    openai_model_default,
)
from evals.options import (
    DEFAULT_OUT,
    AgentKind,
    Api,
    Effort,
    EvalOptions,
    SelectionError,
    check_variants,
)
from evals.plans import (
    FIXTURE_UNREADABLE,
    PLANNED_IDS,
    SELF_CHECK_LABEL,
    SELF_CHECK_OUT,
    build_plans,
    self_check_mismatches,
    self_check_verdicts,
)
from evals.runner import ScenarioResult, run_eval
from evals.scenarios import select
from evals.tokens import (
    ChatApiTokenCounter,
    OfflineTokenCounter,
    ResponsesApiTokenCounter,
    TokenCounter,
)
from evals.variants import build_variants
from tutu_mcp.backend import BackendError
from tutu_mcp.backends import backend_for
from tutu_mcp.config import load_settings
from tutu_mcp.replay.recording import RecordingBackend
from tutu_mcp.replay.store import FixtureStore


def build_agent(
    opts: EvalOptions,
    model: str,
    effort: Effort | None,
    *,
    plans: dict[tuple[str, str], Plan] | None = None,
) -> Agent:
    if opts.agent is AgentKind.SCRIPTED:
        # Never an empty plan. With nothing to replay every scenario ends with no
        # tool calls and an empty answer — every check fails, which is exactly what
        # a genuinely broken harness looks like, so the run proves nothing. The
        # hand-written plans give it something to be right or wrong about.
        assert plans, "scripted agent needs the self-check plans from evals/plans.py"
        return ScriptedAgent(plan=plans, label_=SELF_CHECK_LABEL, key=by_scenario_and_variant)
    value = effort.value if effort else None
    if opts.api is Api.CHAT:
        return ChatCompletionsAgent(model=model, effort=value)
    return ResponsesAgent(model=model, effort=value)


def build_token_counter(opts: EvalOptions, model: str, *, has_key: bool = True) -> TokenCounter:
    # The reason travels with the counter so the report can state it. Ordered by
    # what the reader can act on: a flag they passed, an agent they chose, a key
    # they are missing.
    reason = None
    if opts.estimate_tokens:
        reason = "запрошено флагом --estimate-tokens"
    elif opts.agent is AgentKind.SCRIPTED:
        reason = "агент scripted не обращается к API"
    elif not has_key:
        reason = "нет ключа OpenAI"
    if reason is not None:
        return OfflineTokenCounter(model=model, api=opts.api, reason=reason)
    # Must be the same endpoint the agent runs on — see ResponsesApiTokenCounter.
    if opts.api is Api.CHAT:
        return ChatApiTokenCounter(model=model)
    return ResponsesApiTokenCounter(model=model)


async def list_models() -> int:
    client = make_client()
    # This is the command someone runs precisely when they are unsure their setup
    # works, so a raw traceback is the worst possible answer here.
    try:
        page = await client.models.list()
    except Exception as exc:
        print(f"Не удалось получить список моделей: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    names = sorted(m.id for m in page.data)
    print(f"Доступно моделей: {len(names)}")
    for name in names:
        print(f"  {name}")
    return 0


def progress(result: ScenarioResult) -> None:
    mark = "ok  " if result.success else "FAIL"
    print(f"  [{mark}] {result.variant:<9} {result.scenario.id}", flush=True)


async def run_evals(opts: EvalOptions) -> int:
    settings = load_settings()
    credentials = openai_credentials_source()

    # The self-check runs exactly the scenarios `evals/plans.py` has plans for.
    # Narrowing that set would quietly shrink what the run proves, so a selection
    # flag is rejected rather than ignored.
    self_check = opts.agent is AgentKind.SCRIPTED
    if self_check and (opts.scenarios or opts.domains):
        print(
            "--agent scripted прогоняет фиксированный набор самопроверки "
            f"({', '.join(PLANNED_IDS)}) — только для этих сценариев написаны планы, "
            "так что --scenarios/--domains с ним не сочетаются.",
            file=sys.stderr,
        )
        return 2
    ids = PLANNED_IDS if self_check else opts.scenarios

    # Both checks run before anything is built or connected: a typo in a flag
    # must not cost an upstream request, and an empty selection must not produce
    # a report that looks like a clean run.
    try:
        check_variants(opts.variants)
        scenarios = select(
            ids=list(ids) if ids else None,
            domains=list(opts.domains) if opts.domains else None,
        )
    except SelectionError as exc:
        print(exc, file=sys.stderr)
        return 2

    if opts.record_missing and not opts.live:
        print("--record-missing требует --live", file=sys.stderr)
        return 2

    # `run_eval` builds a semaphore out of this. `Semaphore(0)` does not fail — it
    # simply never lets a scenario through, so the run hangs with a printed plan
    # and no output, which reads as a broken harness rather than a bad flag.
    if opts.concurrency < 1:
        print(f"--concurrency должен быть ≥ 1, получено {opts.concurrency}", file=sys.stderr)
        return 2

    model = opts.model or openai_model_default()
    # --effort wins over OPENAI_EFFORT, which wins over the model's own default —
    # the same precedence --model has over OPENAI_MODEL.
    try:
        effort = opts.effort or openai_effort_default()
    except InvalidEffortError as exc:
        print(exc, file=sys.stderr)
        return 2
    try:
        plans = build_plans(FixtureStore(settings.fixtures_dir)) if self_check else None
    except BackendError as exc:
        print(f"{FIXTURE_UNREADABLE}\n{exc}", file=sys.stderr)
        return 2
    agent = build_agent(opts, model, effort, plans=plans)
    counter = build_token_counter(opts, model, has_key=credentials is not None)

    # Fail before the first request rather than mid-run: a missing key would
    # otherwise surface as an auth error several scenarios deep, after the run
    # has already spent time and (in --live) upstream rate limit. Derived from the
    # objects actually built, so it cannot drift from what they really need.
    needs_credentials = opts.agent is AgentKind.OPENAI or counter.exact
    if needs_credentials and credentials is None:
        print(MISSING_CREDENTIALS_HELP, file=sys.stderr)
        return 2

    if credentials:
        print(f"Ключ OpenAI: {credentials}")
    print(
        f"Сценариев: {len(scenarios)}, варианты: {list(opts.variants)}, "
        f"агент: {agent.label}, endpoint: /v1/{opts.api.value}"
    )

    async with backend_for(settings, live=opts.live, record_missing=opts.record_missing) as wiring:
        backend = wiring.backend
        variants = await build_variants(backend, wiring.instructions(), names=list(opts.variants))
        run = await run_eval(
            agent=agent,
            scenarios=scenarios,
            variants=variants,
            token_counter=counter,
            concurrency=opts.concurrency,
            on_result=progress,
            api=opts.api,
        )

    print(report_mod.render_console(run))

    # A self-check is not a measurement, and `out/eval-results.json` is the file
    # `tutu.py viewer` opens as the last real run — same reason `tutu.py demo`
    # keeps to its own path. An explicit --out still wins.
    out = SELF_CHECK_OUT if self_check and opts.out == DEFAULT_OUT else opts.out
    out_path = report_mod.write_json(run, out)
    print(f"\nПодробный отчёт: {out_path}")

    if isinstance(backend, RecordingBackend):
        if backend.recorded:
            print(f"Дозаписано фикстур: {len(backend.recorded)}")
            for tool, scenario in backend.recorded:
                print(f"  + {tool}/{scenario}")
        if backend.skipped_errors:
            # Named, not just counted: these misses are still open, and a run
            # printing the recorded count alone would read as "holes closed".
            print(
                f"Не записано (upstream ответил ошибкой): {len(backend.skipped_errors)} — "
                "промах остался, повторите запись",
                file=sys.stderr,
            )
            for tool, scenario in backend.skipped_errors:
                print(f"  ! {tool}/{scenario}", file=sys.stderr)

    if self_check:
        mismatches = self_check_mismatches(run)
        print(report_mod.render_self_check(run, mismatches))
        # Non-zero on the slightest disagreement, and on a run that checked
        # nothing at all: the whole point here is that a broken harness must not
        # come out looking like a working one.
        return 1 if mismatches or not self_check_verdicts(run) else 0

    return 0
