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
    ResponsesAgent,
    ScriptedAgent,
    make_client,
)
from evals.config import (
    MISSING_CREDENTIALS_HELP,
    InvalidEffortError,
    openai_credentials_source,
    openai_effort_default,
    openai_model_default,
)
from evals.options import AgentKind, Api, Effort, EvalOptions, SelectionError, check_variants
from evals.runner import ScenarioResult, run_eval
from evals.scenarios import select
from evals.tokens import (
    ChatApiTokenCounter,
    OfflineTokenCounter,
    ResponsesApiTokenCounter,
    TokenCounter,
)
from evals.variants import build_variants
from tutu_mcp.config import load_settings
from tutu_mcp.replay.mock_client import MockUpstreamClient
from tutu_mcp.replay.recording import RecordingBackend
from tutu_mcp.replay.store import FixtureStore
from tutu_mcp.upstream.client import UpstreamClient


def build_agent(opts: EvalOptions, model: str, effort: Effort | None) -> Agent:
    if opts.agent is AgentKind.SCRIPTED:
        # Empty plan: every scenario runs with no tool calls and an empty answer.
        # Useful only to prove the harness wiring end to end, never as a measurement.
        return ScriptedAgent(plan={})
    value = effort.value if effort else None
    if opts.api is Api.CHAT:
        return ChatCompletionsAgent(model=model, effort=value)
    return ResponsesAgent(model=model, effort=value)


def build_token_counter(opts: EvalOptions, model: str) -> TokenCounter:
    if opts.estimate_tokens or opts.agent is AgentKind.SCRIPTED:
        return OfflineTokenCounter(model=model, api=opts.api.value)
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
    store = FixtureStore(settings.fixtures_dir)
    credentials = openai_credentials_source()

    # Both checks run before anything is built or connected: a typo in a flag
    # must not cost an upstream request, and an empty selection must not produce
    # a report that looks like a clean run.
    try:
        check_variants(opts.variants)
        scenarios = select(
            ids=list(opts.scenarios) if opts.scenarios else None,
            domains=list(opts.domains) if opts.domains else None,
        )
    except SelectionError as exc:
        print(exc, file=sys.stderr)
        return 2

    if opts.record_missing and not opts.live:
        print("--record-missing требует --live", file=sys.stderr)
        return 2

    model = opts.model or openai_model_default()
    # --effort wins over OPENAI_EFFORT, which wins over the model's own default —
    # the same precedence --model has over OPENAI_MODEL.
    try:
        effort = opts.effort or openai_effort_default()
    except InvalidEffortError as exc:
        print(exc, file=sys.stderr)
        return 2
    agent = build_agent(opts, model, effort)
    counter = build_token_counter(opts, model)

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

    upstream: UpstreamClient | None = None
    try:
        if opts.live:
            upstream = UpstreamClient(settings.upstream_url, timeout_s=settings.upstream_timeout_s)
            await upstream.connect()
            instructions = upstream.server_info()["instructions"]
            backend = RecordingBackend(store, upstream) if opts.record_missing else upstream
        else:
            backend = MockUpstreamClient(store)
            instructions = store.instructions()

        variants = await build_variants(backend, instructions, names=list(opts.variants))
        run = await run_eval(
            agent=agent,
            scenarios=scenarios,
            variants=variants,
            token_counter=counter,
            concurrency=opts.concurrency,
            on_result=progress,
            api=opts.api.value,
        )
    finally:
        if upstream is not None:
            await upstream.aclose()

    print(report_mod.render_console(run))

    out_path = report_mod.write_json(run, opts.out)
    print(f"\nПодробный отчёт: {out_path}")

    if isinstance(backend, RecordingBackend) and backend.recorded:
        print(f"Дозаписано фикстур: {len(backend.recorded)}")
        for tool, scenario in backend.recorded:
            print(f"  + {tool}/{scenario}")

    return 0
