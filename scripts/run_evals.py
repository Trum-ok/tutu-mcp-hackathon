"""Baseline vs proxy eval runner (OpenAI API).

    uv run python scripts/run_evals.py                       # mock fixtures, real model
    uv run python scripts/run_evals.py --agent scripted      # no model, harness self-check
    uv run python scripts/run_evals.py --live --record-missing    # fill fixture gaps once
    uv run python scripts/run_evals.py --model gpt-5-mini    # small-model breakdown
    uv run python scripts/run_evals.py --list-models         # what this key can reach
    uv run python scripts/run_evals.py --api chat            # OpenAI-compatible gateway

Defaults to fixtures so a run costs nothing upstream and reproduces exactly.
`--live` talks to the real mcp.tutu.ru and is subject to the shared rate limit.
"""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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
    openai_credentials_source,
    openai_model_default,
)
from evals.runner import ScenarioResult, run_eval
from evals.scenarios import select
from evals.tokens import (
    ChatApiTokenCounter,
    OfflineTokenCounter,
    ResponsesApiTokenCounter,
    TokenCounter,
)
from evals.variants import BASELINE, PROXY, build_variants
from tutu_mcp.config import load_settings
from tutu_mcp.replay.mock_client import MockUpstreamClient
from tutu_mcp.replay.recording import RecordingBackend
from tutu_mcp.replay.store import FixtureStore
from tutu_mcp.upstream.client import UpstreamClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", choices=["openai", "scripted"], default="openai")
    parser.add_argument("--model", default=None, help="default: $OPENAI_MODEL, else gpt-5")
    parser.add_argument(
        "--effort",
        default=None,
        choices=["none", "minimal", "low", "medium", "high"],
        help="reasoning effort; omit to let the model use its own default",
    )
    parser.add_argument(
        "--api",
        choices=["responses", "chat"],
        default="responses",
        help=(
            "OpenAI endpoint. 'responses' (default) is the only one that takes "
            "function tools together with reasoning; 'chat' is for OpenAI-compatible "
            "gateways that do not implement /v1/responses"
        ),
    )
    parser.add_argument("--live", action="store_true", help="call the real mcp.tutu.ru")
    parser.add_argument(
        "--record-missing",
        action="store_true",
        help="with --live: record any fixture the agent asks for and we don't have",
    )
    parser.add_argument("--variants", nargs="+", default=[BASELINE, PROXY])
    parser.add_argument("--scenarios", nargs="+", default=None, help="scenario ids to run")
    parser.add_argument("--domains", nargs="+", default=None)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--out", default="eval-results.json")
    parser.add_argument(
        "--estimate-tokens",
        action="store_true",
        help="use the offline tiktoken estimate instead of one real probe request",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="print the models this key can reach, then exit",
    )
    return parser.parse_args()


def build_agent(args: argparse.Namespace, model: str) -> Agent:
    if args.agent == "scripted":
        # Empty plan: every scenario runs with no tool calls and an empty answer.
        # Useful only to prove the harness wiring end to end, never as a measurement.
        return ScriptedAgent(plan={})
    if args.api == "chat":
        return ChatCompletionsAgent(model=model, effort=args.effort)
    return ResponsesAgent(model=model, effort=args.effort)


def build_token_counter(args: argparse.Namespace, model: str) -> TokenCounter:
    if args.estimate_tokens or args.agent == "scripted":
        return OfflineTokenCounter(model=model, api=args.api)
    # Must be the same endpoint the agent runs on — see ResponsesApiTokenCounter.
    if args.api == "chat":
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


async def main() -> int:
    args = parse_args()
    settings = load_settings()
    store = FixtureStore(settings.fixtures_dir)
    credentials = openai_credentials_source()

    if args.list_models:
        if credentials is None:
            print(MISSING_CREDENTIALS_HELP, file=sys.stderr)
            return 2
        return await list_models()

    scenarios = select(ids=args.scenarios, domains=args.domains)

    if args.record_missing and not args.live:
        print("--record-missing требует --live", file=sys.stderr)
        return 2

    model = args.model or openai_model_default()
    agent = build_agent(args, model)
    counter = build_token_counter(args, model)

    # Fail before the first request rather than mid-run: a missing key would
    # otherwise surface as an auth error several scenarios deep, after the run
    # has already spent time and (in --live) upstream rate limit. Derived from the
    # objects actually built, so it cannot drift from what they really need.
    needs_credentials = args.agent == "openai" or counter.exact
    if needs_credentials and credentials is None:
        print(MISSING_CREDENTIALS_HELP, file=sys.stderr)
        return 2

    if credentials:
        print(f"Ключ OpenAI: {credentials}")
    print(
        f"Сценариев: {len(scenarios)}, варианты: {args.variants}, "
        f"агент: {agent.label}, endpoint: /v1/{args.api}"
    )

    upstream: UpstreamClient | None = None
    try:
        if args.live:
            upstream = UpstreamClient(settings.upstream_url)
            await upstream.connect()
            instructions = upstream.server_info()["instructions"]
            backend = RecordingBackend(store, upstream) if args.record_missing else upstream
        else:
            backend = MockUpstreamClient(store)
            instructions = store.instructions()

        variants = await build_variants(backend, instructions, names=args.variants)
        run = await run_eval(
            agent=agent,
            scenarios=scenarios,
            variants=variants,
            token_counter=counter,
            concurrency=args.concurrency,
            on_result=progress,
            api=args.api,
        )
    finally:
        if upstream is not None:
            await upstream.aclose()

    print(report_mod.render_console(run))

    out_path = report_mod.write_json(run, Path(args.out))
    print(f"\nПодробный отчёт: {out_path}")

    if isinstance(backend, RecordingBackend) and backend.recorded:
        print(f"Дозаписано фикстур: {len(backend.recorded)}")
        for tool, scenario in backend.recorded:
            print(f"  + {tool}/{scenario}")

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
