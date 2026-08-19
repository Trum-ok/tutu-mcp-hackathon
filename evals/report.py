"""Console summary + a JSON artifact for the trace viewer to render later."""

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .runner import EvalRun, ScenarioResult, VariantSummary


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:.0f}%"


def _delta(baseline: float | None, other: float | None, *, higher_is_better: bool) -> str:
    if baseline is None or other is None:
        return ""
    diff = other - baseline
    if abs(diff) < 1e-9:
        return "="
    good = diff > 0 if higher_is_better else diff < 0
    return f"{'+' if diff > 0 else ''}{diff * 100:.0f} п.п. {'✓' if good else '✗'}"


def _delta_int(baseline: int, other: int, *, higher_is_better: bool) -> str:
    diff = other - baseline
    if diff == 0:
        return "="
    good = diff > 0 if higher_is_better else diff < 0
    return f"{'+' if diff > 0 else ''}{diff} {'✓' if good else '✗'}"


def render_console(run: EvalRun) -> str:
    lines: list[str] = []
    add = lines.append

    add(f"\nАгент: {run.agent_label}")

    add("\n=== Стоимость поверхности инструментов (до первого поиска) ===")
    if not run.surface_exact:
        add("  ⚠ токены оценены офлайн (tiktoken) — точный замер требует ключа OpenAI")
    add(f"  {'вариант':<12} {'токены':>10} {'байты':>10}")
    base_surface = None
    for summary in run.summaries:
        if summary.surface is None:
            continue
        s = summary.surface
        add(f"  {s.variant:<12} {s.label:>10} {s.bytes_:>10}")
        if base_surface is None:
            base_surface = s
        elif base_surface.tokens:
            saved = 1 - s.tokens / base_surface.tokens
            add(f"  {'':<12} {'экономия':>10} {saved * 100:>9.0f}%")

    add("\n=== Метрики по вариантам ===")
    header = (
        f"  {'вариант':<12} {'успех':>10} {'grounded':>10} {'вызовы':>8} "
        f"{'ошибки':>8} {'промахи':>9} {'in tok':>9} {'out tok':>9} {'p50 с':>7} {'p95 с':>7}"
    )
    add(header)
    for summary in run.summaries:
        add(
            f"  {summary.variant:<12} "
            f"{f'{summary.successes}/{summary.total}':>10} "
            f"{_pct(summary.groundedness_rate):>10} "
            f"{summary.tool_calls:>8} "
            f"{summary.tool_errors:>8} "
            f"{summary.fixture_misses:>9} "
            f"{summary.input_tokens:>9} "
            f"{summary.output_tokens:>9} "
            f"{summary.latency_p50():>7.2f} "
            f"{summary.latency_p95():>7.2f}"
        )

    if len(run.summaries) >= 2:
        base, other = run.summaries[0], run.summaries[1]
        add(f"\n  дельта {other.variant} к {base.variant}:")
        add(
            f"    task success   {_delta(base.success_rate, other.success_rate, higher_is_better=True)}"
        )
        add(
            f"    groundedness   "
            f"{_delta(base.groundedness_rate, other.groundedness_rate, higher_is_better=True)}"
        )
        add(
            f"    вызовов тулов  {_delta_int(base.tool_calls, other.tool_calls, higher_is_better=False)}"
        )
        add(
            f"    входных токенов {_delta_int(base.input_tokens, other.input_tokens, higher_is_better=False)}"
        )

    total_misses = sum(s.fixture_misses for s in run.summaries)
    if total_misses:
        add(
            f"\n  ⚠ промахов по фикстурам: {total_misses}. Это пробел в записи, а не поведение "
            "сервера — перезапишите фикстуры (`--record-missing` в live-режиме)."
        )

    add("\n=== Предпосылки запроса (вход) ===")
    add(
        "  groundedness проверяет ВЫХОД (числа в ответе есть в payload); "
        "эти цифры — про ВХОД:\n  опирался ли расчёт на значение, которого никто не называл."
    )
    add(
        f"  {'вариант':<12} {'гейт сработал':>14} {'на допущении':>14} "
        f"{'раскрыто в начале':>18} {'лишних вопросов':>16}"
    )
    for summary in run.summaries:
        add(
            f"  {summary.variant:<12} "
            f"{summary.gate_fires:>14} "
            f"{summary.runs_with_assumptions:>14} "
            f"{_pct(summary.disclosure_rate):>18} "
            f"{summary.over_asks:>16}"
        )

    add("\n=== Расхождения по сценариям ===")
    add(_render_scenario_matrix(run))

    failures = _render_failures(run)
    if failures:
        add("\n=== Провалы ===")
        add(failures)

    return "\n".join(lines)


def _render_scenario_matrix(run: EvalRun) -> str:
    variants = [s.variant for s in run.summaries]
    by_scenario: dict[str, dict[str, ScenarioResult]] = {}
    for summary in run.summaries:
        for result in summary.results:
            by_scenario.setdefault(result.scenario.id, {})[summary.variant] = result

    lines = [f"  {'сценарий':<28} " + " ".join(f"{v:>10}" for v in variants)]
    for scenario_id, per_variant in by_scenario.items():
        cells = []
        for variant in variants:
            result = per_variant.get(variant)
            if result is None:
                cells.append(f"{'—':>10}")
            elif result.transcript.failure is not None:
                cells.append(f"{'ошибка':>10}")
            else:
                mark = "ok" if result.success else f"{len(result.failed_checks)} fail"
                cells.append(f"{mark:>10}")
        lines.append(f"  {scenario_id:<28} " + " ".join(cells))
    return "\n".join(lines)


def _render_failures(run: EvalRun) -> str:
    lines: list[str] = []
    for summary in run.summaries:
        for result in summary.results:
            if result.success:
                continue
            lines.append(f"  [{summary.variant}] {result.scenario.id} — {result.scenario.probes}")
            if result.transcript.failure:
                lines.append(f"      прогон не завершился: {result.transcript.failure}")
            for check in result.failed_checks:
                lines.append(f"      ✗ {check.name}: {check.detail}")
    return "\n".join(lines)


def to_json(run: EvalRun) -> dict[str, Any]:
    """Full dump — the trace viewer reads this, so it keeps the per-call detail."""
    return {
        "agent": run.agent_label,
        "surface_tokens_exact": run.surface_exact,
        "variants": [_summary_json(s) for s in run.summaries],
    }


def _summary_json(summary: VariantSummary) -> dict[str, Any]:
    return {
        "variant": summary.variant,
        "surface": asdict(summary.surface) if summary.surface else None,
        "metrics": {
            "scenarios": summary.total,
            "successes": summary.successes,
            "success_rate": summary.success_rate,
            "groundedness_rate": summary.groundedness_rate,
            "total_claims": summary.total_claims,
            "grounded_claims": summary.grounded_claims,
            "tool_calls": summary.tool_calls,
            "tool_errors": summary.tool_errors,
            "fixture_misses": summary.fixture_misses,
            "input_tokens": summary.input_tokens,
            "output_tokens": summary.output_tokens,
            "latency_p50_s": summary.latency_p50(),
            "latency_p95_s": summary.latency_p95(),
        },
        "premises": {
            "gate_fires": summary.gate_fires,
            "runs_with_assumptions": summary.runs_with_assumptions,
            "disclosed_assumptions": summary.disclosed_assumptions,
            "disclosure_rate": summary.disclosure_rate,
            "over_asks": summary.over_asks,
        },
        "scenarios": [_result_json(r) for r in summary.results],
    }


def _result_json(result: ScenarioResult) -> dict[str, Any]:
    return {
        "id": result.scenario.id,
        "domain": result.scenario.domain,
        "request": result.scenario.request,
        "probes": result.scenario.probes,
        "success": result.success,
        "failure": result.transcript.failure,
        "answer": result.transcript.answer_text,
        "turns": result.transcript.turns,
        "input_tokens": result.transcript.input_tokens,
        "output_tokens": result.transcript.output_tokens,
        "duration_s": result.transcript.duration_s,
        "checks": [{"name": c.name, "passed": c.passed, "detail": c.detail} for c in result.checks],
        "groundedness": {
            "rate": result.grounding.rate,
            # `status` is the three-state verdict the trace viewer paints:
            # confirmed / assumed / unavailable. `grounded` stays alongside it as
            # the binary roll-up — an assumption is disclosed, not proven, so it
            # must not read as green.
            "claims": [
                {
                    "kind": c.claim.kind,
                    "text": c.claim.text,
                    "status": c.status,
                    "grounded": c.grounded,
                }
                for c in result.grounding.checks
            ],
            "assumptions": result.grounding.assumptions,
            "assumption_disclosed": result.grounding.assumption_disclosed,
        },
        "tool_calls": [
            {
                "name": c.name,
                "arguments": c.arguments,
                "is_error": c.is_error,
                "fixture_miss": c.fixture_miss,
                "duration_s": c.duration_s,
                "result_text": c.result_text,
            }
            for c in result.transcript.tool_calls
        ],
    }


def write_json(run: EvalRun, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_json(run), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path
