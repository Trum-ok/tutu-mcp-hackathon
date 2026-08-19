"""`evals/report.py` — единственный мост между прогоном и страницей.

Консольный отчёт читает человек и заметит, если он поедет. JSON читает
`viewer/app.js`, и заметить некому: переименованное поле не падает, а рисует
`undefined` в клетке метрики — на питче, на чужом ноутбуке, без консоли.
Поэтому форма дампа зафиксирована здесь целиком, а не выборочно; отрисовку
этого же дампа настоящим вьювером проверяет `tests/test_viewer_contract.py`.
"""

import json
import re
from pathlib import Path

import pytest

from evals.plans import Mismatch, self_check_mismatches
from evals.report import render_console, render_self_check, to_json, write_json

VIEWER_APP = Path(__file__).resolve().parent.parent / "viewer" / "app.js"
VIEWER_STYLES = Path(__file__).resolve().parent.parent / "viewer" / "styles.css"

RUN_KEYS = {"agent", "surface_tokens_exact", "variants"}
VARIANT_KEYS = {"variant", "surface", "metrics", "premises", "scenarios"}
SURFACE_KEYS = {"variant", "tokens", "bytes_", "exact"}
METRIC_KEYS = {
    "scenarios",
    "successes",
    "success_rate",
    "groundedness_rate",
    "total_claims",
    "checkable_claims",
    "grounded_claims",
    "fabricated_claims",
    "tool_calls",
    "tool_errors",
    "fixture_misses",
    "input_tokens",
    "output_tokens",
    "latency_p50_s",
    "latency_p95_s",
}
PREMISE_KEYS = {
    "gate_fires",
    "runs_with_assumptions",
    "disclosed_assumptions",
    "disclosure_rate",
    "over_asks",
}
SCENARIO_KEYS = {
    "id",
    "domain",
    "request",
    "probes",
    "success",
    "failure",
    "answer",
    "turns",
    "input_tokens",
    "output_tokens",
    "duration_s",
    "checks",
    "groundedness",
    "tool_calls",
}
CHECK_KEYS = {"name", "passed", "detail"}
GROUNDEDNESS_KEYS = {"rate", "claims", "assumptions", "assumption_disclosed"}
CLAIM_KEYS = {"kind", "text", "status", "grounded"}
CALL_KEYS = {"name", "arguments", "is_error", "fixture_miss", "duration_s", "result_text"}


@pytest.fixture
def dump(planned_run):
    return to_json(planned_run)


def test_the_dump_keeps_exactly_the_documented_shape(dump):
    """Ключи перечислены поимённо, а не проверены на вхождение: вьювер читает дамп
    по именам, и потерянное поле так же ломает страницу, как переименованное."""
    assert set(dump) == RUN_KEYS
    assert dump["variants"], "прогон без вариантов нечего показывать"

    for variant in dump["variants"]:
        assert set(variant) == VARIANT_KEYS
        assert set(variant["surface"]) == SURFACE_KEYS
        assert set(variant["metrics"]) == METRIC_KEYS
        assert set(variant["premises"]) == PREMISE_KEYS
        assert variant["scenarios"], f"вариант {variant['variant']} без сценариев"

        for scenario in variant["scenarios"]:
            assert set(scenario) == SCENARIO_KEYS
            assert set(scenario["groundedness"]) == GROUNDEDNESS_KEYS
            for check in scenario["checks"]:
                assert set(check) == CHECK_KEYS
            for claim in scenario["groundedness"]["claims"]:
                assert set(claim) == CLAIM_KEYS
            for call in scenario["tool_calls"]:
                assert set(call) == CALL_KEYS


def test_every_claim_status_is_one_the_viewer_can_paint(dump):
    """Статус приезжает из `tutu_mcp/groundedness.py`, а подпись и цвет к нему —
    из вьювера. Новый статус без пары там рисуется бесцветным `undefined`."""
    table = re.search(r"const STATUS_RU = \{(.+?)\};", VIEWER_APP.read_text(encoding="utf-8"), re.S)
    assert table, "в viewer/app.js больше нет таблицы STATUS_RU — проверка ослепла"
    known = set(re.findall(r"(\w+):", table.group(1)))
    styles = VIEWER_STYLES.read_text(encoding="utf-8")

    statuses = {
        claim["status"]
        for variant in dump["variants"]
        for scenario in variant["scenarios"]
        for claim in scenario["groundedness"]["claims"]
    }

    assert statuses, "планы написаны так, что утверждения в них есть всегда"
    assert statuses <= known, f"вьювер не знает подписи для {statuses - known}"
    for status in statuses:
        assert f".{status}" in styles, f"у статуса {status} нет цвета в styles.css"


def test_the_dump_is_plain_json(dump):
    """`allow_nan=False` ловит то, чего JSON.parse на странице не переживёт: NaN и
    Infinity питон сериализует молча, а браузер на них падает."""
    text = json.dumps(dump, ensure_ascii=False, allow_nan=False)

    assert json.loads(text) == dump


def test_written_json_reads_back_as_it_was_dumped(planned_run, tmp_path):
    path = write_json(planned_run, tmp_path / "nested" / "eval-results.json")
    text = path.read_text(encoding="utf-8")

    assert json.loads(text) == to_json(planned_run)
    assert text.endswith("\n")


def test_the_console_report_states_the_agent_and_every_variant(planned_run):
    out = render_console(planned_run)

    assert planned_run.agent_label in out
    for summary in planned_run.summaries:
        assert summary.variant in out
    assert "Стоимость поверхности" in out
    assert "Предпосылки запроса" in out


def test_the_console_report_marks_estimated_surface_tokens(planned_run):
    """Офлайновый счётчик — оценка, и отчёт обязан это сказать: те же цифры без
    оговорки читаются как замер токенизатором модели."""
    assert planned_run.surface_exact is False
    assert "⚠ токены оценены офлайн" in render_console(planned_run)


def test_failed_scenarios_are_named_in_the_console_report(planned_run):
    """Половина планов написана так, чтобы провалиться."""
    failed = [
        (summary.variant, result.scenario.id)
        for summary in planned_run.summaries
        for result in summary.results
        if not result.success
    ]

    out = render_console(planned_run)

    assert failed, "планы обязаны содержать заведомо проваленные — иначе проверять нечего"
    assert "=== Провалы ===" in out
    for variant, scenario_id in failed:
        assert f"[{variant}] {scenario_id}" in out


def test_the_self_check_verdict_says_it_matched(planned_run):
    out = render_self_check(planned_run, self_check_mismatches(planned_run))

    assert "Самопроверка харнесса" in out
    assert "Совпало вердиктов" in out
    assert "✗" not in out


def test_the_self_check_verdict_shows_what_diverged(planned_run):
    mismatches = self_check_mismatches(planned_run)
    assert mismatches == []

    broken = planned_run.summaries[0].results[0]

    out = render_self_check(
        planned_run,
        [
            Mismatch(
                scenario_id=broken.scenario.id,
                variant=planned_run.summaries[0].variant,
                expected=frozenset({"grounded"}),
                actual=frozenset(),
                failure=None,
            )
        ],
    )

    assert "Разошлось вердиктов: 1" in out
    assert broken.scenario.id in out


def test_fabrications_are_counted_apart_from_the_rate(planned_run):
    """Процент прячет то, что доходит до пользователя: 4 выдумки из 191 утверждения
    и 1 из 185 — это 96,9 % против 98,9 %, разрыв читается как шум, хотя неверных
    фактов вчетверо больше."""
    summaries = {s.variant: s for s in planned_run.summaries}
    fabricated = {name: s.fabricated_claims for name, s in summaries.items()}

    assert sum(fabricated.values()) > 0, "в планах есть заведомо выдуманные значения"
    for name, summary in summaries.items():
        counted = sum(1 for r in summary.results for c in r.grounding.checks if c.fabricated)
        assert fabricated[name] == counted


def test_the_rate_ignores_what_the_user_said_themselves(planned_run):
    """Знаменатель — только те утверждения, которые payload в принципе мог
    подтвердить. Порог из запроса пользователя туда не входит, иначе одна и та же
    величина считалась бы здесь и в отчёте сценария по-разному."""
    for summary in planned_run.summaries:
        user_stated = sum(
            1 for r in summary.results for c in r.grounding.checks if c.status == "user_stated"
        )
        assert summary.checkable_claims == summary.total_claims - user_stated
