"""Контракт `evals/report.py` → `viewer/app.js`.

Тесты выше проверяют, что дамп имеет форму, о которой договаривались. Этот
проверяет вторую половину договора: что вьювер эту форму читает. Разошедшееся
поле страницу не роняет — она рисуется дальше и пишет `undefined` в клетке
метрики или в объяснении утверждения, а увидеть это можно было только глазами
и только после деплоя.

Настоящий `app.js` из СОБРАННОЙ страницы исполняется в node поверх заглушки DOM
(`tests/viewer_smoke.mjs`) и отрисовывает каждый вариант, каждый сценарий,
обзор, сравнение и ящик доказательств по каждому утверждению.
"""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from evals.report import to_json
from viewer import build as viewer_build

REPO_ROOT = Path(__file__).resolve().parent.parent
SMOKE = Path(__file__).resolve().parent / "viewer_smoke.mjs"

STATUS_LABELS = {
    "confirmed": "подтверждено",
    "assumed": "допущение",
    "user_stated": "слова пользователя",
    "unavailable": "нет в данных",
}

pytestmark = pytest.mark.skipif(
    shutil.which("node") is None, reason="нужен node для запуска app.js"
)


def render(data: dict, tmp_path: Path) -> dict:
    """Печёт страницу и возвращает то, что вьювер из этих данных нарисовал."""
    page = tmp_path / "trace-viewer.html"
    page.write_text(
        viewer_build.bake(data, viewer_build.TEMPLATE.read_text(encoding="utf-8")),
        encoding="utf-8",
    )
    done = subprocess.run(
        ["node", str(SMOKE), str(page)],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        check=False,
    )
    assert done.returncode == 0, f"вьювер упал на этих данных:\n{done.stderr}"
    return json.loads(done.stdout)


@pytest.fixture
def dump(planned_run):
    return to_json(planned_run)


@pytest.fixture
def rendered(dump, tmp_path):
    return render(dump, tmp_path)


def test_the_viewer_draws_every_variant_and_scenario(dump, rendered):
    expected = [f"{v['variant']}/{s['id']}" for v in dump["variants"] for s in v["scenarios"]]

    assert rendered["ids"] == expected
    for name in expected:
        variant, scenario_id = name.split("/", 1)
        assert variant in rendered["html"]
        assert scenario_id in rendered["html"]


def test_no_field_of_the_dump_reaches_the_page_as_undefined(rendered):
    """Главное, ради чего этот тест существует: переименованный ключ виден здесь,
    а не на проекторе."""
    for hole in ("undefined", "NaN", "[object Object]"):
        assert hole not in rendered["html"], f"вьювер нарисовал `{hole}` — поле дампа разошлось"


def test_the_agent_label_is_stamped_on_the_page(dump, rendered):
    """Рукописный прогон обязан быть подписан как не-замер — иначе демо-трейсы
    читаются со сцены как измерение."""
    assert dump["agent"] in rendered["html"]
    assert "НЕ ЗАМЕР" in rendered["html"]


def test_every_claim_status_gets_its_russian_verdict(dump, rendered):
    statuses = {
        claim["status"]
        for variant in dump["variants"]
        for scenario in variant["scenarios"]
        for claim in scenario["groundedness"]["claims"]
    }

    assert statuses
    for status in statuses:
        assert STATUS_LABELS[status] in rendered["html"]


def test_the_evidence_drawer_quotes_the_payload_it_found_the_value_in(dump, rendered):
    """Ящик доказательств читает `tool_calls[].result_text` — то самое поле, которое
    доказывает, что подсветка ищет по настоящему ответу сервера."""
    called = {
        call["name"]
        for variant in dump["variants"]
        for scenario in variant["scenarios"]
        for call in scenario["tool_calls"]
    }

    assert called
    assert "найдено в ответе" in rendered["html"]
    for name in called:
        assert name in rendered["html"]


def test_a_renamed_field_really_does_show_up_as_a_hole(dump, tmp_path):
    """Проверка, которая не умеет краснеть, ничего не охраняет."""
    for variant in dump["variants"]:
        variant["metrics"]["grounded"] = variant["metrics"].pop("grounded_claims")

    assert "undefined" in render(dump, tmp_path)["html"]
