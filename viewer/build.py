"""Bakes an eval run into a single self-contained HTML file.

    uv run python tutu.py viewer [--data out/eval-results.json] [--out viewer/trace-viewer.html]

Вёрстка живёт в трёх файлах — `template.html`, `styles.css`, `app.js`, — а сборка
склеивает их в один документ. Self-contained результат нужен потому, что питч
идёт с чьего-то ноутбука, возможно на конференц-вайфае: страница, которая тянет
данные или ассеты через `file://`, упрётся в CORS и покажет пустоту, а страница,
которой нужен дев-сервер, — это ещё одна вещь, способная упасть на сцене.
Двойного клика по выходному файлу должно быть достаточно.
"""

import base64
import json
import re
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
VIEWER_DIR = Path(__file__).resolve().parent
TEMPLATE = VIEWER_DIR / "template.html"
TOKENS = VIEWER_DIR / "tokens.css"
STYLES = VIEWER_DIR / "styles.css"
SCRIPT = VIEWER_DIR / "app.js"

DEFAULT_DATA = REPO_ROOT / "out" / "eval-results.json"
DEFAULT_OUT = REPO_ROOT / "viewer" / "trace-viewer.html"

PLACEHOLDER = "__TRACE_DATA__"
FONT_REF = re.compile(r"""url\(["']?(fonts/[^"')]+)["']?\)""")


def inline_fonts(css: str) -> str:
    """Ссылки на woff2 → data: URI, иначе страница пошла бы за шрифтами в сеть.

    Общая для вьювера и pages/build.py: обе страницы шьют шрифты из viewer/fonts/.
    """

    def embed(match: re.Match[str]) -> str:
        path = VIEWER_DIR / match.group(1)
        if not path.is_file():
            raise FileNotFoundError(f"{path} не найден: шрифт нечем вшить")
        payload = base64.b64encode(path.read_bytes()).decode("ascii")
        return f'url("data:font/woff2;base64,{payload}")'

    return FONT_REF.sub(embed, css)


def load_styles() -> str:
    """tokens.css (шрифты + цвета, общие с pages/styles.css) + styles.css вьювера."""
    css = TOKENS.read_text(encoding="utf-8") + "\n" + STYLES.read_text(encoding="utf-8")
    return inline_fonts(css.rstrip("\n"))


def inline_assets(template: str, assets: Sequence[tuple[str, str, str]], source: Path | str) -> str:
    """Плейсхолдер → содержимое, с проверкой, что ассет не закрывает свой же блок.

    Общая для вьювера и `pages/build.py`: обе страницы шьют CSS и JS одинаково,
    и раньше обе несли по своей копии этого цикла — включая проверку `closer`,
    единственную защиту от `</script>` внутри вшиваемого файла. Копия, у которой
    такую проверку однажды забудут поправить, молча собирает битую страницу.
    """
    for placeholder, asset, closer in assets:
        if placeholder not in template:
            raise ValueError(f"{placeholder} not found in {source}")
        if closer in asset.lower():
            raise ValueError(f"{placeholder} содержит `{closer}`: это закроет блок раньше времени")
        template = template.replace(placeholder, asset)
    return template


def page_assets(styles: str, script_path: Path) -> tuple[tuple[str, str, str], ...]:
    """Пара ассетов, одинаковая для обеих страниц."""
    return (
        ("__STYLES__", styles, "</style"),
        ("__SCRIPT__", script_path.read_text(encoding="utf-8").rstrip("\n"), "</script"),
    )


def bake(data: dict[str, Any], template: str) -> str:
    # Ассеты подставляются до данных: иначе строка вида `__STYLES__`, случайно
    # попавшая в трейс, была бы принята за плейсхолдер.
    template = inline_assets(template, page_assets(load_styles(), SCRIPT), TEMPLATE)

    # `</script>` inside a tool result would close the data block early; escaping
    # `<` at the JSON level keeps the payload inert wherever it lands in the page.
    payload = json.dumps(data, ensure_ascii=False).replace("<", "\\u003c")
    if PLACEHOLDER not in template:
        raise ValueError(f"{PLACEHOLDER} not found in {TEMPLATE}")
    return template.replace(PLACEHOLDER, payload)


class RunDataError(Exception):
    """`--data` указывает на файл, из которого страницу собирать нельзя."""


HOW_TO_GET_DATA = (
    "  make evals        — реальный прогон (нужен OPENAI_API_KEY)\n"
    "  make demo-traces  — демо-трейсы на фикстурах, без модели"
)


def read_run(path: Path) -> dict[str, Any]:
    """Читает отчёт прогона и проверяет форму, которую ждёт `app.js`.

    Форму приходится проверять здесь, а не в шаблоне: `bake` вшивает любой
    валидный JSON, и страница из объекта не той формы собирается без единой
    ошибки — просто рисует пустой список и печатает `(0 трейсов, агент: None)`.
    Это читается как «прогон вышел пустым», хотя на самом деле вьюверу дали не
    тот файл. Проверка «файла нет» рядом ровно про это же: сказать, что не так и
    чем это чинится, вместо трейсбека или молчаливой пустой страницы.
    """
    if not path.is_file():
        raise RunDataError(f"{path} не найден.\nСначала соберите данные:\n{HOW_TO_GET_DATA}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RunDataError(f"{path} не разбирается как JSON ({exc}).\n{HOW_TO_GET_DATA}") from exc
    except OSError as exc:
        raise RunDataError(f"{path} не читается ({exc.strerror or exc}).") from exc

    if not isinstance(data, dict) or not isinstance(data.get("agent"), str):
        raise RunDataError(
            f"{path} — не отчёт прогона: ожидался объект с полем `agent`.\n{HOW_TO_GET_DATA}"
        )
    variants = data.get("variants")
    if not isinstance(variants, list) or not variants:
        raise RunDataError(f"{path}: в `variants` пусто — вьюверу нечего рисовать.")
    for index, variant in enumerate(variants):
        if not isinstance(variant, dict) or not isinstance(variant.get("scenarios"), list):
            raise RunDataError(
                f"{path}: variants[{index}] без списка `scenarios` — файл собран не этой версией "
                f"харнесса.\n{HOW_TO_GET_DATA}"
            )
    return data


def build_viewer(data_path: Path, out: Path) -> int:
    try:
        data = read_run(data_path)
    except RunDataError as exc:
        print(exc, file=sys.stderr)
        return 2

    html = bake(data, TEMPLATE.read_text(encoding="utf-8"))

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")

    scenarios = sum(len(v["scenarios"]) for v in data["variants"])
    print(f"{out}  ({len(html) / 1024:.0f} КБ, {scenarios} трейсов, агент: {data['agent']})")
    return 0
