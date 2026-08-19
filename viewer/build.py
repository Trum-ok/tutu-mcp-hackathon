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


def build_viewer(data_path: Path, out: Path) -> int:
    if not data_path.is_file():
        print(
            f"{data_path} не найден.\n"
            "Сначала соберите данные:\n"
            "  make evals        — реальный прогон (нужен OPENAI_API_KEY)\n"
            "  make demo-traces  — демо-трейсы на фикстурах, без модели",
            file=sys.stderr,
        )
        return 2

    data = json.loads(data_path.read_text(encoding="utf-8"))
    html = bake(data, TEMPLATE.read_text(encoding="utf-8"))

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")

    scenarios = sum(len(v.get("scenarios", [])) for v in data.get("variants", []))
    print(f"{out}  ({len(html) / 1024:.0f} КБ, {scenarios} трейсов, агент: {data.get('agent')})")
    return 0
