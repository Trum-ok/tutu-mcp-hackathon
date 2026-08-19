"""Сборка страниц: `viewer/build.py` и надстроенный над ним `pages/build.py`.

Обе страницы собираются один раз — в CI перед деплоем или руками перед питчем, —
и до этих тестов их защиты (проверка `closer` в `inline_assets`, экранирование
`<` в данных, вшивание шрифтов) впервые исполнялись именно там. Здесь они
исполняются на каждом прогоне, включая случаи, которых на демо-данных не бывает:
`</script>` внутри ответа сервера и плейсхолдер, попавший в трейс текстом.
"""

import json
import re

import pytest

from pages import build as pages_build
from viewer import build as viewer_build

TRACE_DATA = re.compile(r'<script id="trace-data" type="application/json">(.*?)</script>', re.S)

RUN = {
    "agent": "demo:hand-written",
    "surface_tokens_exact": False,
    "variants": [
        {
            "variant": "proxy",
            "surface": {"variant": "proxy", "tokens": 10, "bytes_": 20, "exact": False},
            "metrics": {},
            "premises": {},
            "scenarios": [],
        }
    ],
}


def bake_viewer(data: dict) -> str:
    return viewer_build.bake(data, viewer_build.TEMPLATE.read_text(encoding="utf-8"))


def trace_payload(html: str) -> dict:
    match = TRACE_DATA.search(html)
    assert match, "блок с данными не найден — вьювер нечего будет разбирать"
    return json.loads(match.group(1))


def test_assets_land_where_the_placeholders_were():
    out = viewer_build.inline_assets(
        "<style>__A__</style><script>__B__</script>",
        (("__A__", "body{color:red}", "</style"), ("__B__", "let x = 1;", "</script")),
        "шаблон",
    )

    assert out == "<style>body{color:red}</style><script>let x = 1;</script>"


def test_a_missing_placeholder_is_an_error_not_a_silent_pass():
    """Шаблон без плейсхолдера собрался бы в страницу без стилей или без скрипта —
    открывается, выглядит сломанной, ничего об этом не говорит."""
    with pytest.raises(ValueError, match="__A__"):
        viewer_build.inline_assets("<style></style>", (("__A__", "x", "</style"),), "шаблон")


def test_an_asset_that_closes_its_own_block_is_refused():
    with pytest.raises(ValueError, match="закроет блок"):
        viewer_build.inline_assets(
            "<script>__B__</script>", (("__B__", "s = '</script>';", "</script"),), "шаблон"
        )


def test_the_closer_check_ignores_case():
    """`</SCRIPT>` закрывает блок ровно так же, как `</script>`."""
    with pytest.raises(ValueError, match="закроет блок"):
        viewer_build.inline_assets(
            "<script>__B__</script>", (("__B__", "s = '</SCRIPT>';", "</script"),), "шаблон"
        )


def test_fonts_are_inlined_as_data_uris():
    css = viewer_build.inline_fonts('src: url("fonts/onest-400-700-latin.woff2") format("woff2");')

    assert "data:font/woff2;base64," in css
    assert "fonts/onest" not in css


def test_a_missing_font_stops_the_build():
    with pytest.raises(FileNotFoundError):
        viewer_build.inline_fonts('src: url("fonts/does-not-exist.woff2")')


@pytest.mark.parametrize("html", ["viewer", "pages"])
def test_the_built_page_goes_nowhere_for_assets(html):
    """Страницу открывают двойным кликом с `file://`, возможно без сети. Любая
    внешняя ссылка на ассет — это пустая страница на сцене."""
    built = (
        bake_viewer(RUN)
        if html == "viewer"
        else pages_build.bake(pages_build.TEMPLATE.read_text(encoding="utf-8"))
    )
    head = built.split("</head>", 1)[0]

    assert "<script src=" not in built
    assert "stylesheet" not in head
    assert "@import" not in built
    assert "url(fonts/" not in built and 'url("fonts/' not in built
    assert re.search(r"url\(['\"]?https?:", built) is None


def test_the_trace_payload_survives_a_closing_script_tag():
    """`</script>` в ответе сервера закрыл бы блок данных на середине: остаток
    payload утёк бы в разметку, а `JSON.parse` упал бы на огрызке."""
    data = {**RUN, "leak": "<script>alert(1)</script> и </SCRIPT> тоже"}

    html = bake_viewer(data)

    assert trace_payload(html) == data
    assert "</script>" not in TRACE_DATA.search(html).group(1)


def test_a_placeholder_inside_the_data_stays_text():
    """Ассеты подставляются раньше данных, поэтому строка `__STYLES__`, попавшая
    в трейс, остаётся строкой, а не превращается в копию таблицы стилей."""
    data = {**RUN, "note": "__STYLES__ __SCRIPT__ __TRACE_DATA__"}

    payload = trace_payload(bake_viewer(data))

    assert payload["note"] == "__STYLES__ __SCRIPT__ __TRACE_DATA__"


def test_the_data_placeholder_is_required():
    with pytest.raises(ValueError, match="__TRACE_DATA__"):
        viewer_build.bake(RUN, "<style>__STYLES__</style><script>__SCRIPT__</script>")


def test_the_viewer_writes_a_single_file(tmp_path):
    data = tmp_path / "eval-results.json"
    data.write_text(json.dumps(RUN, ensure_ascii=False), encoding="utf-8")
    out = tmp_path / "site" / "trace-viewer.html"

    assert viewer_build.build_viewer(data, out) == 0
    assert trace_payload(out.read_text(encoding="utf-8")) == RUN


def test_the_viewer_says_where_to_get_the_data_instead_of_traceback(tmp_path, capsys):
    code = viewer_build.build_viewer(tmp_path / "нет.json", tmp_path / "out.html")

    assert code == 2
    assert "make demo-traces" in capsys.readouterr().err


def test_the_docs_page_points_at_the_deployed_server(monkeypatch):
    monkeypatch.setenv("MCP_PUBLIC_URL", "https://example.invalid/mcp")

    html = pages_build.bake(pages_build.TEMPLATE.read_text(encoding="utf-8"))

    assert "https://example.invalid/mcp" in html
    assert "__DEPLOY_URL__" not in html


def test_the_docs_page_falls_back_to_the_local_address(monkeypatch):
    """Локальная сборка и форк без переменной не должны показывать читателю
    сырой плейсхолдер — там стоит адрес, на котором слушает `tutu.py serve`."""
    monkeypatch.delenv("MCP_PUBLIC_URL", raising=False)

    html = pages_build.bake(pages_build.TEMPLATE.read_text(encoding="utf-8"))

    assert pages_build.LOCAL_DEPLOY_URL in html
    assert "__DEPLOY_URL__" not in html


def test_an_empty_deploy_url_is_not_taken_for_an_address(monkeypatch):
    monkeypatch.setenv("MCP_PUBLIC_URL", "")

    assert pages_build.LOCAL_DEPLOY_URL in pages_build.bake(
        pages_build.TEMPLATE.read_text(encoding="utf-8")
    )


def test_the_deploy_placeholder_is_required():
    with pytest.raises(ValueError, match="__DEPLOY_URL__"):
        pages_build.bake("<style>__STYLES__</style><script>__SCRIPT__</script>")


def test_the_docs_page_is_written_whole(tmp_path):
    out = tmp_path / "site" / "index.html"

    assert pages_build.build_docs(out) == 0
    assert out.read_text(encoding="utf-8").startswith("<!doctype html>")


def test_every_anchor_on_the_docs_page_lands_somewhere():
    """Содержание доки — это якоря внутри одного файла. Переименованный `id`
    ничего не ломает при сборке: ссылка просто перестаёт прокручивать."""
    template = pages_build.TEMPLATE.read_text(encoding="utf-8")
    ids = set(re.findall(r'\bid="([^"]+)"', template))

    anchors = {a for a in re.findall(r'href="#([^"]+)"', template) if a}

    assert anchors, "содержание пустое — проверять нечего"
    assert anchors <= ids, f"ссылки в никуда: {sorted(anchors - ids)}"


def test_the_table_of_contents_still_matches_the_selector_the_script_uses():
    """`pages/app.js` подсвечивает пункт по `nav.toc a[href^="#"]`. Переехавший
    класс оставляет скрипт без единого пункта — молча, без ошибки в консоли."""
    template = pages_build.TEMPLATE.read_text(encoding="utf-8")

    toc = re.search(r'<nav class="toc">(.+?)</nav>', template, re.S)

    assert toc, "блока `nav.toc` больше нет — подсветка содержания мертва"
    assert re.findall(r'href="#', toc.group(1))
