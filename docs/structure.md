# Структура репозитория

Дополнение к [README](../README.md): где что лежит и почему стрелки зависимостей идут
именно так.

```
tutu_mcp/                  сам прокси — это то, что поднимает `make run-mock`
  backend.py               протокол ToolBackend, общий для live- и mock-клиента
  backends.py              единственное место, где выбирается бэкенд по настройкам
  upstream/client.py       live-бэкенд — обёртка над официальным SDK `mcp`
  replay/store.py          файловое хранилище фикстур (VCR-стиль: по тулу + точным аргументам)
  replay/mock_client.py    mock-бэкенд — отвечает из fixtures/ вместо похода в Туту
  replay/recording.py      записывает fixtures/ с живого сервера по ходу прогона
  replay/bootstrap.py      драйвер записи фикстур; список вызовов — в evals/fixtures_recipe.py
  proxy/dispatch.py        единый пайплайн tools/call — общий для сервера и proxy-варианта эвалов
  proxy/surface.py         свои тулы прокси (assess_request, check_groundedness) + сборка каталога
  proxy/compact_tools.py   урезание описаний + сплайсинг проза-в-результат
  proxy/empty_results.py   объясняет пустую выдачу счётчиками post_filter_dropped_*
  proxy/server.py          сборка MCPServer прокси (tools/list, tools/call, check_groundedness)
  groundedness.py          извлечение утверждений + проверка обоснованности
  premises.py              premise gate: происхождение значений, детектор опечаток, assess_request
  toolspec.py              Pydantic -> дескрипторы tools/list для своих тулов
  text.py                  поиск значения по границам слова — общий для гейта и groundedness
  config.py                настройки TUTU_*; читает .env при импорте
  main.py                  точка входа
evals/                     харнесс, который измеряет прокси — импортирует его, не наоборот
  options.py               перечисления и опции одного прогона эвалов
  scenarios.py             сценарии, через которые прогоняются обе поверхности
  fixtures_recipe.py       список вызовов для `tutu.py record` — те же даты, что в сценариях
  variants.py               два сравниваемых варианта: baseline и proxy
  agent.py                 агент под тестом (OpenAI либо скриптованная заглушка для CI)
  runner.py                прогоняет один сценарий, собирает transcript
  transcript.py            общая запись одного прогона агента
  checks.py                детерминированные проверки по сценарию
  report.py                консольная сводка + JSON для трейс-вьювера
  tokens.py                подсчёт токенов (точный через API либо офлайн через tiktoken)
  measure.py               воспроизводимый подсчёт байт каталога — источник цифр в этом README
  run.py                   один прогон целиком: агент + счётчик + поверхности -> измерение
  plans.py                 рукописные планы + таблица ожидаемых вердиктов SELF_CHECK
  demo.py                  рукописные трейсы поверх настоящих фикстур, без модели
  config.py                учётные данные OPENAI_* и модель по умолчанию — только для харнесса
tutu.py                    единая точка входа: serve / evals / record / demo / viewer / docs / measure
viewer/                    UI трейс-вьювера: template.html + styles.css + app.js + fonts/
  tokens.css               шрифты и цветовые токены, общие с pages/styles.css
  build.py                 запекает прогон эвалов и ассеты в один самодостаточный файл
pages/                     пользовательская дока: тот же паттерн сборки, что у вьювера
  build.py                 запекает pages/template.html + styles.css + app.js в site/index.html
site/                      только генерируется — дока и трейс-вьювер, публикуются на Pages
docs/
  findings.md              сырые замеры и мотивирующий кейс, датировано и привязано к версии сервера
  trace-viewer-preview.png скриншот вьювера для README
fixtures/                  записанные ответы (tools/list, инструкции, поиски, ...)
tests/                     pytest, целиком по записанным фикстурам, без сети
  viewer_smoke.mjs         прогоняет viewer/app.js в node: контракт report.py -> вьювер
```
