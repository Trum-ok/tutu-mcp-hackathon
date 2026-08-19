# Настройка и команды

Дополнение к [README](../README.md): переменные окружения и полный список make-целей.

Все настройки — переменные окружения. `tutu_mcp/config.py` читает `.env` из корня репозитория
при импорте — и никогда не перекрывает то, что уже экспортировано в шелле. `.env` в
`.gitignore`; `.env.example` — шаблон:

```bash
cp .env.example .env      # править не обязательно: ключ нужен только для make evals
```

| Переменная                            | Для чего                                                                                  | По умолчанию                |
|---------------------------------------|-------------------------------------------------------------------------------------------|-----------------------------|
| `OPENAI_API_KEY`                      | **только эвалы** — агент под тестом и точный подсчёт токенов                              | —                           |
| `OPENAI_MODEL`                        | модель по умолчанию для эвалов (`--model` переопределяет)                                 | `gpt-5`                     |
| `OPENAI_EFFORT`                       | усилие рассуждения по умолчанию (`--effort` переопределяет)                               | своё у модели               |
| `OPENAI_BASE_URL`                     | для OpenAI-совместимого шлюза                                                             | `https://api.openai.com/v1` |
| `TUTU_PROXY_MODE`                     | `mock` (фикстуры, без сети) или `live`                                                    | `mock`                      |
| `TUTU_UPSTREAM_URL`                   | адрес upstream MCP-сервера                                                                | `https://mcp.tutu.ru/mcp`   |
| `TUTU_UPSTREAM_TIMEOUT_S`             | таймаут одного запроса к живому серверу, секунды                                          | `20`                        |
| `TUTU_FIXTURES_DIR`                   | где лежат записанные фикстуры                                                             | `./fixtures`                |
| `TUTU_CATALOG_TTL_S`                  | как долго закешированный `tools/list` считается свежим, секунды                           | `900`                       |
| `TUTU_PROXY_HOST` / `TUTU_PROXY_PORT` | адрес прослушивания                                                                       | `127.0.0.1` / `8800`        |
| `PORT`                                | фолбэк для PaaS (Render/Railway/Fly/…); `TUTU_PROXY_PORT` в приоритете — см. `Dockerfile` | задаёт платформа            |

Кривое значение любой из `TUTU_*` отвергается на старте — с указанием переменной и кодом
возврата 2, а не трейсбеком из глубины: `TUTU_PROXY_MODE=moc`, нечисловой таймаут, порт вне
1–65535. Занятый порт прокси проверяет своим bind'ом до запуска uvicorn, поэтому строки
`listening on ...` не бывает без сервера за ней.

**Самому прокси ключ OpenAI не нужен** — только эвал-харнессу, потому что это единственная
часть, которая запускает модель. `make run-mock`, `make run-live` и весь набор тестов работают
без ключа. Раннер проверяет наличие ключа до первого запроса и сразу отказывает с инструкцией,
а не падает по авторизации на середине прогона. Какие модели видит ключ —
`uv run python tutu.py evals --list-models`.

## Команды

```bash
make lint          # ruff check + format --check + ty
make format        # ruff check --fix + format
make lint-front    # biome ci over viewer/ + pages/ (app.js + styles.css)
make format-front  # biome check --write over viewer/ + pages/
make test           # pytest
make fixtures       # перезаписать fixtures/ с живого сервера
make run-mock       # прокси на записанных фикстурах
make run-live       # прокси на настоящем mcp.tutu.ru
make evals          # baseline против proxy (нужен OPENAI_API_KEY)
make evals-dry      # самопроверка харнесса на рукописных планах, без ключей
make evals-record   # один живой проход, дописывает недостающие фикстуры
make demo-traces    # рукописные трейсы поверх настоящих фикстур, без модели
make viewer          # trace-viewer.html из последнего настоящего прогона
make viewer-demo    # trace-viewer.html из демо-трейсов
make docs            # site/index.html, пользовательская дока
make site            # дока + трейс-вьювер вместе в site/
```
