#!/usr/bin/env python3
"""Блокирует Bash-команды, которые ходят в живой mcp.tutu.ru или в OpenAI.

Живой режим тратит общий лимит хакатона, поэтому запускать его может только человек
и только вручную. Всё, что работает на записанных фикстурах (evals-dry, demo-traces,
run-mock, тесты, линт), проходит без ограничений.

Хук вызывается на PreToolUse/Bash, читает JSON со stdin и при совпадении печатает
permissionDecision: deny. Молчание = разрешено.

Проверяется не вся строка, а только её исполняемая часть: тела heredoc'ов и
содержимое кавычек вырезаются, иначе хук блокировал бы написание документации,
в которой эти команды всего лишь упомянуты.
"""

from __future__ import annotations

import json
import re
import sys

REASON = (
    "Живой режим запрещён: команда ходит в mcp.tutu.ru или в OpenAI и тратит общий "
    "лимит хакатона. Работай на фикстурах — make evals-dry, make demo-traces, "
    "make run-mock, make test. Живой прогон запускает человек вручную."
)

# Каждый паттерн проверяется на отдельном сегменте команды, уже очищенном от
# кавычек и heredoc'ов. ^ — начало сегмента, то есть позиция команды.
LIVE_PATTERNS = (
    re.compile(r"^TUTU_PROXY_MODE=live\b"),
    re.compile(r"^(?:\w+=\S*\s+)*make\s+(?:run-live|fixtures|evals-record)(?:\s|$)"),
    re.compile(r"^(?:\w+=\S*\s+)*make\s+evals(?:\s|$)"),  # но не evals-dry
    re.compile(r"tutu\.py\s+record(?:\s|$)"),
    re.compile(r"tutu\.py\s+evals\b[^\n]*--(?:live|record-missing)\b"),
)

HEREDOC = re.compile(r"<<-?\s*(['\"]?)([A-Za-z_][A-Za-z0-9_]*)\1")
SEPARATORS = re.compile(r"&&|\|\||[;&|()\n]")


def strip_heredocs(command: str) -> str:
    """Выбрасывает тела heredoc'ов, оставляя строку с самим вызовом."""
    lines = command.split("\n")
    out: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        out.append(line)
        index += 1
        for match in HEREDOC.finditer(line):
            delimiter = match.group(2)
            while index < len(lines) and lines[index].strip() != delimiter:
                index += 1
            index += 1  # сама строка-ограничитель
    return "\n".join(out)


def strip_quoted(command: str) -> str:
    """Вырезает содержимое кавычек: упоминание команды в тексте — не запуск."""
    command = re.sub(r"'[^']*'", "''", command)
    return re.sub(r'"[^"]*"', '""', command)


def is_live(command: str) -> bool:
    executable = strip_quoted(strip_heredocs(command))
    for segment in SEPARATORS.split(executable):
        segment = segment.strip()
        if any(pattern.search(segment) for pattern in LIVE_PATTERNS):
            return True
    return False


def main() -> None:
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        return  # непонятный вход — не наше дело блокировать
    command = payload.get("tool_input", {}).get("command", "")
    if not isinstance(command, str) or not is_live(command):
        return
    json.dump(
        {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": REASON,
            }
        },
        sys.stdout,
        ensure_ascii=False,
    )


if __name__ == "__main__":
    main()
