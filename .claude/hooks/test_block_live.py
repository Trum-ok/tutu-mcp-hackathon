#!/usr/bin/env python3

import importlib.util
import sys
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "block_live", Path(__file__).with_name("block-live-commands.py")
)
block_live = importlib.util.module_from_spec(spec)
spec.loader.exec_module(block_live)

MUST_BLOCK = [
    "make run-live",
    "make evals",
    "make evals-record",
    "make fixtures",
    "uv run python tutu.py record",
    "uv run python tutu.py evals --live --record-missing",
    "TUTU_PROXY_MODE=live uv run python tutu.py serve",
    "cd /tmp && make evals && echo done",
    "make lint; make evals",
]

MUST_PASS = [
    "make evals-dry",
    "make demo-traces",
    "make run-mock",
    "make test",
    "make lint",
    "uv run python tutu.py evals --agent scripted",
    "uv run python tutu.py viewer",
    'grep -rn "make evals" README.md',
    "echo 'TUTU_PROXY_MODE=live упомянут в тексте, но не запущен'",
    # документация с этими командами внутри heredoc — это запись файла, не запуск
    "cat > doc.md <<'EOF'\nБлокируются: make evals, make run-live,\n"
    "TUTU_PROXY_MODE=live ... — запускает человек.\nEOF",
]


def main() -> int:
    failures = 0
    for command in MUST_BLOCK:
        if not block_live.is_live(command):
            print(f"ПРОПУЩЕНО, а должно блокироваться: {command!r}")
            failures += 1
    for command in MUST_PASS:
        if block_live.is_live(command):
            print(f"ЗАБЛОКИРОВАНО, а должно проходить: {command!r}")
            failures += 1
    total = len(MUST_BLOCK) + len(MUST_PASS)
    print(f"{total - failures}/{total} проверок прошло")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
