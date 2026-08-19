"""Annotates an empty search result with what emptiness actually means.

The single most-cited failure in the source article: a filtered search comes
back with no offers and the agent reports "этот поезд не ходит". The tool
returns what is ON SALE right now, not a timetable, so an empty list supports
neither claim — and when a filter did the emptying, the server says so itself.

`meta.post_filter_dropped_*` is the evidence: `..._wrong_train_number: 48` means
48 trains matched the route and were dropped by the number filter alone. Handing
that to the agent turns a guess into a statement it can actually make.

Deliberately a note, not a rewrite: the payload Tutu returned goes out
byte-identical apart from one added `_`-prefixed key, the same convention as
`_answer_preamble`. `groundedness._flatten` skips those keys, so nothing here can
ever "confirm" a claim.
"""

import json
from typing import Any

EMPTY_RESULT_KEY = "_empty_result_note"

# meta counter -> what the user would call that filter
_DROP_REASONS: dict[str, str] = {
    "post_filter_dropped_wrong_train_number": "фильтр по номеру поезда",
    "post_filter_dropped_wrong_carrier": "фильтр по перевозчику",
    "post_filter_dropped_wrong_seat_category": "фильтр по типу места",
    "post_filter_dropped_not_direct": "требование рейса без пересадок",
    "post_filter_dropped_over_cap": "потолок цены",
}

_NEVER_SAY = (
    "Это значит «нет предложений под заданные фильтры», а НЕ «рейса не существует» "
    "и не «поезд не ходит»: инструмент возвращает то, что есть В ПРОДАЖЕ, а не расписание."
)

# Result lists across the domains. All are the same shape: a list under a known key.
_RESULT_KEYS = ("offers", "results", "items", "hotels", "variants")


def _result_list(payload: dict[str, Any]) -> list[Any] | None:
    for key in _RESULT_KEYS:
        value = payload.get(key)
        if isinstance(value, list):
            return value
    return None


def empty_result_note(payload: Any) -> str | None:
    """The sentence to attach, or `None` when the result is not an empty search."""
    if not isinstance(payload, dict):
        return None
    results = _result_list(payload)
    if results is None or results:
        return None

    meta = payload.get("meta")
    dropped = []
    if isinstance(meta, dict):
        dropped = [
            f"{meta[key]} по причине «{label}»"
            for key, label in _DROP_REASONS.items()
            if isinstance(meta.get(key), int) and meta[key] > 0
        ]

    if dropped:
        return (
            f"Пустой список: сервер нашёл предложения по маршруту и отбросил их фильтрами — "
            f"{', '.join(dropped)}. {_NEVER_SAY} Если пользователю нужен ответ по существу, "
            f"повторите поиск без этого фильтра или прямо скажите, какой фильтр обнулил выдачу."
        )
    return (
        f"Пустой список, ни один фильтр ничего не отбросил. {_NEVER_SAY} "
        f"Скажите, что предложений на эту дату не нашлось, и не делайте вывода о расписании."
    )


def annotate_empty_result(text: str) -> str:
    """Adds the note to a JSON result in place; returns `text` unchanged otherwise."""
    try:
        payload = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return text
    note = empty_result_note(payload)
    if note is None:
        return text
    payload[EMPTY_RESULT_KEY] = note
    return json.dumps(payload, ensure_ascii=False)
