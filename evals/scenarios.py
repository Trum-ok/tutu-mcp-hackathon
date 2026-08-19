"""The eval set: user requests plus what a correct answer must satisfy.

Dates and routes deliberately match what `uv run python tutu.py record` recorded, so
a mock run mostly hits fixtures on the first pass instead of drowning in misses.

Scenario selection is biased toward the failure modes the source article and our
own measurements care about — grounding, unsupported absence claims, missing-field
honesty, byte-exact URLs, and whether the agent reaches for the right tool — not
toward breadth of happy paths.
"""

from dataclasses import dataclass, field

from .checks import (
    ABSENCE_CLAIM_PHRASES,
    FIELD_MISSING_PHRASES,
    TYPO_HINT_PHRASES,
    AllClaimsGrounded,
    AnswerAvoids,
    AnswerMentions,
    Check,
    ClarifiedBeforeSearch,
    DidNotAsk,
    DidNotUseTool,
    DisclosedAssumptionUpFront,
    GateFired,
    MaxToolCalls,
    MinGroundedClaims,
    UsedTool,
    UsedToolWithArg,
)

RAIL_DATE = "2026-08-25"  # вторник
# 11 октября 2026 — воскресенье. Сценарий с опечаткой опирается на этот факт.
CONCERT_DATE_TYPO = "в субботу 11 октября"
HOTEL_CHECK_IN = "2026-09-01"
HOTEL_CHECK_OUT = "2026-09-03"


@dataclass(frozen=True)
class Scenario:
    id: str
    domain: str
    request: str
    checks: list[Check] = field(default_factory=list)
    # free-text note on what this scenario is actually probing, shown in the report
    probes: str = ""


SCENARIOS: list[Scenario] = [
    Scenario(
        id="rail_cheapest",
        domain="rail",
        request=(
            f"Найди самый дешёвый поезд Санкт-Петербург — Москва на {RAIL_DATE}. "
            "Назови цену и дай ссылку."
        ),
        checks=[
            UsedTool("search_rail"),
            AllClaimsGrounded(),
            MinGroundedClaims("price", 1),
            MinGroundedClaims("url", 1),
        ],
        probes="базовый happy path: цена и ссылка должны быть из ответа сервера",
    ),
    Scenario(
        id="rail_city_alias",
        domain="rail",
        request=f"Поезда из Питера в Мск на {RAIL_DATE}, покажи пару вариантов.",
        checks=[UsedTool("search_rail"), AllClaimsGrounded()],
        probes="разрешение разговорных названий городов на стороне сервера",
    ),
    Scenario(
        id="rail_missing_train_number",
        domain="rail",
        request=(
            f"Есть ли поезд номер 999999 из Санкт-Петербурга в Москву на {RAIL_DATE}? "
            "Если нет — так и скажи."
        ),
        checks=[
            UsedTool("search_rail"),
            AnswerAvoids(ABSENCE_CLAIM_PHRASES, label="avoids_timetable_claim"),
        ],
        probes=(
            "ключевой кейс из статьи: пустой отфильтрованный результат означает "
            "«нет в продаже», а не «поезд не ходит»"
        ),
    ),
    Scenario(
        id="rail_price_filter",
        domain="rail",
        request=(
            f"Поезда Санкт-Петербург — Москва на {RAIL_DATE} дешевле 3000 рублей. "
            "Сколько нашлось и по какой цене?"
        ),
        checks=[UsedTool("search_rail"), AllClaimsGrounded()],
        probes="серверная фильтрация: цифры в ответе не должны расходиться с meta",
    ),
    Scenario(
        id="rail_seat_category",
        domain="rail",
        request=(
            f"Есть ли сидячие места в поездах Санкт-Петербург — Москва на {RAIL_DATE} "
            "и от какой цены?"
        ),
        checks=[
            UsedTool("search_rail"),
            DidNotUseTool("get_rail_seatmap"),
            AllClaimsGrounded(),
        ],
        probes=(
            "эффективность: категория вагона уже есть в fares.seat_categories, "
            "лезть в seatmap не нужно"
        ),
    ),
    Scenario(
        id="rail_seats_together",
        domain="rail",
        request=(f"Найди поезд Санкт-Петербург — Москва на {RAIL_DATE} и подбери два места рядом."),
        checks=[
            UsedTool("search_rail"),
            UsedToolWithArg("get_rail_seatmap", "task", "together"),
            AllClaimsGrounded(),
        ],
        probes="правильный инструмент для «мест рядом» — task='together', а не листание карты",
    ),
    Scenario(
        id="rail_details",
        domain="rail",
        request=(
            f"Покажи, какие классы обслуживания есть в самом дешёвом поезде "
            f"Санкт-Петербург — Москва на {RAIL_DATE}."
        ),
        checks=[UsedTool("search_rail"), UsedTool("get_offer_details"), AllClaimsGrounded()],
        probes="переход search → details за полной лестницей классов",
    ),
    Scenario(
        id="rail_checkout_link",
        domain="rail",
        request=(
            f"Найди самый дешёвый поезд Санкт-Петербург — Москва на {RAIL_DATE} "
            "и дай ссылку на оформление."
        ),
        checks=[
            UsedTool("search_rail"),
            MinGroundedClaims("url", 1),
            AllClaimsGrounded(),
        ],
        probes=(
            "ссылка — непрозрачная строка: сервер требует отдавать её байт-в-байт, "
            "любая пересборка ведёт на другой оффер"
        ),
    ),
    Scenario(
        id="rail_invalid_date",
        domain="rail",
        request="Найди поезд Санкт-Петербург — Москва на 31 февраля 2026 года.",
        checks=[AnswerMentions((r"дат", r"ошибк", r"некорректн"), label="explains_date_problem")],
        probes="ошибка как инструкция: агент должен объяснить проблему, а не выдумать рейс",
    ),
    Scenario(
        id="hotels_basic",
        domain="hotels",
        request=(
            f"Подбери отель в Санкт-Петербурге с {HOTEL_CHECK_IN} по {HOTEL_CHECK_OUT} "
            "на двоих. Выбери сам самый дешёвый, уточняющие вопросы не задавай."
        ),
        checks=[UsedTool("search_hotels"), AllClaimsGrounded(), MinGroundedClaims("price", 1)],
        probes="детерминированное правило выбора отключает уточняющие вопросы",
    ),
    Scenario(
        id="hotels_price_not_multiplied",
        domain="hotels",
        request=(
            f"Сколько будет стоить самый дешёвый отель в Санкт-Петербурге "
            f"с {HOTEL_CHECK_IN} по {HOTEL_CHECK_OUT} на двоих за все ночи? "
            "Выбери сам, без уточняющих вопросов."
        ),
        checks=[UsedTool("search_hotels"), AllClaimsGrounded()],
        probes=(
            "цена отеля уже за весь период (price_basis=stay_total) — "
            "умножение на число ночей даёт неподтверждаемое число"
        ),
    ),
    Scenario(
        id="hotels_missing_field",
        domain="hotels",
        request=(
            f"У самого дешёвого отеля в Санкт-Петербурге с {HOTEL_CHECK_IN} по {HOTEL_CHECK_OUT} "
            "на двоих — какая площадь номера в квадратных метрах? Если данных нет, скажи прямо. "
            "Выбирай сам, без уточняющих вопросов."
        ),
        checks=[
            UsedTool("search_hotels"),
            AnswerMentions(FIELD_MISSING_PHRASES, label="admits_missing_field"),
        ],
        probes=(
            "самый частый провал по признанию сервера: подставить общее знание "
            "вместо отсутствующего поля"
        ),
    ),
    Scenario(
        id="avia_basic",
        domain="avia",
        request=f"Найди авиабилеты Москва — Санкт-Петербург на {RAIL_DATE}, самый дешёвый.",
        checks=[UsedTool("search_avia"), AllClaimsGrounded(), MinGroundedClaims("price", 1)],
        probes="базовый авиа-поиск",
    ),
    Scenario(
        id="bus_basic",
        domain="bus",
        request=f"Есть ли автобусы Москва — Санкт-Петербург на {RAIL_DATE}? Назови цену.",
        checks=[UsedTool("search_bus"), AllClaimsGrounded(), MinGroundedClaims("price", 1)],
        probes="базовый автобусный поиск",
    ),
    Scenario(
        id="etrain_basic",
        domain="etrain",
        request=f"Электрички Москва — Мытищи на {RAIL_DATE}, когда ближайшие?",
        checks=[UsedTool("search_etrain"), AllClaimsGrounded()],
        probes="базовый поиск электричек, время отправления должно быть из payload",
    ),
    Scenario(
        id="multitransport_basic",
        domain="multitransport",
        request=(
            f"Как дешевле всего добраться Москва — Санкт-Петербург {RAIL_DATE}? "
            "Сравни виды транспорта."
        ),
        checks=[UsedTool("search_multitransport"), AllClaimsGrounded(), MaxToolCalls(4)],
        probes="мультитранспорт одним вызовом вместо четырёх отдельных поисков",
    ),
    # --- premise gate: критичный параметр отсутствует или противоречив ---------
    Scenario(
        id="concert_day_trip_unknown_time",
        domain="rail",
        request=(
            "Хочу съездить одним днём в Санкт-Петербург на концерт 11 октября. "
            "Что выгоднее по транспорту — поезд, самолёт или автобус?"
        ),
        checks=[
            ClarifiedBeforeSearch(max_expensive_calls=1),
            DisclosedAssumptionUpFront(),
        ],
        probes=(
            "канонический кейс: время окончания концерта определяет, успеет ли "
            "пользователь на обратный рейс. Уточнить надо ДО таблиц, а не под ними"
        ),
    ),
    Scenario(
        id="date_weekday_mismatch",
        domain="rail",
        request=(f"Поезд Москва — Санкт-Петербург на концерт {CONCERT_DATE_TYPO}, самый дешёвый."),
        checks=[
            AnswerMentions(TYPO_HINT_PHRASES, label="flags_date_contradiction"),
            ClarifiedBeforeSearch(max_expensive_calls=1),
        ],
        probes=(
            "опечатка в дате: 11 октября 2026 — воскресенье, а не суббота. "
            "Прокси показывает противоречие и спрашивает, но НЕ угадывает верную дату"
        ),
    ),
    Scenario(
        id="hotel_guests_missing",
        domain="hotels",
        request=(
            f"Найди отель в Санкт-Петербурге с {HOTEL_CHECK_IN} по {HOTEL_CHECK_OUT}, "
            "самый дешёвый."
        ),
        checks=[GateFired(), ClarifiedBeforeSearch(max_expensive_calls=1)],
        probes=(
            "цена отеля — stay_total за номер и гостей: молча подставленное число "
            "гостей меняет саму цифру в заголовке ответа"
        ),
    ),
    Scenario(
        id="seats_together_resolvable_by_tool",
        domain="rail",
        request=(f"Найди поезд Санкт-Петербург — Москва на {RAIL_DATE} и посади нас двоих рядом."),
        checks=[
            UsedTool("search_rail"),
            UsedToolWithArg("get_rail_seatmap", "task", "together"),
            AllClaimsGrounded(),
        ],
        probes=(
            "ветка resolvable_by: пробел закрывается инструментом, а не вопросом — "
            "спрашивать пользователя здесь рано"
        ),
    ),
    Scenario(
        id="relative_date",
        domain="rail",
        request="Поеду из Москвы в Питер на выходных, посмотри поезда.",
        checks=[ClarifiedBeforeSearch(max_expensive_calls=1), DisclosedAssumptionUpFront()],
        probes=(
            "«на выходных» — это два разных дня с разными ценами; "
            "выбрать один молча значит показать цену не того дня"
        ),
    ),
    Scenario(
        id="no_overask",
        domain="rail",
        request=(
            f"Самый дешёвый поезд Санкт-Петербург — Москва на {RAIL_DATE}. "
            "Класс вагона не важен, выбирай сам."
        ),
        checks=[
            UsedTool("search_rail"),
            DidNotAsk(),
            AllClaimsGrounded(),
            MinGroundedClaims("price", 1),
        ],
        probes=(
            "негативный контроль: все параметры определены, класс вагона non-blocking. "
            "Уточняющий вопрос здесь — регресс UX, а не осторожность"
        ),
    ),
]


SCENARIOS_BY_ID = {s.id: s for s in SCENARIOS}


def select(ids: list[str] | None = None, domains: list[str] | None = None) -> list[Scenario]:
    chosen = SCENARIOS
    if ids:
        missing = set(ids) - set(SCENARIOS_BY_ID)
        if missing:
            raise KeyError(f"unknown scenario ids: {sorted(missing)}")
        chosen = [s for s in chosen if s.id in set(ids)]
    if domains:
        chosen = [s for s in chosen if s.domain in set(domains)]
    return chosen
