const DATA = JSON.parse(document.getElementById('trace-data').textContent);
const STATUS_RU = {
  confirmed: 'подтверждено',
  assumed: 'допущение',
  user_stated: 'слова пользователя',
  unavailable: 'нет в данных',
};
const state = {
  variant: DATA.variants[0]?.variant,
  scenario: null,
  payloadView: 'pretty',
  claim: null,
  claimVariant: null,
  compare: false,
  overview: false,
  failedOnly: false,
  collapsed: new Set(),
  evidenceSteps: 0,
  evidenceAll: false,
};

const esc = (s) =>
  String(s).replace(
    /[&<>"]/g,
    (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c],
  );
const pct = (v) => (v == null ? '—' : `${Math.round(v * 100)}%`);
const variantOf = (n) => DATA.variants.find((v) => v.variant === n);

const num = (n) => (n == null ? '—' : n.toLocaleString('ru-RU'));

const scenarioOf = (variant, id) => {
  const v = variantOf(variant);
  return v?.scenarios.find((s) => s.id === id) || v?.scenarios[0] || null;
};

/* Ответ сервера показывается двумя способами: как пришёл по проводу и разложенным
   по строкам. Raw остаётся доступным всегда — именно он доказывает, что подсветка
   в ящике доказательств ищет по настоящему payload, а не по нашей перепечатке. */
const prettyPayload = (text) => {
  try {
    return JSON.stringify(JSON.parse(text), null, 2);
  } catch {
    return null;
  }
};

const payloadText = (call) => {
  const raw = call.result_text || '';
  if (state.payloadView === 'raw') return raw;
  return prettyPayload(raw) ?? raw;
};

/* Задержки на фикстурах — миллисекунды, на живом прогоне — секунды.
   Одна шкала на оба случая читалась бы либо как «0.00 с», либо как «3400 мс». */
const secs = (s) => {
  if (s == null) return '—';
  if (s < 0.0005) return '<1 мс';
  if (s < 1) return `${Math.round(s * 1000)} мс`;
  return `${s.toFixed(s < 10 ? 1 : 0)} с`;
};

const plural = (n, [one, few, many]) => {
  const mod10 = n % 10;
  const mod100 = n % 100;
  if (mod10 === 1 && mod100 !== 11) return one;
  if (mod10 >= 2 && mod10 <= 4 && (mod100 < 12 || mod100 > 14)) return few;
  return many;
};

/* Дельта к базовой линии. На самой базовой линии сравнивать не с чем — там пустая
   строка-заглушка: место она держит, поэтому шапка не прыгает при переключении
   вкладок, но и не сообщает того, что и так видно по выбранной вкладке. */
const NO_DELTA = '<span class="d" aria-hidden="true"></span>';

const delta = (value, isBaseline, unit) => {
  if (isBaseline || !Number.isFinite(value)) return NO_DELTA;
  if (value === 0) return '<span class="d">как в baseline</span>';
  const sign = value > 0 ? '+' : '−';
  return `<span class="d ${value > 0 ? 'good' : 'bad'}">${sign}${Math.abs(value)}${unit}</span>`;
};

/* Which raw strings would prove this claim. Prices need normalising: the answer
   writes "1 301,88 ₽" where the payload holds 1301.88. */
function needles(claim) {
  if (claim.kind !== 'price') return [claim.text];
  const digits = claim.text
    .replace(/[^\d.,\s]/g, '')
    .replace(/\s/g, '')
    .replace(',', '.');
  const n = parseFloat(digits);
  if (Number.isNaN(n)) return [digits];
  const out = [digits, String(n)];
  if (Number.isInteger(n)) out.push(n.toFixed(1), n.toFixed(2));
  return [...new Set(out)];
}

function findEvidence(claim, calls) {
  for (const call of calls) {
    for (const needle of needles(claim)) {
      const at = (call.result_text || '').indexOf(needle);
      if (at !== -1) return { call, needle, at };
    }
  }
  return null;
}

/* Non-overlapping highlight ranges. Longest match wins where two claims collide,
   so a train number inside a longer token never steals its highlight. */
function ranges(answer, claims) {
  const found = [];
  claims.forEach((claim, i) => {
    let from = 0;
    for (;;) {
      const at = answer.indexOf(claim.text, from);
      if (at === -1 || !claim.text) break;
      found.push({ start: at, end: at + claim.text.length, claim, i });
      from = at + claim.text.length;
    }
  });
  found.sort((a, b) => a.start - b.start || b.end - b.start - (a.end - a.start));
  const out = [];
  let last = -1;
  for (const r of found) {
    if (r.start >= last) {
      out.push(r);
      last = r.end;
    }
  }
  return out;
}

function renderAnswer(scn, variant) {
  const answer = scn.answer || '';
  if (!answer) return '<span class="empty">агент не дал текстового ответа</span>';
  const rs = ranges(answer, scn.groundedness.claims);
  let html = '',
    cursor = 0;
  for (const r of rs) {
    html += esc(answer.slice(cursor, r.start));
    html += `<span class="claim ${r.claim.status}" data-claim="${r.i}"
      data-variant="${esc(variant)}" tabindex="0"
      title="${STATUS_RU[r.claim.status]}">${esc(answer.slice(r.start, r.end))}</span>`;
    cursor = r.end;
  }
  return html + esc(answer.slice(cursor));
}

function renderVerdict() {
  const v = variantOf(state.variant);
  const base = DATA.variants[0];
  const m = v.metrics,
    s = v.surface;
  const tok = s ? (s.exact ? s.tokens : `~${s.tokens}`) : '—';

  // Say what THIS tab's number means relative to baseline — the same sentence on
  // both tabs reads as if baseline were saving something against itself.
  const isBase = v.variant === base?.variant;
  const bm = base?.metrics;
  const tokenDelta =
    base?.surface && s && base.surface.tokens
      ? -Math.round((1 - s.tokens / base.surface.tokens) * 100)
      : null;
  const groundedDelta =
    bm && m.groundedness_rate != null && bm.groundedness_rate != null
      ? Math.round((m.groundedness_rate - bm.groundedness_rate) * 100)
      : null;
  const successDelta = bm ? m.successes - bm.successes : null;
  // Fewer is better here, so the sign is flipped for the shared `delta` renderer.
  const fabricatedDelta =
    bm && m.fabricated_claims != null && bm.fabricated_claims != null
      ? bm.fabricated_claims - m.fabricated_claims
      : null;

  const cards = [
    {
      k: 'Поверхность, токенов',
      v: tok,
      d: `<div class="d">${s ? `${s.bytes_} байт` : ''}</div>
          ${delta(tokenDelta, isBase, '%')}`,
    },
    {
      k: 'Groundedness',
      v: pct(m.groundedness_rate),
      d: `<div class="d">${m.grounded_claims}/${m.checkable_claims ?? m.total_claims} ${plural(m.checkable_claims ?? m.total_claims, ['утверждение', 'утверждения', 'утверждений'])}</div>
          ${delta(groundedDelta, isBase, ' пп')}`,
    },
    {
      // The percentage hides this: 4 fabrications out of 191 claims vs 1 out of 185
      // is a fourfold difference that reads as two points. The count is what a user
      // actually receives as wrong facts.
      k: 'Выдумано',
      v: `${m.fabricated_claims ?? 0}`,
      d: `<div class="d">значений, которых нет в ответах сервера</div>
          ${delta(fabricatedDelta, isBase, '')}`,
    },
    {
      k: 'Задач решено',
      v: `${m.successes}<small>/${m.scenarios}</small>`,
      d: `<div class="d">${m.tool_calls} ${plural(m.tool_calls, ['вызов', 'вызова', 'вызовов'])} инструментов</div>
          ${delta(successDelta, isBase, '')}`,
    },
  ];
  document.getElementById('verdict').innerHTML = cards
    .map(
      (c) =>
        `<div class="metric"><div class="k">${c.k}</div><div class="v">${c.v}</div>${c.d}</div>`,
    )
    .join('');
  renderRunBar(v);
}

const big = (v) => `<b>${v}</b>`;

/* Раньше здесь была лента «ключ — число». Числа были верные, но подписи ехали из
   отчёта CLI, где рядом идёт объясняющий абзац, а на экране его нет: получалась
   строка жаргона, где ноль читался как поломка. Теперь тот же набор чисел, но
   предложениями — блок объясняет себя без легенды. */
function costNarrative(m) {
  const tokens = `Диалог с моделью стоил ${big(num(m.input_tokens))} токенов на входе
    и ${big(num(m.output_tokens))} на выходе.`;
  if (m.latency_p50_s == null) return tokens;
  return `${tokens} Половина сценариев уложилась в ${big(secs(m.latency_p50_s))},
    самый долгий — ${big(secs(m.latency_p95_s))}.`;
}

function premiseNarrative(p, scenarios, isBase) {
  // гейт — часть того, чем прокси ОТЛИЧАЕТСЯ от базовой линии; нули в базовой
  // означают «механики нет», а не «сработала вхолостую», и врать об этом нельзя
  if (isBase) {
    return `Базовая линия ходит в MCP напрямую — гейта предпосылок в ней нет.
      Как он срабатывает, видно на вкладке прокси.`;
  }
  const total = `${big(scenarios)} ${plural(scenarios, ['сценарии', 'сценариях', 'сценариях'])}`;
  const fired = p.gate_fires
    ? `Сработал в ${big(p.gate_fires)} ${plural(p.gate_fires, ['сценарии', 'сценариях', 'сценариях'])}
       из ${big(scenarios)}: вернул уточняющий вопрос вместо данных.`
    : `Ни разу не понадобился — в ${total} все параметры были названы явно.`;
  const assumed = p.runs_with_assumptions
    ? `На допущении построено ${big(p.runs_with_assumptions)}
       ${plural(p.runs_with_assumptions, ['ответ', 'ответа', 'ответов'])}, из них
       ${big(p.disclosure_rate == null ? '—' : pct(p.disclosure_rate))} объявлены сразу.`
    : 'Ни один ответ не построен на скрытом допущении.';
  const asked = p.over_asks
    ? `Зря переспросил в ${big(p.over_asks)}
       ${plural(p.over_asks, ['сценарии', 'сценариях', 'сценариях'])}.`
    : 'Лишних переспросов нет.';
  return `${fired} ${assumed} ${asked}`;
}

function renderRunBar(v) {
  const isBase = v.variant === DATA.variants[0]?.variant;
  const blocks = [
    {
      k: 'Цена прогона',
      title: 'токены самого диалога, без схемы инструментов, и время одного сценария',
      text: costNarrative(v.metrics),
    },
    {
      k: 'Гейт предпосылок',
      title:
        'groundedness проверяет выход — что числа из ответа есть в payload. ' +
        'Гейт проверяет вход: не считал ли агент от значения, которого никто не называл',
      text: premiseNarrative(v.premises || {}, v.metrics.scenarios, isBase),
    },
  ];
  document.getElementById('runbar').innerHTML = blocks
    .map(
      (b) => `<div class="note">
        <div class="note-k" tabindex="0">${b.k}<i class="q-mark" aria-hidden="true">?</i>
          <span class="tip" role="tooltip">${esc(b.title)}</span></div>
        <p class="note-t">${b.text}</p></div>`,
    )
    .join('');
}

function renderTabs() {
  const tabs = DATA.variants
    .map(
      (v) =>
        `<button class="tab" role="tab" data-variant="${esc(v.variant)}"
      aria-selected="${!state.overview && v.variant === state.variant}">${esc(v.variant)}</button>`,
    )
    .join('');
  // сравнение — состояние («показывать оба»), а не отдельный экран, поэтому
  // переключатель, а не кнопка: вкладки при нём остаются рабочими
  const compare =
    DATA.variants.length > 1
      ? `<button class="switch" id="cmp-toggle" role="switch" aria-checked="${state.compare}">
           <span class="switch-l">бок о бок</span>
           <span class="switch-track"><span class="switch-thumb"></span></span></button>`
      : '';
  const overview = `<span class="tabs-sep" aria-hidden="true"></span>
    <button class="cmp" id="ov-toggle" aria-pressed="${state.overview}">
    <svg class="cmp-icon" viewBox="0 0 12 12" aria-hidden="true">
      <path d="M0.5 0.5h11v11h-11z M0.5 4h11 M0.5 8h11 M4.5 0.5v11" /></svg>обзор</button>`;
  document.getElementById('tabs').innerHTML = tabs + compare + overview;
}

function scenarioButton(s) {
  const bars =
    s.groundedness.claims.map((c) => `<i class="${c.status}"></i>`).join('') || '<i></i>';
  return `<button class="scn" data-scenario="${esc(s.id)}" aria-current="${s.id === state.scenario}">
      <span class="row"><span class="id">${esc(s.id)}</span>
      <span class="st ${s.success ? 'ok' : 'fail'}">${s.success ? 'ok' : 'fail'}</span></span>
      <span class="bars">${bars}</span></button>`;
}

/* Группы идут в порядке первого появления домена, а не по алфавиту: порядок
   сценариев в прогоне осмыслен, и пересортировка его бы стёрла. Единственная
   группа не получает заголовка — подписывать нечего, если делить не на что. */
function groupByDomain(scenarios) {
  const groups = new Map();
  for (const s of scenarios) {
    const key = s.domain || 'прочее';
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(s);
  }
  return groups;
}

/* Провал считается по тому, что сейчас на экране: на одиночной вкладке — по её
   варианту, в обзоре и сравнении — по любому из них. Иначе на вкладке прокси
   фильтр показывал бы кейсы, которые прокси как раз и починил. */
const isFailing = (id) => {
  const scope = state.overview || state.compare ? DATA.variants : [variantOf(state.variant)];
  return scope.some((v) => v?.scenarios.some((s) => s.id === id && !s.success));
};

function renderNav() {
  const v = variantOf(state.variant);
  const all = v.scenarios;
  const shown = state.failedOnly ? all.filter((s) => isFailing(s.id)) : all;
  document.getElementById('scn-count').textContent = state.failedOnly
    ? `(${shown.length} из ${all.length})`
    : `(${all.length})`;
  document.getElementById('scn-filter').innerHTML =
    `<button class="filter" id="fail-filter" aria-pressed="${state.failedOnly}">
      <svg class="filter-icon" viewBox="0 0 12 12" aria-hidden="true">
        <path d="M0.8 1.6h10.4L7.1 6.6v4.1L4.9 9.2V6.6z" /></svg>только провалы</button>`;
  if (!shown.length) {
    document.getElementById('scenarios').innerHTML =
      '<p class="nav-empty">провалов нет — все сценарии прошли</p>';
    return;
  }
  const groups = groupByDomain(shown);
  const html =
    groups.size < 2
      ? shown.map(scenarioButton).join('')
      : [...groups]
          .map(([domain, items]) => {
            const failed = items.filter((s) => !s.success).length;
            const note = failed
              ? `<span class="grp-fail">${failed} ${plural(failed, ['провал', 'провала', 'провалов'])}</span>`
              : `<span class="grp-n">${items.length}</span>`;
            const open = !state.collapsed.has(domain);
            return `<div class="grp">
              <button class="grp-h" data-domain="${esc(domain)}" aria-expanded="${open}">
                <svg class="grp-caret" viewBox="0 0 8 10" aria-hidden="true">
                  <path d="M1 0.6 L7 5 L1 9.4 Z" /></svg>${esc(domain)}${note}</button>
              ${open ? items.map(scenarioButton).join('') : ''}</div>`;
          })
          .join('');
  document.getElementById('scenarios').innerHTML = html;
}

/* Легенда объясняет только те цвета, которые в этом ответе действительно есть:
   строка про «объявленное допущение» под ответом без единого допущения заставляет
   искать на экране то, чего там нет. */
const LEGEND = [
  ['confirmed', 'подтверждено в tool_result'],
  ['assumed', 'объявленное допущение'],
  ['user_stated', 'условие из запроса пользователя'],
  ['unavailable', 'в данных нет'],
];

function renderLegend(claims) {
  if (!claims.length) {
    return `<div class="legend"><span class="hint">В ответе нет цен, времён, номеров и ссылок —
      подсвечивать нечего. Такой ответ судят проверки сценария ниже: подсветка ловит
      типизированные утверждения, но не каждую выдумку.</span></div>`;
  }
  const present = new Set(claims.map((c) => c.status));
  const items = LEGEND.filter(([status]) => present.has(status))
    .map(([status, label]) => `<span><i class="${status}"></i>${label}</span>`)
    .join('');
  return `<div class="legend">${items}
    <span class="hint">— кликните значение, чтобы увидеть источник</span></div>`;
}

function payloadSwitch(call) {
  const raw = call.result_text || '';
  if (!raw) return '';
  const parses = prettyPayload(raw) !== null;
  const button = (mode, label, title) =>
    `<button class="vs" data-mode="${mode}" ${parses ? '' : 'disabled'}
       aria-selected="${state.payloadView === mode}" title="${esc(title)}">${label}</button>`;
  if (!parses) return '<span class="viewsw"><span class="vs-note">не JSON</span></span>';
  return `<span class="viewsw">
    ${button('pretty', 'форматированный', 'разложить JSON по строкам')}
    ${button('raw', 'raw', 'ровно то, что вернул сервер')}</span>`;
}

/* Переключение режима не перерисовывает панель: перерисовка схлопнула бы
   раскрытые вызовы, а читают их как раз в раскрытом виде. */
function applyPayloadView() {
  const scn = scenarioOf(state.variant, state.scenario);
  if (!scn) return;
  for (const el of document.querySelectorAll('pre.payload')) {
    const call = scn.tool_calls[Number(el.dataset.call)];
    if (call) el.textContent = payloadText(call);
  }
  for (const b of document.querySelectorAll('.vs[data-mode]')) {
    b.setAttribute('aria-selected', String(b.dataset.mode === state.payloadView));
  }
}

function rateLineOf(g) {
  const bad = g.claims.filter((c) => c.status === 'unavailable').length;
  return g.claims.length
    ? `${pct(g.rate)} · ${bad ? `${bad} не подтверждено` : 'всё подтверждено'}`
    : 'проверяемых утверждений нет';
}

function scenarioMeta(scn) {
  return `${scn.turns} ${plural(scn.turns, ['ход', 'хода', 'ходов'])} ·
    ${num(scn.input_tokens)} ↑ · ${num(scn.output_tokens)} ↓ · ${secs(scn.duration_s)}`;
}

function checkRow(c, diff = false) {
  return `<div class="check ${c.passed ? 'pass' : 'fail'}${diff ? ' diff' : ''}">
    <span class="m">${c.passed ? '✓' : '✕'}</span>
    <span class="n">${esc(c.name)}</span><span class="dt">${esc(c.detail || '')}</span></div>`;
}

/* Сравнение бок о бок: один сценарий, два варианта. Ради него всё и затевалось —
   разницу между «придумал цену» и «взял цену из ответа» видно, только когда оба
   ответа на экране одновременно. Проверки сопоставляются по имени, и те, где
   варианты разошлись, помечены — это и есть ответ на вопрос «что дал прокси». */
function renderCompare() {
  const pairs = DATA.variants
    .map((v) => ({ variant: v.variant, scn: scenarioOf(v.variant, state.scenario) }))
    .filter((p) => p.scn);
  const names = [...new Set(pairs.flatMap((p) => p.scn.checks.map((c) => c.name)))];
  const verdictOf = (scn, name) => scn.checks.find((c) => c.name === name);
  const differs = (name) => new Set(pairs.map((p) => verdictOf(p.scn, name)?.passed)).size > 1;

  const answers = pairs
    .map(
      ({ variant, scn }) => `<div class="cmp-col">
        <div class="cmp-h"><span class="cmp-name">${esc(variant)}</span>
          <span class="rate">${rateLineOf(scn.groundedness)}</span></div>
        <div class="cmp-meta">${scenarioMeta(scn)}</div>
        <div class="answer-wrap"><div class="answer">${renderAnswer(scn, variant)}</div></div>
      </div>`,
    )
    .join('');

  const checks = pairs
    .map(({ variant, scn }) => {
      const rows = names
        .map((name) => {
          const c = verdictOf(scn, name);
          return c
            ? checkRow(c, differs(name))
            : `<div class="check absent"><span class="m">·</span>
               <span class="n">${esc(name)}</span><span class="dt">проверка не выполнялась</span></div>`;
        })
        .join('');
      return `<div class="cmp-col">
        <div class="cmp-sub">${esc(variant)}</div>
        <div class="checks">${rows}</div></div>`;
    })
    .join('');

  // одна легенда на оба ответа: цвета в колонках означают одно и то же, а два
  // одинаковых списка рядом читаются как будто чем-то различаются
  const allClaims = pairs.flatMap((p) => p.scn.groundedness.claims);

  // проверки живут в отдельной сетке, поэтому начинаются на одной высоте, какой
  // бы длины ни были ответы — иначе сравнивать их пришлось бы, водя глазами
  return `<div class="cmp-grid">${answers}</div>
    ${renderLegend(allClaims)}
    <div class="h2">Проверки сценария</div>
    <div class="cmp-grid cmp-grid-checks">${checks}</div>`;
}

/* Обзорная матрица: сценарии × варианты. В отчёте CLI она есть (_render_scenario_matrix),
   а на экране её не было — общую картину приходилось собирать, открывая трейсы по одному. */
function renderOverview() {
  const ids = [...new Set(DATA.variants.flatMap((v) => v.scenarios.map((s) => s.id)))];
  const shown = state.failedOnly ? ids.filter(isFailing) : ids;
  const head = DATA.variants.map((v) => `<th scope="col">${esc(v.variant)}</th>`).join('');

  const cell = (variant, id) => {
    const scn = variantOf(variant)?.scenarios.find((s) => s.id === id);
    if (!scn) return '<td class="mx-cell"><span class="mx-none">—</span></td>';
    const failed = scn.checks.filter((c) => !c.passed).length;
    const mark = scn.failure
      ? { cls: 'err', text: 'ошибка' }
      : scn.success
        ? { cls: 'ok', text: 'ok' }
        : { cls: 'bad', text: `${failed} ${plural(failed, ['провал', 'провала', 'провалов'])}` };
    const rate = scn.groundedness.claims.length ? pct(scn.groundedness.rate) : '—';
    return `<td class="mx-cell"><button class="mx-btn ${mark.cls}"
      data-variant="${esc(variant)}" data-scenario="${esc(id)}">
      <span class="mx-mark">${mark.text}</span><span class="mx-rate">${rate}</span></button></td>`;
  };

  const rows = shown
    .map((id) => {
      const any = DATA.variants.flatMap((v) => v.scenarios).find((s) => s.id === id);
      return `<tr><th scope="row"><span class="mx-id">${esc(id)}</span>
        ${any?.domain ? `<span class="case-domain">${esc(any.domain)}</span>` : ''}</th>
        ${DATA.variants.map((v) => cell(v.variant, id)).join('')}</tr>`;
    })
    .join('');

  const totals = DATA.variants
    .map((v) => {
      const m = v.metrics;
      return `<td class="mx-cell mx-total">${m.successes}/${m.scenarios} · ${pct(m.groundedness_rate)}</td>`;
    })
    .join('');

  return `<div class="fade">
    <div class="h2 mx-h2">Все сценарии</div>
    <p class="mx-note">Клетка — один трейс: статус проверок и доля подтверждённых утверждений.
      Нажмите, чтобы открыть разбор.</p>
    <table class="mx">
      <thead><tr><th scope="col">сценарий</th>${head}</tr></thead>
      <tbody>${rows}</tbody>
      <tfoot><tr><th scope="row">решено · groundedness</th>${totals}</tr></tfoot>
    </table>
  </div>`;
}

function renderPane() {
  if (state.overview) {
    document.getElementById('pane').innerHTML = renderOverview();
    return;
  }
  const v = variantOf(state.variant);
  const scn = v.scenarios.find((s) => s.id === state.scenario) || v.scenarios[0];
  if (!scn) {
    document.getElementById('pane').innerHTML = '<p>нет данных</p>';
    return;
  }
  state.scenario = scn.id;

  const g = scn.groundedness;
  const rateLine = rateLineOf(g);
  const checks = scn.checks.map((c) => checkRow(c)).join('');

  const calls =
    scn.tool_calls
      .map(
        (c, i) => `
    <details class="call">
      <summary><span class="nm">${esc(c.name)}</span>
        <span class="meta">${c.fixture_miss ? 'нет фикстуры · ' : ''}${c.is_error ? 'ошибка · ' : ''}${(c.result_text || '').length} симв.</span>
      </summary>
      <div class="body">
        <div class="lbl-sm">Аргументы</div>
        <pre>${esc(JSON.stringify(c.arguments, null, 2))}</pre>
        <div class="lbl-sm">Ответ сервера${payloadSwitch(c)}</div>
        <pre class="payload" data-call="${i}">${esc(payloadText(c))}</pre>
      </div>
    </details>`,
      )
      .join('') || '<p style="color:var(--paper-faint)">инструменты не вызывались</p>';

  const failNote = scn.failure
    ? `<div class="note-fail">
        <span class="lbl">Прогон не завершился</span>${esc(scn.failure)}</div>`
    : '';

  const single = `
      <div class="answer-wrap">
        <div class="answer-head"><span>Ответ агента</span>
          <span class="scn-meta">${scenarioMeta(scn)}</span>
          <span class="rate">${rateLine}</span></div>
        <div class="answer" id="answer">${renderAnswer(scn, state.variant)}</div>
      </div>
      ${renderLegend(g.claims)}
      <div class="h2">Проверки сценария</div>
      <div class="checks">${checks}</div>
      <div class="h2">Вызовы инструментов</div>
      ${calls}`;

  document.getElementById('pane').innerHTML = `
    <div class="fade">
      <div class="case">
        <div class="case-h">
          <span class="case-id">${esc(scn.id)}</span>
          ${scn.domain ? `<span class="case-domain">${esc(scn.domain)}</span>` : ''}
        </div>
        ${scn.probes ? `<p class="case-note">${esc(scn.probes)}</p>` : ''}
      </div>
      <div class="q"><span class="lbl">Запрос пользователя</span>
        <div class="bubble">${esc(scn.request)}</div></div>
      ${failNote}
      ${state.compare ? renderCompare() : single}
    </div>`;
}

/* Фрагмент вокруг найденного значения. В сыром тексте вырезаем окно по символам,
   в форматированном — по строкам: JSON с отступами наполовину состоит из пробелов,
   и то же окно в 260 символов показало бы почти пустоту. */
const CHARS_AROUND = 260;
const LINES_AROUND = 6;
/* Каждое нажатие удваивает окно. Линейный шаг выглядел разумно, пока не считаешь:
   ответ на поиск — это тысяча с лишним строк, и до конца пришлось бы кликать
   десятки раз. Рядом «весь ответ» — для тех, кому нужно сразу целиком. */
const windowSize = (base) => (state.evidenceAll ? Infinity : base * 2 ** state.evidenceSteps);
const charWindow = () => windowSize(CHARS_AROUND);
const lineWindow = () => windowSize(LINES_AROUND);

function markup(before, needle, after, hiddenHead, hiddenTail, unit) {
  const hidden = hiddenHead + hiddenTail;
  const more = hidden
    ? `<div class="more-row">
         <button class="more" id="d-more">показать ещё
           <span class="more-n">скрыто ${num(hidden)} ${plural(hidden, unit)}</span></button>
         <button class="more" id="d-all">весь ответ</button>
       </div>`
    : '';
  return `<pre>${hiddenHead ? '…' : ''}${esc(before)}<mark>${esc(needle)}</mark>${esc(after)}${hiddenTail ? '…' : ''}</pre>${more}`;
}

function evidenceFragment(ev) {
  const raw = ev.call.result_text;
  const pretty = state.payloadView === 'raw' ? null : prettyPayload(raw);
  if (pretty === null) {
    const from = Math.max(0, ev.at - charWindow());
    const tailAt = ev.at + ev.needle.length;
    const to = Math.min(raw.length, tailAt + charWindow());
    return markup(raw.slice(from, ev.at), ev.needle, raw.slice(tailAt, to), from, raw.length - to, [
      'символ',
      'символа',
      'символов',
    ]);
  }

  const at = pretty.indexOf(ev.needle);
  if (at === -1) {
    // форматирование разорвало искомую строку — честнее сказать, чем показать «не найдено»
    return `<pre>${esc(pretty.slice(0, 900))}…</pre>
      <div class="lbl-sm">точное место видно в режиме raw</div>`;
  }

  const lines = pretty.split('\n');
  let start = 0;
  let idx = 0;
  while (idx < lines.length && at >= start + lines[idx].length + 1) {
    start += lines[idx].length + 1;
    idx += 1;
  }
  const from = Math.max(0, idx - lineWindow());
  const to = Math.min(lines.length, idx + lineWindow() + 1);
  const head = lines.slice(from, idx).join('\n');
  const line = lines[idx];
  const inLine = at - start;
  const tail = lines.slice(idx + 1, to).join('\n');
  return markup(
    (head ? `${head}\n` : '') + line.slice(0, inLine),
    ev.needle,
    line.slice(inLine + ev.needle.length) + (tail ? `\n${tail}` : ''),
    from,
    lines.length - to,
    ['строка', 'строки', 'строк'],
  );
}

function openDrawer(claimIndex, resetWindow = true) {
  state.claim = claimIndex;
  if (resetWindow) {
    state.evidenceSteps = 0;
    state.evidenceAll = false;
  }
  const scn = scenarioOf(state.claimVariant || state.variant, state.scenario);
  const claim = scn.groundedness.claims[claimIndex];
  const ev = findEvidence(claim, scn.tool_calls);

  document.querySelectorAll('.claim.sel').forEach((e) => {
    e.classList.remove('sel');
  });
  document.querySelector(`.claim[data-claim="${claimIndex}"]`)?.classList.add('sel');

  const pill = document.getElementById('d-pill');
  pill.className = `pill ${claim.status}`;
  pill.textContent = STATUS_RU[claim.status];

  let body;
  if (ev) {
    body = `<div class="verdict-line">Значение <b>${esc(claim.text)}</b> найдено в ответе
        <b>${esc(ev.call.name)}</b>, позиция ${ev.at}.</div>
      <div class="lbl-sm">Фрагмент ответа сервера${payloadSwitch(ev.call)}</div>
      ${evidenceFragment(ev)}`;
  } else if (claim.status === 'user_stated') {
    body = `<div class="verdict-line">Значение <b>${esc(claim.text)}</b> назвал сам пользователь —
      это условие запроса, процитированное в ответе. Сервер его подтверждать не обязан,
      и выдумкой оно не является.</div>
      <div class="lbl-sm">Запрос сценария</div><pre>${esc(scn.request)}</pre>`;
  } else if (claim.status === 'assumed') {
    body = `<div class="verdict-line">Значение <b>${esc(claim.text)}</b> сервер не возвращал — это
      допущение, которое агент объявил заранее.</div>
      ${(scn.groundedness.assumptions || []).map((a) => `<pre>${esc(a)}</pre>`).join('')}`;
  } else {
    body = `<div class="verdict-line">Значение <b>${esc(claim.text)}</b> не найдено ни в одном
      ответе сервера за этот ход. Агент его придумал.</div>
      <div class="lbl-sm">Проверено вызовов: ${scn.tool_calls.length}</div>`;
  }
  document.getElementById('d-body').innerHTML = body;
  document.getElementById('drawer').classList.add('open');
}

function closeDrawer() {
  document.getElementById('drawer').classList.remove('open');
  document.querySelectorAll('.claim.sel').forEach((e) => {
    e.classList.remove('sel');
  });
}

document.addEventListener('click', (e) => {
  const tab = e.target.closest('.tab');
  if (tab) {
    state.variant = tab.dataset.variant;
    state.overview = false;
    closeDrawer();
    render();
    return;
  }
  if (e.target.closest('#fail-filter')) {
    state.failedOnly = !state.failedOnly;
    render();
    return;
  }
  if (e.target.closest('#ov-toggle')) {
    state.overview = !state.overview;
    if (state.overview) state.compare = false;
    closeDrawer();
    render();
    return;
  }
  const mxBtn = e.target.closest('.mx-btn');
  if (mxBtn) {
    // из матрицы уходим прямо в разбор той клетки, по которой нажали
    state.overview = false;
    state.variant = mxBtn.dataset.variant;
    state.scenario = mxBtn.dataset.scenario;
    render();
    return;
  }
  if (e.target.closest('#cmp-toggle')) {
    state.compare = !state.compare;
    state.overview = false;
    closeDrawer();
    render();
    return;
  }
  const group = e.target.closest('.grp-h');
  if (group) {
    const { domain } = group.dataset;
    // свёрнутая группа не меняет выбранный трейс: он остаётся открытым справа
    if (state.collapsed.has(domain)) state.collapsed.delete(domain);
    else state.collapsed.add(domain);
    renderNav();
    return;
  }
  const scn = e.target.closest('.scn');
  if (scn) {
    state.scenario = scn.dataset.scenario;
    closeDrawer();
    render();
    return;
  }
  const view = e.target.closest('.vs[data-mode]');
  if (view) {
    state.payloadView = view.dataset.mode;
    applyPayloadView();
    // ящик перерисовываем целиком: там меняется не только текст, но и само окно
    if (state.claim != null && document.getElementById('drawer').classList.contains('open')) {
      openDrawer(state.claim, false);
    }
    return;
  }
  if (e.target.closest('#d-more')) {
    state.evidenceSteps += 1;
    openDrawer(state.claim, false);
    return;
  }
  if (e.target.closest('#d-all')) {
    state.evidenceAll = true;
    openDrawer(state.claim, false);
    return;
  }
  const claim = e.target.closest('.claim');
  if (claim) {
    state.claimVariant = claim.dataset.variant || state.variant;
    openDrawer(Number(claim.dataset.claim));
    return;
  }
  if (e.target.id === 'd-close') closeDrawer();
});
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') closeDrawer();
  if (e.key === 'Enter' && e.target.classList?.contains('claim')) {
    state.claimVariant = e.target.dataset.variant || state.variant;
    openDrawer(Number(e.target.dataset.claim));
  }
});

/* Хэш вида #proxy/rail_cheapest: перезагрузка возвращает на тот же экран,
   а конкретный трейс можно открыть ссылкой, не ища его руками. */
function readHash() {
  const raw = decodeURIComponent(location.hash.slice(1));
  if (!raw) return false;
  const slash = raw.indexOf('/');
  const variant = slash === -1 ? raw : raw.slice(0, slash);
  const scenario = slash === -1 ? null : raw.slice(slash + 1);
  if (variant === 'overview') {
    state.overview = true;
    return true;
  }
  if (variant === 'compare') {
    state.compare = true;
    if (scenario) {
      const known = DATA.variants.some((x) => x.scenarios.some((s) => s.id === scenario));
      if (known) state.scenario = scenario;
    }
    return true;
  }
  const v = variantOf(variant);
  if (!v) return false;
  state.compare = false;
  state.variant = variant;
  if (scenario && v.scenarios.some((s) => s.id === scenario)) state.scenario = scenario;
  return true;
}

function writeHash() {
  if (state.overview) {
    setHash('#overview');
    return;
  }
  if (!state.scenario) return;
  const head = state.compare ? 'compare' : encodeURIComponent(state.variant);
  setHash(`#${head}/${encodeURIComponent(state.scenario)}`);
}

function setHash(hash) {
  if (location.hash === hash) return;
  try {
    history.replaceState(null, '', hash);
  } catch {
    // file:// в части браузеров запрещает replaceState — тогда пишем напрямую
    location.hash = hash;
  }
}

window.addEventListener('hashchange', () => {
  if (readHash()) {
    closeDrawer();
    render();
  }
});

function render() {
  // nav рисуется раньше pane, поэтому выбранный сценарий фиксируем до отрисовки —
  // иначе на первой загрузке и после смены варианта подсветка в списке пустая
  const scenarios = variantOf(state.variant)?.scenarios ?? [];
  if (!scenarios.some((s) => s.id === state.scenario)) {
    state.scenario = scenarios[0]?.id ?? null;
  }
  renderTabs();
  renderVerdict();
  renderNav();
  renderPane();
  writeHash();
}

const tag = document.getElementById('agent-tag');
const synthetic = /^(demo|scripted)/.test(DATA.agent || '');
tag.className = `agent-tag${synthetic ? ' synthetic' : ''}`;
tag.innerHTML = `${(synthetic ? 'НЕ ЗАМЕР · агент: <b>' : 'агент: <b>') + esc(DATA.agent || '—')}</b>`;

readHash();
render();
