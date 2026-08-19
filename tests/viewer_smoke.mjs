/* Прогоняет `viewer/app.js` из СОБРАННОЙ страницы поверх минимальной заглушки DOM.

   Смысл — контракт между `evals/report.py` и вьювером. Переименованное в дампе
   поле не роняет страницу: она рисуется с `undefined` в клетке метрики, и узнать
   об этом можно только глазами и только после деплоя. Здесь тот же скрипт
   отрисовывает каждый вариант, каждый сценарий и каждое утверждение, а разметку
   забирает питон и проверяет, что дыр в ней нет.

   Использование: node tests/viewer_smoke.mjs <trace-viewer.html>
   На stdout — JSON `{html, ids}`; ненулевой код возврата означает исключение. */

import { readFileSync } from 'node:fs';
import vm from 'node:vm';

const html = readFileSync(process.argv[2], 'utf8');

const data = html.match(/<script id="trace-data" type="application\/json">([\s\S]*?)<\/script>/);
if (!data) throw new Error('в собранной странице нет блока trace-data');

const scripts = [...html.matchAll(/<script(?![^>]*\bid=)[^>]*>([\s\S]*?)<\/script>/g)];
if (scripts.length !== 1)
  throw new Error(`ожидался один скрипт приложения, найдено ${scripts.length}`);

const elements = new Map();
const makeElement = (id) => ({
  id,
  className: '',
  textContent: '',
  dataset: {},
  style: {},
  classList: {
    _set: new Set(),
    add(c) {
      this._set.add(c);
    },
    remove(c) {
      this._set.delete(c);
    },
    contains(c) {
      return this._set.has(c);
    },
  },
  addEventListener() {},
  closest: () => null,
  focus() {},
  scrollIntoView() {},
  querySelector: () => null,
  querySelectorAll: () => [],
  set innerHTML(value) {
    this._html = String(value);
  },
  get innerHTML() {
    return this._html ?? '';
  },
});

const document = {
  getElementById(id) {
    if (!elements.has(id)) {
      const el = makeElement(id);
      if (id === 'trace-data') el.textContent = data[1];
      elements.set(id, el);
    }
    return elements.get(id);
  },
  querySelector: () => null,
  querySelectorAll: () => [],
  addEventListener() {},
};

const context = vm.createContext({
  document,
  window: { addEventListener() {} },
  location: { hash: '' },
  history: { replaceState() {} },
  console,
  JSON,
  Math,
  Infinity,
  Number,
  String,
  Set,
  Map,
  Array,
  Object,
});

vm.runInContext(scripts[0][1], context, { filename: 'app.js' });

/* Стартовый render() рисует только первый вариант и первый сценарий. Остальное —
   вкладки, обзор, сравнение, ящик доказательств по каждому утверждению — трогает
   те поля дампа, до которых первый экран не доходит. */
const probe = `
const seen = [];
const dump = () => seen.push(...[...elementsHtml()]);
function* elementsHtml() {
  for (const id of ['agent-tag', 'tabs', 'verdict', 'runbar', 'scn-count', 'scn-filter', 'scenarios', 'pane', 'd-body', 'd-pill']) {
    const el = document.getElementById(id);
    yield el.innerHTML;
    yield el.textContent;
  }
}
const ids = [];
for (const v of DATA.variants) {
  state.variant = v.variant;
  state.compare = false;
  state.overview = false;
  for (const s of v.scenarios) {
    ids.push(v.variant + '/' + s.id);
    state.scenario = s.id;
    state.claimVariant = v.variant;
    render();
    dump();
    s.groundedness.claims.forEach((_, i) => {
      openDrawer(i);
      dump();
      state.payloadView = state.payloadView === 'raw' ? 'pretty' : 'raw';
      openDrawer(i, false);
      dump();
    });
    state.failedOnly = true;
    render();
    dump();
    state.failedOnly = false;
    state.compare = true;
    render();
    dump();
    state.compare = false;
  }
}
state.overview = true;
render();
dump();
JSON.stringify({ html: seen.join('\\n'), ids });
`;

process.stdout.write(vm.runInContext(probe, context, { filename: 'probe.js' }));
