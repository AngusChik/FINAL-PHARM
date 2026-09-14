const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

// Run the real shared script in browser lifecycle order. Cached history never
// fires DOMContentLoaded; popstate precedes queued restoration timers.
const source = readFileSync(path.resolve(__dirname, '../page_scroll.js'), 'utf8');
const ORIGIN = 'https://pharmacy.test';
const INVENTORY = '/inventory/';
const PREFIX = 'scroll:page:v1:';
const PREVIOUS = 'scroll:last-visible-page';
const saved = (y, containers = {}) => JSON.stringify({ y, containers });

test('Inventory return links preserve pagination for reordered or duplicate categories only', () => {
  const template = readFileSync(path.resolve(__dirname, '../../../app/templates/inventory_display.html'), 'utf8');
  const start = template.indexOf('function currentInventoryReturnUrl()');
  const end = template.indexOf('function syncStockFilterSummary', start);
  assert.ok(start >= 0 && end > start, 'Run the actual Inventory return-link builder');

  for (const categoryQuery of [
    'category_id=2&category_id=1',
    'category_id=2&category_id=1&category_id=2',
  ]) {
    const location = new URL(INVENTORY + '?' + categoryQuery + '&page=3&sort=name&direction=desc&q=cream&stock_filter=expired#stock', ORIGIN);
    const context = {
      URLSearchParams,
      window: { location },
      lookupInput: { value: 'cream' },
      checkedCats: () => ['1', '2'],
      selectedStockFilter: () => 'expired',
    };
    vm.createContext(context);
    vm.runInContext(template.slice(start, end), context);
    const unchanged = new URL(context.currentInventoryReturnUrl(), ORIGIN);
    assert.equal(unchanged.searchParams.get('page'), '3', categoryQuery);
    assert.deepEqual(unchanged.searchParams.getAll('category_id'), ['1', '2']);
    assert.equal(unchanged.searchParams.get('sort'), 'name');
    assert.equal(unchanged.searchParams.get('direction'), 'desc');
    assert.equal(unchanged.searchParams.get('q'), 'cream');
    assert.equal(unchanged.searchParams.get('stock_filter'), 'expired');
    assert.equal(unchanged.hash, '#stock');

    for (const changedCategories of [['1'], ['1', '3'], ['1', '2', '3'], []]) {
      context.checkedCats = () => changedCategories;
      const changed = new URL(context.currentInventoryReturnUrl(), ORIGIN);
      assert.equal(changed.searchParams.has('page'), false, changedCategories.join(','));
      assert.deepEqual(changed.searchParams.getAll('category_id'), changedCategories);
    }
  }
});

function events() {
  const handlers = new Map();
  return {
    addEventListener(name, callback) {
      if (!handlers.has(name)) handlers.set(name, []);
      handlers.get(name).push(callback);
    },
    removeEventListener(name, callback) {
      handlers.set(name, (handlers.get(name) || []).filter((item) => item !== callback));
    },
    dispatch(name, event = {}) {
      event.type = name;
      event.defaultPrevented ??= false;
      event.preventDefault ??= () => { event.defaultPrevented = true; };
      for (const handler of handlers.get(name) || []) handler(event);
      return event;
    },
  };
}

function page({ url = INVENTORY, referrer = '', navigation = 'navigate',
  store = new Map(), blocked = false, initialY = 0, parentPage = null,
  parentPath = '/dashboard/', opener = null, historyLength = 1, tables = [],
  historyState = null, historyBlocked = false } = {}) {
  const location = new URL(url, ORIGIN);
  const timers = new Map();
  const observers = new Set();
  let timerId = 0;
  const scrolls = [];
  const scrollBehaviors = [];
  const document = {
    ...events(), referrer, readyState: 'loading',
    documentElement: { scrollTop: initialY }, body: { scrollTop: initialY, style: {} },
    querySelector() { return null; },
    querySelectorAll(selector) {
      assert.equal(selector, '#main-content table');
      return tables;
    },
  };
  const window = {
    ...events(), location, opener, scrollY: initialY, pageYOffset: initialY,
    getComputedStyle(element) {
      return { overflowX: 'visible', overflowY: 'visible', ...element.style };
    },
    scrollTo(x, y) {
      scrollBehaviors.push(typeof x === 'object' ? x.behavior : 'auto');
      if (typeof x === 'object') y = x.top;
      scrolls.push([0, y]);
      window.scrollY = window.pageYOffset = y;
      document.documentElement.scrollTop = document.body.scrollTop = y;
    },
  };
  window.self = window;
  window.parent = parentPage?.window || (parentPage === false
    ? { location: new URL(parentPath, ORIGIN) } : window);
  window.top = window.parent;
  const history = { scrollRestoration: 'auto', length: historyLength, state: historyState,
    replaceState(state) {
      if (historyBlocked) throw new Error('History writes blocked');
      this.state = JSON.parse(JSON.stringify(state));
    } };
  const sessionStorage = {
    getItem(key) {
      if (blocked) throw new Error('Storage blocked');
      return store.has(key) ? store.get(key) : null;
    },
    setItem(key, value) {
      if (blocked) throw new Error('Storage blocked');
      store.set(key, String(value));
    },
    removeItem(key) {
      if (blocked) throw new Error('Storage blocked');
      store.delete(key);
    },
  };
  const performance = { getEntriesByType: () => navigation ? [{ type: navigation }] : [] };
  const setTimeout = (callback) => { timers.set(++timerId, callback); return timerId; };
  const clearTimeout = (id) => timers.delete(id);
  class MutationObserver {
    constructor(callback) { this.callback = callback; }
    observe(target) { this.target = target; observers.add(this); }
    disconnect() { observers.delete(this); }
  }
  Object.assign(window, { document, history, sessionStorage, performance, setTimeout,
    clearTimeout, requestAnimationFrame: setTimeout, cancelAnimationFrame: clearTimeout });
  const context = vm.createContext({ window, self: window, document, history, location,
    sessionStorage, URL, performance, setTimeout, clearTimeout,
    requestAnimationFrame: setTimeout, cancelAnimationFrame: clearTimeout, MutationObserver, console });
  vm.runInContext(source, context, { filename: 'page_scroll.js' });
  const result = {
    window, document, history, store, scrolls, scrollBehaviors,
    flush() {
      let iterations = 0;
      while (timers.size) {
        assert.ok(++iterations < 100, 'Restoration timers should settle');
        const [id, callback] = timers.entries().next().value;
        timers.delete(id);
        callback();
      }
    },
    load({ flush = true } = {}) {
      document.readyState = 'interactive';
      document.dispatch('DOMContentLoaded');
      document.readyState = 'complete';
      window.dispatch('pageshow', { persisted: false });
      if (flush) result.flush();
      return result;
    },
    hide() { window.dispatch('pagehide', { persisted: true }); },
    unlockBody(oldOffset) {
      document.body.style.position = '';
      document.body.style.top = '';
      // Modal close handlers restore their captured offset synchronously;
      // MutationObserver notifications run only after that handler finishes.
      window.scrollTo(0, oldOffset);
      for (const observer of [...observers]) observer.callback([{ target: document.body }]);
    },
    showCached({ flush = true } = {}) {
      window.dispatch('pageshow', { persisted: true });
      if (flush) result.flush();
    },
    container(name, top = 0, left = 0) {
      const element = { ...events(), scrollTop: top, scrollLeft: left };
      window.PageScroll.registerContainer(name, element);
      return element;
    },
    submit({ excluded = false, prevented = false } = {}) {
      const form = {
        tagName: 'FORM', matches: () => excluded,
        hasAttribute: (name) => name === 'data-no-scroll-restore' && excluded,
        getAttribute: (name) => name === 'data-no-scroll-restore' && excluded ? '' : null,
        closest: () => form, dataset: excluded ? { noScrollRestore: '' } : {},
      };
      const event = document.dispatch('submit', { target: form });
      // A later form handler can cancel after the central submit listener.
      if (prevented) event.preventDefault();
      result.flush();
      return event;
    },
  };
  return result;
}

function tableFixture({ id = 'rpTable', wrapperId = '', dialog = false, scrollable = true } = {}) {
  const main = { id: 'main-content', parentElement: null, style: {} };
  const wrapper = { id: wrapperId, parentElement: main, scrollTop: 0, scrollLeft: 0,
    style: { overflowX: scrollable ? 'auto' : 'visible', overflowY: scrollable ? 'auto' : 'visible' } };
  const table = { id, parentElement: wrapper, getAttribute: () => null,
    closest(selector) {
      const ancestorSelectors = selector.split(',').map((part) => part.trim());
      return dialog && ancestorSelectors.includes('[role="dialog"]') ? {} : null;
    } };
  return { table, wrapper, main };
}

for (const pathname of [INVENTORY, '/dashboard/', '/orders/', '/ordering-sheet/',
  '/check-in/', '/sales-analytics/', '/daily-report/', '/product/41/']) {
  test(`${pathname}: query navigation preserves page and container positions`, () => {
    const first = page({ url: pathname }).load();
    const grid = first.container('table');
    first.window.scrollY = 825;
    grid.scrollTop = 420;
    grid.scrollLeft = 125;
    first.hide();
    assert.deepEqual(JSON.parse(first.store.get(PREFIX + pathname)), {
      y: 825, containers: { table: { top: 420, left: 125 } },
    });
    const sorted = page({ url: pathname + '?sort=name',
      referrer: first.window.location.href, store: first.store });
    const sortedGrid = sorted.container('table');
    sorted.load();
    assert.equal(sorted.window.scrollY, 825);
    assert.equal(sortedGrid.scrollTop, 420);
    assert.equal(sortedGrid.scrollLeft, 125);
    assert.equal(sorted.history.scrollRestoration, 'manual');
  });
  test(`${pathname}: cross-page entry clears stale positions and starts at top`, () => {
    const store = new Map([[PREFIX + pathname, saved(925, { table: { top: 420, left: 125 } })],
      [PREVIOUS, pathname]]);
    const current = page({ url: pathname, referrer: ORIGIN + '/different-page/', store, initialY: 925 });
    const grid = current.container('table', 420, 125);
    current.load();
    assert.equal(current.window.scrollY, 0);
    assert.equal(grid.scrollTop, 0);
    assert.equal(grid.scrollLeft, 0);
    assert.equal(store.has(PREFIX + pathname), false);
  });
}

for (const referrer of ['', 'https://elsewhere.test/inventory/']) {
  test(`Direct or external entry (${referrer || 'no referrer'}) ignores stale positions`, () => {
    const store = new Map([[PREFIX + INVENTORY, saved(900)], [PREVIOUS, INVENTORY]]);
    const current = page({ referrer, store, initialY: 900 }).load();
    assert.equal(current.window.scrollY, 0);
    assert.equal(store.has(PREFIX + INVENTORY), false);
  });
}

test('Reload preserves the latest position despite a different original referrer', () => {
  const first = page({ referrer: ORIGIN + '/dashboard/' }).load();
  first.window.scrollY = 678;
  first.hide();
  const reloaded = page({ referrer: first.document.referrer, navigation: 'reload', store: first.store }).load();
  assert.equal(reloaded.window.scrollY, 678);
});

test('Opening a link without leaving does not save or change the visible-page marker', () => {
  const current = page().load();
  current.window.scrollY = 350;
  const link = { target: '_blank', href: ORIGIN + '/dashboard/', getAttribute: () => '/dashboard/' };
  current.document.dispatch('click', { target: { closest: () => link } });
  assert.equal(current.store.get(PREVIOUS), INVENTORY);
  assert.equal(current.store.has(PREFIX + INVENTORY), false);
});

test('A new tab with an opener ignores positions inherited from its source tab', () => {
  const store = new Map([[PREFIX + INVENTORY, saved(715)], [PREVIOUS, INVENTORY]]);
  const current = page({ referrer: ORIGIN + INVENTORY, store, opener: {}, initialY: 715 }).load();
  assert.equal(current.window.scrollY, 0);
  assert.equal(store.has(PREFIX + INVENTORY), false);
});

test('An opener does not prevent preserving the position on reload', () => {
  const current = page({ navigation: 'reload', opener: {},
    store: new Map([[PREFIX + INVENTORY, saved(715)]]) }).load();
  assert.equal(current.window.scrollY, 715);
});

test('Later same-page navigation in a tab with an opener preserves its own position', () => {
  const current = page({ referrer: ORIGIN + INVENTORY, opener: {}, historyLength: 3,
    store: new Map([[PREFIX + INVENTORY, saved(715)]]) }).load();
  assert.equal(current.window.scrollY, 715);
});

test('Reload with a modal open preserves the underlying position rather than frozen window zero', () => {
  const first = page().load();
  first.document.body.style.position = 'fixed';
  first.document.body.style.top = '-840.5px';
  first.window.scrollY = 0;
  first.hide();
  assert.equal(JSON.parse(first.store.get(PREFIX + INVENTORY)).y, 840.5);
  const reloaded = page({ navigation: 'reload', store: first.store }).load();
  assert.equal(reloaded.window.scrollY, 840.5);
});

test('A cross-page cached return with a modal open keeps its original position after closing', () => {
  const current = page().load();
  current.document.body.style.position = 'fixed';
  current.document.body.style.top = '-840px';
  current.window.scrollY = 0;
  current.hide();
  current.store.set(PREVIOUS, '/dashboard/');
  current.showCached();
  assert.equal(current.document.body.style.top, '-840px');
  current.unlockBody(840);
  assert.equal(current.window.scrollY, 840);
});

test('A cached modal restores its own history entry instead of another visit to the same page', () => {
  const current = page().load();
  current.document.body.style.position = 'fixed';
  current.document.body.style.top = '-400px';
  current.window.scrollY = 0;
  current.hide();
  current.store.set(PREFIX + INVENTORY, saved(650));
  current.showCached();
  assert.equal(current.document.body.style.top, '-400px');
  current.unlockBody(400);
  assert.equal(current.window.scrollY, 400);
});

test('Cached history from another page preserves the document and registered containers', () => {
  const inventory = page().load();
  const grid = inventory.container('table');
  inventory.window.scrollY = 960;
  grid.scrollTop = 410;
  grid.scrollLeft = 50;
  inventory.hide();
  const other = page({ url: '/dashboard/', store: inventory.store }).load();
  other.hide();
  inventory.showCached();
  assert.equal(inventory.window.scrollY, 960);
  assert.equal(grid.scrollTop, 410);
  assert.equal(grid.scrollLeft, 50);
  assert.equal(inventory.store.has(PREFIX + INVENTORY), true);
  assert.equal(inventory.store.get(PREVIOUS), INVENTORY);
});

test('Back and Forward between queries restore each entry independently', () => {
  const first = page({ url: '/inventory/?page=1' }).load();
  const firstGrid = first.container('table');
  first.window.scrollY = 400;
  firstGrid.scrollTop = 200;
  first.hide();
  const second = page({ url: '/inventory/?page=2', referrer: first.window.location.href, store: first.store }).load();
  const secondGrid = second.container('table');
  second.window.scrollY = 855;
  secondGrid.scrollTop = 650;
  secondGrid.scrollLeft = 70;
  second.hide();
  first.showCached();
  assert.equal(first.window.scrollY, 400);
  assert.equal(firstGrid.scrollTop, 200);
  assert.equal(firstGrid.scrollLeft, 0);
  first.hide();
  second.showCached();
  assert.equal(second.window.scrollY, 855);
  assert.equal(secondGrid.scrollTop, 650);
  assert.equal(secondGrid.scrollLeft, 70);
});

test('Uncached history restores a previously visited page regardless of the last visible page', () => {
  for (const [previous, referrer, expected] of [
    ['/dashboard/', ORIGIN + INVENTORY, 715],
    [INVENTORY, ORIGIN + '/dashboard/', 715],
  ]) {
    const store = new Map([[PREFIX + INVENTORY, saved(715)], [PREVIOUS, previous]]);
    const current = page({ navigation: 'back_forward', referrer, store }).load();
    assert.equal(current.window.scrollY, expected);
    assert.equal(store.get(PREVIOUS), INVENTORY);
  }
});

test('Local history listeners cannot override the shared position during a cached return', () => {
  const current = page().load();
  current.window.scrollY = 800;
  current.hide();
  current.store.set(PREVIOUS, '/dashboard/');
  current.showCached({ flush: false });
  let allowedDuringPopstate;
  current.window.addEventListener('popstate', () => {
    allowedDuringPopstate = current.window.PageScroll.allowHistoryScroll();
    if (allowedDuringPopstate) current.window.scrollTo(0, 800);
  });
  current.window.dispatch('popstate', { state: { scrollY: 800 } });
  assert.equal(allowedDuringPopstate, false);
  current.flush();
  assert.equal(current.window.scrollY, 800);
  assert.equal(current.window.PageScroll.allowHistoryScroll(), true);
});

test('Local history listeners defer to the shared restore during same-page cached returns', () => {
  const current = page().load();
  current.hide();
  current.showCached({ flush: false });
  assert.equal(current.window.PageScroll.allowHistoryScroll(), false);
  current.flush();
});

test('Uncached cross-page history also suppresses local popstate restoration until its timer', () => {
  const current = page({ navigation: 'back_forward', referrer: ORIGIN + INVENTORY,
    store: new Map([[PREFIX + INVENTORY, saved(715)], [PREVIOUS, '/dashboard/']]) });
  current.load({ flush: false });
  assert.equal(current.window.PageScroll.allowHistoryScroll(), false);
  assert.equal(current.window.scrollY, 715);
  current.flush();
  assert.equal(current.window.PageScroll.allowHistoryScroll(), true);
});

test('A real excluded form submission clears all positions on exit', () => {
  const current = page().load();
  current.container('table', 120, 25);
  current.window.scrollY = 610;
  current.submit({ excluded: true });
  current.hide();
  assert.equal(current.store.has(PREFIX + INVENTORY), false);
  const returned = page({ referrer: ORIGIN + INVENTORY, store: current.store, initialY: 610 }).load();
  assert.equal(returned.window.scrollY, 0);
});

test('A prevented excluded form still saves the actual position when leaving later', () => {
  const current = page().load();
  current.submit({ excluded: true, prevented: true });
  current.window.scrollY = 610;
  current.hide();
  assert.equal(JSON.parse(current.store.get(PREFIX + INVENTORY)).y, 610);
});

test('Ordinary form submission preserves the latest scroll at pagehide', () => {
  const current = page().load();
  current.window.scrollY = 210;
  current.submit();
  current.window.scrollY = 610;
  current.hide();
  assert.equal(JSON.parse(current.store.get(PREFIX + INVENTORY)).y, 610);
});

test('Anchors remain in browser control while containers follow the entry rule', () => {
  for (const [referrer, expectedTop] of [[ORIGIN + INVENTORY, 125], [ORIGIN + '/dashboard/', 0]]) {
    const current = page({ url: '/inventory/#product-41', referrer, initialY: 420,
      store: new Map([[PREFIX + INVENTORY, saved(810, { table: { top: 125, left: 30 } })]]) });
    const grid = current.container('table', 700, 100);
    current.load();
    assert.deepEqual(current.scrolls, []);
    assert.equal(current.window.scrollY, 420);
    assert.equal(grid.scrollTop, expectedTop);
    assert.equal(grid.scrollLeft, expectedTop ? 30 : 0);
  }
});

test('Containers registered after load immediately receive the restore policy', () => {
  const current = page({ navigation: 'reload',
    store: new Map([[PREFIX + INVENTORY, saved(610, { table: { top: 120, left: 25 } })]]) }).load();
  const grid = current.container('table');
  assert.equal(grid.scrollTop, 120);
  assert.equal(grid.scrollLeft, 25);
});

test('A container registered by a later DOMContentLoaded listener restores immediately', () => {
  const current = page({ navigation: 'reload',
    store: new Map([[PREFIX + INVENTORY, saved(610, { table: { top: 120, left: 25 } })]]) });
  let grid;
  current.document.addEventListener('DOMContentLoaded', () => { grid = current.container('table'); });
  current.load();
  assert.equal(grid.scrollTop, 120);
  assert.equal(grid.scrollLeft, 25);
});

test('A late container on cross-page entry clears its old position', () => {
  const current = page({ referrer: ORIGIN + '/dashboard/' }).load();
  const grid = current.container('table', 120, 25);
  assert.equal(grid.scrollTop, 0);
  assert.equal(grid.scrollLeft, 0);
});

test('Missing navigation timing still allows a valid same-page referrer', () => {
  const current = page({ navigation: null, referrer: ORIGIN + INVENTORY,
    store: new Map([[PREFIX + INVENTORY, saved(715)]]) }).load();
  assert.equal(current.window.scrollY, 715);
});

test('A standard page table automatically restores its wrapper by table ID', () => {
  const { table, wrapper } = tableFixture();
  const current = page({ navigation: 'reload', tables: [table], store: new Map([
    [PREFIX + INVENTORY, saved(610, { 'table:rpTable:0': { top: 125, left: 40 } })],
  ]) }).load();
  assert.equal(wrapper.scrollTop, 125);
  assert.equal(wrapper.scrollLeft, 40);
  wrapper.scrollTop = 230;
  wrapper.scrollLeft = 80;
  current.hide();
  assert.deepEqual(JSON.parse(current.store.get(PREFIX + INVENTORY)).containers,
    { 'table:rpTable:0': { top: 230, left: 80 } });
});

test('Automatically discovered nested table wrappers retain their positions on a cached return', () => {
  const { table, wrapper, main } = tableFixture({ wrapperId: 'inner-grid' });
  const outer = { id: 'outer-grid', parentElement: main, scrollTop: 0, scrollLeft: 0,
    style: { overflowX: 'scroll', overflowY: 'auto' } };
  wrapper.parentElement = outer;
  const current = page({ tables: [table] }).load();
  wrapper.scrollTop = 230;
  wrapper.scrollLeft = 80;
  outer.scrollTop = 650;
  outer.scrollLeft = 150;
  current.hide();
  assert.deepEqual(JSON.parse(current.store.get(PREFIX + INVENTORY)).containers,
    { 'table:inner-grid:0': { top: 230, left: 80 }, 'table:outer-grid:1': { top: 650, left: 150 } });
  current.store.set(PREVIOUS, '/dashboard/');
  current.showCached();
  assert.equal(wrapper.scrollTop, 230);
  assert.equal(wrapper.scrollLeft, 80);
  assert.equal(outer.scrollTop, 650);
  assert.equal(outer.scrollLeft, 150);
});

test('A table wrapper created by a later ready listener is restored on pageshow', () => {
  const { table, wrapper } = tableFixture({ scrollable: false });
  const current = page({ navigation: 'reload', tables: [table], store: new Map([
    [PREFIX + INVENTORY, saved(610, { 'table:rpTable:0': { top: 125, left: 40 } })],
  ]) });
  current.document.addEventListener('DOMContentLoaded', () => {
    wrapper.style.overflowX = 'auto';
    wrapper.style.overflowY = 'auto';
  });
  current.load();
  assert.equal(wrapper.scrollTop, 125);
  assert.equal(wrapper.scrollLeft, 40);
});

test('A dynamically introduced table saves its current offsets and synchronizes the top scrollbar on reload', () => {
  const tables = [];
  const current = page({ tables }).load();
  const dynamic = tableFixture({ wrapperId: 'dynamic-grid' });
  dynamic.wrapper.scrollTop = 325;
  dynamic.wrapper.scrollLeft = 90;
  tables.push(dynamic.table);
  current.hide();
  assert.equal(dynamic.wrapper.scrollTop, 325);
  assert.equal(dynamic.wrapper.scrollLeft, 90);
  assert.deepEqual(JSON.parse(current.store.get(PREFIX + INVENTORY)).containers,
    { 'table:dynamic-grid:0': { top: 325, left: 90 } });

  const next = tableFixture({ wrapperId: 'dynamic-grid' });
  const topScrollbar = { scrollLeft: 0 };
  next.wrapper._uiTopScrollUpdate = function () { topScrollbar.scrollLeft = this.scrollLeft; };
  page({ navigation: 'reload', tables: [next.table], store: current.store }).load();
  assert.equal(next.wrapper.scrollTop, 325);
  assert.equal(next.wrapper.scrollLeft, 90);
  assert.equal(topScrollbar.scrollLeft, 90);
});

test('Tables within semantic dialogs are excluded from page location retention', () => {
  const { table, wrapper } = tableFixture({ dialog: true });
  wrapper.scrollTop = 230;
  wrapper.scrollLeft = 80;
  const current = page({ tables: [table] }).load();
  assert.equal(wrapper.scrollTop, 230);
  assert.equal(wrapper.scrollLeft, 80);
  current.hide();
  assert.deepEqual(JSON.parse(current.store.get(PREFIX + INVENTORY)).containers, {});
});

test('A manually named table container is not duplicated by automatic table discovery', () => {
  const { table, wrapper } = tableFixture();
  const current = page({ tables: [table] });
  current.window.PageScroll.registerContainer('ordering-sheet', wrapper);
  current.load();
  wrapper.scrollTop = 230;
  wrapper.scrollLeft = 80;
  current.hide();
  assert.deepEqual(JSON.parse(current.store.get(PREFIX + INVENTORY)).containers,
    { 'ordering-sheet': { top: 230, left: 80 } });
});

test('Blocked storage does not throw during registration or any lifecycle event', () => {
  const current = page({ blocked: true, navigation: 'reload', initialY: 300 });
  assert.doesNotThrow(() => {
    current.container('table', 200, 70);
    current.load();
    current.submit({ excluded: true, prevented: true });
    current.hide();
    current.showCached();
  });
  assert.equal(current.window.scrollY, 0);
});

for (const value of ['not-json', '{}', 'null', '[]', saved(-1), saved('600'), saved(null),
  '{"y":1e999,"containers":{}}', '{"y":600,"containers":{"table":{"top":-2,"left":"30"}}}']) {
  test(`Invalid saved positions fail safely: ${value}`, () => {
    const current = page({ navigation: null, referrer: ORIGIN + INVENTORY,
      store: new Map([[PREFIX + INVENTORY, value]]), initialY: 300 });
    const grid = current.container('table', 200, 70);
    assert.doesNotThrow(() => current.load());
    assert.equal(typeof current.window.scrollY, 'number');
    assert.ok(Number.isFinite(current.window.scrollY) && current.window.scrollY >= 0);
    if (!value.includes('"y":600')) assert.equal(current.window.scrollY, 0);
    assert.equal(grid.scrollTop, 0);
    assert.equal(grid.scrollLeft, 0);
  });
}

test('Zero and fractional saved positions remain valid', () => {
  const current = page({ navigation: 'reload',
    store: new Map([[PREFIX + INVENTORY, saved(0, { table: { top: 2.5, left: 0 } })]]), initialY: 300 });
  const grid = current.container('table', 200, 70);
  current.load();
  assert.equal(current.window.scrollY, 0);
  assert.equal(grid.scrollTop, 2.5);
  assert.equal(grid.scrollLeft, 0);
});

test('Legacy page and one-shot form keys cannot restore stale positions', () => {
  const store = new Map([
    ['scrollPos:/inventory/', '900'], ['scroll:/inventory/?page=2', '850'],
    ['scroll-after-submit', JSON.stringify({ p: INVENTORY, y: 800 })],
    ['osScroll:/inventory/', JSON.stringify({ y: 750, top: 600, left: 30 })],
  ]);
  const current = page({ url: '/inventory/?page=2', referrer: ORIGIN + INVENTORY, store, initialY: 300 }).load();
  assert.equal(current.window.scrollY, 0);
});

test('Embedded pages never overwrite top-level positions or the previous-page marker', () => {
  const store = new Map([[PREVIOUS, INVENTORY], [PREFIX + '/ordering-sheet/', saved(950)]]);
  const embedded = page({ url: '/ordering-sheet/?embed=1', store, parentPage: false }).load();
  embedded.window.scrollY = 410;
  embedded.hide();
  embedded.showCached();
  assert.equal(store.get(PREVIOUS), INVENTORY);
  assert.equal(JSON.parse(store.get(PREFIX + '/ordering-sheet/')).y, 950);
  assert.equal(JSON.parse(store.get(PREFIX + 'frame:/dashboard/:/ordering-sheet/')).y, 410);
});

test('An embedded reload restores only its parent-scoped position', () => {
  const frameKey = PREFIX + 'frame:/dashboard/:/ordering-sheet/';
  const store = new Map([[frameKey, saved(410)],
    [PREFIX + 'frame:/inventory/:/ordering-sheet/', saved(730)],
    [PREFIX + '/ordering-sheet/', saved(950)], [PREVIOUS, INVENTORY]]);
  const embedded = page({ url: '/ordering-sheet/?embed=1', store, parentPage: false,
    parentPath: '/dashboard/', navigation: 'reload' }).load();
  assert.equal(embedded.window.scrollY, 410);
  assert.equal(store.get(PREVIOUS), INVENTORY);
});

test('Cached embedded pages restore alongside their parent when returning from another page', () => {
  const parent = page({ url: '/dashboard/' }).load();
  const embedded = page({ url: '/ordering-sheet/?embed=1', store: parent.store, parentPage: parent }).load();
  embedded.window.scrollY = 410;
  embedded.hide();
  parent.hide();
  const other = page({ url: INVENTORY, store: parent.store }).load();
  other.hide();
  parent.showCached();
  embedded.showCached();
  assert.equal(embedded.window.scrollY, 410);
  assert.equal(parent.store.get(PREVIOUS), '/dashboard/');
});

test('An embedded cached pageshow before its parent still restores its saved position', () => {
  const parent = page({ url: '/dashboard/', navigation: 'reload' }).load();
  const embedded = page({ url: '/ordering-sheet/?embed=1', store: parent.store, parentPage: parent }).load();
  embedded.window.scrollY = 410;
  embedded.hide();
  parent.hide();
  const other = page({ url: INVENTORY, store: parent.store }).load();
  other.hide();
  embedded.showCached();
  parent.showCached();
  assert.equal(embedded.window.scrollY, 410);
  assert.equal(parent.store.get(PREVIOUS), '/dashboard/');
});

test('An embedded cached pageshow before its parent retains the latest same-parent offset', () => {
  const parent = page({ url: '/dashboard/' }).load();
  const embedded = page({ url: '/ordering-sheet/?embed=1', store: parent.store, parentPage: parent }).load();
  embedded.window.scrollY = 410;
  embedded.hide();
  parent.hide();
  embedded.showCached();
  parent.showCached();
  assert.equal(embedded.window.scrollY, 410);
  assert.equal(parent.store.get(PREVIOUS), '/dashboard/');
});

test('Uncached Back and Forward use each history entry, not the latest pathname offset', () => {
  const first = page({ url: '/inventory/?page=1' }).load();
  first.window.scrollY = 400;
  const firstGrid = first.container('table');
  firstGrid.scrollTop = 200;
  firstGrid.scrollLeft = 30;
  first.hide();
  const firstState = first.history.state;
  const second = page({ url: '/inventory/?page=2', referrer: first.window.location.href, store: first.store }).load();
  second.window.scrollY = 900;
  const secondGrid = second.container('table');
  secondGrid.scrollTop = 650;
  secondGrid.scrollLeft = 70;
  second.hide();
  const secondState = second.history.state;

  for (const [url, historyState, y, top, left] of [
    ['/inventory/?page=1', firstState, 400, 200, 30],
    ['/inventory/?page=2', secondState, 900, 650, 70],
  ]) {
    const current = page({ url, navigation: 'back_forward', store: first.store, historyState });
    const grid = current.container('table');
    current.load();
    assert.equal(current.window.scrollY, y);
    assert.equal(grid.scrollTop, top);
    assert.equal(grid.scrollLeft, left);
    assert.deepEqual(current.scrolls, [[0, y]], 'No reset to zero or repeated delayed scroll');
    assert.deepEqual(current.scrollBehaviors, ['instant']);
    current.hide();
  }
});

test('Saving positions preserves page-specific history data', () => {
  const current = page({ historyState: { rpSuggestionsBoard: true, rpSuggestionsFrontScroll: 540 } }).load();
  current.window.scrollY = 540;
  current.document.dispatch('scroll');
  current.flush();
  assert.equal(current.history.state.rpSuggestionsBoard, true);
  assert.equal(current.history.state.rpSuggestionsFrontScroll, 540);
  assert.equal(current.history.state.pharmacyPageScroll.position.y, 540);
});

test('A saved history entry remains available when session storage is blocked', () => {
  const first = page({ blocked: true }).load();
  first.window.scrollY = 760;
  first.hide();
  const returned = page({ blocked: true, navigation: 'back_forward', historyState: first.history.state }).load();
  assert.equal(returned.window.scrollY, 760);
  assert.deepEqual(returned.scrollBehaviors, ['instant']);
});

test('A cached page keeps its own position when history and storage writes are blocked', () => {
  const current = page({ blocked: true, historyBlocked: true }).load();
  current.window.scrollY = 760;
  current.hide();
  current.window.scrollY = 0;
  assert.doesNotThrow(() => current.showCached());
  assert.equal(current.window.scrollY, 760);
});

test('Returning from cache does not move an already restored page through the top', () => {
  const current = page().load();
  current.window.scrollY = 840;
  current.hide();
  current.showCached();
  assert.equal(current.window.scrollY, 840);
  assert.deepEqual(current.scrolls, []);
});

test('Registered containers restore instantly even when their CSS uses smooth scrolling', () => {
  const current = page({ navigation: 'reload', store: new Map([
    [PREFIX + INVENTORY, saved(500, { table: { top: 320, left: 60 } })],
  ]) });
  const grid = current.container('table');
  const calls = [];
  grid.scrollTo = options => {
    calls.push(options.behavior);
    grid.scrollTop = options.top;
    grid.scrollLeft = options.left;
  };
  current.load();
  assert.equal(grid.scrollTop, 320);
  assert.equal(grid.scrollLeft, 60);
  assert.ok(calls.length > 0 && calls.every(behavior => behavior === 'instant'));
});

test('User scrolling cancels a queued layout restoration', () => {
  for (const [event, detail] of [['wheel', {}], ['touchstart', {}], ['pointerdown', {}], ['keydown', { key: 'PageDown' }]]) {
    const current = page({ navigation: 'reload', store: new Map([[PREFIX + INVENTORY, saved(500)]]) });
    current.load({ flush: false });
    current.window.dispatch(event, detail);
    current.window.scrollY = 740;
    current.flush();
    assert.equal(current.window.scrollY, 740, event);
  }
});

test('Excluded form submissions clear the history entry as well as pathname storage', () => {
  const current = page().load();
  current.window.scrollY = 500;
  current.submit({ excluded: true });
  current.document.dispatch('scroll');
  current.flush();
  current.hide();
  assert.equal(current.history.state.pharmacyPageScroll.position, null);
  current.showCached();
  assert.equal(current.window.scrollY, 0);
});

test('Recently Purchased restores its front face once and without smooth scrolling', () => {
  const template = readFileSync(path.resolve(__dirname, '../../../app/templates/low_stock.html'), 'utf8');
  const restore = template.match(/function restoreFrontScrollPosition\(generation\) \{[\s\S]*?\n  \}/)[0];
  const calls = [];
  const context = { frontWindowScroll: 840, isShowingSuggestions: false, boardStateGeneration: 2,
    stage: { style: {} }, front: { scrollHeight: 1800 }, window: {
      scrollTo: options => calls.push(options),
      requestAnimationFrame() { throw new Error('No delayed second scroll'); },
      setTimeout() { throw new Error('No delayed second scroll'); },
    } };
  vm.createContext(context);
  vm.runInContext(restore, context);
  context.restoreFrontScrollPosition(2);
  assert.equal(calls.length, 1);
  assert.equal(calls[0].top, 840);
  assert.equal(calls[0].behavior, 'instant');
  assert.equal(context.stage.style.height, '1800px');
  context.restoreFrontScrollPosition(1);
  assert.equal(calls.length, 1, 'An obsolete transition cannot move the page');
});
