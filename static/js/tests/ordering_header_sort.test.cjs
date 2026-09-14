const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

const template = readFileSync(path.join(__dirname, '../../../app/templates/partials/_ordering_sheet.html'), 'utf8');
const filtersStart = template.indexOf('    // ── Combined filtering:');
const filtersEnd = template.indexOf('    // ── Row selection + bulk delete', filtersStart);
const sortStart = template.indexOf('    // ── Click-to-sort column headers');
const sortEnd = template.indexOf('    initializeCustomStatusEditors();', sortStart);
assert.ok(filtersStart >= 0 && filtersEnd > filtersStart && sortStart >= 0 && sortEnd > sortStart);
const source = template.slice(filtersStart, filtersEnd) + template.slice(sortStart, sortEnd);
const tableHead = template.slice(template.indexOf('<thead>'), template.indexOf('</thead>'));
const headerMarkup = [...tableHead.matchAll(/<th\b([^>]*)>([\s\S]*?)<\/th>/g)];

function harness(saved = {}) {
  function node(dataset = {}) {
    const classes = new Set();
    return {
      dataset, value: '', textContent: '', style: {}, attributes: {}, listeners: {},
      classList: {
        toggle(name, enabled) { if (enabled) classes.add(name); else classes.delete(name); },
        remove(name) { classes.delete(name); },
        contains(name) { return classes.has(name); },
      },
      addEventListener(name, fn) { this.listeners[name] = fn; },
      getAttribute(name) { return name.startsWith('data-') ? dataset[name.slice(5)] : this.attributes[name]; },
      setAttribute(name, value) { this.attributes[name] = value; },
      removeAttribute(name) { delete this.attributes[name]; },
      closest() { return null; },
    };
  }
  const definitions = [
    ['old', 'Benadryl', 'Order for stock', '2026-08-31', 'BB', 'drug', 'pending', 'low'],
    ['same-day-late', '5 Zantac', 'Order for basket', '2026-09-12', 'ZZ', 'drug', 'ordered', 'high'],
    ['same-day-early', '5 Advil', 'OTC · Left', '2026-09-12', 'AA', 'otc', 'pending', 'low'],
    ['new', 'Zyrtec', 'Order for stock', '2026-09-13', 'ZZ', 'drug', 'pending', 'low'],
  ];
  const rows = definitions.map(([id, name, reason, entryDate, initials, entryType, status, urgency]) => {
    const row = node({ id, name, initials, entryType, status, urgency, patient: '', note: '', customStatus: '' });
    row.dataset.initialOrder = ['new', 'same-day-late', 'same-day-early', 'old'].indexOf(id);
    row.cells = Array.from({ length: 8 }, () => node());
    row.cells[2].dataset.sort = name;
    row.cells[3].textContent = reason;
    row.cells[6].dataset.sort = entryDate.replaceAll('-', '');
    row.cells[6].textContent = `${initials} - ${entryDate}`;
    return row;
  });
  const allHeaders = headerMarkup.map(([markup, attributes], index) => {
    const header = node();
    header.cellIndex = index;
    header.sortable = /class="[^"]*\bos-sortable\b/.test(attributes);
    const indicator = node();
    header.querySelector = () => indicator;
    return header;
  });
  const headers = allHeaders.filter(header => header.sortable);
  const ids = {};
  for (const id of ['os-search', 'os-count', 'os-high-flag', 'os-high-n',
    'os-clear-filters', 'os-nomatch', 'os-nomatch-clear', 'os-empty', 'os-table', 'os-tbody']) {
    ids[id] = node();
  }
  ids['os-table'].querySelectorAll = () => headers;
  ids['os-table'].querySelector = selector => selector === 'th[data-column-key="initial-date"]' ? allHeaders[6] : null;
  ids['os-tbody'].querySelectorAll = () => rows;
  ids['os-tbody'].appendChild = fragment => rows.splice(0, rows.length, ...fragment.children);
  const typeButtons = ['all', 'drug', 'otc'].map(filter => node({ filter }));
  const statusButtons = ['all', 'pending', 'ordered', 'backordered', 'not_for_sale'].map(status => node({ status }));
  const counts = statusButtons.map(button => node({ count: button.dataset.status }));
  const selectors = { '#os-tbody tr': rows, '.os-filter-btn': typeButtons, '.os-stat': statusButtons, '.os-stat-n': counts };
  const storage = new Map([['orderingSheetTableState:v3:active', JSON.stringify(saved)]]);
  const context = {
    URLSearchParams,
    sessionStorage: { getItem: key => storage.get(key), setItem: (key, value) => storage.set(key, value) },
    window: { location: { search: '' }, listeners: {}, addEventListener(name, fn) { this.listeners[name] = fn; } },
    document: {
      getElementById: id => ids[id] || null,
      querySelectorAll: selector => selectors[selector] || [],
      addEventListener() {},
      createDocumentFragment: () => ({ children: [], appendChild(row) { this.children.push(row); } }),
    },
  };
  vm.createContext(context);
  vm.runInContext(source, context);
  return {
    context, rows, ids, headers, allHeaders, typeButtons, statusButtons,
    visible: () => rows.filter(row => row.style.display !== 'none').map(row => row.dataset.id),
    saved: () => JSON.parse(storage.get('orderingSheetTableState:v3:active')),
    sort(index) {
      const header = allHeaders[index];
      header.listeners.click({ target: header });
    },
  };
}

test('only Drug Name, Reasoning and Initial - Date headers offer sorting', () => {
  const h = harness();
  assert.deepEqual(h.headers.map(header => header.cellIndex), [2, 3, 6]);
  for (const [index, label] of [[2, 'drug name'], [3, 'reasoning'], [6, 'entry date']]) {
    assert.match(headerMarkup[index][2], new RegExp(`<button type="button" class="os-sort-button" aria-label="Sort by ${label}">`));
    assert.equal(typeof h.allHeaders[index].listeners.click, 'function');
  }
  for (const index of [0, 1, 4, 5, 7]) {
    assert.equal(h.allHeaders[index].listeners.click, undefined);
  }
  assert.doesNotMatch(tableHead, /type="date"|os-entry-date-filter/);
});

test('initial table order is newest date first with a descending date indicator', () => {
  const h = harness();
  assert.deepEqual(h.visible(), ['new', 'same-day-late', 'same-day-early', 'old']);
  assert.equal(h.allHeaders[6].attributes['aria-sort'], 'descending');
  assert.equal(h.allHeaders[6].querySelector('.sort-ind').textContent, ' ▼');
});

test('drug name sorts alphabetically both ways including names beginning with the same number', () => {
  const h = harness();
  h.sort(2);
  assert.deepEqual(h.visible(), ['same-day-early', 'same-day-late', 'old', 'new']);
  assert.equal(h.allHeaders[2].attributes['aria-sort'], 'ascending');
  h.sort(2);
  assert.deepEqual(h.visible(), ['new', 'old', 'same-day-late', 'same-day-early']);
  assert.equal(h.allHeaders[2].attributes['aria-sort'], 'descending');
});

test('reasoning sorts both ways and changing column starts ascending', () => {
  const h = harness();
  h.sort(2);
  h.sort(2);
  h.sort(3);
  assert.deepEqual(h.visible(), ['same-day-late', 'new', 'old', 'same-day-early']);
  assert.equal(h.allHeaders[3].attributes['aria-sort'], 'ascending');
  assert.equal(h.allHeaders[2].attributes['aria-sort'], undefined);
  h.sort(3);
  assert.deepEqual(h.visible(), ['same-day-early', 'new', 'old', 'same-day-late']);
  assert.equal(h.allHeaders[3].attributes['aria-sort'], 'descending');
});

test('date sorting ignores initials and preserves the original same-day order', () => {
  const h = harness();
  h.sort(6);
  assert.deepEqual(h.visible(), ['old', 'same-day-late', 'same-day-early', 'new']);
  assert.equal(h.allHeaders[6].attributes['aria-sort'], 'ascending');
  h.sort(6);
  assert.deepEqual(h.visible(), ['new', 'same-day-late', 'same-day-early', 'old']);
  assert.equal(h.allHeaders[6].attributes['aria-sort'], 'descending');
});

test('manual header sorting survives a targeted row refresh but resets on a fresh visit', () => {
  for (const index of [2, 3, 6]) {
    const h = harness();
    h.sort(index);
    h.sort(index);
    const manualOrder = h.visible();
    h.context.applyCurrentSort();
    h.context.applyFilters();
    assert.deepEqual(h.visible(), manualOrder);
    const reloaded = harness(h.saved());
    assert.deepEqual(reloaded.visible(), ['new', 'same-day-late', 'same-day-early', 'old']);
    assert.equal(reloaded.allHeaders[6].attributes['aria-sort'], 'descending');
  }
});

test('previously saved sorts never override the newest-first default', () => {
  for (const index of [0, 1, 2, 3, 4, 5, 6, 7, 99]) {
    for (const direction of ['asc', 'desc', 'invalid']) {
      const h = harness({ sortIndex: index, sortDirection: direction });
      assert.deepEqual(h.visible(), ['new', 'same-day-late', 'same-day-early', 'old']);
      assert.equal(Object.hasOwn(h.saved(), 'sortIndex'), false);
      assert.equal(Object.hasOwn(h.saved(), 'sortDirection'), false);
      assert.equal(h.allHeaders[6].attributes['aria-sort'], 'descending');
      assert.equal(h.allHeaders[2].attributes['aria-sort'], undefined);
      assert.equal(h.allHeaders[3].attributes['aria-sort'], undefined);
    }
  }
});

test('returning from the browser cache also starts with the newest date first', () => {
  const h = harness();
  h.sort(2);
  const manualOrder = h.visible();
  h.context.window.listeners.pageshow({ persisted: false });
  assert.deepEqual(h.visible(), manualOrder);
  h.context.window.listeners.pageshow({ persisted: true });
  assert.deepEqual(h.visible(), ['new', 'same-day-late', 'same-day-early', 'old']);
  assert.equal(h.allHeaders[6].attributes['aria-sort'], 'descending');
});

test('an obsolete saved calendar filter never hides rows and is dropped on the next save', () => {
  const h = harness({ entryDate: '2026-09-12' });
  assert.deepEqual(h.visible(), ['new', 'same-day-late', 'same-day-early', 'old']);
  assert.equal(h.ids['os-clear-filters'].classList.contains('visible'), false);
  h.sort(6);
  assert.equal(Object.hasOwn(h.saved(), 'entryDate'), false);
  assert.equal(h.visible().length, 4);
});

test('existing text, status, type and urgency filters still combine with header sorting', () => {
  const h = harness({ search: '5 ', type: 'drug', status: 'ordered', urgencyOnly: true, entryDate: '2026-01-01' });
  assert.deepEqual(h.visible(), ['same-day-late']);
  h.sort(2);
  assert.deepEqual(h.visible(), ['same-day-late']);
  h.ids['os-clear-filters'].listeners.click();
  assert.deepEqual(h.visible(), ['same-day-early', 'same-day-late', 'old', 'new']);
  assert.equal(h.allHeaders[2].attributes['aria-sort'], 'ascending');
});
