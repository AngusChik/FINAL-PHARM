const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

const template = readFileSync(path.join(__dirname, '../../../app/templates/low_stock.html'), 'utf8').replace(/\r\n/g, '\n');
const scriptStart = template.indexOf('(function() {\n  const CSRF');
assert.ok(scriptStart >= 0, 'The Recently Purchased table script must exist.');
const source = template.slice(scriptStart, template.indexOf('</script>', scriptStart));

// Minimal DOM surface for the real table script. AJAX HTML fixtures install row
// nodes; filtering, selection, sorting and removal all run the production IIFE.
function element(tag = 'div', attributes = {}) {
  const listeners = {};
  const node = {
    tagName: tag.toUpperCase(), children: [], parentElement: null,
    attributes: {}, dataset: {}, style: {}, value: '', checked: false,
    disabled: false, hidden: false, _text: '',
    setAttribute(name, value) {
      this.attributes[name] = String(value);
      if (name.startsWith('data-')) this.dataset[name.slice(5).replace(/-([a-z])/g, (_, char) => char.toUpperCase())] = String(value);
    },
    getAttribute(name) { return this.attributes[name] ?? null; },
    removeAttribute(name) { delete this.attributes[name]; },
    appendChild(child) { child.remove(); child.parentElement = this; this.children.push(child); return child; },
    remove() {
      if (this.parentElement) this.parentElement.children = this.parentElement.children.filter(child => child !== this);
      this.parentElement = null;
    },
    addEventListener(type, callback) { (listeners[type] ||= []).push(callback); },
    emit(type, target = this, extra = {}) {
      const event = { target, preventDefault() {}, stopPropagation() {}, ...extra };
      return Promise.all((listeners[type] || []).map(callback => callback.call(this, event)));
    },
    matches(selector) {
      return selector.split(',').some(part => {
        part = part.trim();
        if (part.endsWith(':checked')) {
          if (!this.checked) return false;
          part = part.slice(0, -8);
        }
        const attrs = [...part.matchAll(/\[([^=\]]+)(?:="([^"]*)")?\]/g)];
        if (attrs.some(([, name, value]) => this.getAttribute(name) === null || (value !== undefined && this.getAttribute(name) !== value))) return false;
        part = part.replace(/\[[^\]]+\]/g, '');
        const id = part.match(/#([\w-]+)/);
        if (id && this.id !== id[1]) return false;
        if ([...part.matchAll(/\.([\w-]+)/g)].some(([, name]) => !this.classList.contains(name))) return false;
        const tagName = part.match(/^[\w-]+/);
        return !tagName || this.tagName === tagName[0].toUpperCase();
      });
    },
    closest(selector) { return this.matches(selector) ? this : this.parentElement?.closest(selector) || null; },
    querySelectorAll(selector) {
      const result = [];
      const visit = parent => parent.children.forEach(child => {
        if (selector.split(',').some(part => {
          const parts = part.trim().split(/\s+/);
          return child.matches(parts.at(-1)) && (parts.length === 1 || child.parentElement?.closest(parts.slice(0, -1).join(' ')));
        })) result.push(child);
        visit(child);
      });
      visit(this);
      return result;
    },
    querySelector(selector) { return this.querySelectorAll(selector)[0] || null; },
    focus() {},
    get nextElementSibling() { return this.parentElement?.children[this.parentElement.children.indexOf(this) + 1] || null; },
    get id() { return this.attributes.id || ''; }, set id(value) { this.setAttribute('id', value); },
    get className() { return this.attributes.class || ''; }, set className(value) { this.setAttribute('class', value); },
    get textContent() { return this._text + this.children.map(child => child.textContent).join(''); },
    set textContent(value) { this._text = String(value); this.children.forEach(child => { child.parentElement = null; }); this.children = []; },
    get innerHTML() { return this._html || ''; },
    set innerHTML(value) {
      this.children.forEach(child => { child.parentElement = null; });
      this.children = [];
      this._text = '';
      this._html = value;
      if (this.installFixture) this.installFixture(value);
    },
  };
  node.classList = {
    contains(name) { return node.className.split(/\s+/).includes(name); },
    add(...names) { node.className = [...new Set([...node.className.split(/\s+/).filter(Boolean), ...names])].join(' '); },
    remove(...names) { node.className = node.className.split(/\s+/).filter(name => !names.includes(name)).join(' '); },
    toggle(name, enabled) {
      if (enabled === undefined) enabled = !this.contains(name);
      if (enabled) this.add(name); else this.remove(name);
      return enabled;
    },
  };
  Object.entries(attributes).forEach(([name, value]) => node.setAttribute(name, value));
  return node;
}

function harness(initialRows, { selected = [], search = '' } = {}) {
  const root = element('body');
  const add = (parent, tag, attrs) => parent.appendChild(element(tag, attrs));
  const card = add(root, 'section', { class: 'rp-card' });
  const toolbar = add(card, 'div', { class: 'rp-toolbar' });
  const ids = [
    'rp-select-all', 'rp-selected-badge', 'rp-total-count', 'rp-dd-delete-selected',
    'rp-ignore-snacks', 'rp-ignore-braces', 'rp-delete-btn', 'rp-delete-menu',
    'rp-filter-btn', 'rp-filter-menu', 'rp-filter-badge', 'rp-filter-clear',
    'rp-dd-delete-all', 'rp-search-input', 'rp-clear-btn', 'rp-results-status',
    'rp-export-modal', 'rp-export-btn', 'rp-export-modal-close', 'rp-export-cancel',
    'rp-export-go', 'rp-export-sel-count',
  ];
  ids.forEach(id => add(toolbar, id === 'rp-search-input' ? 'input' : 'button', { id }));
  const id = name => root.querySelector('#' + name);
  id('rp-delete-menu').className = 'rp-dropdown-menu';
  id('rp-filter-menu').className = 'rp-filter-menu';
  id('rp-search-input').value = search;
  const notice = add(card, 'div', { class: 'rp-filter-notice' });
  add(notice, 'strong');
  add(notice, 'span', { class: 'rp-filter-count' });
  const wrapper = add(card, 'div', { id: 'rp-table-wrap' });
  const table = add(wrapper, 'table', { class: 'rp-table' });
  const thead = add(table, 'thead');
  const headerRow = add(thead, 'tr');
  for (let column = 1; column <= 6; column++) add(headerRow, 'th', { 'data-sort': column, ...(column >= 5 ? { 'data-sort-type': 'num' } : {}) });
  const tbody = add(table, 'tbody', { id: 'rp-tbody' });
  const empty = add(card, 'div', { id: 'rp-no-results' });
  add(empty, 'div', { class: 'rp-empty-text' });
  empty.hidden = Boolean(initialRows.length);

  function installRows(rows) {
    for (const row of rows) {
      const product = add(tbody, 'tr', { 'data-id': row.id, class: 'rp-product-row' });
      const cells = ['', row.brand || 'Brand', row.name, row.barcode || '123', row.item || '', row.bought || 0, row.stock || 0, 'No demand', ''];
      cells.forEach(value => { const cell = add(product, 'td'); cell.textContent = value; });
      const checkbox = add(product.children[0], 'input', { class: 'rp-row-check' });
      checkbox.value = String(row.id);
      add(tbody, 'tr', { class: 'rp-detail-row', 'data-detail-for': row.id });
    }
  }
  installRows(initialRows);
  tbody.installFixture = html => installRows(JSON.parse(html));
  const storage = new Map([['rp_selected_ids', JSON.stringify(selected)]]);
  const responses = [];
  const requests = [];
  const toasts = [];
  const timers = new Map();
  let timerId = 0;
  const location = { href: 'http://localhost/low-stock/?return_to=%2Fsales%2F', search: '?return_to=%2Fsales%2F' };
  const context = vm.createContext({
    document: {
      getElementById: id, querySelector: selector => root.querySelector(selector),
      querySelectorAll: selector => root.querySelectorAll(selector), createElement: tag => element(tag),
      addEventListener: (...args) => root.addEventListener(...args), body: root,
    },
    window: {
      location, uiConfirm: async () => true,
      history: { state: {}, replaceState(state, unused, url) { this.state = state; location.href = String(url); location.search = new URL(url).search; } },
    },
    localStorage: { getItem: key => storage.get(key) ?? null, setItem: (key, value) => storage.set(key, value) },
    fetch: async (url, options = {}) => {
      requests.push({ url, options });
      assert.ok(responses.length, 'Each request must have an explicit test response.');
      const payload = responses.shift();
      return { ok: true, json: async () => payload };
    },
    showToast: (message, level) => toasts.push({ message, level }),
    setTimeout(callback) { const token = ++timerId; timers.set(token, callback); return token; },
    clearTimeout(token) { timers.delete(token); },
    URL, URLSearchParams, AbortController, console,
  });
  const rendered = source.replace(/\{\{\s*recent_count\s*\}\}/g, String(initialRows.length))
    .replace(/\{\{[^}]+\}\}/g, '')
    .replace(/\{%\s*url\s+["']([^"']+)["'][^%]*%\}/g, '/$1/');
  vm.runInContext(rendered, context);
  return {
    id, card, tbody, thead, root, requests, responses, toasts, storage, location,
    rowIds: () => tbody.querySelectorAll('.rp-product-row').map(row => Number(row.dataset.id)),
    replyRows(rows, q = '') { responses.push({ html: JSON.stringify(rows), count: rows.length, q, categories: [] }); },
    async settle() { await new Promise(resolve => setImmediate(resolve)); },
    async flushTimers() {
      for (const [token, callback] of [...timers]) { timers.delete(token); callback(); }
      await new Promise(resolve => setImmediate(resolve));
    },
  };
}

test('empty filtered results retain controls and clearing the search restores the full table', async () => {
  const h = harness([{ id: 1, name: 'Alpha' }]);
  const originalSearch = h.id('rp-search-input');
  h.replyRows([], 'missing');
  originalSearch.value = 'missing';
  await originalSearch.emit('input');
  await h.flushTimers();
  assert.equal(h.id('rp-no-results').hidden, false);
  assert.equal(h.id('rp-table-wrap').style.display, 'none');
  assert.equal(h.id('rp-search-input'), originalSearch);
  assert.ok(h.card.querySelector('#rp-ignore-braces'));
  assert.equal(h.id('rp-results-status').textContent, '0 items shown');
  assert.equal(new URL(h.location.href).searchParams.get('return_to'), '/sales/');

  h.replyRows([{ id: 1, name: 'Alpha' }, { id: 2, name: 'Beta' }]);
  await h.id('rp-clear-btn').emit('click');
  await h.settle();
  assert.deepEqual(h.rowIds(), [1, 2]);
  assert.equal(originalSearch.value, '');
  assert.equal(h.id('rp-no-results').hidden, true);
  assert.equal(h.id('rp-table-wrap').style.display, '');
  assert.equal(h.id('rp-results-status').textContent, '2 items shown');
  assert.equal(h.toasts.length, 0);
});

test('ascending and descending column sorts survive AJAX filters and keep detail rows paired', async () => {
  const h = harness([{ id: 1, name: 'Beta' }, { id: 2, name: 'Alpha' }]);
  const header = h.thead.querySelector('th[data-sort="2"]');
  await h.thead.emit('click', header);
  assert.deepEqual(h.rowIds(), [2, 1]);
  h.replyRows([{ id: 3, name: 'Zulu' }, { id: 4, name: 'Gamma' }]);
  await h.id('rp-ignore-braces').emit('click');
  await h.settle();
  assert.deepEqual(h.rowIds(), [4, 3]);
  assert.equal(header.getAttribute('aria-sort'), 'ascending');
  assert.equal(new URL(h.requests.at(-1).url, h.location.href).searchParams.get('hide_braces'), '1');

  await header.emit('keydown', header, { key: 'Enter' });
  h.replyRows([{ id: 5, name: 'Delta' }, { id: 6, name: 'Theta' }, { id: 7, name: 'Alpha' }]);
  await h.id('rp-ignore-snacks').emit('click');
  await h.settle();
  assert.deepEqual(h.rowIds(), [6, 5, 7]);
  assert.equal(header.getAttribute('aria-sort'), 'descending');
  for (const row of h.tbody.querySelectorAll('.rp-product-row')) assert.equal(row.nextElementSibling.dataset.detailFor, row.dataset.id);
  assert.equal(h.toasts.length, 0);
});

test('bulk removal of hidden selected rows changes the displayed count only for visible removals', async () => {
  const h = harness([{ id: 1, name: 'Visible selected' }, { id: 2, name: 'Visible retained' }], { selected: [1, 3, 4, 5] });
  assert.equal(h.id('rp-selected-badge').textContent, '4 selected');
  h.responses.push({ success: true, deleted_count: 4 });
  await h.id('rp-dd-delete-selected').emit('click');
  await h.settle();
  await h.flushTimers();
  assert.deepEqual(JSON.parse(h.requests[0].options.body), { ids: [1, 3, 4, 5] });
  assert.deepEqual(h.rowIds(), [2]);
  assert.equal(h.id('rp-total-count').textContent, '1 Item');
  assert.equal(h.id('rp-results-status').textContent, '1 item shown');
  assert.equal(h.id('rp-no-results').hidden, true);
  assert.deepEqual(JSON.parse(h.storage.get('rp_selected_ids')), []);
  assert.equal(h.id('rp-dd-delete-selected').disabled, true);
  assert.equal(h.toasts.some(toast => toast.level === 'error'), false);
});
