const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

const root = path.resolve(__dirname, '../../..');
const settle = () => new Promise(setImmediate);

test('stock log boundary jumps preserve filters and use returned page bounds', async () => {
  const nodes = new Map();
  function element() {
    const listeners = {};
    return {
      value: '', style: {}, innerHTML: '', hidden: false, disabled: false,
      classList: { add() {}, remove() {}, contains() { return false; }, toggle() {} },
      contains() { return false; },
      getAttribute() { return '/stock-log/api/'; },
      addEventListener(event, listener) { listeners[event] = listener; },
      async click() {
        assert.equal(this.disabled, false);
        if (listeners.click) listeners.click();
        await settle();
      },
    };
  }
  const document = {
    body: element(), addEventListener() {}, dispatchEvent() {}, createElement: element,
    getElementById(id) {
      if (!nodes.has(id)) nodes.set(id, element());
      return nodes.get(id);
    },
  };
  const requests = [];
  vm.runInNewContext(readFileSync(path.join(root, 'static/js/stock_log.js'), 'utf8'), {
    document, URLSearchParams, CustomEvent: class {},
    window: { location: { pathname: '/' }, scrollY: 0, scrollTo() {} },
    sessionStorage: { getItem() {}, setItem() {}, removeItem() {} },
    fetch(url) {
      requests.push(url);
      const query = new URL(url, 'https://example.test').searchParams;
      const page = Math.min(3, Number(query.get('log_page')) || 1);
      return Promise.resolve({ json: async () => ({
        page, num_pages: 3, has_prev: page > 1, has_next: page < 3,
        entries: [{ time: '', name: 'Product', barcode: '1', qty: 1, action: 'Check-in', note: '' }],
      }) });
    },
  });
  document.getElementById('slFilterProduct').value = 'A & B';
  document.getElementById('slFilterType').value = 'checkin';
  document.getElementById('slFilterDateFrom').value = '2026-01-01';
  await nodes.get('slSliderToggle').click();
  assert.equal(nodes.get('slFirstBtn').hidden, true);
  assert.equal(nodes.get('slLastBtn').hidden, true);
  await nodes.get('slNextBtn').click();
  assert.equal(nodes.get('slFirstBtn').hidden, false);
  assert.equal(nodes.get('slLastBtn').hidden, false);
  await nodes.get('slLastBtn').click();
  const last = new URL(requests.at(-1), 'https://example.test').searchParams;
  assert.equal(last.get('log_page'), '3');
  assert.equal(last.get('log_product'), 'A & B');
  assert.equal(last.get('log_type'), 'checkin');
  assert.equal(last.get('log_date_from'), '2026-01-01');
  assert.equal(nodes.get('slNextBtn').disabled, true);
  assert.equal(nodes.get('slLastBtn').disabled, true);
  await nodes.get('slFirstBtn').click();
  assert.equal(new URL(requests.at(-1), 'https://example.test').searchParams.has('log_page'), false);
  assert.equal(nodes.get('slPageInfo').textContent, 'Page 1 of 3');
  assert.equal(nodes.get('slFirstBtn').hidden, true);
  assert.equal(nodes.get('slLastBtn').hidden, true);
});

test('label preview can jump to first and last sheets with keyboard focus retained', () => {
  const template = readFileSync(path.join(root, 'app/templates/label_printing.html'), 'utf8').replace(/\r\n/g, '\n');
  const start = template.indexOf('  function renderSheetPreview(');
  const end = template.indexOf('\n  }\n', start) + 4;
  assert.ok(start >= 0 && end > start);
  let buttons = [];
  let focused;
  const container = {
    set innerHTML(html) {
      buttons = [...html.matchAll(/<button\b([^>]*)>/g)].map(([, attrs]) => ({
        direction: /data-dir="([^"]+)"/.exec(attrs)[1], disabled: /\sdisabled/.test(attrs),
        getAttribute() { return this.direction; },
        addEventListener(type, handler) { this.handler = handler; },
        focus() { focused = this.direction; },
      }));
    },
    querySelectorAll() { return buttons; },
    querySelector(selector) {
      const direction = /data-dir="([^"]+)"/.exec(selector)[1];
      return buttons.find((button) => button.direction === direction && (!selector.includes(':not') || !button.disabled));
    },
  };
  const context = vm.createContext({
    document: { getElementById() { return container; } },
    getExpandedLabels: () => Array(25).fill({}), PREVIEW_PER_PAGE: 10,
    renderOnePage: () => '',
  });
  vm.runInContext(template.slice(start, end), context);
  context.renderSheetPreview(container);
  function click(direction) {
    const button = buttons.find((candidate) => candidate.direction === direction);
    assert.ok(button);
    assert.equal(button.disabled, false);
    button.handler.call(button, { preventDefault() {} });
  }
  assert.deepEqual(buttons.map((button) => button.direction), ['prev', 'next']);
  click('next');
  assert.deepEqual(buttons.map((button) => button.direction), ['first', 'prev', 'next', 'last']);
  click('last');
  assert.equal(container._previewState.currentPage, 2);
  assert.equal(buttons.find((button) => button.direction === 'last').disabled, true);
  assert.equal(focused, 'prev');
  click('first');
  assert.equal(container._previewState.currentPage, 0);
  assert.deepEqual(buttons.map((button) => button.direction), ['prev', 'next']);
  assert.equal(focused, 'next');
});
