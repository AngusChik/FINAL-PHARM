const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

const source = readFileSync(path.join(__dirname, '..', 'expired_log_layout.js'), 'utf8');

function harness({ selected = 0, count = 5, width = 1366, height = 768 } = {}) {
  const nodes = new Map();
  const resizeListeners = [];
  const lotChangeListeners = [];
  let changes = 0;
  function node(id) {
    const listeners = new Map();
    const element = {
      disabled: false, hidden: false, textContent: '',
      setAttribute() {},
      addEventListener(type, listener) { listeners.set(type, listener); },
      click() { if (!this.disabled) listeners.get('click')?.(); },
    };
    nodes.set(id, element);
    return element;
  }
  const rows = Array.from({ length: count }, (_, index) => {
    const input = {
      checked: index === selected,
      matches() { return this.checked; },
      dispatchEvent(event) {
        changes += 1;
        event.target = input;
        lotChangeListeners.forEach(listener => listener(event));
        return true;
      },
    };
    return {
      input, hidden: false,
      querySelector(selector) { return selector === 'input:checked' && input.checked ? input : null; },
      contains(candidate) { return candidate === input; },
    };
  });
  const lots = {
    querySelectorAll: () => rows,
    addEventListener(type, listener) { if (type === 'change') lotChangeListeners.push(listener); },
  };
  nodes.set('expiryCollection', { dataset: { userId: '123' }, querySelector: () => lots });
  ['expiryLotPrev', 'expiryLotNext', 'expiryLotPage', 'expiryLotPagination'].forEach(node);
  const window = {
    innerHeight: height,
    matchMedia: () => ({ matches: width <= 1100 }),
    sessionStorage: { getItem: () => null },
    addEventListener(type, listener) { if (type === 'resize') resizeListeners.push(listener); },
  };
  vm.runInNewContext(source, {
    document: { readyState: 'complete', getElementById: id => nodes.get(id) || null },
    window,
    Event: class { constructor(type, options) { this.type = type; this.bubbles = options.bubbles; } },
  });
  return {
    rows, nodes,
    get changes() { return changes; },
    select(index) {
      rows.forEach((row, position) => { row.input.checked = position === index; });
      rows[index].input.dispatchEvent({ type: 'change', bubbles: true });
    },
    resize(nextWidth, nextHeight) {
      width = nextWidth;
      window.innerHeight = nextHeight;
      resizeListeners.forEach(listener => listener());
    },
  };
}

test('paging away clears the previously selected lot and notifies collection controls', () => {
  const page = harness();
  assert.equal(page.rows[0].input.checked, true);
  assert.equal(page.changes, 0);
  page.nodes.get('expiryLotNext').click();
  assert.equal(page.rows[0].hidden, true);
  assert.equal(page.rows[0].input.checked, false);
  assert.equal(page.rows.some(row => row.input.checked), false);
  assert.equal(page.changes, 1);
  page.nodes.get('expiryLotPrev').click();
  assert.equal(page.rows[0].hidden, false);
  assert.equal(page.rows[0].input.checked, false, 'Returning to a page must not silently reselect its lot.');
});

test('selecting a visible lot remains valid until a later page hides that exact row', () => {
  const page = harness({ selected: -1 });
  page.nodes.get('expiryLotNext').click();
  page.select(2);
  assert.equal(page.rows[2].hidden, false);
  assert.equal(page.rows[2].input.checked, true);
  page.nodes.get('expiryLotNext').click();
  assert.equal(page.rows[2].input.checked, false);
  assert.equal(page.rows[4].hidden, false);
  assert.equal(page.rows.some(row => row.hidden && row.input.checked), false);
});

test('initial pagination reveals a selected lot beyond the first page without clearing it', () => {
  const page = harness({ selected: 4 });
  assert.equal(page.rows[4].hidden, false);
  assert.equal(page.rows[4].input.checked, true);
  assert.equal(page.changes, 0);
  assert.equal(page.nodes.get('expiryLotPage').textContent, '5–5 of 5');
});

test('changing page size preserves the selected lot rather than hiding or clearing it', () => {
  const page = harness({ selected: 3, width: 390, height: 844 });
  assert.equal(page.rows[3].hidden, false);
  page.resize(1366, 768);
  assert.equal(page.rows[3].hidden, false);
  assert.equal(page.rows[3].input.checked, true);
  assert.equal(page.nodes.get('expiryLotPage').textContent, '3–4 of 5');
  page.resize(1440, 900);
  assert.equal(page.rows[3].hidden, false);
  assert.equal(page.rows[3].input.checked, true);
  assert.equal(page.nodes.get('expiryLotPage').textContent, '4–5 of 5');
  assert.equal(page.changes, 0);
});
