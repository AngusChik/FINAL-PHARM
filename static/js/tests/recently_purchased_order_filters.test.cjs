const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

const source = readFileSync(path.join(__dirname, '../recently_purchased_order_filters.js'), 'utf8');
const template = readFileSync(path.join(__dirname, '../../../app/templates/low_stock.html'), 'utf8');
const automationSource = template.slice(template.indexOf('// ── Automate Ordering:'), template.lastIndexOf('</script>'));

function element() {
  const listeners = {};
  const classes = new Set();
  return {
    dataset: {}, style: {}, attributes: {}, value: '', checked: false, disabled: false, hidden: false, textContent: '',
    classList: {
      contains: name => classes.has(name), add: name => classes.add(name), remove: name => classes.delete(name),
      toggle(name, enabled) { if (enabled === undefined) enabled = !classes.has(name); if (enabled) classes.add(name); else classes.delete(name); },
    },
    setAttribute(name, value) { this.attributes[name] = String(value); },
    addEventListener(type, listener) { (listeners[type] ||= []).push(listener); },
    emit(type, target = this) { for (const listener of listeners[type] || []) listener({ target }); },
    querySelectorAll() { return []; }, querySelector() { return null; },
    get innerHTML() { return this.html || this.textContent; }, set innerHTML(value) { this.html = value; },
  };
}

function memoryStorage(initial) {
  const data = new Map(initial ? [['rp_ao_category_filters:v1', initial]] : []);
  return { getItem: key => data.get(key) || null, setItem: (key, value) => data.set(key, value) };
}

function harness(names = ['Snacks', 'Braces', 'Vitamins'], storage = memoryStorage(), setup = true) {
  const nodes = new Map();
  function id(name) { if (!nodes.has(name)) nodes.set(name, element()); return nodes.get(name); }
  const checks = names.map((name, index) => {
    const cb = element();
    cb.value = String(index + 1);
    cb.dataset = { name: name.toLowerCase(), label: name };
    cb.classList.add('rp-ao-cat-check');
    cb.row = element();
    cb.closest = () => cb.row;
    return cb;
  });
  const vendors = ['mck', 'kf'].map(value => Object.assign(element(), { value, checked: true }));
  id('rp-ao-cats').querySelectorAll = () => checks;
  id('rp-ao-vendors').querySelectorAll = () => vendors;
  id('rp-ao-modal').querySelector = selector => id(selector.slice(1));
  const context = vm.createContext({
    window: { localStorage: storage }, document: { getElementById: id, createElement: element },
    showToast() {}, setTimeout, clearTimeout, console,
  });
  vm.runInContext(source, context);
  const filters = setup ? context.window.RecentlyPurchasedOrderFilters.init(id('rp-ao-modal')) : null;
  return {
    id, checks, vendors, context, filters, storage,
    excluded: () => Array.from(filters.excludedCategoryIds()),
    check(index, checked) { checks[index].checked = checked; id('rp-ao-cats').emit('change', checks[index]); },
  };
}

test('snacks stay excluded by default and braces match category names without case sensitivity', () => {
  const h = harness(['SNACKS', 'bRaCeS', 'Vitamins']);
  assert.deepEqual(h.excluded(), [1]);
  h.id('rp-ao-braces-toggle').emit('click');
  assert.deepEqual(h.excluded(), [1, 2]);
  assert.equal(h.id('rp-ao-braces-toggle').attributes['aria-checked'], 'true');
  h.check(1, false);
  assert.deepEqual(h.excluded(), [1]);
  assert.equal(h.id('rp-ao-braces-toggle').attributes['aria-checked'], 'false');
});

test('custom choices survive reopening and page reload, including explicit snacks inclusion', () => {
  const h = harness();
  h.check(0, false);
  h.check(1, true);
  h.check(2, true);
  h.filters.refresh();
  assert.deepEqual(h.excluded(), [2, 3]);
  const reloaded = harness(undefined, h.storage);
  assert.deepEqual(reloaded.excluded(), [2, 3]);
  assert.equal(reloaded.id('rp-ao-snacks-toggle').attributes['aria-checked'], 'false');
  assert.match(reloaded.id('rp-ao-filter-summary').textContent, /Saved on this browser/);
});

test('exclude all, include all and reset synchronize both quick switches and the custom checklist', () => {
  const h = harness();
  h.id('rp-ao-cat-all').emit('click');
  assert.deepEqual(h.excluded(), [1, 2, 3]);
  assert.equal(h.id('rp-ao-braces-toggle').attributes['aria-checked'], 'true');
  h.id('rp-ao-cat-none').emit('click');
  assert.deepEqual(h.excluded(), []);
  assert.equal(h.id('rp-ao-snacks-toggle').attributes['aria-checked'], 'false');
  h.id('rp-ao-cat-reset').emit('click');
  assert.deepEqual(h.excluded(), [1]);
  assert.equal(h.id('rp-ao-snacks-toggle').attributes['aria-checked'], 'true');
  assert.equal(h.id('rp-ao-braces-toggle').attributes['aria-checked'], 'false');
  assert.deepEqual(harness(undefined, h.storage).excluded(), [1]);
});

test('category search keeps hidden exclusions and all-category controls retain their full scope', () => {
  const h = harness();
  h.id('rp-ao-cat-search').value = 'BRACE';
  h.id('rp-ao-cat-search').emit('input');
  assert.deepEqual(h.checks.map(cb => cb.row.hidden), [true, false, true]);
  assert.deepEqual(h.excluded(), [1]);
  h.id('rp-ao-cat-all').emit('click');
  assert.deepEqual(h.excluded(), [1, 2, 3]);
  h.id('rp-ao-cat-search').value = 'does not exist';
  h.id('rp-ao-cat-search').emit('input');
  assert.equal(h.id('rp-ao-cat-search-empty').hidden, false);
  h.id('rp-ao-cat-reset').emit('click');
  assert.equal(h.id('rp-ao-cat-search').value, '');
  assert.deepEqual(h.checks.map(cb => cb.row.hidden), [false, false, false]);
});

test('missing categories are visibly unavailable and cannot create phantom exclusions', () => {
  const h = harness(['Vitamins']);
  for (const name of ['snacks', 'braces']) {
    assert.equal(h.id('rp-ao-' + name + '-toggle').disabled, true);
    assert.equal(h.id('rp-ao-' + name + '-toggle').attributes['aria-checked'], 'false');
    assert.match(h.id('rp-ao-' + name + '-description').textContent, /No .* category in the current list/);
    h.id('rp-ao-' + name + '-toggle').emit('click');
  }
  assert.deepEqual(h.excluded(), []);
});

test('storage failures and invalid saved values do not break ordering filters', () => {
  const blocked = { getItem() { throw Error('blocked'); }, setItem() { throw Error('blocked'); } };
  const h = harness(undefined, blocked);
  h.id('rp-ao-braces-toggle').emit('click');
  assert.deepEqual(h.excluded(), [1, 2]);
  const malformed = harness(undefined, memoryStorage('{not json'));
  assert.deepEqual(malformed.excluded(), [1]);
  const invalid = harness(undefined, memoryStorage(JSON.stringify({ choices: { 1: 'false', 2: true }, defaults: { snacks: 'no' } })));
  assert.deepEqual(invalid.excluded(), [1, 2]);
});

test('preview sends synchronized exclusions and start preserves the reviewed items and edited quantities', async () => {
  const dependency = template.indexOf("{% static 'js/recently_purchased_order_filters.js' %}");
  assert.ok(dependency >= 0 && dependency < template.indexOf('// ── Automate Ordering:'), 'Load filter controls before the ordering workflow.');
  const h = harness(undefined, undefined, false);
  const requests = [];
  const reviewed = { product_id: 35, barcode: '12345678', name: 'Vitamin', quantity: 3 };
  h.context.fetch = async (url, options = {}) => {
    const body = options.body ? JSON.parse(options.body) : null;
    requests.push({ url, body });
    if (url.includes('mckesson_order_preview')) return { json: async () => ({ ok: true, items: [reviewed], skipped: [] }) };
    if (url.includes('supplier_order_plan') && body) {
      // Stop at plan creation so this test never starts a supplier runner.
      return { ok: false, status: 409, json: async () => ({ error: 'Test stops before supplier execution.' }) };
    }
    return { json: async () => ({ state: 'idle', plan: null }) };
  };
  vm.runInContext(automationSource, h.context);
  h.id('rp-ao-braces-toggle').emit('click');
  h.id('rp-ao-btn').emit('click');
  h.id('rp-ao-preview-btn').emit('click');
  await new Promise(resolve => setImmediate(resolve));
  const preview = requests.find(request => request.url.includes('mckesson_order_preview'));
  assert.deepEqual(preview.body, { exclude_category_ids: [1, 2] });
  assert.equal(h.id('rp-ao-preview-filters').textContent, 'Excluded: Snacks, Braces.');
  h.id('rp-ao-preview-list').querySelector = () => ({ value: '7' });
  h.id('rp-ao-start').emit('click');
  await new Promise(resolve => setImmediate(resolve));
  const plan = requests.find(request => request.body?.action === 'create');
  assert.deepEqual(plan.body.items, [{ ...reviewed, quantity: 7 }]);
  assert.deepEqual(plan.body.seq, ['mck', 'kf']);
  assert.equal(requests.some(request => request.url.includes('order_start')), false);
});
