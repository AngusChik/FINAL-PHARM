const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

const source = fs.readFileSync(path.join(__dirname, '..', 'sales_suggestions.js'), 'utf8');

class Element {
  constructor() {
    this.listeners = {};
    this.attributes = {};
    this.value = '';
    this.checked = false;
    this.hidden = false;
    this.children = [];
    this.dataset = {};
  }
  addEventListener(name, fn) { (this.listeners[name] ||= []).push(fn); }
  emit(name, extra = {}) {
    for (const fn of this.listeners[name] || []) fn({ preventDefault() {}, target: this, ...extra });
  }
  setAttribute(name, value) { this.attributes[name] = value; }
  appendChild(child) { child.parent = this; this.children.push(child); }
  remove() { this.parent.children = this.parent.children.filter(child => child !== this); }
  querySelectorAll() { return []; }
}

function setup({ url = 'http://localhost/sales/?tab=suggestions', active = true } = {}) {
  const ids = {};
  for (const id of ['sa-panel-suggestions', 'sa-suggestions-filters', 'sa-suggestions-content',
    'sa-suggestions-status', 'sa-suggestions-refresh', 'sa-suggestions-query',
    'sa-suggestions-category-count', 'sa-suggestions-clear', 'sa-filters']) ids[id] = new Element();
  const page = new Element();
  const panel = ids['sa-panel-suggestions'];
  panel.hidden = !active;
  panel.dataset.suggestionsUrl = '/low-stock/suggestions/';
  const snacks = new Element();
  const braces = new Element();
  const categories = ['10', '20'].map(value => Object.assign(new Element(), { value }));
  const form = ids['sa-suggestions-filters'];
  form.querySelector = selector => selector.includes('hide_snacks') ? snacks : braces;
  form.querySelectorAll = selector => selector.includes('category') ? categories : [];
  form.reset = () => {
    ids['sa-suggestions-query'].value = '';
    [snacks, braces, ...categories].forEach(input => { input.checked = false; });
  };
  ids['sa-filters'].querySelectorAll = () => ids['sa-filters'].children;
  const window = new Element();
  window.location = { href: url };
  const pushed = [];
  window.history = {
    state: { existing: true },
    pushState(state, title, next) { pushed.push(String(next)); window.location.href = String(next); }
  };
  const calls = [];
  const fetch = (requestedURL, options) => new Promise((resolve, reject) => {
    calls.push({ url: new URL(requestedURL), options, resolve, reject });
  });
  vm.runInNewContext(source, {
    URL, AbortController, window, fetch,
    document: {
      querySelector: () => page,
      getElementById: id => ids[id],
      createElement: () => new Element()
    }
  });
  const resolve = (index, html = '<article>Suggestions</article>') => calls[index].resolve({
    ok: true, headers: { get: () => 'application/json' },
    json: () => Promise.resolve({ html, count: 2 })
  });
  return { ids, page, panel, form, window, snacks, braces, categories, calls, pushed, resolve };
}

const flush = () => new Promise(resolve => setImmediate(resolve));

test('loads only on Suggestions and requests independent product filters', async () => {
  const app = setup({ active: false, url: 'http://localhost/sales/?tab=overview&start=2020-01-01&q=Alpha&category=10,20&hide_braces=1' });
  assert.equal(app.calls.length, 0);
  app.panel.hidden = false;
  app.page.emit('sales:tabchange', { detail: { key: 'suggestions' } });
  assert.equal(app.calls.length, 1);
  const params = app.calls[0].url.searchParams;
  assert.equal(params.get('q'), 'Alpha');
  assert.equal(params.get('category'), '10,20');
  assert.equal(params.get('hide_braces'), '1');
  assert.equal(params.has('start'), false);
  assert.equal(app.braces.checked, true);
  app.resolve(0);
  await flush();
  app.page.emit('sales:tabchange', { detail: { key: 'suggestions' } });
  assert.equal(app.calls.length, 1);
  assert.equal(app.ids['sa-suggestions-content'].attributes['aria-busy'], 'false');
});

test('applies combined filters, preserves chart dates, and clears only suggestion filters', async () => {
  const app = setup({ url: 'http://localhost/sales/?tab=suggestions&start=2026-09-01&ignore_snacks=1' });
  app.ids['sa-suggestions-query'].value = '  tablets  ';
  app.categories[0].checked = true;
  app.snacks.checked = true;
  app.braces.checked = true;
  app.form.emit('submit');
  const applied = new URL(app.window.location.href).searchParams;
  assert.equal(applied.get('q'), 'tablets');
  assert.equal(applied.get('category'), '10');
  assert.equal(applied.get('hide_braces'), '1');
  assert.equal(applied.get('hide_snacks'), '1');
  assert.equal(applied.get('start'), '2026-09-01');
  assert.equal(applied.get('ignore_snacks'), '1');
  assert.equal(app.calls[0].options.signal.aborted, true);
  assert.equal(app.ids['sa-filters'].children.length, 4);
  app.ids['sa-suggestions-clear'].emit('click');
  const cleared = new URL(app.window.location.href).searchParams;
  assert.equal(cleared.has('hide_braces'), false);
  assert.equal(cleared.has('q'), false);
  assert.equal(cleared.get('start'), '2026-09-01');
  assert.equal(app.ids['sa-filters'].children.length, 0);
});

test('back navigation during a pending request cannot display a stale filter result', async () => {
  const app = setup();
  app.resolve(0, '<article>All products</article>');
  await flush();
  app.braces.checked = true;
  app.braces.emit('change');
  assert.equal(app.calls.length, 2);
  app.window.location.href = 'http://localhost/sales/?tab=suggestions';
  app.window.emit('popstate');
  assert.equal(app.braces.checked, false);
  assert.equal(app.calls.length, 3);
  assert.equal(app.calls[1].options.signal.aborted, true);
  app.resolve(2, '<article>Restored all products</article>');
  await flush();
  app.resolve(1, '<article>Stale braces filter</article>');
  await flush();
  assert.equal(app.ids['sa-suggestions-content'].innerHTML, '<article>Restored all products</article>');
});

test('non-JSON login response shows an accessible error and can retry', async () => {
  const app = setup();
  app.calls[0].resolve({ ok: true, headers: { get: () => 'text/html' } });
  await flush();
  assert.match(app.ids['sa-suggestions-content'].innerHTML, /data-suggestions-retry/);
  assert.equal(app.ids['sa-suggestions-status'].textContent, 'Ordering suggestions could not be loaded.');
  assert.equal(app.ids['sa-suggestions-refresh'].disabled, false);
  app.ids['sa-suggestions-content'].emit('click', { target: { closest: () => true } });
  app.resolve(1);
  await flush();
  assert.equal(app.ids['sa-suggestions-status'].textContent, '2 ordering suggestions are ready.');
});
