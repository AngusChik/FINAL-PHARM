const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

const source = readFileSync(path.join(__dirname, '..', 'dashboard_reorder.js'), 'utf8');

// Exercise the real browser script using a small DOM and controllable requests.
class Element {
  constructor(document, tag = 'div', id = '') {
    this.ownerDocument = document;
    this.tagName = tag;
    this.id = id;
    this.dataset = {};
    this.attributes = {};
    this.children = [];
    this.listeners = new Map();
    this.parentElement = null;
    this.disabled = false;
    this.hidden = false;
    this.value = '';
    this._text = '';
    const classes = new Set();
    this.classList = {
      add: (value) => classes.add(value),
      contains: (value) => classes.has(value),
      toggle(value, enabled) { if (enabled) classes.add(value); else classes.delete(value); },
    };
  }
  set textContent(value) { this._text = String(value); this.children = []; }
  get textContent() { return this._text + this.children.map((child) => child.textContent).join(''); }
  set innerHTML(_value) { throw new Error('Add-to-list messages must not render HTML.'); }
  append(child) { child.parentElement = this; this.children.push(child); return child; }
  setAttribute(name, value) {
    this.attributes[name] = String(value);
    if (name.startsWith('data-')) this.dataset[name.slice(5).replace(/-([a-z])/g, (_, letter) => letter.toUpperCase())] = String(value);
  }
  removeAttribute(name) { delete this.attributes[name]; }
  matches(selector) {
    if (selector.startsWith('#')) return this.id === selector.slice(1);
    const className = selector.match(/^\.([\w-]+)/);
    if (className && !this.classList.contains(className[1])) return false;
    for (const match of selector.matchAll(/\[([\w-]+)\]/g)) if (!(match[1] in this.attributes)) return false;
    return !!className || selector.startsWith('[');
  }
  closest(selector) { return this.matches(selector) ? this : this.parentElement?.closest(selector) || null; }
  querySelectorAll(selector) {
    return this.children.flatMap((child) => [...(child.matches(selector) ? [child] : []), ...child.querySelectorAll(selector)]);
  }
  querySelector(selector) { return this.querySelectorAll(selector)[0] || null; }
  addEventListener(type, listener) {
    this.listeners.set(type, [...(this.listeners.get(type) || []), listener]);
  }
  dispatch(type, detail) {
    const event = { type, target: this, detail, preventDefault() { this.defaultPrevented = true; } };
    for (let node = this; node; node = node.parentElement) {
      for (const listener of node.listeners.get(type) || []) listener(event);
    }
  }
  focus() { this.ownerDocument.activeElement = this; }
}

function harness(initiallyAdded = false) {
  const document = new Element(null);
  document.ownerDocument = document;
  document.body = new Element(document, 'body');
  const nodes = new Map();
  document.getElementById = (id) => nodes.get(id) || null;
  function node(tag, id, parent = document) {
    const element = parent.append(new Element(document, tag, id));
    if (id) nodes.set(id, element);
    return element;
  }
  const card = node('div', 'card');
  card.classList.add('sidebar-reorder');
  card.dataset.addUrl = '/api/dashboard/reorder/add/';
  card.dataset.csrf = 'test-csrf-token';
  const list = node('div', 'reorderList', card);
  const boardStatus = node('div', 'reorderAddStatus', card);
  const modal = node('div', 'reorderModal');
  const modalBody = node('div', 'reorderModalBody', modal);
  const modalStatus = node('div', 'reorderModalStatus', modal);
  function control(id, parent = list, isAdded = false, quantity = '3') {
    const container = node('div', '', parent);
    container.classList.add('reorder-add-control');
    container.setAttribute('data-product-id', id);
    container.dataset.added = String(isAdded);
    const input = node('input', '', container);
    input.setAttribute('data-reorder-quantity', '');
    input.value = quantity;
    const button = node('button', '', container);
    button.setAttribute('data-reorder-add', '');
    button.setAttribute('data-requires-admin', '');
    const label = node('span', '', button);
    label.setAttribute('data-reorder-label', '');
    label.textContent = isAdded ? 'Added to Recently Purchased' : 'Add to Recently Purchased';
    const marker = node('span', '', button);
    marker.classList.add('ui-access-marker');
    marker.textContent = 'Admin';
    return { container, input, button, label, marker };
  }
  const board = control(1, list, initiallyAdded);
  const expanded = control(1, modalBody, initiallyAdded);
  const requests = [];
  const clock = { time: 10 };
  const fetch = (url, options) => new Promise((resolve, reject) => {
    requests.push({ url, options, body: JSON.parse(options.body), reject,
      resolve(data, status = 200, extra = {}) {
        resolve({ ok: status >= 200 && status < 300, redirected: false, json: async () => data, ...extra });
      }
    });
  });
  const context = vm.createContext({ document, fetch, performance: { now: () => clock.time }, TypeError });
  vm.runInContext(source, context);
  return { document, card, list, modal, modalBody, boardStatus, modalStatus, board, expanded, requests, clock, control,
    rerun: () => vm.runInContext(source, context),
    rendered(scope = modalBody, requestedAt) { document.dispatch('dashboard:reorder-rendered', { scope, requestedAt }); }
  };
}

async function settle() { await new Promise(setImmediate); }
function success(id = 1, extra = {}) { return { ok: true, product_id: id, recent_id: 20, quantity: 3, ...extra }; }

test('one CSRF-protected request synchronizes matching rows and retains admin indicators', async () => {
  const env = harness();
  env.rerun();
  env.board.button.focus();
  env.board.label.dispatch('click');
  env.expanded.button.dispatch('click');
  assert.equal(env.requests.length, 1);
  const request = env.requests[0];
  assert.deepEqual(request.body, { product_id: 1, quantity: 3 });
  assert.equal(request.options.method, 'POST');
  assert.equal(request.options.credentials, 'same-origin');
  assert.equal(request.options.cache, 'no-store');
  assert.equal(request.options.headers['X-CSRFToken'], 'test-csrf-token');
  assert.equal(request.options.headers['X-Requested-With'], 'XMLHttpRequest');
  for (const row of [env.board, env.expanded]) {
    assert.equal(row.button.disabled, true);
    assert.equal(row.input.hidden, false, 'row and quantity remain visible while saving');
    assert.equal(row.label.textContent, 'Adding…');
    assert.ok(row.button.children.includes(row.marker));
  }
  request.resolve(success());
  await settle();
  for (const row of [env.board, env.expanded]) {
    assert.equal(row.container.dataset.added, 'true');
    assert.equal(row.input.hidden, true);
    assert.equal(row.input.disabled, true);
    assert.equal(row.button.disabled, true);
    assert.equal(row.label.textContent, 'Added to Recently Purchased');
  }
  assert.equal(env.document.activeElement, env.boardStatus);
  assert.equal(env.boardStatus.textContent, 'Added to Recently Purchased.');
  env.board.button.dispatch('click');
  assert.equal(env.requests.length, 1);
});

test('invalid quantities never send a request and focus the quantity field', () => {
  const env = harness();
  for (const value of ['', '0', '-1', '2.5', '1e2', '10000', 'not-a-number']) {
    env.board.input.value = value;
    env.board.button.dispatch('click');
    assert.equal(env.requests.length, 0);
    assert.equal(env.document.activeElement, env.board.input);
    assert.equal(env.board.input.attributes['aria-invalid'], 'true');
    assert.equal(env.board.button.disabled, false);
    assert.match(env.boardStatus.textContent, /1 to 9,999/);
  }
  env.board.input.value = '9999';
  env.board.button.dispatch('click');
  assert.equal(env.requests[0].body.quantity, 9999);
  assert.equal(env.board.input.attributes['aria-invalid'], undefined);
});

test('keyboard focus follows the result after disabling the button unless the user has moved elsewhere', async () => {
  for (const movedElsewhere of [false, true]) {
    const env = harness();
    env.expanded.button.focus();
    env.expanded.button.dispatch('click');
    // Native browsers commonly move focus to body when an active button is disabled.
    env.document.activeElement = env.document.body;
    const nextInput = env.control(2).input;
    if (movedElsewhere) nextInput.focus();
    env.requests[0].resolve(success());
    await settle();
    assert.equal(env.document.activeElement, movedElsewhere ? nextInput : env.modalStatus);
  }
});

test('server errors remain safe text and retry uses the edited quantity', async () => {
  const env = harness();
  env.expanded.input.value = '7';
  env.expanded.button.dispatch('click');
  const message = 'Unlock admin access. <img src=x onerror=alert(1)>';
  env.requests[0].resolve({ ok: false, error: message }, 403);
  await settle();
  assert.equal(env.modalStatus.textContent, message);
  assert.equal(env.modalStatus.children.length, 0);
  assert.equal(env.modalStatus.classList.contains('reorder-add-error'), true);
  for (const row of [env.board, env.expanded]) {
    assert.equal(row.input.value, '7');
    assert.equal(row.input.hidden, false);
    assert.equal(row.input.disabled, false);
    assert.equal(row.button.disabled, false);
    assert.equal(row.label.textContent, 'Retry add');
  }
  assert.equal(env.requests.length, 1, 'failed requests are never replayed automatically');
  env.board.input.value = '4';
  env.board.button.dispatch('click');
  assert.equal(env.requests[1].body.quantity, 4);
  env.requests[1].resolve(success(1, { already_added: true, quantity: 2, message: 'Already on Recently Purchased.' }));
  await settle();
  assert.equal(env.boardStatus.textContent, 'Already on Recently Purchased.');
  assert.equal(env.modalStatus.classList.contains('reorder-add-error'), false);
  assert.equal(env.expanded.container.dataset.added, 'true');
});

test('network, login redirects, unreadable responses and wrong-product confirmations retain retry controls', async () => {
  for (const failure of ['network', 'redirect', 'html', 'wrong-product']) {
    const env = harness();
    env.board.button.dispatch('click');
    if (failure === 'network') env.requests[0].reject(new TypeError('Failed to fetch'));
    if (failure === 'redirect') env.requests[0].resolve({}, 200, { redirected: true });
    if (failure === 'html') env.requests[0].resolve({}, 500, { json: async () => { throw new Error('Invalid JSON'); } });
    if (failure === 'wrong-product') env.requests[0].resolve(success(2));
    await settle();
    assert.equal(env.board.label.textContent, 'Retry add', failure);
    assert.equal(env.board.button.disabled, false, failure);
    assert.equal(env.board.input.value, '3', failure);
    assert.equal(env.board.input.hidden, false, failure);
    assert.equal(env.requests.length, 1, failure);
    assert.equal(env.board.container.dataset.added, 'false', failure);
    if (failure === 'network') assert.match(env.boardStatus.textContent, /Check your connection/);
    if (failure === 'redirect') assert.match(env.boardStatus.textContent, /sign in/);
  }
});

test('new modal rows inherit pending saves, and stale responses cannot undo a local success', async () => {
  const env = harness();
  env.board.button.dispatch('click');
  const incoming = env.control(1, env.modalBody, false, '8');
  env.rendered(env.modalBody, 9);
  assert.equal(incoming.button.disabled, true);
  assert.equal(incoming.label.textContent, 'Adding…');
  assert.equal(incoming.input.value, '3');
  env.clock.time = 20;
  env.requests[0].resolve(success());
  await settle();
  assert.equal(incoming.container.dataset.added, 'true');
  const staleScope = new Element(env.document);
  env.modal.append(staleScope);
  const stale = env.control(1, staleScope, false);
  env.rendered(staleScope, 10);
  assert.equal(stale.container.dataset.added, 'true');
  assert.equal(stale.button.disabled, true);
  const freshScope = new Element(env.document);
  env.modal.append(freshScope);
  const fresh = env.control(1, freshScope, false);
  env.rendered(freshScope, 30);
  assert.equal(fresh.container.dataset.added, 'false');
  assert.equal(env.board.button.disabled, false, 'a later server response reflects removal elsewhere');
  assert.equal(env.board.input.hidden, false);
  assert.equal(env.requests.length, 1, 'rendering a modal does not add products');
});

test('server-rendered existing additions synchronize and plain render events preserve local saves', async () => {
  const env = harness(true);
  assert.equal(env.board.button.disabled, true);
  assert.equal(env.expanded.input.hidden, true);
  env.board.button.dispatch('click');
  assert.equal(env.requests.length, 0);
  const another = env.control(2, env.list);
  env.rendered(env.list);
  another.button.dispatch('click');
  env.requests[0].resolve(success(2));
  await settle();
  const incoming = env.control(2, env.modalBody, false);
  env.document.dispatch('dashboard:reorder-rendered');
  assert.equal(incoming.button.disabled, true);
  assert.equal(incoming.container.dataset.added, 'true');
});
