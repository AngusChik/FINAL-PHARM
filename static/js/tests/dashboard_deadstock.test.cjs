const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

const source = readFileSync(path.join(__dirname, '..', 'dashboard_deadstock.js'), 'utf8');

// This deliberately small DOM supplies only the browser surfaces used here.
// Tests run the real script and interact through events and deferred fetches.
class Element {
  constructor(document, id = '', tag = 'div') {
    this.ownerDocument = document;
    this.id = id;
    this.tagName = tag;
    this.dataset = {};
    this.attributes = {};
    this.children = [];
    this.listeners = new Map();
    this.parentElement = null;
    this.hidden = false;
    this.disabled = false;
    this.tabIndex = 0;
    this.offsetParent = {};
    this.textContent = '';
    this._html = '';
    const classes = new Set();
    this.classList = {
      add: (name) => classes.add(name),
      remove: (name) => classes.delete(name),
      contains: (name) => classes.has(name),
      toggle: (name, force) => {
        const enabled = force === undefined ? !classes.has(name) : force;
        if (enabled) classes.add(name); else classes.delete(name);
        return enabled;
      },
    };
  }

  append(child) {
    child.parentElement = this;
    this.children.push(child);
    return child;
  }

  setAttribute(name, value) {
    this.attributes[name] = String(value);
    if (name.startsWith('data-')) {
      const key = name.slice(5).replace(/-([a-z])/g, (_, letter) => letter.toUpperCase());
      this.dataset[key] = String(value);
    }
    if (name === 'disabled') this.disabled = true;
  }

  set innerHTML(value) {
    this._html = String(value);
    for (const child of this.children) child.parentElement = null;
    this.children = [];
    // Dynamic controls need real nodes for delegated clicks and focus handling.
    for (const match of this._html.matchAll(/<button\b([^>]*)>/g)) {
      const button = this.append(new Element(this.ownerDocument, '', 'button'));
      for (const attr of match[1].matchAll(/([\w-]+)(?:="([^"]*)")?/g)) {
        button.setAttribute(attr[1], attr[2] || '');
      }
    }
  }

  get innerHTML() { return this._html; }

  insertAdjacentHTML(_position, value) {
    this.innerHTML = this._html + value;
  }

  matches(selector) {
    return selector.split(',').some((part) => {
      part = part.trim();
      if (part.startsWith('#')) return this.id === part.slice(1);
      if (part.includes(' ')) return false;
      if (part.startsWith('button') && this.tagName !== 'button') return false;
      if (part.includes(':not([disabled])') && this.disabled) return false;
      part = part.replace(':not([disabled])', '');
      for (const match of part.matchAll(/\[([\w-]+)(?:="([^"]*)")?\]/g)) {
        if (!(match[1] in this.attributes)) return false;
        if (match[2] !== undefined && this.attributes[match[1]] !== match[2]) return false;
      }
      return part.startsWith('[') || part.startsWith('button');
    });
  }

  querySelectorAll(selector) {
    const found = [];
    for (const child of this.children) {
      if (child.matches(selector)) found.push(child);
      found.push(...child.querySelectorAll(selector));
    }
    return found;
  }

  querySelector(selector) { return this.querySelectorAll(selector)[0] || null; }
  closest(selector) {
    return this.matches(selector) ? this : this.parentElement?.closest(selector) || null;
  }
  contains(element) {
    return element === this || this.children.some((child) => child.contains(element));
  }
  focus() { this.ownerDocument.activeElement = this; }
  addEventListener(type, listener) {
    const listeners = this.listeners.get(type) || [];
    listeners.push(listener);
    this.listeners.set(type, listeners);
  }
  dispatch(type, properties = {}) {
    const event = {
      type, target: this, ...properties,
      preventDefault() { this.defaultPrevented = true; },
      stopPropagation() { this.stopped = true; },
    };
    let element = this;
    while (element) {
      event.currentTarget = element;
      for (const listener of element.listeners.get(type) || []) listener(event);
      if (event.stopped) break;
      element = element.parentElement;
    }
  }
}

function harness() {
  const document = new Element(null, 'document');
  document.ownerDocument = document;
  const nodes = new Map();
  document.getElementById = (id) => nodes.get(id) || null;
  function add(id, parent = document, tag = 'div') {
    const element = parent.append(new Element(document, id, tag));
    nodes.set(id, element);
    return element;
  }
  const list = add('dsListContainer');
  list.dataset.apiUrl = '/api/dashboard/deadstock/';
  list.dataset.expandUrl = '/dashboard/expand/';
  list.dataset.csrf = 'test-csrf-token';
  const modal = add('deadstockModal');
  const modalBody = add('deadstockModalBody', modal);
  add('deadstockModalSub', modal);
  add('dsModalStatus', modal);
  add('dsModalRetry', modal, 'button').hidden = true;
  const close = add('close', modal, 'button');
  close.setAttribute('data-dx-close', '');
  const tabs = {};
  for (const tab of ['all', 'dismissed']) {
    tabs[tab] = add(`tab-${tab}`, modal, 'button');
    tabs[tab].setAttribute('data-ds-tab', tab);
  }
  for (const id of ['dsFirst', 'dsPrev', 'dsNext', 'dsLast']) add(id, modal, 'button');
  add('dsPageLabel', modal);
  for (const id of ['dsRetry', 'dsUndo', 'ds-ignore-snacks', 'ds-ignore-braces', 'deadstockExpand']) {
    add(id, document, 'button');
  }
  nodes.get('dsRetry').hidden = true;
  nodes.get('dsUndo').hidden = true;
  for (const id of ['dsStatus', 'dsDismissedCount', 'dsVisibleCount', 'dsSummaryText']) add(id);
  const window = new Element(document, 'window');
  const storage = new Map();
  const requests = [];
  function fetch(url, options) {
    return new Promise((resolve, reject) => {
      requests.push({
        url, options, body: options.body ? JSON.parse(options.body) : null,
        resolve(data, status = 200) {
          resolve({ ok: status >= 200 && status < 300, json: async () => data });
        },
        reject,
      });
    });
  }
  const context = vm.createContext({
    document, window, fetch, AbortController,
    localStorage: {
      getItem: (key) => storage.get(key) || null,
      setItem: (key, value) => storage.set(key, value),
    },
  });
  vm.runInContext(source, context);
  return {
    document, window, nodes, list, modal, modalBody, tabs, close, requests,
    rerun: () => vm.runInContext(source, context),
  };
}

function item(productId, overrides = {}) {
  return {
    product_id: productId, name: `Product ${productId}`, quantity_in_stock: 12,
    capital_tied: 24, days_since_sale: 'Never', category_name: 'Health', dismissed: false,
    ...overrides,
  };
}

function batch(ids, extra = {}) {
  return {
    ok: true, items: ids.map((id) => item(id)), available_count: ids.length,
    total_count: ids.length, filtered_count: 0, dismissed_count: 0, ...extra,
  };
}

async function settle() {
  await new Promise(setImmediate);
  await new Promise(setImmediate);
}

async function click(element) {
  assert.ok(element, 'the requested control exists');
  assert.equal(element.disabled, false, 'the requested control is enabled');
  element.dispatch('click');
  await settle();
}

function rowIds(env) {
  return env.list.querySelectorAll('[data-ds-action]').map((node) => Number(node.dataset.productId));
}

test('dismissed history can jump to both ends and disables Last on the last page', async () => {
  const env = harness();
  await enter(env);
  await click(env.nodes.get('deadstockExpand'));
  await click(env.tabs.dismissed);
  function resolvePage(page) {
    env.requests.at(-1).resolve({
      ok: true, items: [item(page, { dismissed: true })], count: 60,
      dismissed_count: 60, page, num_pages: 3,
      has_previous: page > 1, has_next: page < 3,
    });
  }
  resolvePage(1);
  await settle();
  assert.equal(env.nodes.get('dsFirst').hidden, true);
  assert.equal(env.nodes.get('dsLast').hidden, true);
  await click(env.nodes.get('dsNext'));
  assert.equal(env.requests.at(-1).url, '/api/dashboard/deadstock/?page=2');
  resolvePage(2);
  await settle();
  assert.equal(env.nodes.get('dsFirst').hidden, false);
  assert.equal(env.nodes.get('dsLast').hidden, false);
  await click(env.nodes.get('dsLast'));
  assert.equal(env.requests.at(-1).url, '/api/dashboard/deadstock/?page=3');
  resolvePage(3);
  await settle();
  assert.equal(env.nodes.get('dsLast').disabled, true);
  assert.equal(env.nodes.get('dsNext').disabled, true);
  await click(env.nodes.get('dsFirst'));
  assert.equal(env.requests.at(-1).url, '/api/dashboard/deadstock/?page=1');
  resolvePage(1);
  await settle();
  assert.equal(env.nodes.get('dsFirst').hidden, true);
  assert.equal(env.nodes.get('dsLast').hidden, true);
});

async function enter(env, ids = [1, 2, 3]) {
  env.window.dispatch('pageshow', { persisted: false });
  await settle();
  env.requests.at(-1).resolve(batch(ids));
  await settle();
}

test('rotates once on each page entry, including cached Back, and ignores focus/visibility events', async () => {
  const env = harness();
  assert.equal(env.requests.length, 0, 'script evaluation itself does not advance');
  env.document.dispatch('DOMContentLoaded');
  env.window.dispatch('DOMContentLoaded');
  await settle();
  assert.equal(env.requests.length, 0);

  await enter(env);
  assert.equal(env.requests.length, 1);
  assert.equal(env.requests[0].body.action, 'next');
  env.document.dispatch('focus');
  env.window.dispatch('focus');
  env.document.dispatch('visibilitychange');
  await settle();
  assert.equal(env.requests.length, 1, 'background/foreground changes do not consume suggestions');

  env.window.dispatch('pageshow', { persisted: true });
  await settle();
  assert.equal(env.requests.length, 2, 'cached Back navigation requests exactly one new batch');
  assert.equal(env.requests[1].body.action, 'next');
  env.requests[1].resolve(batch([4, 5, 6]));
  await settle();
  assert.deepEqual(rowIds(env), [4, 5, 6]);

  env.rerun();
  env.window.dispatch('pageshow', { persisted: true });
  await settle();
  assert.equal(env.requests.length, 3, 'a duplicate script inclusion does not duplicate lifecycle listeners');
  env.requests[2].resolve(batch([7, 8, 9]));
  await settle();
});

test('failed dismiss keeps rows; retry sends CSRF/current IDs and success supports shared undo', async () => {
  const env = harness();
  await enter(env);
  await click(env.list.querySelector('button[data-product-id="2"]'));
  const attempt = env.requests[1];
  assert.equal(attempt.options.method, 'POST');
  assert.equal(attempt.options.headers['X-CSRFToken'], 'test-csrf-token');
  assert.equal(attempt.options.credentials, 'same-origin');
  assert.deepEqual(attempt.body, {
    action: 'dismiss', product_id: 2, current_ids: [1, 2, 3],
    exclude_snacks: false, exclude_braces: false,
  });
  assert.deepEqual(rowIds(env), [1, 2, 3], 'the row remains visible while saving');
  attempt.reject(new Error('Connection lost'));
  await settle();
  assert.deepEqual(rowIds(env), [1, 2, 3], 'an unsuccessful dismissal never removes the row');
  assert.equal(env.nodes.get('dsRetry').hidden, false);
  assert.match(env.nodes.get('dsStatus').textContent, /Connection lost/);
  assert.equal(env.list.querySelector('button[data-product-id="2"]').disabled, false);

  await click(env.nodes.get('dsRetry'));
  const retry = env.requests[2];
  assert.deepEqual(retry.body, attempt.body, 'retry repeats the same requested dismissal');
  retry.resolve(batch([1, 4, 3], {
    dismissed_count: 1, product_id: 2, dismissed: true,
    expires_at: '2026-10-08T12:00:00Z',
  }));
  await settle();
  assert.deepEqual(rowIds(env), [1, 4, 3], 'the replacement occupies the dismissed row slot');
  assert.equal(env.nodes.get('dsRetry').hidden, true);
  assert.equal(env.nodes.get('dsUndo').hidden, false);
  assert.match(env.nodes.get('dsStatus').textContent, /Product 2 dismissed for everyone until/);

  await click(env.nodes.get('dsUndo'));
  const restore = env.requests[3];
  assert.equal(restore.body.action, 'restore');
  assert.equal(restore.body.product_id, 2);
  assert.deepEqual(restore.body.current_ids, [1, 4, 3]);
  assert.equal(restore.options.headers['X-CSRFToken'], 'test-csrf-token');
  restore.resolve(batch([1, 4, 3], { product_id: 2, dismissed: false, expires_at: null }));
  await settle();
  assert.equal(env.nodes.get('dsUndo').hidden, true);
  assert.match(env.nodes.get('dsStatus').textContent, /restored for everyone/);
});

test('ignores stale modal results after tab changes and closing, even if aborted fetch resolves', async () => {
  const env = harness();
  await enter(env);
  await click(env.nodes.get('deadstockExpand'));
  const oldReport = env.requests[1];
  assert.equal(oldReport.url, '/dashboard/expand/?section=deadstock');
  await click(env.tabs.dismissed);
  const dismissed = env.requests[2];
  assert.equal(dismissed.url, '/api/dashboard/deadstock/?page=1');
  assert.equal(oldReport.options.signal.aborted, true);
  dismissed.resolve({
    ok: true, items: [item(9, { dismissed: true, expires_at: '2026-10-08T12:00:00Z' })],
    count: 1, dismissed_count: 1, page: 1, num_pages: 1, has_previous: false, has_next: false,
  });
  await settle();
  assert.match(env.modalBody.innerHTML, /Product 9/);
  assert.match(env.modalBody.innerHTML, /data-ds-action="restore"/);
  const currentHtml = env.modalBody.innerHTML;
  oldReport.resolve({ ok: true, items: [item(8)], count: 1, dismissed_count: 1 });
  await settle();
  assert.equal(env.modalBody.innerHTML, currentHtml, 'late full report cannot replace the selected Dismissed tab');

  await click(env.tabs.all);
  const closedRequest = env.requests[3];
  await click(env.close);
  assert.equal(env.modal.classList.contains('open'), false);
  assert.equal(closedRequest.options.signal.aborted, true);
  const closedHtml = env.modalBody.innerHTML;
  closedRequest.resolve({ ok: true, items: [item(7)], count: 1, dismissed_count: 1 });
  await settle();
  assert.equal(env.modalBody.innerHTML, closedHtml, 'closing invalidates pending modal rendering');
  assert.equal(env.document.activeElement, env.nodes.get('deadstockExpand'));
  assert.equal(env.requests.filter((request) => request.body?.action === 'next').length, 1,
    'opening, changing and closing report tabs never advances dashboard rotation');
});

for (const action of ['dismiss', 'restore']) {
  test(`failed modal ${action} keeps the table and retries the same action without rotating the board`, async () => {
    const env = harness();
    const initiallyDismissed = action === 'restore';
    const expiresAt = '2026-10-08T12:00:00Z';
    await enter(env);
    await click(env.nodes.get('deadstockExpand'));
    env.requests[1].resolve({
      ok: true,
      items: [item(7, { dismissed: initiallyDismissed, expires_at: initiallyDismissed ? expiresAt : null }), item(8)],
      count: 2, dismissed_count: Number(initiallyDismissed),
    });
    await settle();

    const originalTable = env.modalBody.innerHTML;
    const modalButton = env.modalBody.querySelector('button[data-product-id="7"]');
    assert.equal(modalButton.dataset.dsAction, action);
    await click(modalButton);
    const attempt = env.requests[2];
    assert.equal(attempt.body.action, action);
    assert.equal(attempt.body.product_id, 7, 'the expanded list can act on a product outside the current board');
    assert.deepEqual(attempt.body.current_ids, [1, 2, 3]);
    assert.equal(env.modalBody.innerHTML, originalTable, 'the table stays visible while saving');
    assert.equal(modalButton.disabled, true);
    assert.deepEqual(rowIds(env), [1, 2, 3]);

    const failure = 'Could not save the product. Please try again.';
    if (action === 'dismiss') attempt.resolve({ ok: false, error: failure }, 503);
    else attempt.reject(new Error(failure));
    await settle();
    assert.equal(env.modalBody.innerHTML, originalTable, 'failed mutations keep all expanded table rows visible');
    assert.deepEqual(rowIds(env), [1, 2, 3]);
    assert.equal(modalButton.disabled, false);
    assert.equal(env.nodes.get('dsModalStatus').textContent, failure);
    assert.equal(env.nodes.get('dsModalStatus').classList.contains('ds-error'), true);
    assert.equal(env.nodes.get('dsModalRetry').hidden, false, 'retry is available inside the still-open modal');
    assert.equal(env.modal.classList.contains('open'), true);

    await click(env.nodes.get('dsModalRetry'));
    const retry = env.requests[3];
    assert.deepEqual(retry.body, attempt.body, 'modal retry preserves the original action, product, filters and retained IDs');
    assert.equal(retry.options.headers['X-CSRFToken'], 'test-csrf-token');
    assert.equal(env.nodes.get('dsModalRetry').hidden, true);
    retry.resolve(batch([1, 2, 3], {
      product_id: 7, dismissed: !initiallyDismissed,
      expires_at: initiallyDismissed ? null : expiresAt,
      dismissed_count: Number(!initiallyDismissed),
    }));
    await settle();
    assert.deepEqual(rowIds(env), [1, 2, 3], 'a successful modal action preserves unaffected board suggestions');
    const refreshedReport = env.requests[4];
    assert.equal(refreshedReport.url, '/dashboard/expand/?section=deadstock');
    assert.equal(refreshedReport.body, null, 'success refreshes only the expanded report');
    refreshedReport.resolve({
      ok: true,
      items: [item(7, { dismissed: !initiallyDismissed, expires_at: initiallyDismissed ? null : expiresAt }), item(8)],
      count: 2, dismissed_count: Number(!initiallyDismissed),
    });
    await settle();

    const updatedButton = env.modalBody.querySelector('button[data-product-id="7"]');
    assert.equal(updatedButton.dataset.dsAction, initiallyDismissed ? 'dismiss' : 'restore');
    assert.equal(updatedButton.disabled, false);
    assert.equal(env.nodes.get('dsModalRetry').hidden, true);
    assert.equal(env.nodes.get('dsModalStatus').classList.contains('ds-error'), false);
    assert.match(env.nodes.get('dsModalStatus').textContent, initiallyDismissed ? /restored for everyone/ : /Product 7 dismissed for everyone until/);
    assert.deepEqual(rowIds(env), [1, 2, 3]);
    assert.equal(env.requests.filter((request) => request.body?.action === 'next').length, 1,
      'report opening, mutation failures and retries never advance board rotation');
  });
}
