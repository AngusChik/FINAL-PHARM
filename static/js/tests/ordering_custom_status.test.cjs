const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

const template = readFileSync(path.join(__dirname, '../../../app/templates/partials/_ordering_sheet.html'), 'utf8');
// Execute the actual editor helpers and delegated handlers with controlled
// network responses, so immediate UI and failure behavior can be asserted.
const source = template.slice(template.indexOf('    function customStatusFormForSelect('),
  template.indexOf('        // Shift-click selects')) + '\n}';

function harness(status = 'pending', saved = '') {
  const listeners = {}, timers = [], requests = [];
  const document = { body: {}, activeElement: null };
  document.activeElement = document.body;
  function node(classes = '') {
    const names = new Set(classes.split(' '));
    return {
      dataset: {}, attributes: {}, hidden: false, disabled: false, value: '', textContent: '', isConnected: true,
      classList: { contains: name => names.has(name), toggle(name, yes) { if (yes) names.add(name); else names.delete(name); } },
      setAttribute(name, value) { this.attributes[name] = value; },
      removeAttribute(name) { delete this.attributes[name]; },
      focus() { document.activeElement = this; }, select() {},
    };
  }
  const input = node('os-custom-status-input'), form = node('os-custom-status-form');
  const select = node('os-status-select'), row = node(), feedback = node(), updated = node();
  const option = node(), saveButton = node();
  option.textContent = saved || 'Custom text';
  const options = { custom: option };
  input.value = saved;
  input.dataset.serverValue = saved;
  input.form = form;
  input.setCustomValidity = value => { input.error = value; };
  input.checkValidity = () => !input.error && input.value.length <= 80;
  input.reportValidity = () => { input.reported = true; };
  Object.defineProperty(input, 'disabled', { get() { return this._disabled; }, set(value) {
    this._disabled = value;
    if (value && document.activeElement === this) document.activeElement = document.body;
  } });
  select.value = status;
  select.dataset.current = status;
  select.form = { querySelector: () => ({ value: '1' }), requestSubmit() {} };
  select.querySelector = selector => options[selector.match(/value="([^"]+)"/)[1]] || null;
  select.appendChild = child => { options[child.value] = child; };
  const selectors = { '.os-custom-status-form': form, '.os-status-select': select,
    '.os-custom-status-feedback': feedback, '.os-status-updated': updated };
  row.dataset = { entryId: '12', status, customStatus: saved };
  row.querySelector = selector => selectors[selector] || null;
  row.remove = () => { row.isConnected = false; };
  for (const element of [select, form, input]) element.closest = () => row;
  form.querySelector = selector => selector === '.os-custom-status-input' ? input : null;
  form.contains = target => target === input || target === saveButton;
  form.getAttribute = () => '';
  document.querySelectorAll = () => [select];
  document.createElement = () => node();
  const context = {
    document, console, Promise, tableView: 'active', totalRows: 1,
    window: { location: { href: '/ordering-sheet/?embed=1' },
      requestAnimationFrame: fn => fn(), setTimeout: (fn, delay) => timers.push({ fn, delay }) },
    FormData: class { constructor(currentForm) { this.text = currentForm.querySelector('.os-custom-status-input').value; } },
    fetch(url, config) { return new Promise((resolve, reject) => requests.push({ url, config, resolve, reject })); },
    osTbody: { addEventListener(name, callback) { listeners[name] = callback; } },
    allRows: () => row.isConnected ? [row] : [],
    refreshCounts() {}, applyCurrentSort() {}, applyFilters() {}, updateOsSelection() {},
  };
  vm.createContext(context);
  vm.runInContext(source, context);
  function open(text = 'Waiting for supplier') {
    select.value = status === 'custom' ? 'edit_custom' : 'custom';
    context.saveSelectedStatus(select);
    input.value = text;
  }
  function event(name, extra = {}) {
    const value = { target: input, preventDefault() { this.defaultPrevented = true; }, ...extra };
    listeners[name](value);
    return value;
  }
  function succeed(request = requests[0], text = request.config.body.text) {
    request.resolve({ ok: true, redirected: false, headers: { get: () => 'application/json' },
      json: async () => ({ ok: true, entry_id: 12, status: 'custom', custom_status_text: text, updated_text: 'Updated now' }) });
  }
  function flushBlur() { timers.filter(timer => timer.delay === 0).splice(0).forEach(timer => timer.fn()); }
  return { context, document, input, form, select, row, option, options, feedback, updated,
    saveButton, open, event, succeed, flushBlur, requests, timers };
}

test('saving immediately closes the editor and shows safely rendered pending text', async () => {
  const h = harness();
  h.open('  <b>Call supplier</b>  ');
  const pending = h.context.saveCustomStatus(h.form, true);
  assert.equal(h.form.hidden, true);
  assert.equal(h.option.textContent, '<b>Call supplier</b>');
  assert.equal(h.feedback.textContent, 'Saving…');
  assert.equal(h.select.dataset.current, 'pending');
  assert.equal(h.requests[0].config.headers['X-Ordering-Status'], 'custom');
  assert.equal(h.requests[0].config.body.text, '<b>Call supplier</b>');
  h.succeed();
  await pending;
  assert.equal(h.feedback.textContent, 'Saved');
  assert.equal(h.select.dataset.current, 'custom');
  assert.equal(h.input.dataset.serverValue, '<b>Call supplier</b>');
  assert.equal(h.options.edit_custom.textContent, 'Edit custom text…');
  assert.equal(h.updated.textContent, 'Updated now');
  assert.equal(h.document.activeElement, h.select);
});

test('duplicate submissions are ignored until the first save completes', async () => {
  const h = harness(); h.open();
  const first = h.context.saveCustomStatus(h.form, false);
  h.context.saveCustomStatus(h.form, false);
  assert.equal(h.requests.length, 1);
  h.succeed(); await first;
});

test('failed saves restore the old dropdown and preserve the draft without stealing focus', async () => {
  const h = harness('custom', 'Previously saved'); h.open('New draft');
  const pending = h.context.saveCustomStatus(h.form, false);
  const otherInput = {}; h.document.activeElement = otherInput;
  h.requests[0].reject(new Error('Network unavailable'));
  await pending;
  assert.equal(h.form.hidden, false);
  assert.equal(h.input.value, 'New draft');
  assert.equal(h.option.textContent, 'Previously saved');
  assert.equal(h.input.dataset.serverValue, 'Previously saved');
  assert.equal(h.feedback.textContent, 'Network unavailable');
  assert.equal(h.document.activeElement, otherInput);
  assert.equal(h.select.disabled, false);
  const retry = h.context.saveCustomStatus(h.form, false);
  h.succeed(h.requests[1]); await retry;
  assert.equal(h.input.dataset.serverValue, 'New draft');
});

test('server validation errors do not mark unsaved text as saved', async () => {
  const h = harness(); h.open();
  const pending = h.context.saveCustomStatus(h.form, false);
  h.requests[0].resolve({ ok: false, headers: { get: () => 'application/json' },
    json: async () => ({ ok: false, error: 'Permission expired' }) });
  await pending;
  assert.equal(h.select.value, 'pending');
  assert.equal(h.form.hidden, false);
  assert.equal(h.feedback.textContent, 'Permission expired');
});

test('session redirects preserve the draft for recovery', async () => {
  const h = harness(); h.open();
  const pending = h.context.saveCustomStatus(h.form, false);
  h.requests[0].resolve({ ok: true, redirected: true });
  await pending;
  assert.equal(h.form.hidden, false);
  assert.equal(h.input.value, 'Waiting for supplier');
  assert.match(h.feedback.textContent, /session may have expired/);
});

test('Enter saves and Escape cancels, but composition Enter never submits', () => {
  const h = harness(); h.open();
  h.event('keydown', { key: 'Enter', isComposing: true });
  assert.equal(h.requests.length, 0);
  h.event('keydown', { key: 'Escape' });
  assert.equal(h.form.hidden, true);
  assert.equal(h.select.value, 'pending');
  h.open();
  assert.equal(h.event('keydown', { key: 'Enter' }).defaultPrevented, true);
  assert.equal(h.requests.length, 1);
});

test('Tab or clicking outside saves, while moving to Save does not double-submit', () => {
  const h = harness(); h.open();
  assert.match(template, /class="os-custom-status-save" tabindex="-1"/);
  h.document.activeElement = h.saveButton;
  h.event('focusout'); h.flushBlur();
  assert.equal(h.requests.length, 0);
  h.document.activeElement = {};
  h.event('focusout'); h.flushBlur();
  assert.equal(h.requests.length, 1);
});

test('empty click-away cancels, unchanged labels close without a request', () => {
  const blank = harness(); blank.open('  ');
  blank.document.activeElement = {};
  blank.event('focusout'); blank.flushBlur();
  assert.equal(blank.requests.length, 0);
  assert.equal(blank.form.hidden, true);
  const same = harness('custom', 'Already saved'); same.open(' Already saved ');
  same.context.saveCustomStatus(same.form, true);
  assert.equal(same.requests.length, 0);
  assert.equal(same.form.hidden, true);
});

test('completed view removes a successfully reopened row', async () => {
  const h = harness('not_for_sale'); h.context.tableView = 'completed'; h.open();
  const pending = h.context.saveCustomStatus(h.form, false);
  h.succeed(); await pending;
  assert.equal(h.row.isConnected, false);
  assert.equal(h.context.totalRows, 0);
});
