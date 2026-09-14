const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');
const { moneyCents, quantityUnits, decimalMoney, formatMoney } = require('../business_loss.js');

test('money retains exact cents when multiplying quantities', () => {
  assert.equal(decimalMoney(quantityUnits('3') * moneyCents('0.10')), '0.30');
  assert.equal(decimalMoney(quantityUnits('7') * moneyCents('1.99')), '13.93');
  assert.equal(moneyCents(' 0012.3 '), 1230n);
  assert.equal(moneyCents('.50'), 50n);
  assert.equal(moneyCents('1.'), 100n);
});

test('largest allowed quantity and unit cost retain cents beyond Number precision', () => {
  const total = quantityUnits('2147483647') * moneyCents('99999999.99');
  assert.equal(decimalMoney(total), '214748364678525163.53');
  assert.equal(formatMoney(total), '$214,748,364,678,525,163.53');
  assert.equal(moneyCents('999999999999999999.99'), 99999999999999999999n);
});

test('blank and malformed money values stay missing instead of becoming zero', () => {
  for (const value of ['', ' ', null, undefined, 'not a cost', '-1', '.', '1e2', '1.234', '1,000.00', '$2.00', 'NaN', 'Infinity']) {
    assert.equal(moneyCents(value), null, String(value));
  }
});

test('quantities require nonnegative whole units', () => {
  assert.equal(quantityUnits(' 00042 '), 42n);
  assert.equal(quantityUnits('2147483647'), 2147483647n);
  for (const value of ['', ' ', null, undefined, '-1', '1e2', '1.5', '1.0', '1,000', 'NaN', 'Infinity']) {
    assert.equal(quantityUnits(value), null, String(value));
  }
});

test('zero is valid and currency formatting preserves two digits and thousands', () => {
  assert.equal(quantityUnits('0'), 0n);
  assert.equal(moneyCents('0'), 0n);
  assert.equal(moneyCents('0.00'), 0n);
  assert.equal(decimalMoney(0n), '0.00');
  assert.equal(decimalMoney(1n), '0.01');
  assert.equal(formatMoney(0n), '$0.00');
  assert.equal(formatMoney(100001n), '$1,000.01');
});

const source = readFileSync(path.join(__dirname, '..', 'business_loss.js'), 'utf8');

function harness(initialRows) {
  const listeners = {};
  const nodes = Object.fromEntries(['businessLossRows', 'businessLossUnits', 'businessLossTotal', 'businessLossMissing', 'businessLossGenerate', 'businessLossEntryCount', 'businessLossEmpty', 'businessLossTable', 'businessLossRemovalStatus', 'businessLossDate']
    .map(id => [id, { value: '', textContent: '', disabled:false, hidden:false, focus() { this.focused = true; } }]));
  const rows = initialRows.map(values => {
    const fields = Object.fromEntries(Object.entries(values).map(([name, value]) => [name, {
      dataset: { field: name }, value, style: {}, scrollHeight: 40, removeAttribute() {},
      validationMessage: '',
      setCustomValidity(message) { this.validationMessage = message; },
      closest: selector => selector === '[data-loss-row]' ? row : null,
    }]));
    const row = {
      fields,
      remove() { this.removed = true; },
      querySelector: selector => selector === '[data-remove-loss-row]' ? row.button : fields[/data-field="([^"]+)"/.exec(selector)[1]],
      querySelectorAll: () => Object.values(fields),
    };
    row.button = {
      closest: selector => selector === '[data-remove-loss-row]' ? row.button : row,
      focus() { this.focused = true; },
    };
    return row;
  });
  nodes.businessLossForm = {
    querySelectorAll: () => rows.filter(row => !row.removed),
    addEventListener: (event, callback) => { listeners[event] = callback; },
  };
  vm.runInNewContext(source, { document: { getElementById: id => nodes[id] } });
  return {
    rows, nodes,
    edit(index, name, value) {
      const input = rows[index].fields[name];
      input.value = value;
      listeners.input({ target: input });
    },
    submit() {
      let prevented = false;
      listeners.submit({preventDefault() { prevented = true; }});
      if (prevented) return null;
      return JSON.parse(nodes.businessLossRows.value);
    },
    remove(index) { listeners.click({target: rows[index].button}); },
  };
}

const initialRow = () => ({ product: 'Example product', quantity: '3', cost_per_unit: '0.10', total_cost: '0.30', year: '2026' });

test('editing total directly updates the summary and survives changes to product and year', () => {
  const form = harness([initialRow()]);
  form.edit(0, 'total_cost', '10.99');
  form.edit(0, 'product', 'Corrected product');
  form.edit(0, 'year', '2025');
  assert.equal(form.nodes.businessLossTotal.textContent, '$10.99');
  assert.equal(form.rows[0].fields.total_cost.value, '10.99');
  assert.deepEqual(form.submit(), [{ product: 'Corrected product', quantity: '3', cost_per_unit: '0.10', total_cost: '10.99', year: '2025' }]);
});

test('quantity and unit cost changes recalculate a previously edited total with exact cents', () => {
  const form = harness([initialRow()]);
  form.edit(0, 'total_cost', '10.99');
  form.edit(0, 'quantity', '7');
  assert.equal(form.rows[0].fields.total_cost.value, '0.70');
  form.edit(0, 'cost_per_unit', '1.99');
  assert.equal(form.rows[0].fields.total_cost.value, '13.93');
  assert.equal(form.nodes.businessLossUnits.textContent, '7');
  assert.equal(form.nodes.businessLossTotal.textContent, '$13.93');
});

test('clearing a cost removes its stale calculated total and restoring zero clears the missing notice', () => {
  const form = harness([initialRow()]);
  form.edit(0, 'cost_per_unit', '');
  assert.equal(form.rows[0].fields.total_cost.value, '');
  assert.equal(form.nodes.businessLossTotal.textContent, '$0.00');
  assert.match(form.nodes.businessLossMissing.textContent, /1 row needs/);
  form.edit(0, 'cost_per_unit', '0.00');
  assert.equal(form.rows[0].fields.total_cost.value, '0.00');
  assert.equal(form.nodes.businessLossMissing.textContent, '');
});

test('ordinary shorthand decimal inputs recalculate successfully', () => {
  const form = harness([initialRow()]);
  form.edit(0, 'cost_per_unit', '.50');
  assert.equal(form.rows[0].fields.total_cost.value, '1.50');
  assert.equal(form.rows[0].fields.cost_per_unit.validationMessage, '');
  form.edit(0, 'cost_per_unit', '1.');
  assert.equal(form.rows[0].fields.total_cost.value, '3.00');
});

test('unsupported numeric notation is invalid and correcting dependent fields clears stale total errors', () => {
  const form = harness([initialRow()]);
  form.edit(0, 'quantity', '1e2');
  assert.match(form.rows[0].fields.quantity.validationMessage, /whole number/);
  form.edit(0, 'quantity', '4');
  assert.equal(form.rows[0].fields.quantity.validationMessage, '');
  form.edit(0, 'total_cost', '1e2');
  assert.match(form.rows[0].fields.total_cost.validationMessage, /decimal places/);
  form.edit(0, 'cost_per_unit', '1.25');
  assert.equal(form.rows[0].fields.total_cost.value, '5.00');
  assert.equal(form.rows[0].fields.total_cost.validationMessage, '');
});

test('summaries include all rows and edited fields serialize as unrounded strings', () => {
  const form = harness([initialRow(), {
    product: 'Large quantity', quantity: '2147483647', cost_per_unit: '99999999.99',
    total_cost: '214748364678525163.53', year: '2024',
  }]);
  assert.equal(form.nodes.businessLossUnits.textContent, '2,147,483,650');
  assert.equal(form.nodes.businessLossTotal.textContent, '$214,748,364,678,525,163.83');
  assert.equal(form.submit()[1].total_cost, '214748364678525163.53');
  form.edit(0, 'product', 'A "quoted" product\nwith a second line');
  assert.equal(form.submit()[0].product, 'A "quoted" product\nwith a second line');
});

test('deleting an edited report row updates totals and excludes it from the PDF payload', () => {
  const form = harness([initialRow(), {...initialRow(), product:'Keep this product', quantity:'2', total_cost:'0.20'}]);
  form.edit(0, 'total_cost', '99.99');
  form.remove(0);
  assert.equal(form.nodes.businessLossUnits.textContent, '2');
  assert.equal(form.nodes.businessLossTotal.textContent, '$0.20');
  assert.equal(form.nodes.businessLossEntryCount.textContent, '1 log entry in this report');
  assert.equal(form.nodes.businessLossGenerate.disabled, false);
  assert.equal(form.rows[1].button.focused, true);
  assert.deepEqual(form.submit(), [{product:'Keep this product',quantity:'2',cost_per_unit:'0.10',total_cost:'0.20',year:'2026'}]);
});

test('deleting the last row shows an empty report, clears missing cost warnings and blocks submission', () => {
  const form = harness([initialRow()]);
  form.edit(0, 'cost_per_unit', '');
  form.remove(0);
  assert.equal(form.nodes.businessLossUnits.textContent, '0');
  assert.equal(form.nodes.businessLossTotal.textContent, '$0.00');
  assert.equal(form.nodes.businessLossMissing.textContent, '');
  assert.equal(form.nodes.businessLossEmpty.hidden, false);
  assert.equal(form.nodes.businessLossTable.hidden, true);
  assert.equal(form.nodes.businessLossGenerate.disabled, true);
  assert.equal(form.nodes.businessLossDate.focused, true);
  assert.equal(form.submit(), null);
});
