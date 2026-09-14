const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');
const store = require('../expired_collection_store.js');

function lot(overrides = {}) {
  return {
    product_id: '1', lot_id: '10', quantity: '2', product_name: 'Example product',
    lot_name: 'LOT-A', expiry: '2026-08-31', available: '10', ...overrides,
  };
}

test('normalization returns clean canonical rows without mutating saved data', () => {
  const original = Object.freeze(lot({ product_id: '001', lot_id: '010', quantity: '002', available: '010', extra: 'discard' }));
  const rows = Object.freeze([original]);
  assert.deepEqual(store.normalize(rows), [lot()]);
  assert.equal(original.product_id, '001');
  assert.equal(original.extra, 'discard');
});

test('malformed saved payloads restore an empty collection without throwing', () => {
  for (const payload of [null, undefined, {}, '[]', [null], [1], [[]], [lot(), null]]) {
    assert.deepEqual(store.normalize(payload), []);
  }
  const corrupt = [
    { product_id: '' }, { product_id: '0' }, { lot_id: '-1' }, { lot_id: '1e2' },
    { quantity: 1 }, { quantity: '1.5' }, { quantity: '0' }, { quantity: '11' },
    { available: '' }, { available: '0' }, { available: 'NaN' }, { product_name: null },
    { expiry: undefined }, { quantity: '9007199254740992', available: '9007199254740992' },
  ];
  for (const invalid of corrupt) assert.deepEqual(store.normalize([lot(invalid)]), [], JSON.stringify(invalid));
});

test('duplicate saved identities are rejected including alternate leading zeros', () => {
  assert.deepEqual(store.normalize([lot(), lot({ product_id: '01', lot_id: '010' })]), []);
});

test('identity uses product and database lot IDs rather than shared lot labels', () => {
  assert.equal(store.key({ product_id: '001', lot_id: '010' }), '1:10');
  assert.equal(store.key({ product_id: '1', lot_id: 'legacy' }), '1:legacy');
  const rows = store.add([lot()], lot({ lot_id: '11', expiry: '2026-09-30' }));
  assert.equal(rows.length, 2);
  assert.notEqual(store.key(rows[0]), store.key(rows[1]));
  assert.equal(rows[0].lot_name, rows[1].lot_name);
});

test('adding another scan increments only the matching lot and refreshes available details', () => {
  const rows = Object.freeze([Object.freeze(lot()), Object.freeze(lot({ lot_id: '11', quantity: '3' }))]);
  const next = store.add(rows, lot({ quantity: '1', available: '8', product_name: 'Updated name' }));
  assert.equal(next[0].quantity, '3');
  assert.equal(next[0].available, '8');
  assert.equal(next[0].product_name, 'Updated name');
  assert.equal(next[1].quantity, '3');
  assert.equal(rows[0].quantity, '2');
});

test('duplicate additions cannot exceed current lot stock and preserve the existing list on failure', () => {
  const rows = [lot({ quantity: '8' })];
  assert.throws(() => store.add(rows, lot({ quantity: '3' })), /cannot exceed/);
  assert.throws(() => store.add(rows, lot({ quantity: '1', available: '5' })), /cannot exceed/);
  assert.equal(rows[0].quantity, '8');
});

test('editing quantity requires a positive integer within the selected lot availability', () => {
  const rows = Object.freeze([Object.freeze(lot())]);
  assert.equal(store.setQuantity(rows, '1:10', '010')[0].quantity, '10');
  for (const value of ['', '0', '-1', '1.5', '1e2', '11', 3]) {
    assert.throws(() => store.setQuantity(rows, '1:10', value), /quantity/i);
  }
  assert.throws(() => store.setQuantity(rows, '1:99', '1'), /no longer/);
  assert.equal(rows[0].quantity, '2');
});

test('removal addresses exact identity and safely handles an already removed item', () => {
  const rows = [lot(), lot({ lot_id: '11' }), lot({ product_id: '2' })];
  const next = store.remove(rows, '1:10');
  assert.deepEqual(next.map(store.key), ['1:11', '2:10']);
  assert.deepEqual(store.remove(next, '1:10'), next);
  assert.equal(rows.length, 3);
});

test('summary counts unique products and all lot quantities including legacy rows', () => {
  const rows = [lot(), lot({ lot_id: '11', quantity: '4' }), lot({ product_id: '2', lot_id: 'legacy', quantity: '3' })];
  assert.deepEqual(store.summarize(rows), { products: 2, lots: 3, units: 9 });
  assert.deepEqual(store.summarize([]), { products: 0, lots: 0, units: 0 });
});

test('backend serialization includes only canonical IDs and quantity strings', () => {
  const result = JSON.parse(store.serialize([lot({ product_id: '001', quantity: '002' }), lot({ product_id: '2', lot_id: 'legacy' })]));
  assert.deepEqual(result, [
    { product_id: '1', lot_id: '10', quantity: '2' },
    { product_id: '2', lot_id: 'legacy', quantity: '2' },
  ]);
  assert.equal(store.serialize([]), '[]');
});

test('a collection can contain 400 lots but adding a 401st is rejected', () => {
  const rows = Array.from({ length: 400 }, (_, index) => lot({ lot_id: String(index + 1) }));
  assert.equal(store.normalize(rows).length, 400);
  assert.throws(() => store.add(rows, lot({ lot_id: '401' })), /400 lots/);
  assert.deepEqual(store.normalize([...rows, lot({ lot_id: '401' })]), []);
  assert.equal(store.add(rows, lot({ lot_id: '1', quantity: '1' }))[0].quantity, '3');
});

test('a collection can contain 200 products and more lots for an existing product', () => {
  const rows = Array.from({ length: 200 }, (_, index) => lot({ product_id: String(index + 1) }));
  assert.equal(store.summarize(rows).products, 200);
  assert.equal(store.add(rows, lot({ lot_id: '11' })).length, 201);
  assert.throws(() => store.add(rows, lot({ product_id: '201' })), /200 products/);
  assert.deepEqual(store.normalize([...rows, lot({ product_id: '201' })]), []);
});

test('large quantities remain exact and unsafe aggregate totals are rejected', () => {
  const large = lot({ quantity: '9007199254740991', available: '9007199254740991' });
  assert.equal(store.summarize([large]).units, Number.MAX_SAFE_INTEGER);
  assert.equal(JSON.parse(store.serialize([large]))[0].quantity, '9007199254740991');
  assert.throws(() => store.add([large], lot({ quantity: '1', available: '9007199254740991' })), /supported range/);
  assert.throws(() => store.add([large], lot({ lot_id: '11', quantity: '1' })), /total collected quantity/);
});

test('mutations and serialization reject corrupted current data instead of silently submitting part of it', () => {
  const invalid = [lot(), null];
  assert.throws(() => store.add(invalid, lot()), /valid lot/);
  assert.throws(() => store.setQuantity(invalid, '1:10', '1'), /valid lot/);
  assert.throws(() => store.remove(invalid, '1:10'), /valid lot/);
  assert.throws(() => store.serialize(invalid), /valid lot/);
});

test('browser export exposes the same pure API without requiring a document or storage', () => {
  const browser = {};
  const source = readFileSync(path.join(__dirname, '..', 'expired_collection_store.js'), 'utf8');
  vm.runInNewContext(source, { window: browser });
  assert.deepEqual(Object.keys(browser.ExpiredCollectionStore).sort(), Object.keys(store).sort());
  assert.equal(browser.ExpiredCollectionStore.serialize([]), '[]');
});

test('logging part of a collected lot keeps the remainder with updated availability', () => {
  const rows = Object.freeze([Object.freeze(lot({ quantity: '7', available: '10' }))]);
  const next = store.applyReceipt(rows, [{ product_id: '1', lot_id: '10', quantity: '3' }]);
  assert.deepEqual(next, [lot({ quantity: '4', available: '7' })]);
  assert.deepEqual(store.summarize(next), { products: 1, lots: 1, units: 4 });
  assert.equal(rows[0].quantity, '7');
  assert.equal(rows[0].available, '10');
});

test('receipt removes only logged exact identities and retains unchecked products and same-name lots', () => {
  const rows = [
    lot({ quantity: '3' }),
    lot({ lot_id: '11', expiry: '2026-09-30', quantity: '4' }),
    lot({ product_id: '2', quantity: '5' }),
    lot({ product_id: '3', lot_id: 'legacy', quantity: '6' }),
  ];
  const next = store.applyReceipt(rows, [
    { product_id: '1', lot_id: '10', quantity: '3' },
    { product_id: '3', lot_id: 'legacy', quantity: '2' },
  ]);
  assert.deepEqual(next, [rows[1], rows[2], lot({ product_id: '3', lot_id: 'legacy', quantity: '4', available: '8' })]);
  assert.deepEqual(store.summarize(next), { products: 3, lots: 3, units: 13 });
});

test('receipt tolerates an increased final quantity and other lots added at review without negative remainders', () => {
  const rows = [lot({ quantity: '2', available: '10' }), lot({ product_id: '2', quantity: '4' })];
  const next = store.applyReceipt(rows, [
    { product_id: '1', lot_id: '10', quantity: '7' },
    { product_id: '1', lot_id: '99', quantity: '3' },
  ]);
  assert.deepEqual(next, [rows[1]]);
  assert.deepEqual(store.applyReceipt([rows[0]], [{ product_id: '1', lot_id: '10', quantity: '12' }]), []);
});

test('receipt canonicalizes identities and an empty receipt preserves the original collection', () => {
  assert.deepEqual(store.applyReceipt([lot()], [{ product_id: '001', lot_id: '010', quantity: '001' }]), [lot({ quantity: '1', available: '9' })]);
  const rows = [lot()];
  const unchanged = store.applyReceipt(rows, []);
  assert.deepEqual(unchanged, rows);
  assert.notEqual(unchanged, rows);
  assert.notEqual(unchanged[0], rows[0]);
});

test('malformed or duplicate receipt entries cannot silently clear a collection', () => {
  for (const receipt of [null, {}, '[]', [null], [{ product_id: '1', lot_id: '10', quantity: '0' }], [{ product_id: '1', lot_id: '10', quantity: 1 }]]) {
    assert.throws(() => store.applyReceipt([lot()], receipt));
  }
  assert.throws(() => store.applyReceipt([lot()], [
    { product_id: '1', lot_id: '10', quantity: '1' },
    { product_id: '01', lot_id: '010', quantity: '1' },
  ]), /same lot/);
});
