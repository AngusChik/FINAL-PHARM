(function () {
  'use strict';

  const MAX_LOTS = 400;
  const MAX_PRODUCTS = 200;

  function positiveInteger(value, label) {
    if (typeof value !== 'string' || !/^[0-9]+$/.test(value)) {
      throw new Error(label + ' must be a positive whole number.');
    }
    const number = Number(value);
    if (!Number.isSafeInteger(number) || number <= 0) {
      throw new Error(label + ' must be a positive whole number within the supported range.');
    }
    return String(number);
  }

  function identity(row) {
    if (!row || typeof row !== 'object' || Array.isArray(row)) {
      throw new Error('Each collected item must be a valid lot.');
    }
    return {
      product_id: positiveInteger(row.product_id, 'Product ID'),
      lot_id: row.lot_id === 'legacy' ? 'legacy' : positiveInteger(row.lot_id, 'Lot ID'),
    };
  }

  function key(row) {
    const ids = identity(row);
    return ids.product_id + ':' + ids.lot_id;
  }

  function checkedRow(row) {
    const ids = identity(row);
    const quantity = positiveInteger(row.quantity, 'Collected quantity');
    const available = positiveInteger(row.available, 'Available quantity');
    if (Number(quantity) > Number(available)) {
      throw new Error('Collected quantity cannot exceed the ' + available + ' units available in this lot.');
    }
    const result = { ...ids, quantity };
    ['product_name', 'lot_name', 'expiry'].forEach(name => {
      if (typeof row[name] !== 'string') {
        throw new Error('Collected item details are incomplete. Select the lot again.');
      }
      result[name] = row[name];
    });
    result.available = available;
    return result;
  }

  function checkedRows(rows) {
    if (!Array.isArray(rows)) throw new Error('The collected list must be an array of lots.');
    if (rows.length > MAX_LOTS) throw new Error('Collect up to 400 lots at a time.');
    const lots = new Set();
    const products = new Set();
    let units = 0;
    const result = [];
    for (const source of rows) {
      const row = checkedRow(source);
      const rowKey = key(row);
      if (lots.has(rowKey)) throw new Error('The collected list contains the same lot more than once.');
      lots.add(rowKey);
      products.add(row.product_id);
      if (products.size > MAX_PRODUCTS) throw new Error('Collect up to 200 products at a time.');
      units += Number(row.quantity);
      if (!Number.isSafeInteger(units)) throw new Error('The total collected quantity exceeds the supported range.');
      result.push(row);
    }
    return result;
  }

  // Reject the entire saved payload when it is malformed; never restore a
  // partial collection that could be mistaken for everything staff collected.
  function normalize(rows) {
    try { return checkedRows(rows); }
    catch (_error) { return []; }
  }

  function add(rows, source) {
    const result = checkedRows(rows);
    const incoming = checkedRow(source);
    const rowKey = key(incoming);
    const index = result.findIndex(row => key(row) === rowKey);
    if (index === -1) {
      result.push(incoming);
    } else {
      const quantity = Number(result[index].quantity) + Number(incoming.quantity);
      if (!Number.isSafeInteger(quantity)) throw new Error('Collected quantity exceeds the supported range.');
      result[index] = checkedRow({ ...incoming, quantity: String(quantity) });
    }
    return checkedRows(result);
  }

  function setQuantity(rows, rowKey, quantity) {
    const result = checkedRows(rows);
    const index = result.findIndex(row => key(row) === rowKey);
    if (index === -1) throw new Error('This lot is no longer in the collected list.');
    result[index] = checkedRow({ ...result[index], quantity });
    return checkedRows(result);
  }

  function remove(rows, rowKey) {
    return checkedRows(rows).filter(row => key(row) !== rowKey);
  }

  function summarize(rows) {
    const valid = checkedRows(rows);
    return {
      products: new Set(valid.map(row => row.product_id)).size,
      lots: valid.length,
      units: valid.reduce((sum, row) => sum + Number(row.quantity), 0),
    };
  }

  function serialize(rows) {
    return JSON.stringify(checkedRows(rows).map(row => ({
      product_id: row.product_id,
      lot_id: row.lot_id,
      quantity: row.quantity,
    })));
  }

  function applyReceipt(rows, loggedRows) {
    const result = checkedRows(rows);
    if (!Array.isArray(loggedRows) || loggedRows.length > MAX_LOTS) {
      throw new Error('The logged collection receipt must contain at most 400 lots.');
    }
    const logged = new Map();
    const products = new Set();
    loggedRows.forEach(row => {
      const ids = identity(row);
      const rowKey = key(ids);
      if (logged.has(rowKey)) throw new Error('The logged collection receipt contains the same lot more than once.');
      logged.set(rowKey, Number(positiveInteger(row.quantity, 'Logged quantity')));
      products.add(ids.product_id);
      if (products.size > MAX_PRODUCTS) throw new Error('The logged collection receipt contains more than 200 products.');
    });
    return result.flatMap(row => {
      const actual = logged.get(key(row));
      if (actual === undefined) return [row];
      const available = Math.max(0, Number(row.available) - actual);
      const quantity = Math.min(available, Math.max(0, Number(row.quantity) - actual));
      return quantity ? [{ ...row, quantity: String(quantity), available: String(available) }] : [];
    });
  }

  const api = { normalize, add, setQuantity, remove, summarize, serialize, key, applyReceipt };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  if (typeof window !== 'undefined') window.ExpiredCollectionStore = api;
}());
