(function () {
  'use strict';

  function moneyCents(value) {
    const match = /^(?:(\d+)(?:\.(\d{0,2}))?|\.(\d{1,2}))$/.exec(String(value).trim());
    return match ? BigInt(match[1] || '0') * 100n + BigInt((match[2] || match[3] || '').padEnd(2, '0')) : null;
  }

  function quantityUnits(value) {
    return /^\d+$/.test(String(value).trim()) ? BigInt(String(value).trim()) : null;
  }

  function decimalMoney(cents) {
    return (cents / 100n).toString() + '.' + (cents % 100n).toString().padStart(2, '0');
  }

  function formatMoney(cents) {
    const parts = decimalMoney(cents).split('.');
    return '$' + parts[0].replace(/\B(?=(\d{3})+(?!\d))/g, ',') + '.' + parts[1];
  }

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = { moneyCents, quantityUnits, decimalMoney, formatMoney };
  }
  if (typeof document === 'undefined') return;
  const form = document.getElementById('businessLossForm');
  if (!form) return;
  const rows = () => Array.from(form.querySelectorAll('[data-loss-row]'));
  const generate = document.getElementById('businessLossGenerate');
  const generationBlocked = generate.disabled;
  const field = (row, name) => row.querySelector('[data-field="' + name + '"]');

  function resizeProduct(input) {
    input.style.height = 'auto';
    input.style.height = input.scrollHeight + 2 + 'px';
  }

  function updateSummary() {
    let units = 0n;
    let total = 0n;
    let missing = 0;
    const currentRows = rows();
    currentRows.forEach(row => {
      const quantity = quantityUnits(field(row, 'quantity').value);
      const cost = moneyCents(field(row, 'cost_per_unit').value);
      const line = moneyCents(field(row, 'total_cost').value);
      if (quantity !== null) units += quantity;
      if (line !== null) total += line;
      if (cost === null || line === null) missing += 1;
    });
    document.getElementById('businessLossUnits').textContent = units.toLocaleString('en-CA');
    document.getElementById('businessLossTotal').textContent = formatMoney(total);
    document.getElementById('businessLossMissing').textContent = missing
      ? missing + (missing === 1 ? ' row needs' : ' rows need') + ' a valid unit cost or total. Complete these fields before generating the PDF.'
      : '';
    document.getElementById('businessLossEntryCount').textContent = currentRows.length + (currentRows.length === 1 ? ' log entry' : ' log entries') + ' in this report';
    generate.disabled = generationBlocked || currentRows.length === 0;
    const empty = document.getElementById('businessLossEmpty');
    const table = document.getElementById('businessLossTable');
    if (empty) empty.hidden = currentRows.length > 0;
    if (table) table.hidden = currentRows.length === 0;
  }

  form.addEventListener('click', event => {
    const button = event.target.closest('[data-remove-loss-row]');
    if (!button) return;
    const row = button.closest('[data-loss-row]');
    if (!row) return;
    const index = rows().indexOf(row);
    const product = field(row, 'product').value.trim() || 'Product';
    row.remove();
    updateSummary();
    document.getElementById('businessLossRemovalStatus').textContent = product + ' deleted from this report.';
    const remaining = rows();
    const next = remaining[Math.min(index, remaining.length - 1)];
    const focusTarget = next ? next.querySelector('[data-remove-loss-row]') : document.getElementById('businessLossDate');
    focusTarget.focus({preventScroll:true});
  });

  form.addEventListener('input', event => {
    const input = event.target;
    const row = input.closest('[data-loss-row]');
    if (!row) return;
    input.removeAttribute('aria-invalid');
    const name = input.dataset.field;
    if (name === 'cost_per_unit' || name === 'total_cost') {
      input.setCustomValidity(input.value && moneyCents(input.value) === null ? 'Enter a cost using up to two decimal places.' : '');
    } else if (name === 'quantity' || name === 'year') {
      input.setCustomValidity(input.value && quantityUnits(input.value) === null ? 'Enter a whole number.' : '');
    }
    if (input.dataset.field === 'product') resizeProduct(input);
    if (input.dataset.field === 'quantity' || input.dataset.field === 'cost_per_unit') {
      const quantity = quantityUnits(field(row, 'quantity').value);
      const cost = moneyCents(field(row, 'cost_per_unit').value);
      field(row, 'total_cost').value = quantity !== null && cost !== null ? decimalMoney(quantity * cost) : '';
      field(row, 'total_cost').setCustomValidity('');
    }
    updateSummary();
  });

  form.addEventListener('submit', event => {
    const currentRows = rows();
    if (!currentRows.length || generationBlocked) { event.preventDefault(); return; }
    document.getElementById('businessLossRows').value = JSON.stringify(currentRows.map(row => {
      const values = {};
      row.querySelectorAll('[data-field]').forEach(input => { values[input.dataset.field] = input.value; });
      return values;
    }));
  });
  rows().forEach(row => resizeProduct(field(row, 'product')));
  if (typeof window !== 'undefined') {
    let resizeFrame;
    window.addEventListener('resize', () => {
      window.cancelAnimationFrame(resizeFrame);
      resizeFrame = window.requestAnimationFrame(() => {
        rows().forEach(row => resizeProduct(field(row, 'product')));
      });
    });
  }
  updateSummary();
}());
