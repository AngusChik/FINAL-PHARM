(function () {
  'use strict';
  function init() {
    const root = document.getElementById('expiryCollection');
    const store = window.ExpiredCollectionStore;
    if (!root || !store) return;
    const storageKey = 'expired-collection:v1:' + root.dataset.userId;
    const list = document.getElementById('collectionRows');
    const message = document.getElementById('collectionMessage');
    const reviewForm = document.getElementById('reviewCollectionForm');
    const reviewButton = document.getElementById('reviewCollection');
    const addForm = document.getElementById('retireLotForm');
    const quantity = document.getElementById('retire_quantity');
    const addButton = document.getElementById('addToCollection');
    const allButton = document.getElementById('collectAllLot');
    const clearButton = document.getElementById('clearCollection');
    let rows = [];
    let processedReceipts = [];
    let storageReady = true;

    function announce(text, error) {
      message.textContent = text;
      message.hidden = !text;
      message.classList.toggle('is-error', Boolean(error));
    }
    function persist() {
      try { window.sessionStorage.setItem(storageKey, JSON.stringify({rows, receipts:processedReceipts})); }
      catch (_) {
        storageReady = false;
        announce('This browser cannot save your collection between scans. Enable session storage and reload before continuing.', true);
      }
    }
    function selectedLot() {
      return addForm && addForm.querySelector('input[name="retire_lot_id"]:checked:not(:disabled)');
    }
    function selectedState() {
      const lot = selectedLot();
      if (!lot) return null;
      const rowKey = store.key({product_id: addForm.dataset.productId, lot_id: lot.value});
      const collected = rows.find(row => store.key(row) === rowKey);
      const pending = collected ? Number(collected.quantity) : 0;
      const available = Number(lot.dataset.lotQuantity);
      return {lot, pending, available, remaining: Math.max(0, available - pending)};
    }
    function updateSelection() {
      if (!addForm) return;
      const state = selectedState();
      const enabled = storageReady && state && state.remaining > 0;
      quantity.disabled = !enabled;
      quantity.max = state ? String(state.remaining) : '0';
        allButton.disabled = !enabled;
        allButton.textContent = state && state.pending > 0 ? 'Use remaining lot quantity' : 'Use full lot quantity';
      addButton.disabled = !enabled;
      const hint = document.getElementById('selectedLotHint');
      if (state) {
        hint.textContent = state.remaining > 0
          ? `Up to ${state.remaining} more unit${state.remaining === 1 ? '' : 's'} from lot ${state.lot.dataset.lotNumber}.`
          : 'All available units from this lot are already in your collection.';
      } else {
        hint.textContent = addForm.querySelector('input[name="retire_lot_id"]:not(:disabled)')
          ? 'Choose a lot on this page to enter its collected quantity.'
          : 'No lot is eligible for collection.';
      }
      const draft = enabled && quantity.validity.valid ? Number(quantity.value || 0) : 0;
      document.getElementById('lotCollectedCount').textContent = state ? String(state.pending) : '—';
      document.getElementById('lotRemainingCount').textContent = state ? String(Math.max(0, state.remaining - draft)) : '—';
    }
    function updateTotals() {
      const totals = store.summarize(rows);
      document.getElementById('collectionProducts').textContent = String(totals.products);
      document.getElementById('collectionLots').textContent = String(totals.lots);
      document.getElementById('collectionUnits').textContent = String(totals.units);
      document.getElementById('collectionEmpty').hidden = rows.length > 0;
      clearButton.hidden = !rows.length;
      reviewButton.disabled = !storageReady || !rows.length || Array.from(list.querySelectorAll('input')).some(input => !input.validity.valid);
      reviewButton.textContent = rows.length ? `Review collection · ${totals.units} unit${totals.units === 1 ? '' : 's'}` : 'Review collection';
      updateSelection();
    }
    function element(tag, className, text) {
      const node = document.createElement(tag);
      if (className) node.className = className;
      if (text !== undefined) node.textContent = text;
      return node;
    }
    function render() {
      list.replaceChildren();
      rows.forEach(row => {
        const rowKey = store.key(row);
        const item = element('li');
        item.append(element('strong', 'expiry-collected-name', row.product_name));
        item.append(element('span', 'expiry-collected-lot', `Lot ${row.lot_name} · Expiry ${row.expiry || 'not recorded'}`));
        const controls = element('div', 'expiry-collected-controls');
        const label = element('label', '', 'Qty');
        const input = element('input');
        input.type = 'number'; input.min = '1'; input.max = row.available; input.step = '1'; input.required = true; input.value = row.quantity;
        input.setAttribute('form', 'reviewCollectionForm');
        input.setAttribute('aria-label', `Collected quantity for ${row.product_name}, lot ${row.lot_name}, expiry ${row.expiry}`);
        input.addEventListener('input', () => {
          input.setCustomValidity('');
          if (input.validity.valid) {
            try { rows = store.setQuantity(rows, rowKey, input.value); persist(); }
            catch (error) { input.setCustomValidity(error.message); }
          }
          updateTotals();
        });
        const remove = element('button', 'expiry-text-button', 'Remove');
        remove.type = 'button';
        remove.setAttribute('aria-label', `Remove ${row.product_name}, lot ${row.lot_name}, expiry ${row.expiry} from collection`);
        remove.addEventListener('click', () => {
          rows = store.remove(rows, rowKey); persist(); render();
          announce(`${row.product_name}, lot ${row.lot_name}, removed from your collection.`);
          (list.querySelector('button') || document.getElementById('product_lookup')).focus({preventScroll:true});
        });
        label.append(input); controls.append(label, remove); item.append(controls); list.append(item);
      });
      updateTotals();
    }
    function restore() {
      try {
        const raw = window.sessionStorage.getItem(storageKey);
        const parsed = raw ? JSON.parse(raw) : [];
        const savedRows = Array.isArray(parsed) ? parsed : parsed && parsed.rows;
        processedReceipts = parsed && Array.isArray(parsed.receipts) ? parsed.receipts.filter(id => typeof id === 'string').slice(-100) : [];
        rows = store.normalize(savedRows);
        if (raw && (!Array.isArray(savedRows) || (savedRows.length && !rows.length))) {
          announce('The saved collection could not be restored. Scan the products again; no stock was removed.', true);
        }
        const receiptElement = document.getElementById('logged-collection-receipt');
        const receipt = receiptElement ? JSON.parse(receiptElement.textContent) : null;
        if (receipt && receipt.id && Array.isArray(receipt.rows)) {
          if (!processedReceipts.includes(receipt.id)) {
            rows = store.applyReceipt(rows, receipt.rows);
            processedReceipts = [...processedReceipts, receipt.id].slice(-100);
            persist();
            if (storageReady) announce(rows.length ? 'Stock logged. Your remaining collection is ready to review.' : 'Collection logged. You can start the next collection.');
          }
          const url = new URL(window.location.href);
          url.searchParams.delete('collection_logged');
          url.searchParams.delete('collection_receipt');
          window.history.replaceState(window.history.state, '', url.toString());
        }
        persist();
      } catch (_) {
        rows = [];
        announce('The saved collection could not be restored. Scan the products again; no stock was removed.', true);
        persist();
      }
      render();
    }
    if (addForm) {
      addForm.querySelectorAll('input[name="retire_lot_id"]').forEach(input => {
        input.addEventListener('change', () => {
          quantity.value = '';
          quantity.setCustomValidity('');
          updateSelection();
          if (!quantity.disabled) quantity.focus({preventScroll:true});
        });
      });
      quantity.addEventListener('input', () => { quantity.setCustomValidity(''); updateSelection(); });
      allButton.addEventListener('click', () => {
        const state = selectedState();
        if (state) { quantity.value = String(state.remaining); quantity.setCustomValidity(''); updateSelection(); }
      });
      addForm.addEventListener('submit', event => {
        event.preventDefault();
        const state = selectedState();
        if (!storageReady || !state || !addForm.reportValidity()) return;
        try {
          rows = store.add(rows, {
            product_id:addForm.dataset.productId, product_name:addForm.dataset.productName,
            lot_id:state.lot.value, lot_name:state.lot.dataset.lotNumber, expiry:state.lot.dataset.lotExpiry,
            available:state.lot.dataset.lotQuantity, quantity:quantity.value
          });
          persist();
          quantity.value = '';
          render();
          if (storageReady) announce('Added to collection. Choose another product or review.');
          const lookup = document.getElementById('product_lookup');
          lookup.value = '';
          lookup.dispatchEvent(new Event('input', {bubbles:true}));
          lookup.focus({preventScroll:true});
        } catch (error) { quantity.setCustomValidity(error.message); quantity.reportValidity(); }
      });
    }
    clearButton.addEventListener('click', () => {
      rows = []; persist(); render(); announce('Collection cleared. No stock was removed.');
      document.getElementById('product_lookup').focus({preventScroll:true});
    });
    reviewForm.addEventListener('submit', event => {
      if (!storageReady || !rows.length || !reviewForm.reportValidity()) { event.preventDefault(); return; }
      document.getElementById('collectedRowsPayload').value = store.serialize(rows);
      reviewButton.disabled = true;
      reviewButton.textContent = 'Checking collection…';
    });
    window.addEventListener('pageshow', restore);
    restore();
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
}());
