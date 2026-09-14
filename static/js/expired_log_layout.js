(function () {
  'use strict';

  function init() {
    const root = document.getElementById('expiryCollection');
    if (!root) return;
    const storageKey = 'expired-picker:v1:' + root.dataset.userId;
    const pickerList = document.getElementById('expiredPickerList');
    const pickerSearch = document.getElementById('expiredPickerSearch');
    const pickerCount = document.getElementById('expiredPickerCount');
    const lotList = root.querySelector('.retire-lot-list');
    const queueList = document.getElementById('collectionRows');
    let pickerPage = 0;
    let lotPage = 0;
    let queuePage = 0;
    let query = '';
    let restoredPicker = false;
    let lastQueueCount = queueList ? queueList.children.length : 0;
    let sizes = pageSizes();

    function pageSizes() {
      if (window.matchMedia('(max-width: 1100px)').matches) return { picker: 4, lots: 4, queue: 4 };
      return {
        picker: window.innerHeight < 800 ? 1 : (window.innerHeight < 1000 ? 2 : 3),
        lots: window.innerHeight < 800 ? 2 : 3,
        queue: window.innerHeight < 1000 ? 1 : 2,
      };
    }

    function clampPage(page, count, size) {
      return Math.max(0, Math.min(page, Math.max(0, Math.ceil(count / size) - 1)));
    }

    function status(id, previousId, nextId, page, count, size, label) {
      const text = document.getElementById(id);
      const previous = document.getElementById(previousId);
      const next = document.getElementById(nextId);
      const pages = Math.ceil(count / size);
      if (text) {
        text.textContent = count ? `${page * size + 1}–${Math.min((page + 1) * size, count)} of ${count}` : `0 ${label}`;
        text.setAttribute('role', 'status');
        text.setAttribute('aria-live', 'polite');
        text.setAttribute('aria-atomic', 'true');
        text.setAttribute('aria-label', count ? `${label}, page ${page + 1} of ${pages}` : `No ${label}`);
      }
      if (previous) previous.disabled = count === 0 || page === 0;
      if (next) next.disabled = count === 0 || page + 1 >= pages;
    }

    function bind(id, callback) {
      const button = document.getElementById(id);
      if (button) button.addEventListener('click', callback);
    }

    function savePicker() {
      try { window.sessionStorage.setItem(storageKey, JSON.stringify({ search: query, page: pickerPage })); }
      catch (_error) { /* The picker remains usable when preferences cannot be saved. */ }
    }

    try {
      const saved = JSON.parse(window.sessionStorage.getItem(storageKey));
      if (saved && typeof saved.search === 'string' && Number.isSafeInteger(saved.page) && saved.page >= 0) {
        query = saved.search;
        pickerPage = saved.page;
        restoredPicker = true;
      }
    } catch (_error) { /* Ignore malformed or unavailable picker preferences. */ }

    let products = [];
    const productData = document.getElementById('expiry-picker-products');
    try {
      const parsed = productData ? JSON.parse(productData.textContent) : [];
      if (Array.isArray(parsed)) {
        products = parsed.filter(product => product && typeof product === 'object'
          && /^[0-9]+$/.test(String(product.product_id)) && Number(product.product_id) > 0);
      }
    } catch (_error) { /* Render an empty picker without disrupting the collection. */ }

    function matchingProducts() {
      const term = query.trim().toLocaleLowerCase();
      if (!term) return products;
      return products.filter(product => [product.name, product.barcode, product.item_number]
        .some(value => String(value || '').toLocaleLowerCase().includes(term)));
    }

    function appendText(parent, tag, className, text) {
      const child = document.createElement(tag);
      child.className = className;
      child.textContent = text;
      parent.append(child);
      return child;
    }

    function expiryLabel(value) {
      const match = /^([0-9]{4})-([0-9]{2})-([0-9]{2})$/.exec(String(value || ''));
      return match ? `${match[3]}/${match[2]}/${match[1]}` : 'not recorded';
    }

    function renderPicker() {
      if (!pickerList) return;
      const matches = matchingProducts();
      pickerPage = clampPage(pickerPage, matches.length, sizes.picker);
      pickerList.replaceChildren();
      matches.slice(pickerPage * sizes.picker, (pickerPage + 1) * sizes.picker).forEach(product => {
        const item = document.createElement('li');
        item.className = 'expiry-picker-item';
        const link = appendText(item, 'a', 'expiry-picker-link', '');
        const url = new URL(root.dataset.productsUrl || window.location.pathname, window.location.origin);
        url.search = '';
        url.searchParams.set('mode', 'log');
        url.searchParams.set('pid', String(product.product_id));
        link.href = url.toString();
        if (String(product.product_id) === root.dataset.productId) {
          link.classList.add('is-current');
          link.setAttribute('aria-current', 'page');
        }
        appendText(link, 'strong', 'expiry-picker-name', String(product.name || 'Unnamed product'));
        const units = Number(product.expired_quantity) || 0;
        appendText(link, 'span', 'expiry-picker-meta', `${units.toLocaleString()} expired unit${units === 1 ? '' : 's'} · First expiry ${expiryLabel(product.earliest_expiry)}`);
        pickerList.append(item);
      });
      if (!matches.length) appendText(pickerList, 'li', 'expiry-picker-empty', query.trim() ? 'No matching expired products.' : 'No expired products on shelf.');
      if (pickerCount) {
        pickerCount.textContent = query.trim()
          ? `${matches.length} of ${products.length} products`
          : `${products.length} expired product${products.length === 1 ? '' : 's'}`;
        pickerCount.setAttribute('role', 'status');
        pickerCount.setAttribute('aria-live', 'polite');
        pickerCount.setAttribute('aria-atomic', 'true');
      }
      status('expiredPickerPage', 'expiredPickerPrev', 'expiredPickerNext', pickerPage, matches.length, sizes.picker, 'products');
      savePicker();
    }

    function lotRows() {
      return lotList ? Array.from(lotList.querySelectorAll('.retire-lot-option')) : [];
    }

    function renderLots() {
      const rows = lotRows();
      lotPage = clampPage(lotPage, rows.length, sizes.lots);
      const cleared = [];
      rows.forEach((row, index) => {
        row.hidden = Math.floor(index / sizes.lots) !== lotPage;
        const selected = row.hidden && row.querySelector('input:checked');
        if (selected) {
          selected.checked = false;
          cleared.push(selected);
        }
      });
      const pagination = document.getElementById('expiryLotPagination');
      if (pagination) pagination.hidden = rows.length <= sizes.lots;
      status('expiryLotPage', 'expiryLotPrev', 'expiryLotNext', lotPage, rows.length, sizes.lots, 'lots');
      // Notify the collection controls after paging removes their selected lot.
      // A quantity must never be added against a lot staff can no longer see.
      cleared.forEach(input => input.dispatchEvent(new Event('change', { bubbles: true })));
    }

    function queueRows() {
      return queueList ? Array.from(queueList.children) : [];
    }

    function invalidQueueIndex(rows) {
      return rows.findIndex(row => Array.from(row.querySelectorAll('input')).some(input => !input.disabled && !input.validity.valid));
    }

    function queueCanPage() {
      const invalid = queueRows().filter(row => !row.hidden)
        .flatMap(row => Array.from(row.querySelectorAll('input')))
        .find(input => !input.disabled && !input.validity.valid);
      if (invalid) { invalid.reportValidity(); return false; }
      return true;
    }

    function renderQueue() {
      const rows = queueRows();
      const invalid = invalidQueueIndex(rows);
      if (invalid !== -1) queuePage = Math.floor(invalid / sizes.queue);
      queuePage = clampPage(queuePage, rows.length, sizes.queue);
      const focused = document.activeElement;
      const focusedRow = rows.find(row => row.contains(focused));
      rows.forEach((row, index) => { row.hidden = Math.floor(index / sizes.queue) !== queuePage; });
      if (focusedRow && focusedRow.hidden) {
        const visible = rows.find(row => !row.hidden);
        const target = visible && visible.querySelector(focused.tagName === 'BUTTON' ? 'button' : 'input');
        if (target) target.focus({ preventScroll: true });
      }
      const pagination = document.getElementById('expiryQueuePagination');
      if (pagination) pagination.hidden = rows.length <= sizes.queue;
      status('expiryQueuePage', 'expiryQueuePrev', 'expiryQueueNext', queuePage, rows.length, sizes.queue, 'collected lots');
    }

    if (pickerSearch) {
      pickerSearch.value = query;
      pickerSearch.addEventListener('input', () => { query = pickerSearch.value; pickerPage = 0; renderPicker(); });
    }
    if (!restoredPicker) {
      const selected = matchingProducts().findIndex(product => String(product.product_id) === root.dataset.productId);
      if (selected !== -1) pickerPage = Math.floor(selected / sizes.picker);
    }
    bind('expiredPickerPrev', () => { pickerPage -= 1; renderPicker(); });
    bind('expiredPickerNext', () => { pickerPage += 1; renderPicker(); });

    const selectedLot = lotRows().findIndex(row => row.querySelector('input:checked'));
    if (selectedLot !== -1) lotPage = Math.floor(selectedLot / sizes.lots);
    bind('expiryLotPrev', () => { lotPage -= 1; renderLots(); });
    bind('expiryLotNext', () => { lotPage += 1; renderLots(); });
    if (lotList) lotList.addEventListener('change', event => {
      if (event.target.matches('input[name="retire_lot_id"]:checked')) {
        const selected = lotRows().findIndex(row => row.contains(event.target));
        if (selected !== -1) { lotPage = Math.floor(selected / sizes.lots); renderLots(); }
      }
    });

    bind('expiryQueuePrev', () => { if (queueCanPage()) { queuePage -= 1; renderQueue(); } });
    bind('expiryQueueNext', () => { if (queueCanPage()) { queuePage += 1; renderQueue(); } });
    if (queueList) new MutationObserver(() => {
      const count = queueList.children.length;
      if (count > lastQueueCount) queuePage = Math.max(0, Math.ceil(count / sizes.queue) - 1);
      lastQueueCount = count;
      renderQueue();
    }).observe(queueList, { childList: true });

    window.addEventListener('resize', () => {
      const next = pageSizes();
      if (next.picker === sizes.picker && next.lots === sizes.lots && next.queue === sizes.queue) return;
      const pickerStart = pickerPage * sizes.picker;
      const selectedLotIndex = lotRows().findIndex(row => row.querySelector('input:checked'));
      const lotStart = selectedLotIndex === -1 ? lotPage * sizes.lots : selectedLotIndex;
      const focusedQueueRow = queueRows().findIndex(row => row.contains(document.activeElement));
      const queueStart = focusedQueueRow === -1 ? queuePage * sizes.queue : focusedQueueRow;
      sizes = next;
      pickerPage = Math.floor(pickerStart / sizes.picker);
      lotPage = Math.floor(lotStart / sizes.lots);
      queuePage = Math.floor(queueStart / sizes.queue);
      renderPicker();
      renderLots();
      renderQueue();
    });

    renderPicker();
    renderLots();
    renderQueue();
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
}());
