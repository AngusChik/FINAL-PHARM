/* Shared deadstock review state; a fresh batch is requested only on page entry. */
(function () {
  'use strict';
  const list = document.getElementById('dsListContainer');
  if (!list || list.dataset.dsInitialized) return;
  list.dataset.dsInitialized = 'true';
  const api = list.dataset.apiUrl;
  const expandUrl = list.dataset.expandUrl;
  const modal = document.getElementById('deadstockModal');
  const modalBody = document.getElementById('deadstockModalBody');
  const modalSub = document.getElementById('deadstockModalSub');
  const status = document.getElementById('dsStatus');
  const modalStatus = document.getElementById('dsModalStatus');
  const modalRetry = document.getElementById('dsModalRetry');
  const retry = document.getElementById('dsRetry');
  const undo = document.getElementById('dsUndo');
  const snacks = document.getElementById('ds-ignore-snacks');
  const braces = document.getElementById('ds-ignore-braces');
  let items = [];
  let modalItems = [];
  let tab = 'all';
  let page = 1;
  let totalPages = 1;
  let lastDismissal = null;
  let retryAction = null;
  let pending = false;
  let generation = 0;
  let modalGeneration = 0;
  let modalAbort = null;
  let requestQueue = Promise.resolve();
  let returnFocus = null;

  function preference(key) {
    try { return localStorage.getItem(key) === '1'; } catch (_) { return false; }
  }
  const filters = { exclude_snacks: preference('ds_hide_snacks'), exclude_braces: preference('ds_hide_braces') };
  function syncFilters() {
    [[snacks, filters.exclude_snacks], [braces, filters.exclude_braces]].forEach(function (pair) {
      pair[0].classList.toggle('active', pair[1]);
      pair[0].setAttribute('aria-checked', String(pair[1]));
    });
  }
  function esc(value) {
    return String(value == null ? '' : value).replace(/[&<>"']/g, function (character) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[character];
    });
  }
  function money(value) {
    return '$' + Number(value || 0).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  }
  function idle(item) {
    return item.days_since_sale === 'Never' ? 'Never sold' : item.days_since_sale + ' days idle';
  }
  function returnDate(value) {
    return new Date(value).toLocaleString(undefined, { year: 'numeric', month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit' });
  }
  function actionButton(item) {
    const action = item.dismissed ? 'restore' : 'dismiss';
    const label = item.dismissed ? 'Restore' : 'Dismiss for 30 days';
    return '<button type="button" class="ds-' + action + '-btn" data-ds-action="' + action +
      '" data-product-id="' + esc(item.product_id) + '" aria-label="' + label + ': ' + esc(item.name) +
      '"' + (pending ? ' disabled' : '') + '>' + label + '</button>';
  }
  function setStatus(message, error) {
    status.textContent = message;
    status.classList.toggle('ds-error', !!error);
    if (modal.classList.contains('open')) {
      modalStatus.textContent = message;
      modalStatus.classList.toggle('ds-error', !!error);
    }
  }
  function setBusy(value) {
    pending = value;
    list.setAttribute('aria-busy', String(value));
    document.querySelectorAll('[data-ds-action], #dsUndo, #ds-ignore-snacks, #ds-ignore-braces, #dsRetry, #dsModalRetry').forEach(function (button) {
      button.disabled = value;
    });
  }
  async function jsonRequest(url, options) {
    const response = await fetch(url, Object.assign({ credentials: 'same-origin', cache: 'no-store' }, options));
    let data;
    try { data = await response.json(); } catch (_) {
      throw new Error('Could not load deadstock. Your session may have expired; refresh the page and try again.');
    }
    if (!response.ok || !data.ok) throw new Error(data.error || 'Could not update deadstock. Please try again.');
    return data;
  }
  function updateDismissedCount(count) {
    const badge = document.getElementById('dsDismissedCount');
    if (badge) badge.textContent = String(count || 0);
  }
  function renderBoard(data) {
    items = data.items || [];
    document.getElementById('dsVisibleCount').textContent = String(items.length);
    document.getElementById('dsSummaryText').textContent = 'Showing ' + items.length + ' of ' + data.available_count + ' available';
    updateDismissedCount(data.dismissed_count);
    if (!items.length) {
      let message = 'No deadstock detected.';
      if (data.total_count && data.filtered_count === data.total_count) message = 'All deadstock is excluded by your category filters.';
      else if (data.total_count) message = 'All matching products are dismissed. Open Dismissed to restore a product early.';
      list.innerHTML = '<div class="deadstock-empty">' + esc(message) + '</div>';
      return;
    }
    list.innerHTML = items.map(function (item) {
      return '<div class="deadstock-item" data-product-id="' + esc(item.product_id) + '">' +
        '<div class="ds-row-main"><span class="deadstock-name">' + esc(item.name) + '</span>' +
        '<div class="deadstock-meta"><span>' + esc(item.quantity_in_stock) + (Number(item.quantity_in_stock) === 1 ? ' unit' : ' units') + '</span><span>Stock value ' + money(item.capital_tied) + '</span></div>' +
        '<div class="deadstock-days">' + esc(idle(item)) + '</div></div><div class="ds-row-actions">' + actionButton(item) + '</div></div>';
    }).join('');
  }
  function focusAfterAction(source, productId) {
    const container = source === 'modal' && modal.classList.contains('open') ? modalBody : list;
    const matching = container.querySelector('[data-product-id="' + productId + '"] button, button[data-product-id="' + productId + '"]');
    const next = matching || container.querySelector('[data-ds-action]');
    if (next) next.focus({ preventScroll: true });
    else if (source === 'modal') modal.querySelector('[data-ds-tab="' + tab + '"]').focus({ preventScroll: true });
    else status.focus({ preventScroll: true });
  }
  function update(action, productId, source) {
    const token = ++generation;
    const selectedFilters = Object.assign({}, filters);
    setBusy(true);
    retry.hidden = true;
    modalRetry.hidden = true;
    setStatus(action === 'next' ? 'Loading suggestions…' : 'Saving…');
    const work = requestQueue.catch(function () {}).then(async function () {
      if (token !== generation) return;
      const payload = Object.assign({ action: action, current_ids: items.map(function (item) { return item.product_id; }) }, selectedFilters);
      if (productId != null) payload.product_id = productId;
      const original = items.concat(modalItems).find(function (item) { return item.product_id === productId; });
      const data = await jsonRequest(api, {
        method: 'POST', headers: { 'Content-Type': 'application/json', 'X-CSRFToken': list.dataset.csrf }, body: JSON.stringify(payload)
      });
      if (token !== generation) return;
      renderBoard(data);
      retryAction = null;
      if (action === 'dismiss') {
        lastDismissal = { id: productId, name: original ? original.name : 'Product' };
        undo.hidden = false;
        undo.textContent = 'Undo last dismissal';
        setStatus(lastDismissal.name + ' dismissed for everyone until ' + returnDate(data.expires_at) + '.');
      } else if (action === 'restore') {
        if (lastDismissal && lastDismissal.id === productId) { lastDismissal = null; undo.hidden = true; }
        setStatus('Product restored for everyone. It can appear again if it still qualifies.');
      } else setStatus('Suggestions updated. Dismissals apply to everyone for 30 days.');
      if (modal.classList.contains('open')) await loadModal(source === 'modal' ? productId : null);
      if (action !== 'next') {
        setBusy(false);
        focusAfterAction(source, productId);
      }
    }).catch(function (error) {
      if (token !== generation) return;
      setStatus(error.message, true);
      retryAction = function () { update(action, productId, source); };
      retry.hidden = false;
      modalRetry.hidden = false;
    }).finally(function () { if (token === generation) setBusy(false); });
    requestQueue = work;
    return work;
  }
  function renderModal(data) {
    modalItems = data.items || [];
    updateDismissedCount(data.dismissed_count);
    const dismissed = tab === 'dismissed';
    page = Number(data.page) || 1;
    totalPages = Number(data.num_pages) || 1;
    modalSub.textContent = dismissed ? data.count + ' products dismissed for everyone' :
      'Showing ' + modalItems.length + ' of ' + data.count + ' — no sales in 69+ days';
    document.getElementById('dsPrev').hidden = !dismissed;
    document.getElementById('dsNext').hidden = !dismissed;
    document.getElementById('dsPrev').disabled = !data.has_previous;
    document.getElementById('dsNext').disabled = !data.has_next;
    document.getElementById('dsFirst').hidden = !dismissed || page <= 1;
    document.getElementById('dsLast').hidden = !dismissed || page <= 1;
    document.getElementById('dsFirst').disabled = !data.has_previous;
    document.getElementById('dsLast').disabled = !data.has_next;
    document.getElementById('dsPageLabel').textContent = dismissed ? 'Page ' + data.page + ' of ' + data.num_pages : '';
    if (!modalItems.length) {
      modalBody.innerHTML = '<div class="dx-empty">' + (dismissed ? 'No products are currently dismissed.' : 'No deadstock detected.') + '</div>';
      return;
    }
    const rows = modalItems.map(function (item) {
      const badge = item.dismissed ? '<span class="ds-dismissed-badge">Dismissed until <time datetime="' + esc(item.expires_at) + '">' + esc(returnDate(item.expires_at)) + '</time></span>' : '';
      return '<tr data-product-id="' + esc(item.product_id) + '"><td class="dx-name">' + esc(item.name) + '</td>' +
        '<td>' + esc(item.category_name || '—') + '</td><td class="dx-num">' + esc(item.quantity_in_stock) + '</td>' +
        '<td class="dx-num">' + money(item.capital_tied) + '</td><td class="dx-num">' + esc(idle(item)) + '</td>' +
        '<td class="ds-table-action">' + badge + actionButton(item) + '</td></tr>';
    }).join('');
    modalBody.innerHTML = '<div data-table-scroll style="overflow-x:auto"><table class="dx-table" data-personalize-table data-table-key="' +
      (dismissed ? 'dead-stock-dismissed' : 'dead-stock-details') + '" data-table-label="' + (dismissed ? 'Dismissed deadstock' : 'Dead stock details') + '">' +
      '<thead><tr><th>Product</th><th>Category</th><th class="dx-num">Units</th><th class="dx-num">Stock Value</th><th class="dx-num">Idle</th><th>Dashboard</th></tr></thead>' +
      '<tbody>' + rows + '</tbody></table></div>';
    if (!dismissed) {
      const total = modalItems.reduce(function (sum, item) { return sum + Number(item.capital_tied || 0); }, 0);
      modalBody.insertAdjacentHTML('beforeend', '<div class="dx-foot">Stock value (shown): <strong>' + money(total) + '</strong></div>');
    }
  }
  async function loadModal(focusId) {
    const token = ++modalGeneration;
    if (modalAbort) modalAbort.abort();
    modalAbort = new AbortController();
    modalBody.setAttribute('aria-busy', 'true');
    modalBody.innerHTML = '<div class="dx-loading">Loading…</div>';
    document.getElementById('dsPrev').disabled = true;
    document.getElementById('dsNext').disabled = true;
    document.getElementById('dsFirst').disabled = true;
    document.getElementById('dsLast').disabled = true;
    const url = tab === 'dismissed' ? api + '?page=' + page : expandUrl + '?section=deadstock';
    try {
      const data = await jsonRequest(url, { signal: modalAbort.signal });
      if (token !== modalGeneration) return;
      page = data.page || 1;
      renderModal(data);
      if (focusId != null) focusAfterAction('modal', focusId);
    } catch (error) {
      if (token !== modalGeneration || error.name === 'AbortError') return;
      modalBody.innerHTML = '<div class="dx-empty" role="status">' + esc(error.message) + ' <button type="button" data-ds-modal-retry>Retry</button></div>';
    } finally {
      if (token === modalGeneration) modalBody.setAttribute('aria-busy', 'false');
    }
  }
  function selectTab(value) {
    tab = value;
    page = 1;
    modal.querySelectorAll('[data-ds-tab]').forEach(function (button) {
      const selected = button.dataset.dsTab === tab;
      button.setAttribute('aria-selected', String(selected));
      button.tabIndex = selected ? 0 : -1;
      button.classList.toggle('active', selected);
      if (selected && button.id) modalBody.setAttribute('aria-labelledby', button.id);
    });
    loadModal();
  }
  function closeModal() {
    modal.classList.remove('open');
    ++modalGeneration;
    if (modalAbort) modalAbort.abort();
    if (returnFocus) returnFocus.focus({ preventScroll: true });
  }
  document.getElementById('deadstockExpand').addEventListener('click', function (event) {
    returnFocus = event.currentTarget;
    modalStatus.textContent = '';
    modal.classList.add('open');
    selectTab('all');
    modal.querySelector('[data-ds-tab="all"]').focus();
  });
  modal.addEventListener('click', function (event) {
    if (event.target === modal || event.target.closest('[data-dx-close]')) closeModal();
    const button = event.target.closest('[data-ds-tab]');
    if (button) selectTab(button.dataset.dsTab);
    if (event.target.closest('[data-ds-modal-retry]')) loadModal();
  });
  document.addEventListener('click', function (event) {
    const button = event.target.closest('[data-ds-action]');
    if (!button || pending || (!list.contains(button) && !modal.contains(button))) return;
    update(button.dataset.dsAction, Number(button.dataset.productId), modal.contains(button) ? 'modal' : 'board');
  });
  modal.addEventListener('keydown', function (event) {
    if (event.key === 'Escape') { event.preventDefault(); event.stopPropagation(); closeModal(); }
    if (event.target.matches('[data-ds-tab]') && ['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) {
      event.preventDefault();
      const value = event.key === 'Home' ? 'all' : event.key === 'End' ? 'dismissed' : tab === 'all' ? 'dismissed' : 'all';
      selectTab(value);
      modal.querySelector('[data-ds-tab="' + value + '"]').focus();
    }
    if (event.key === 'Tab') {
      const controls = Array.from(modal.querySelectorAll('button:not([disabled]), a[href], input, select, [tabindex="0"]')).filter(function (element) { return !element.hidden && element.offsetParent !== null && element.tabIndex >= 0; });
      const first = controls[0];
      const last = controls[controls.length - 1];
      if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last.focus(); }
      else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first.focus(); }
    }
  });
  [[snacks, 'exclude_snacks', 'ds_hide_snacks'], [braces, 'exclude_braces', 'ds_hide_braces']].forEach(function (entry) {
    entry[0].addEventListener('click', function () {
      if (pending) return;
      filters[entry[1]] = !filters[entry[1]];
      try { localStorage.setItem(entry[2], filters[entry[1]] ? '1' : '0'); } catch (_) { /* Browsing without persistent preferences is supported. */ }
      syncFilters();
      update('next');
    });
  });
  undo.addEventListener('click', function () { if (lastDismissal && !pending) update('restore', lastDismissal.id, 'board'); });
  retry.addEventListener('click', function () { if (retryAction && !pending) retryAction(); });
  modalRetry.addEventListener('click', function () { if (retryAction && !pending) retryAction(); });
  document.getElementById('dsPrev').addEventListener('click', function () { page = Math.max(1, page - 1); loadModal(); });
  document.getElementById('dsFirst').addEventListener('click', function () { page = 1; loadModal(); });
  document.getElementById('dsNext').addEventListener('click', function () { page = Math.min(totalPages, page + 1); loadModal(); });
  document.getElementById('dsLast').addEventListener('click', function () { page = totalPages; loadModal(); });
  syncFilters();
  // pageshow runs once for initial load, reload, and cached back/forward entry.
  window.addEventListener('pageshow', function () { update('next'); });
})();
