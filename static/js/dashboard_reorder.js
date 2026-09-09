/* Put reorder suggestions on the shared ordering list without creating a purchase. */
(function () {
  'use strict';
  const list = document.getElementById('reorderList');
  const config = list && list.closest('.sidebar-reorder');
  if (!config || !config.dataset.addUrl || config.dataset.reorderInitialized) return;
  config.dataset.reorderInitialized = 'true';

  const selector = '.reorder-add-control[data-product-id]';
  const pending = new Map();
  const added = new Map();
  const failures = new Map();
  const statuses = [document.getElementById('reorderAddStatus'), document.getElementById('reorderModalStatus')];
  const now = function () { return performance.now(); };

  function productId(control) {
    const id = Number(control.dataset.productId);
    return Number.isSafeInteger(id) && id > 0 ? id : null;
  }
  function controlsFor(id) {
    return Array.from(document.querySelectorAll(selector)).filter(function (control) { return productId(control) === id; });
  }
  function label(button, text) {
    const target = button.querySelector('[data-reorder-label]') || button;
    target.textContent = text;
  }
  function status(message, error) {
    statuses.forEach(function (element) {
      if (!element) return;
      element.textContent = message;
      element.classList.toggle('reorder-add-error', !!error);
    });
  }
  function render(id) {
    const saving = pending.get(id);
    const success = added.get(id);
    const failure = failures.get(id);
    controlsFor(id).forEach(function (control) {
      const input = control.querySelector('[data-reorder-quantity]');
      const button = control.querySelector('[data-reorder-add]');
      if (!input || !button) return;
      control.dataset.added = success ? 'true' : 'false';
      control.setAttribute('aria-busy', String(!!saving));
      input.hidden = !!success;
      input.disabled = !!saving || !!success;
      button.disabled = !!saving || !!success;
      if (saving || failure) input.value = String((saving || failure).quantity);
      if (saving || success) input.removeAttribute('aria-invalid');
      label(button, saving ? 'Adding…' : success ? 'Added to Recently Purchased' : failure ? 'Retry add' : 'Add to Recently Purchased');
    });
  }
  function reconcile(event) {
    const detail = event && event.detail;
    const scope = detail && detail.scope && detail.scope.querySelectorAll ? detail.scope : document;
    const requestedAt = detail && Number.isFinite(detail.requestedAt) ? detail.requestedAt : null;
    const ids = new Set();
    scope.querySelectorAll(selector).forEach(function (control) {
      const id = productId(control);
      if (!id) return;
      ids.add(id);
      if (pending.has(id)) return;
      const previous = added.get(id);
      // A fresh full-list response is authoritative, unless this page saved
      // the product after that response's request began.
      if (previous && requestedAt !== null && previous.savedAt > requestedAt) return;
      if (control.dataset.added === 'true') {
        added.set(id, previous || { savedAt: -Infinity });
        failures.delete(id);
      } else if (requestedAt !== null) {
        added.delete(id);
      }
    });
    ids.forEach(render);
  }

  async function add(control, button) {
    const id = productId(control);
    if (!id || pending.has(id) || added.has(id) || button.disabled) return;
    const input = control.querySelector('[data-reorder-quantity]');
    if (!input) return;
    const raw = input.value.trim();
    const quantity = Number(raw);
    if (!/^\d+$/.test(raw) || !Number.isSafeInteger(quantity) || quantity < 1 || quantity > 9999) {
      input.setAttribute('aria-invalid', 'true');
      status('Enter a whole-number quantity from 1 to 9,999.', true);
      input.focus();
      return;
    }
    const sourceStatus = control.closest('#reorderModal') ? statuses[1] : statuses[0];
    const wasFocused = document.activeElement === button;
    failures.delete(id);
    pending.set(id, { quantity: quantity });
    render(id);
    status('Adding to Recently Purchased…', false);
    try {
      const response = await fetch(config.dataset.addUrl, {
        method: 'POST',
        credentials: 'same-origin',
        cache: 'no-store',
        headers: {
          'Content-Type': 'application/json',
          'X-CSRFToken': config.dataset.csrf,
          'X-Requested-With': 'XMLHttpRequest'
        },
        body: JSON.stringify({ product_id: id, quantity: quantity })
      });
      if (response.redirected) throw new Error('Your session may have expired. Refresh the page, sign in, and try again.');
      let data;
      try { data = await response.json(); } catch (_) {
        throw new Error('Could not add this product. Refresh the page to check your session, then try again.');
      }
      if (!response.ok || !data || !data.ok) {
        throw new Error(data && typeof data.error === 'string' ? data.error : 'Could not add this product. Please try again.');
      }
      if (Number(data.product_id) !== id) throw new Error('Could not confirm this product was added. Please try again.');
      added.set(id, { savedAt: now() });
      status(typeof data.message === 'string' ? data.message : 'Added to Recently Purchased.', false);
    } catch (error) {
      failures.set(id, { quantity: quantity });
      status(error instanceof TypeError ? 'Could not connect. Check your connection, then select Retry add.' : error.message || 'Could not add this product. Please try again.', true);
    } finally {
      pending.delete(id);
      render(id);
      // Keep keyboard users near the result when their button becomes disabled.
      const active = document.activeElement;
      if (added.has(id) && wasFocused && (active === button || active === document.body) && sourceStatus) sourceStatus.focus();
    }
  }

  document.addEventListener('click', function (event) {
    const button = event.target.closest && event.target.closest('[data-reorder-add]');
    const control = button && button.closest(selector);
    if (!control || event.defaultPrevented) return;
    event.preventDefault();
    add(control, button);
  });
  document.addEventListener('dashboard:reorder-rendered', reconcile);
  document.addEventListener('focusin', function (event) {
    const target = event.target;
    if (target.closest && target.closest('#reorderModal') &&
        target.matches('[data-reorder-add], [data-reorder-quantity]') && target.scrollIntoView) {
      target.scrollIntoView({ block: 'nearest', inline: 'nearest' });
    }
  });
  reconcile();
}());
