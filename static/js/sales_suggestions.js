(function () {
  'use strict';

  const page = document.querySelector('.sa-page');
  const panel = document.getElementById('sa-panel-suggestions');
  if (!page || !panel) return;

  const form = document.getElementById('sa-suggestions-filters');
  const content = document.getElementById('sa-suggestions-content');
  const status = document.getElementById('sa-suggestions-status');
  const refresh = document.getElementById('sa-suggestions-refresh');
  const query = document.getElementById('sa-suggestions-query');
  const categoryCount = document.getElementById('sa-suggestions-category-count');
  const categories = Array.from(form.querySelectorAll('input[name="category"]'));
  const snacks = form.querySelector('input[name="hide_snacks"]');
  const braces = form.querySelector('input[name="hide_braces"]');
  const filterNames = ['q', 'category', 'hide_snacks', 'hide_braces'];
  let loadedSignature = '';
  let pendingSignature = '';
  let activeRequest = null;
  let sequence = 0;

  function selectedCategories() {
    return categories.filter(input => input.checked).map(input => input.value);
  }

  function updateCategoryCount() {
    const count = selectedCategories().length;
    categoryCount.textContent = count ? count + ' selected' : 'All';
  }

  function preserveFiltersForCharts(url) {
    const chartForm = document.getElementById('sa-filters');
    chartForm.querySelectorAll('[data-suggestion-filter]').forEach(input => input.remove());
    filterNames.forEach(name => {
      const value = url.searchParams.get(name);
      if (!value) return;
      const input = document.createElement('input');
      input.type = 'hidden';
      input.name = name;
      input.value = value;
      input.setAttribute('data-suggestion-filter', '');
      chartForm.appendChild(input);
    });
  }

  function restoreFilters() {
    const url = new URL(window.location.href);
    query.value = url.searchParams.get('q') || '';
    const selected = (url.searchParams.get('category') || '').split(',');
    categories.forEach(input => { input.checked = selected.includes(input.value); });
    snacks.checked = url.searchParams.get('hide_snacks') === '1';
    braces.checked = url.searchParams.get('hide_braces') === '1';
    updateCategoryCount();
    preserveFiltersForCharts(url);
  }

  function applyFilters() {
    const url = new URL(window.location.href);
    filterNames.forEach(name => url.searchParams.delete(name));
    if (query.value.trim()) url.searchParams.set('q', query.value.trim());
    const selected = selectedCategories();
    if (selected.length) url.searchParams.set('category', selected.join(','));
    if (snacks.checked) url.searchParams.set('hide_snacks', '1');
    if (braces.checked) url.searchParams.set('hide_braces', '1');
    url.searchParams.set('tab', 'suggestions');
    if (url.hash.startsWith('#sa-panel-')) url.hash = '';
    if (url.toString() !== window.location.href) {
      window.history.pushState(window.history.state, '', url);
    }
    preserveFiltersForCharts(url);
    updateCategoryCount();
    form.querySelectorAll('details').forEach(details => { details.open = false; });
    loadSuggestions(false);
  }

  function loadSuggestions(force) {
    const url = new URL(panel.dataset.suggestionsUrl, window.location.href);
    const current = new URL(window.location.href);
    filterNames.forEach(name => {
      const value = current.searchParams.get(name);
      if (value) url.searchParams.set(name, value);
    });
    const signature = url.toString();
    if (!force && (signature === loadedSignature || signature === pendingSignature)) return;
    if (activeRequest) activeRequest.abort();
    const requestNumber = ++sequence;
    loadedSignature = '';
    pendingSignature = signature;
    activeRequest = typeof AbortController === 'function' ? new AbortController() : null;
    content.setAttribute('aria-busy', 'true');
    content.innerHTML = '<div class="rp-suggestions-loading" aria-hidden="true">' +
      '<div></div><div></div></div>';
    status.textContent = 'Calculating ordering suggestions.';
    refresh.disabled = true;

    fetch(signature, {
      method: 'GET',
      credentials: 'same-origin',
      cache: 'no-store',
      headers: { 'Accept': 'application/json', 'X-Requested-With': 'XMLHttpRequest' },
      signal: activeRequest ? activeRequest.signal : undefined
    }).then(response => {
      if (!response.ok || !(response.headers.get('content-type') || '').includes('application/json')) {
        throw new Error('Suggestions could not be loaded');
      }
      return response.json();
    }).then(data => {
      if (requestNumber !== sequence) return;
      if (!data || typeof data.html !== 'string') throw new Error('Incomplete suggestions');
      content.innerHTML = data.html;
      loadedSignature = signature;
      const count = Number(data.count || 0);
      status.textContent = count === 1 ? '1 ordering suggestion is ready.' : count + ' ordering suggestions are ready.';
    }).catch(error => {
      if (requestNumber !== sequence || (error && error.name === 'AbortError')) return;
      loadedSignature = '';
      content.innerHTML = '<div class="rp-suggestions-error"><div>' +
        '<strong>Suggestions could not be loaded.</strong>Try the calculation again.</div>' +
        '<button type="button" class="sa-quick-range" data-suggestions-retry>Try again</button></div>';
      status.textContent = 'Ordering suggestions could not be loaded.';
    }).finally(() => {
      if (requestNumber !== sequence) return;
      content.setAttribute('aria-busy', 'false');
      refresh.disabled = false;
      pendingSignature = '';
      activeRequest = null;
    });
  }

  form.addEventListener('submit', event => { event.preventDefault(); applyFilters(); });
  categories.forEach(input => input.addEventListener('change', updateCategoryCount));
  [snacks, braces].forEach(input => input.addEventListener('change', applyFilters));
  document.getElementById('sa-suggestions-clear').addEventListener('click', () => {
    form.reset();
    applyFilters();
  });
  refresh.addEventListener('click', () => loadSuggestions(true));
  content.addEventListener('click', event => {
    if (event.target.closest('[data-suggestions-retry]')) loadSuggestions(true);
  });
  page.addEventListener('sales:tabchange', event => {
    if (event.detail.key === 'suggestions') loadSuggestions(false);
  });
  window.addEventListener('popstate', () => {
    restoreFilters();
    if (!panel.hidden) loadSuggestions(false);
  });
  restoreFilters();
  if (!panel.hidden) loadSuggestions(false);
})();
