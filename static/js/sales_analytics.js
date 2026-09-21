(function () {
  'use strict';

  const page = document.querySelector('.sa-page');
  if (!page) return;
  const tablist = page.querySelector('.sa-tabs');
  const tabs = Array.from(page.querySelectorAll('[data-sales-tab]'));
  const panels = Array.from(page.querySelectorAll('.sa-panel'));
  const activeInput = document.getElementById('sa-active-tab');
  const form = document.getElementById('sa-filters');
  const from = document.getElementById('sa-date-from');
  const to = document.getElementById('sa-date-to');
  const filterPanel = page.querySelector('.sa-filter-panel');
  const mobileFilters = window.matchMedia('(max-width: 600px)');
  function fitFilters() { filterPanel.open = !mobileFilters.matches; }
  fitFilters();
  mobileFilters.addEventListener('change', fitFilters);

  function selectedFromURL() {
    const url = new URL(window.location.href);
    const fragment = url.hash.replace('#sa-panel-', '');
    return tabs.some(tab => tab.dataset.salesTab === fragment)
      ? fragment : url.searchParams.get('tab');
  }

  function selectTab(key, updateURL) {
    const selected = tabs.find(tab => tab.dataset.salesTab === key) || tabs[0];
    key = selected.dataset.salesTab;
    tabs.forEach(tab => {
      const active = tab === selected;
      tab.setAttribute('aria-selected', String(active));
      tab.tabIndex = active ? 0 : -1;
    });
    panels.forEach(panel => { panel.hidden = panel.id !== 'sa-panel-' + key; });
    filterPanel.hidden = key === 'suggestions';
    page.querySelector('.sa-period-label').hidden = key === 'suggestions';
    activeInput.value = key;
    if (updateURL) {
      const url = new URL(window.location.href);
      url.searchParams.set('tab', key);
      // The query parameter owns tab selection; remove an old panel anchor.
      if (url.hash.startsWith('#sa-panel-')) url.hash = '';
      window.history.pushState(window.history.state, '', url);
    }
    page.dispatchEvent(new CustomEvent('sales:tabchange', { detail: { key } }));
  }

  tablist.setAttribute('role', 'tablist');
  tabs.forEach((tab, index) => {
    tab.setAttribute('role', 'tab');
    tab.setAttribute('aria-controls', 'sa-panel-' + tab.dataset.salesTab);
    tab.addEventListener('click', event => {
      if (event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) return;
      event.preventDefault();
      selectTab(tab.dataset.salesTab, activeInput.value !== tab.dataset.salesTab);
    });
    tab.addEventListener('keydown', event => {
      let next;
      if (event.key === 'ArrowRight') next = (index + 1) % tabs.length;
      else if (event.key === 'ArrowLeft') next = (index + tabs.length - 1) % tabs.length;
      else if (event.key === 'Home') next = 0;
      else if (event.key === 'End') next = tabs.length - 1;
      else if (event.key === ' ') next = index;
      else return;
      event.preventDefault();
      tabs[next].focus({ preventScroll: true });
      selectTab(tabs[next].dataset.salesTab, activeInput.value !== tabs[next].dataset.salesTab);
    });
  });
  panels.forEach(panel => panel.setAttribute('role', 'tabpanel'));
  selectTab(selectedFromURL(), false);
  window.addEventListener('popstate', () => selectTab(selectedFromURL(), false));
  window.addEventListener('hashchange', () => selectTab(selectedFromURL(), false));

  function validateDates() {
    to.setCustomValidity(from.value && to.value && from.value > to.value
      ? 'Choose an end date on or after the start date.' : '');
  }
  [from, to].forEach(input => input.addEventListener('input', validateDates));
  validateDates();

  function applyFilters() {
    validateDates();
    form.requestSubmit();
  }
  form.querySelectorAll('input[name="gran"], input[name="ignore_snacks"], input[name="ignore_braces"]').forEach(input => {
    input.addEventListener('change', () => {
      form.querySelectorAll('input[name="gran"], input[name="ignore_snacks"], input[name="ignore_braces"]').forEach(option => {
        option.closest('label').classList.toggle('active', option.checked);
      });
      applyFilters();
    });
  });

  function localDate(date) {
    return date.getFullYear() + '-' + String(date.getMonth() + 1).padStart(2, '0')
      + '-' + String(date.getDate()).padStart(2, '0');
  }
  page.querySelector('.sa-quick-ranges').hidden = false;
  page.querySelectorAll('[data-range]').forEach(button => {
    button.addEventListener('click', () => {
      const today = new Date();
      const start = new Date(today.getFullYear(), today.getMonth(), today.getDate());
      const range = button.dataset.range;
      if (range === 'month') start.setDate(1);
      else if (range === 'year') start.setMonth(0, 1);
      else if (range !== 'today') start.setDate(start.getDate() - Number(range) + 1);
      from.value = localDate(start);
      to.value = localDate(today);
      applyFilters();
    });
  });

  const search = document.getElementById('sa-category-search');
  if (search) {
    search.closest('.sa-table-search').hidden = false;
    const categories = Array.from(page.querySelectorAll('[data-sales-category]'));
    const count = document.getElementById('sa-category-count');
    const empty = document.getElementById('sa-category-no-results');
    function filterCategories() {
      const query = search.value.trim().toLocaleLowerCase();
      let visible = 0;
      categories.forEach(category => {
        category.hidden = !category.dataset.salesCategory.toLocaleLowerCase().includes(query);
        if (!category.hidden) visible += 1;
      });
      count.textContent = visible + ' of ' + categories.length + ' categories';
      empty.hidden = visible !== 0;
    }
    search.addEventListener('input', filterCategories);
    filterCategories();
  }
})();
