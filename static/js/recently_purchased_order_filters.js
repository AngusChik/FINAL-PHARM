/* Keep the automated-order quick switches and category exclusions in sync. */
(function (root) {
  'use strict';

  function init(modal, storage) {
    var key = 'rp_ao_category_filters:v1';
    var choices = {};
    var defaults = { snacks: true, braces: false };
    var saved = false;
    var cats = modal.querySelector('#rp-ao-cats');
    var checks = Array.from(cats.querySelectorAll('.rp-ao-cat-check'));
    var search = modal.querySelector('#rp-ao-cat-search');
    var summary = modal.querySelector('#rp-ao-filter-summary');
    var emptySearch = modal.querySelector('#rp-ao-cat-search-empty');
    var quick = ['snacks', 'braces'].map(function (name) {
      return {
        name: name,
        button: modal.querySelector('#rp-ao-' + name + '-toggle'),
        description: modal.querySelector('#rp-ao-' + name + '-description'),
        checks: checks.filter(function (cb) { return String(cb.dataset.name || '').trim().toLowerCase() === name; })
      };
    });

    try {
      storage = storage || root.localStorage;
      var stored = JSON.parse(storage.getItem(key));
      if (stored && stored.choices && typeof stored.choices === 'object' && !Array.isArray(stored.choices)) {
        Object.keys(stored.choices).forEach(function (id) {
          if (/^\d+$/.test(id) && typeof stored.choices[id] === 'boolean') choices[id] = stored.choices[id];
        });
        quick.forEach(function (entry) {
          if (stored.defaults && typeof stored.defaults[entry.name] === 'boolean') defaults[entry.name] = stored.defaults[entry.name];
        });
        saved = true;
      }
    } catch (_) { /* Filters still work when browser storage is unavailable. */ }

    function excludedCategoryIds() {
      return checks.filter(function (cb) { return cb.checked; })
        .map(function (cb) { return Number(cb.value); })
        .filter(function (id) { return Number.isInteger(id) && id > 0; });
    }

    function description() {
      var excluded = checks.filter(function (cb) { return cb.checked; });
      return excluded.length
        ? 'Excluded: ' + excluded.map(function (cb) { return cb.dataset.label || cb.dataset.name; }).join(', ') + '.'
        : 'All categories are included.';
    }

    function refresh() {
      quick.forEach(function (entry) {
        var available = entry.checks.length > 0;
        var checked = available && entry.checks.every(function (cb) { return cb.checked; });
        entry.button.disabled = !available;
        entry.button.classList.toggle('active', checked);
        entry.button.setAttribute('aria-checked', String(checked));
        entry.description.textContent = available
          ? 'Skip the ' + entry.name + ' category when building the order.'
          : 'No ' + entry.name + ' category in the current list.';
      });
      var count = excludedCategoryIds().length;
      summary.textContent = count + ' of ' + checks.length + ' categories excluded. ' +
        (saved ? 'Saved on this browser.' : 'Changes are remembered on this browser when available.');
      var query = search.value.trim().toLowerCase();
      var visible = 0;
      checks.forEach(function (cb) {
        var matches = !query || String(cb.dataset.label || cb.dataset.name || '').toLowerCase().indexOf(query) !== -1;
        cb.closest('label').hidden = !matches;
        if (matches) visible += 1;
      });
      emptySearch.hidden = !checks.length || visible > 0;
    }

    function applyChoices() {
      checks.forEach(function (cb) {
        var name = String(cb.dataset.name || '').trim().toLowerCase();
        cb.checked = Object.prototype.hasOwnProperty.call(choices, cb.value)
          ? choices[cb.value] : defaults[name] === true;
      });
      refresh();
    }

    function persist() {
      checks.forEach(function (cb) { choices[cb.value] = cb.checked; });
      try {
        storage.setItem(key, JSON.stringify({ choices: choices, defaults: defaults }));
        saved = true;
      } catch (_) { saved = false; }
      refresh();
    }

    quick.forEach(function (entry) {
      entry.button.addEventListener('click', function () {
        if (!entry.checks.length) return;
        var checked = !entry.checks.every(function (cb) { return cb.checked; });
        defaults[entry.name] = checked;
        entry.checks.forEach(function (cb) { cb.checked = checked; });
        persist();
      });
    });
    cats.addEventListener('change', function (event) {
      if (!event.target.classList.contains('rp-ao-cat-check')) return;
      quick.forEach(function (entry) {
        if (entry.checks.indexOf(event.target) !== -1) defaults[entry.name] = entry.checks.every(function (cb) { return cb.checked; });
      });
      persist();
    });
    function setAll(checked) {
      checks.forEach(function (cb) { cb.checked = checked; });
      quick.forEach(function (entry) { defaults[entry.name] = checked; });
      persist();
    }
    modal.querySelector('#rp-ao-cat-all').addEventListener('click', function () { setAll(true); });
    modal.querySelector('#rp-ao-cat-none').addEventListener('click', function () { setAll(false); });
    modal.querySelector('#rp-ao-cat-reset').addEventListener('click', function () {
      choices = {};
      defaults = { snacks: true, braces: false };
      search.value = '';
      applyChoices();
      persist();
    });
    search.addEventListener('input', refresh);
    applyChoices();
    return { refresh: refresh, excludedCategoryIds: excludedCategoryIds, description: description };
  }

  root.RecentlyPurchasedOrderFilters = { init: init };
})(typeof window === 'undefined' ? globalThis : window);
