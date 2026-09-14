/* Restore each Back/Forward entry directly; fresh cross-page visits start at the top. */
(function () {
  'use strict';

  var pathname = window.location.pathname;
  var embedded = window.top !== window.self;
  var scope = '';
  if (embedded) {
    try { scope = 'frame:' + window.parent.location.pathname + ':'; }
    catch (_) { scope = 'frame:'; }
  }
  var positionKey = 'scroll:page:v1:' + scope + pathname;
  var previousKey = 'scroll:last-visible-page' + (embedded ? ':' + scope : '');
  var historyKey = 'pharmacyPageScroll';
  var containers = Object.create(null);
  var submittedEvent = null;
  var historyScrollAllowed = true;
  var unlockObserver = null;
  var restored = false;
  var savedPosition = null;
  var visible = false;
  var saveTimer = null;
  var restoreFrame = null;

  function read(key) {
    try { return sessionStorage.getItem(key); } catch (_) { return null; }
  }
  function write(key, value) {
    try { sessionStorage.setItem(key, value); } catch (_) {}
  }
  function remove(key) {
    try { sessionStorage.removeItem(key); } catch (_) {}
  }
  function position(value) {
    return typeof value === 'number' && Number.isFinite(value) && value >= 0 ? value : 0;
  }
  function navigationType() {
    try {
      var navigation = performance.getEntriesByType('navigation')[0];
      return navigation ? navigation.type : '';
    } catch (_) { return ''; }
  }
  function historyPosition() {
    try {
      var entry = history.state && history.state[historyKey];
      if (entry && entry.key === positionKey && entry.url === window.location.pathname + window.location.search) {
        return entry;
      }
    } catch (_) {}
    return null;
  }
  function rememberHistory(saved) {
    try {
      var state = Object.assign({}, history.state || {});
      state[historyKey] = {
        key: positionKey,
        url: window.location.pathname + window.location.search,
        position: saved
      };
      history.replaceState(state, '');
    } catch (_) {}
  }
  function mayRestore(fromCache) {
    if (fromCache || navigationType() === 'back_forward') {
      if (embedded) {
        try {
          if (window.parent.PageScroll && !window.parent.PageScroll.canRestoreFromHistory()) return false;
        } catch (_) { return false; }
      }
      return true;
    }
    if (navigationType() === 'reload') return true;
    // A newly opened tab can inherit both its opener's storage and referrer.
    if (window.opener && history.length === 1) return false;
    try {
      var previous = new URL(document.referrer);
      return previous.origin === window.location.origin && previous.pathname === pathname;
    } catch (_) { return false; }
  }
  function manualScroll() {
    if ('scrollRestoration' in history) history.scrollRestoration = 'manual';
  }
  function currentPosition() {
    // Drawers freeze the body at a negative top while window.scrollY becomes 0.
    if (document.body && document.body.style.position === 'fixed') {
      return position(-parseFloat(document.body.style.top));
    }
    return position(window.scrollY);
  }
  function moveWindow(y) {
    if (unlockObserver) { unlockObserver.disconnect(); unlockObserver = null; }
    if (document.body && document.body.style.position === 'fixed') {
      document.body.style.top = -y + 'px';
      // Cached pages can keep a drawer open. Its local unlock handler may still
      // hold the old offset, so apply the decided position after it unfreezes.
      if (typeof MutationObserver === 'function') {
        unlockObserver = new MutationObserver(function () {
          if (document.body.style.position === 'fixed') return;
          unlockObserver.disconnect();
          unlockObserver = null;
          window.scrollTo({ top: y, left: 0, behavior: 'instant' });
        });
        unlockObserver.observe(document.body, { attributes: true, attributeFilter: ['style'] });
      }
    } else {
      if (currentPosition() !== y) window.scrollTo({ top: y, left: 0, behavior: 'instant' });
    }
  }
  function restoreContainer(name) {
    var offsets = savedPosition && savedPosition.containers && savedPosition.containers[name];
    var top = position(offsets && offsets.top);
    var left = position(offsets && offsets.left);
    if (typeof containers[name].scrollTo === 'function') {
      containers[name].scrollTo({ top: top, left: left, behavior: 'instant' });
    } else {
      containers[name].scrollTop = top;
      containers[name].scrollLeft = left;
    }
    if (typeof containers[name]._uiTopScrollUpdate === 'function') containers[name]._uiTopScrollUpdate();
  }
  function registerPageTables(restoreNew) {
    if (typeof window.getComputedStyle !== 'function') return;
    document.querySelectorAll('#main-content table').forEach(function (table, index) {
      if (table.closest('dialog, [role="dialog"], [aria-modal="true"], [class*="modal-overlay"], [class*="-slider-panel"], nav')) return;
      var element = table.parentElement;
      var level = 0;
      while (element && element.id !== 'main-content') {
        var style = window.getComputedStyle(element);
        if (/(auto|scroll)/.test(style.overflowY + ' ' + style.overflowX)) {
          var name = 'table:' + (element.id || table.id || table.getAttribute('data-table-key') || index) + ':' + level++;
          if (!Object.keys(containers).some(function (key) { return containers[key] === element; })) {
            containers[name] = element;
            if (restored && restoreNew !== false) restoreContainer(name);
          }
        }
        element = element.parentElement;
      }
    });
  }
  function restore(fromCache) {
    manualScroll();
    var allowed = mayRestore(fromCache);
    window.PageScroll.canRestore = allowed;
    var saved = null;
    if (allowed) {
      var entry = historyPosition();
      if ((fromCache || navigationType() === 'back_forward' || navigationType() === 'reload') && entry) {
        saved = entry.position;
      } else if (fromCache) {
        saved = savedPosition;
      } else {
        try { saved = JSON.parse(read(positionKey)); } catch (_) {}
      }
    } else {
      remove(positionKey);
    }
    savedPosition = saved;
    registerPageTables();
    restored = true;
    // Explicit fragment links keep their browser-provided destination.
    if (!window.location.hash) moveWindow(position(saved && saved.y));
    Object.keys(containers).forEach(restoreContainer);
    // Later ready/pageshow handlers can finish table layout or reorder rows.
    // Apply the same position before the next paint, without an animated scroll.
    if (restoreFrame !== null) window.cancelAnimationFrame(restoreFrame);
    restoreFrame = window.requestAnimationFrame(function () {
      restoreFrame = null;
      registerPageTables();
      if (!window.location.hash) moveWindow(position(savedPosition && savedPosition.y));
      Object.keys(containers).forEach(restoreContainer);
    });
    return allowed;
  }
  function save() {
    if (saveTimer !== null) { clearTimeout(saveTimer); saveTimer = null; }
    // AJAX can introduce a new table after load; record its current location.
    registerPageTables(false);
    if (submittedEvent && !submittedEvent.defaultPrevented && submittedEvent.target &&
        submittedEvent.target.matches('[data-no-scroll-restore]')) {
      remove(positionKey);
      savedPosition = null;
      rememberHistory(null);
      submittedEvent = null;
      return;
    }
    var offsets = Object.create(null);
    Object.keys(containers).forEach(function (name) {
      offsets[name] = {
        top: position(containers[name].scrollTop),
        left: position(containers[name].scrollLeft)
      };
    });
    savedPosition = { y: currentPosition(), containers: offsets };
    write(positionKey, JSON.stringify(savedPosition));
    rememberHistory(savedPosition);
    submittedEvent = null;
  }
  function cancelQueuedRestore() {
    if (restoreFrame !== null) { window.cancelAnimationFrame(restoreFrame); restoreFrame = null; }
  }

  window.PageScroll = {
    canRestore: mayRestore(false),
    registerContainer: function (name, element) {
      if (element) {
        containers[name] = element;
        if (restored) restoreContainer(name);
      }
    },
    allowHistoryScroll: function () { return historyScrollAllowed; },
    // Child frames can receive pageshow before their cached parent does.
    canRestoreFromHistory: function () {
      return visible ? window.PageScroll.canRestore : mayRestore(true);
    }
  };

  // Load in the head, before native history restoration can run.
  manualScroll();
  document.addEventListener('DOMContentLoaded', function () { restore(false); });
  document.addEventListener('submit', function (event) {
    submittedEvent = event;
    if (saveTimer !== null) { clearTimeout(saveTimer); saveTimer = null; }
  }, true);
  window.addEventListener('wheel', cancelQueuedRestore, { passive: true });
  window.addEventListener('touchstart', cancelQueuedRestore, { passive: true });
  window.addEventListener('pointerdown', cancelQueuedRestore, { passive: true });
  window.addEventListener('keydown', function (event) {
    if (['ArrowUp', 'ArrowDown', 'PageUp', 'PageDown', 'Home', 'End', ' '].indexOf(event.key) !== -1) cancelQueuedRestore();
  });
  document.addEventListener('scroll', function () {
    if (!visible || restoreFrame !== null || (submittedEvent && !submittedEvent.defaultPrevented)) return;
    if (saveTimer !== null) clearTimeout(saveTimer);
    saveTimer = setTimeout(save, 150);
  }, true);
  window.addEventListener('pagehide', function () {
    cancelQueuedRestore();
    save();
    visible = false;
  });
  window.addEventListener('pageshow', function (event) {
    submittedEvent = null;
    if (event.persisted) restore(true);
    visible = true;
    // Shared table styling may add a scroll wrapper in a later ready handler.
    if (!event.persisted) registerPageTables();
    write(previousKey, pathname);
    // The shared restore owns cross-document history. Page-local popstate code
    // may update its UI, but must not apply a second, conflicting scroll.
    historyScrollAllowed = !event.persisted && navigationType() !== 'back_forward';
    setTimeout(function () { historyScrollAllowed = true; }, 0);
  });
})();
