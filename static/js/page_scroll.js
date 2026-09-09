/* Keep position only while navigating within the same page (query changes included). */
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
  var containers = Object.create(null);
  var submittedEvent = null;
  var historyScrollAllowed = true;
  var unlockObserver = null;
  var restored = false;
  var savedPosition = null;
  var visible = false;

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
  function mayRestore(fromCache) {
    if (fromCache || navigationType() === 'back_forward') {
      if (embedded) {
        try {
          if (window.parent.PageScroll && !window.parent.PageScroll.canRestoreFromHistory()) return false;
        } catch (_) { return false; }
      }
      return read(previousKey) === pathname;
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
          window.scrollTo(0, y);
        });
        unlockObserver.observe(document.body, { attributes: true, attributeFilter: ['style'] });
      }
    } else {
      window.scrollTo(0, y);
    }
  }
  function restoreContainer(name) {
    var offsets = savedPosition && savedPosition.containers && savedPosition.containers[name];
    containers[name].scrollTop = position(offsets && offsets.top);
    containers[name].scrollLeft = position(offsets && offsets.left);
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
      try { saved = JSON.parse(read(positionKey)); } catch (_) {}
    } else {
      remove(positionKey);
    }
    savedPosition = saved;
    registerPageTables();
    restored = true;
    // Explicit fragment links keep their browser-provided destination.
    if (!window.location.hash) moveWindow(position(saved && saved.y));
    Object.keys(containers).forEach(restoreContainer);
    return allowed;
  }
  function save() {
    // AJAX can introduce a new table after load; record its current location.
    registerPageTables(false);
    if (submittedEvent && !submittedEvent.defaultPrevented && submittedEvent.target &&
        submittedEvent.target.matches('[data-no-scroll-restore]')) {
      remove(positionKey);
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
    write(positionKey, JSON.stringify({ y: currentPosition(), containers: offsets }));
    submittedEvent = null;
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
  document.addEventListener('submit', function (event) { submittedEvent = event; }, true);
  window.addEventListener('pagehide', function () { save(); visible = false; });
  window.addEventListener('pageshow', function (event) {
    submittedEvent = null;
    var allowed = event.persisted ? restore(true) : window.PageScroll.canRestore;
    visible = true;
    // Shared table styling may add a scroll wrapper in a later ready handler.
    if (!event.persisted) registerPageTables();
    // Read the previous page before marking this one visible, including BFCache.
    write(previousKey, pathname);
    // A page-local popstate handler must not undo the cross-page reset. Native
    // traversal dispatches popstate after pageshow in the same task.
    historyScrollAllowed = allowed || (!event.persisted && navigationType() !== 'back_forward');
    setTimeout(function () { historyScrollAllowed = true; }, 0);
  });
})();
