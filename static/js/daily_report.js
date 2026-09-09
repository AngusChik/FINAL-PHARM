(function () {
  'use strict';
  var report = document.querySelector('.dr-wrap');
  var tablist = report && report.querySelector('.dr-tabs[role="tablist"]');

  if (tablist) {
    var tabs = Array.from(tablist.querySelectorAll('[data-report-tab]'));
    var panels = tabs.map(function (tab) { return document.getElementById(tab.getAttribute('aria-controls')); });
    var selectedTab = null;
    var printState = null;
    var sectionTabs = {
      drOutOfStock: 'stock',
      drLowStock: 'stock',
      drExpiry: 'expiry',
      drExpired: 'expiry'
    };

    function tabForHash(hash) {
      var targetId;
      try { targetId = decodeURIComponent(hash.replace(/^#/, '')); }
      catch (error) { return null; }
      var name = sectionTabs[targetId];
      return tabs.find(function (tab) {
        return tab.getAttribute('aria-controls') === targetId || 'report-' + tab.dataset.reportTab === targetId || tab.dataset.reportTab === name;
      }) || null;
    }

    function keepReportContext(tab) {
      var hash = '#report-' + tab.dataset.reportTab;
      var controls = report.querySelector('.dr-controls');
      if (controls) {
        var action = new URL(controls.getAttribute('action') || window.location.href, window.location.href);
        action.hash = hash;
        controls.action = action.href;
      }
      report.querySelectorAll('.dr-day-nav a').forEach(function (link) {
        var destination = new URL(link.href, window.location.href);
        destination.hash = hash;
        link.href = destination.href;
      });
      report.querySelectorAll('.dr-product-link').forEach(function (link) {
        var destination = new URL(link.href, window.location.href);
        var returnTo = destination.searchParams.get('return_to');
        if (!returnTo) return;
        var returnURL = new URL(returnTo, window.location.href);
        if (returnURL.origin !== window.location.origin) return;
        returnURL.hash = hash;
        destination.searchParams.set('return_to', returnURL.pathname + returnURL.search + returnURL.hash);
        link.href = destination.href;
      });
    }

    function selectTab(tab, updateHash) {
      selectedTab = tab;
      tabs.forEach(function (item, index) {
        var selected = item === tab;
        item.setAttribute('aria-selected', String(selected));
        item.tabIndex = selected ? 0 : -1;
        panels[index].hidden = !selected;
      });
      if (updateHash) {
        var currentURL = new URL(window.location.href);
        currentURL.hash = 'report-' + tab.dataset.reportTab;
        window.history.replaceState(window.history.state, '', currentURL.href);
      }
      keepReportContext(tab);
    }

    // Keep the complete report readable if the template and script disagree.
    if (tabs.length && panels.every(function (panel) { return panel && report.contains(panel); })) {
      var environmentBanner = document.querySelector('.ui-development-banner');
      function positionTabs() {
        var height = environmentBanner ? environmentBanner.getBoundingClientRect().height : 0;
        report.style.setProperty('--dr-banner-height', Math.ceil(height) + 'px');
      }
      positionTabs();
      window.addEventListener('resize', positionTabs);
      if (environmentBanner && typeof ResizeObserver !== 'undefined') {
        new ResizeObserver(positionTabs).observe(environmentBanner);
      }
      tabs.forEach(function (tab, index) {
        tab.setAttribute('role', 'tab');
        panels[index].setAttribute('role', 'tabpanel');
        panels[index].setAttribute('aria-labelledby', tab.id);
        panels[index].tabIndex = 0;
        tab.addEventListener('click', function () { selectTab(tab, true); });
        tab.addEventListener('keydown', function (event) {
          var nextIndex;
          if (event.key === 'ArrowRight') nextIndex = (index + 1) % tabs.length;
          else if (event.key === 'ArrowLeft') nextIndex = (index + tabs.length - 1) % tabs.length;
          else if (event.key === 'Home') nextIndex = 0;
          else if (event.key === 'End') nextIndex = tabs.length - 1;
          else return;
          event.preventDefault();
          selectTab(tabs[nextIndex], true);
          tabs[nextIndex].focus({ preventScroll: true });
        });
      });
      report.classList.add('dr-tabs-ready');
      tablist.hidden = false;
      selectTab(tabForHash(window.location.hash) || tabs[0], false);
      window.addEventListener('hashchange', function () {
        var tab = tabForHash(window.location.hash);
        if (tab || !window.location.hash) selectTab(tab || tabs[0], false);
      });
      report.addEventListener('click', function (event) {
        var link = event.target.closest('a[href^="#"]');
        if (!link || event.button !== 0 || event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) return;
        var tab = tabForHash(link.hash);
        if (!tab) return;
        event.preventDefault();
        selectTab(tab, true);
        var target = document.getElementById(decodeURIComponent(link.hash.slice(1)));
        if (target) {
          if (!target.hasAttribute('tabindex')) target.tabIndex = -1;
          target.focus({ preventScroll: true });
        }
      });
      window.addEventListener('beforeprint', function () {
        if (printState) return;
        printState = {
          tab: selectedTab,
          details: Array.from(report.querySelectorAll('details')).map(function (detail) {
            return { element: detail, open: detail.open };
          })
        };
        panels.forEach(function (panel) { panel.hidden = false; });
        printState.details.forEach(function (detail) { detail.element.open = true; });
      });
      window.addEventListener('afterprint', function () {
        if (!printState) return;
        selectTab(printState.tab, false);
        printState.details.forEach(function (detail) { detail.element.open = detail.open; });
        printState = null;
      });
    }
  }

  var printButton = document.getElementById('drPrintPage');
  if (printButton) printButton.addEventListener('click', function () { window.print(); });
  var frame = document.getElementById('drrPrintFrame');
  if (!frame) return;
  document.querySelectorAll('.dr-history [data-print]').forEach(function (button) {
    button.addEventListener('click', function () {
      var url = button.getAttribute('data-print');
      frame.onload = function () {
        try { frame.contentWindow.focus(); frame.contentWindow.print(); }
        catch (error) { window.open(url, '_blank', 'noopener'); }
      };
      frame.src = url;
    });
  });
})();
