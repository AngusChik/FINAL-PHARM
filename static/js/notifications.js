/* Shared notification contract: transient toasts and persistent action banners. */
(function () {
  'use strict';

  if (window.PharmacyNotifications) return;

  var TITLES = {
    success: 'Success',
    warning: 'Heads up',
    error: 'Something went wrong',
    info: 'Notice'
  };
  var ICONS = {
    success: '✓',
    warning: '!',
    error: '×',
    info: 'i'
  };
  var DEFAULT_DURATION = 3500;
  var BANNER_STORAGE_KEY = 'ui-action-banner:dismissed';

  function normalizeLevel(level) {
    level = String(level || 'info').toLowerCase();
    if (level === 'danger') level = 'error';
    return Object.prototype.hasOwnProperty.call(TITLES, level) ? level : 'info';
  }

  function ensureToastStack() {
    var stack = document.querySelector('[data-ui-toast-stack]');
    if (stack) return stack;
    if (!document.body) return null;

    stack = document.createElement('div');
    stack.className = 'ui-toast-stack toast-stack';
    stack.setAttribute('data-ui-toast-stack', '');
    stack.setAttribute('aria-live', 'polite');
    stack.setAttribute('aria-relevant', 'additions');
    stack.setAttribute('aria-atomic', 'false');
    document.body.appendChild(stack);
    return stack;
  }

  function removeToast(toast) {
    if (!toast || !toast.parentNode) return;
    toast.remove();
  }

  function hideToast(toast) {
    if (!toast || toast.classList.contains('is-hiding')) return;
    if (toast._uiToastTimer) window.clearTimeout(toast._uiToastTimer);
    toast.classList.add('is-hiding');
    toast.addEventListener('animationend', function () { removeToast(toast); }, { once: true });
    window.setTimeout(function () { removeToast(toast); }, 240);
  }

  function armToast(toast, duration) {
    if (!toast || toast.hasAttribute('data-ui-toast-armed')) return toast;
    toast.setAttribute('data-ui-toast-armed', 'true');

    var close = toast.querySelector('[data-ui-toast-dismiss], .toast-close');
    if (close) close.addEventListener('click', function () { hideToast(toast); });

    var requested = Number(duration || toast.getAttribute('data-duration') || DEFAULT_DURATION);
    if (!Number.isFinite(requested) || requested < 0) requested = DEFAULT_DURATION;
    if (requested > 0) {
      toast._uiToastTimer = window.setTimeout(function () { hideToast(toast); }, requested);
    }
    return toast;
  }

  function buildToast(text, level, options) {
    var toast = document.createElement('div');
    toast.className = 'ui-toast toast-msg';
    toast.setAttribute('data-level', level);
    toast.setAttribute('role', level === 'error' ? 'alert' : 'status');
    toast.setAttribute('aria-atomic', 'true');

    var icon = document.createElement('span');
    icon.className = 'ui-toast__icon toast-icon';
    icon.setAttribute('aria-hidden', 'true');
    icon.textContent = ICONS[level];

    var body = document.createElement('div');
    body.className = 'ui-toast__body toast-body';
    var title = document.createElement('div');
    title.className = 'ui-toast__title toast-title';
    title.textContent = options.title || TITLES[level];
    var message = document.createElement('div');
    message.className = 'ui-toast__text toast-text';
    message.textContent = text;
    body.appendChild(title);
    body.appendChild(message);

    var close = document.createElement('button');
    close.type = 'button';
    close.className = 'ui-toast__dismiss toast-close';
    close.setAttribute('data-ui-toast-dismiss', '');
    close.setAttribute('aria-label', 'Dismiss notification');
    close.textContent = '×';

    toast.appendChild(icon);
    toast.appendChild(body);
    toast.appendChild(close);
    return toast;
  }

  function showToast(text, level, options) {
    text = String(text == null ? '' : text).trim();
    if (!text) return null;
    level = normalizeLevel(level);
    options = options || {};

    var stack = ensureToastStack();
    if (!stack) {
      document.addEventListener('DOMContentLoaded', function () {
        showToast(text, level, options);
      }, { once: true });
      return null;
    }

    var toast = buildToast(text, level, options);
    stack.appendChild(toast);
    return armToast(toast, options.duration);
  }

  function bannerSignature(alerts) {
    return alerts.map(function (alert) {
      return [alert.key || '', normalizeLevel(alert.level || alert.type), alert.text || '', alert.url || ''].join('|');
    }).join('||');
  }

  function storedBannerSignature() {
    try { return window.sessionStorage.getItem(BANNER_STORAGE_KEY) || ''; }
    catch (error) { return ''; }
  }

  function storeBannerSignature(signature) {
    try {
      window.sessionStorage.setItem(BANNER_STORAGE_KEY, signature);
      window.sessionStorage.removeItem('alert-dismissed');
    } catch (error) {}
  }

  function safeLocalUrl(raw) {
    try {
      var parsed = new URL(raw, window.location.origin);
      if (parsed.origin !== window.location.origin) return '';
      return parsed.pathname + parsed.search + parsed.hash;
    } catch (error) {
      return '';
    }
  }

  function renderActionBanner(banner, alerts) {
    if (!banner) return null;
    alerts = Array.isArray(alerts) ? alerts.filter(function (alert) {
      return alert && String(alert.text || '').trim() && safeLocalUrl(alert.url);
    }) : [];

    if (!alerts.length) {
      banner.hidden = true;
      banner.classList.remove('visible');
      return banner;
    }

    var signature = bannerSignature(alerts);
    try {
      if (window.sessionStorage.getItem('alert-dismissed') === '1') storeBannerSignature(signature);
    } catch (error) {}
    if (storedBannerSignature() === signature) return banner;

    var items = banner.querySelector('[data-ui-action-banner-items]');
    if (!items) return banner;
    items.replaceChildren();

    alerts.forEach(function (alert) {
      var level = normalizeLevel(alert.level || alert.type);
      var link = document.createElement('a');
      link.className = 'ui-action-banner__action alert-pill ' + level;
      link.setAttribute('data-level', level);
      link.href = safeLocalUrl(alert.url);
      link.textContent = String(alert.text).trim();
      items.appendChild(link);
    });

    banner.setAttribute('data-ui-action-banner-signature', signature);
    banner.hidden = false;
    banner.classList.add('visible');
    return banner;
  }

  function wireActionBanner(banner) {
    if (!banner || banner.hasAttribute('data-ui-action-banner-wired')) return;
    banner.setAttribute('data-ui-action-banner-wired', 'true');

    var dismiss = banner.querySelector('[data-ui-action-banner-dismiss]');
    if (dismiss) dismiss.addEventListener('click', function () {
      storeBannerSignature(banner.getAttribute('data-ui-action-banner-signature') || 'dismissed');
      banner.classList.remove('visible');
      banner.hidden = true;
    });

    var endpoint = banner.getAttribute('data-alert-endpoint');
    if (!endpoint) return;
    window.fetch(endpoint, { headers: { Accept: 'application/json' }, credentials: 'same-origin' })
      .then(function (response) {
        if (!response.ok) throw new Error('Could not load alerts.');
        return response.json();
      })
      .then(function (payload) { renderActionBanner(banner, payload.alerts); })
      .catch(function () {});
  }

  function initialize() {
    document.querySelectorAll('[data-ui-toast-stack] .ui-toast, [data-ui-toast-stack] .toast-msg').forEach(function (toast) {
      armToast(toast);
    });
    document.querySelectorAll('[data-ui-action-banner]').forEach(wireActionBanner);
  }

  window.showToast = showToast;
  window.showActionBanner = function (alerts, options) {
    options = options || {};
    var banner = options.banner || document.querySelector('[data-ui-action-banner]');
    return renderActionBanner(banner, alerts);
  };
  window.PharmacyNotifications = {
    showToast: showToast,
    showActionBanner: window.showActionBanner,
    hideToast: hideToast,
    normalizeLevel: normalizeLevel
  };

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', initialize);
  else initialize();
})();
