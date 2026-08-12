/* Shared, progressive UI behavior. Business field names and URLs stay intact. */
(function () {
  'use strict';

  var headerSelector = [
    '.al-header', '.as-header', '.cd-header', '.dr-header', '.dv-header',
    '.edit-header', '.exs-header', '.home-header', '.il-header', '.lp-header',
    '.ls-header', '.newprod-header', '.oos-header', '.order-header', '.os-header',
    '.page-header', '.ps-header', '.rc-header', '.rp-header', '.sa-header', '.sd-header',
    '.trend-header'
  ].join(',');

  function pathOf(link) {
    try { return new URL(link.href, window.location.href).pathname.replace(/\/+$/, '') || '/'; }
    catch (error) { return ''; }
  }

  function composeWorkflowHeader() {
    var nav = document.querySelector('.container > .workflow-nav');
    if (!nav) return;

    var page = nav.nextElementSibling;
    var header = null;
    while (page && !header) {
      header = page.matches(headerSelector) ? page : page.querySelector(headerSelector);
      if (!header) page = page.nextElementSibling;
    }
    if (!header || header.closest('.workflow-header-stack')) return;

    var dashboard = nav.querySelector('.workflow-dashboard-link');
    if (dashboard) {
      var dashboardPath = pathOf(dashboard);
      header.querySelectorAll('a[href]').forEach(function (link) {
        if (pathOf(link) === dashboardPath) link.remove();
      });
    }

    var stack = document.createElement('div');
    stack.className = 'workflow-header-stack';
    header.parentNode.insertBefore(stack, header);
    stack.appendChild(nav);
    stack.appendChild(header);
  }

  function rgba(value) {
    var match = String(value || '').match(/rgba?\(([^)]+)\)/i);
    if (!match) return null;
    var parts = match[1].split(/[,\s/]+/).filter(Boolean).map(Number);
    if (parts.length < 3 || parts.slice(0, 3).some(function (part) { return Number.isNaN(part); })) return null;
    return {
      r: Math.max(0, Math.min(255, parts[0])),
      g: Math.max(0, Math.min(255, parts[1])),
      b: Math.max(0, Math.min(255, parts[2])),
      a: parts.length > 3 && !Number.isNaN(parts[3]) ? Math.max(0, Math.min(1, parts[3])) : 1
    };
  }

  function blend(top, bottom) {
    var alpha = top.a + bottom.a * (1 - top.a);
    if (!alpha) return { r: 255, g: 255, b: 255, a: 1 };
    return {
      r: (top.r * top.a + bottom.r * bottom.a * (1 - top.a)) / alpha,
      g: (top.g * top.a + bottom.g * bottom.a * (1 - top.a)) / alpha,
      b: (top.b * top.a + bottom.b * bottom.a * (1 - top.a)) / alpha,
      a: alpha
    };
  }

  function effectiveBackground(element) {
    var layers = [];
    var node = element;
    while (node && node.nodeType === 1) {
      var style = window.getComputedStyle(node);
      /* A gradient or image may intentionally provide the contrast. Avoid
         guessing from background-color alone in that case. */
      if (style.backgroundImage && style.backgroundImage !== 'none') return null;
      var layer = rgba(style.backgroundColor);
      if (layer && layer.a > 0) layers.push(layer);
      node = node.parentElement;
    }
    var result = { r: 255, g: 255, b: 255, a: 1 };
    layers.reverse().forEach(function (layer) { result = blend(layer, result); });
    return result;
  }

  function luminance(color) {
    function channel(value) {
      value /= 255;
      return value <= 0.03928 ? value / 12.92 : Math.pow((value + 0.055) / 1.055, 2.4);
    }
    return 0.2126 * channel(color.r) + 0.7152 * channel(color.g) + 0.0722 * channel(color.b);
  }

  function contrast(first, second) {
    var light = Math.max(luminance(first), luminance(second));
    var dark = Math.min(luminance(first), luminance(second));
    return (light + 0.05) / (dark + 0.05);
  }

  function auditControlContrast() {
    document.querySelectorAll([
      'body.app-shell a[href]', 'body.app-shell button',
      'body.app-shell input[type="button"]', 'body.app-shell input[type="submit"]'
    ].join(',')).forEach(function (control) {
      if (control.hasAttribute('data-contrast-lock')) return;
      var style = window.getComputedStyle(control);
      if (style.display === 'none' || style.visibility === 'hidden' || control.getClientRects().length === 0) return;
      var background = effectiveBackground(control);
      var foreground = rgba(style.color);
      if (!background || !foreground) return;
      foreground = blend(foreground, background);
      if (contrast(foreground, background) < 3) control.classList.add('ui-contrast-fix');
    });
  }

  function fieldContainer(field) {
    return field.closest([
      '.form-group', '.field', '.np-field', '.edit-field', '.detail-box',
      '.input-group', '.control-box', '[class$="-field"]'
    ].join(','));
  }

  function markInvalid(field) {
    if (!field || !field.matches('input, select, textarea') || field.type === 'hidden') return;
    field.classList.add('ui-invalid');
    field.setAttribute('aria-invalid', 'true');
    var wrapper = fieldContainer(field);
    if (wrapper) wrapper.classList.add('ui-field-invalid');
  }

  function clearInvalid(field) {
    if (!field || !field.classList.contains('ui-invalid')) return;
    if (field.validity && !field.validity.valid) return;
    field.classList.remove('ui-invalid');
    field.removeAttribute('aria-invalid');
    var wrapper = fieldContainer(field);
    if (wrapper && !wrapper.querySelector('.ui-invalid')) wrapper.classList.remove('ui-field-invalid');
  }

  function wireValidation() {
    var pendingFocus = new WeakSet();

    document.addEventListener('invalid', function (event) {
      var field = event.target;
      var form = field.form;
      markInvalid(field);
      if (!form) return;
      form.classList.add('ui-validation-attempted');
      if (pendingFocus.has(form)) return;
      pendingFocus.add(form);
      window.requestAnimationFrame(function () {
        var first = form.querySelector(':invalid:not([type="hidden"])');
        if (first) {
          markInvalid(first);
          first.scrollIntoView({ behavior: 'smooth', block: 'center' });
          try { first.focus({ preventScroll: true }); } catch (error) { first.focus(); }
        }
        pendingFocus.delete(form);
      });
    }, true);

    document.addEventListener('input', function (event) { clearInvalid(event.target); });
    document.addEventListener('change', function (event) { clearInvalid(event.target); });

    /* Django marks bound fields with aria-invalid and gives field error lists an
       id such as id_name_error. Restore the same clear highlight after a
       server-side validation response. */
    document.querySelectorAll('[aria-invalid="true"], .is-invalid').forEach(markInvalid);
    document.querySelectorAll('.errorlist[id$="_error"], [data-field-error]').forEach(function (error) {
      var field = null;
      if (error.id) field = document.getElementById(error.id.replace(/_error$/, ''));
      if (!field) {
        var scope = error.closest('.form-group, .field, .np-field, .edit-field, .detail-box') || error.parentElement;
        if (scope) field = scope.querySelector('input:not([type="hidden"]), select, textarea');
      }
      markInvalid(field);
      if (field && field.form) field.form.classList.add('ui-validation-attempted');
    });

    /* A rejected check-in inline edit must reopen with its bad field visible. */
    var inlineForm = document.getElementById('inlineEditForm');
    if (inlineForm) {
      var externalInvalid = document.querySelector('[form="inlineEditForm"].ui-invalid');
      if (inlineForm.querySelector('.ui-invalid') || externalInvalid) {
        var card = inlineForm.closest('.product-card');
        if (card) card.classList.add('is-editing');
        var actions = document.getElementById('inlineEditActions');
        var editButton = document.getElementById('toggleEditBtn');
        if (actions) actions.style.display = 'flex';
        if (editButton) editButton.style.display = 'none';
      }
    }
  }

  function ready() {
    document.body.classList.add('ui-ready');
    composeWorkflowHeader();
    auditControlContrast();
    wireValidation();

    /* Give users immediate feedback after a valid form is submitted. We do not
       disable or rename the submitter because its name/value may be required. */
    document.addEventListener('submit', function (event) {
      var form = event.target;
      if (!form || form.hasAttribute('data-no-submit-state')) return;
      if (typeof form.checkValidity === 'function' && !form.checkValidity()) return;
      var submitter = event.submitter;
      if (!submitter || submitter.classList.contains('is-submitting')) return;
      submitter.classList.add('is-submitting');
      submitter.setAttribute('aria-busy', 'true');
    });

    document.querySelectorAll('table thead th').forEach(function (cell) {
      if (!cell.hasAttribute('scope')) cell.setAttribute('scope', 'col');
    });

    var mobileSearch = document.getElementById('mobileGlobalSearch');
    var mobileOrdering = document.getElementById('mobileGlobalOrdering');
    if (mobileSearch) mobileSearch.addEventListener('click', function () {
      var source = document.getElementById('psSliderToggle');
      if (source) source.click();
    });
    if (mobileOrdering) mobileOrdering.addEventListener('click', function () {
      var source = document.getElementById('osSliderToggle');
      if (source) source.click();
    });

    document.addEventListener('keydown', function (event) {
      if (event.key !== 'Escape' || event.defaultPrevented) return;
      var candidates = Array.prototype.slice.call(document.querySelectorAll(
        '[class$="-modal-overlay"].active, [class$="-slider-panel"].open, [class$="-history-panel"].open'
      )).filter(function (node) {
        var style = window.getComputedStyle(node);
        return style.display !== 'none' && style.visibility !== 'hidden';
      });
      if (!candidates.length) return;
      var active = candidates[candidates.length - 1];
      var close = active.querySelector('[class$="-close"], [aria-label="Close"], [aria-label="Dismiss"]');
      if (close) close.click();
    });

    /* Recheck after opening a modal/drawer because controls hidden during the
       initial pass have no rendered background to measure. */
    document.addEventListener('click', function () {
      window.setTimeout(auditControlContrast, 0);
    });
  }

  window.addEventListener('pageshow', function () {
    document.querySelectorAll('.is-submitting').forEach(function (button) {
      button.classList.remove('is-submitting');
      button.removeAttribute('aria-busy');
    });
  });

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', ready);
  else ready();
})();
