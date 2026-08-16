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

  function isMobileLayout() {
    return window.matchMedia('(max-width: 768px)').matches;
  }

  function arrangeMobileTools() {
    var utilityBar = document.querySelector('.mobile-utility-bar');
    var wraps = document.querySelectorAll('.slider-toggles-wrap');
    if (!utilityBar) return;

    wraps.forEach(function (wrap) {
      if (!wrap._uiPlaceholder) {
        wrap._uiPlaceholder = document.createComment('slider tools original position');
        wrap.parentNode.insertBefore(wrap._uiPlaceholder, wrap);
      }

      if (!isMobileLayout()) {
        if (wrap._uiPlaceholder.parentNode) {
          wrap._uiPlaceholder.parentNode.insertBefore(wrap, wrap._uiPlaceholder.nextSibling);
        }
        wrap.classList.remove('ui-mobile-tools', 'ui-no-extra-tools');
        return;
      }

      var extraCount = 0;
      wrap.querySelectorAll('button').forEach(function (button) {
        var label = (button.textContent || '').trim().toUpperCase();
        var duplicate = label === 'SEARCH' || label === 'ORDERING';
        if (duplicate) button.setAttribute('data-mobile-duplicate', 'true');
        else extraCount += 1;
      });
      wrap.classList.add('ui-mobile-tools');
      wrap.classList.toggle('ui-no-extra-tools', extraCount === 0);
      utilityBar.appendChild(wrap);
    });

    document.querySelectorAll('.lp-history-tab, .el-slider-toggle, .sl-slider-toggle').forEach(function (tool) {
      if (tool.closest('.slider-toggles-wrap')) return;
      if (!tool._uiPlaceholder) {
        tool._uiPlaceholder = document.createComment('standalone slider tool original position');
        tool.parentNode.insertBefore(tool._uiPlaceholder, tool);
      }
      if (!isMobileLayout()) {
        if (tool._uiPlaceholder.parentNode) {
          tool._uiPlaceholder.parentNode.insertBefore(tool, tool._uiPlaceholder.nextSibling);
        }
        tool.classList.remove('ui-mobile-standalone-tool');
        return;
      }
      tool.classList.add('ui-mobile-standalone-tool');
      utilityBar.appendChild(tool);
    });
  }

  function revealCurrentNavigation() {
    var mobile = isMobileLayout();
    var workflow = document.querySelector('.workflow-nav');
    if (workflow) {
      var dashboard = workflow.querySelector('.workflow-dashboard-link');
      var label = workflow.querySelector('.workflow-nav-label');
      var active = workflow.querySelector('a.active');
      if (dashboard && label) {
        workflow.style.setProperty('--workflow-dashboard-offset', (dashboard.offsetWidth + 8) + 'px');
        label.setAttribute('title', (label.textContent || '').trim());
      }
      if (mobile && dashboard && label && active) {
        var reserved = dashboard.offsetWidth + label.offsetWidth + 44;
        workflow.scrollLeft = Math.max(0, active.offsetLeft - reserved);
      } else if (!mobile) {
        workflow.scrollLeft = 0;
      }
    }

    var navContent = document.querySelector('nav .nav-content');
    var current = navContent && navContent.querySelector('.nav-links > li.active');
    if (navContent && current) {
      if (mobile && navContent.scrollWidth > navContent.clientWidth) {
        var navRect = navContent.getBoundingClientRect();
        var currentRect = current.getBoundingClientRect();
        navContent.scrollLeft += currentRect.left - navRect.left
          - (navContent.clientWidth - currentRect.width) / 2;
      } else if (!mobile) {
        navContent.scrollLeft = 0;
      }
    }
  }

  function refreshResponsiveLayout() {
    arrangeMobileTools();
    window.requestAnimationFrame(revealCurrentNavigation);
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

  function responseNotice(page) {
    var toast = page && page.querySelector('.toast-msg, .os-msg');
    if (!toast) return null;
    var text = toast.querySelector('.toast-text');
    var level = toast.getAttribute('data-level') || 'info';
    if (toast.classList.contains('os-msg-success')) level = 'success';
    else if (toast.classList.contains('os-msg-warning')) level = 'warning';
    else if (toast.classList.contains('os-msg-error')) level = 'error';
    return {
      text: (text ? text.textContent : toast.textContent || '').trim(),
      level: level
    };
  }

  function showSeamlessToast(message, level) {
    if (!message) return;
    if (window.showToast) {
      window.showToast(message, level || 'info');
      return;
    }
    try {
      if (window.parent && window.parent !== window && window.parent.showToast) {
        window.parent.showToast(message, level || 'info');
      }
    } catch (error) {}
  }

  function selectorList(value) {
    return String(value || '').split(',').map(function (selector) {
      return selector.trim();
    }).filter(Boolean);
  }

  /* Copy only the requested page regions from a freshly rendered response.
     Forms still work as ordinary Django posts without JavaScript; pages opt in
     with data-seamless when a small action should not replace the whole view. */
  function refreshRegions(page, selectors) {
    var refreshed = [];
    selectorList(selectors).forEach(function (selector) {
      var currentNodes;
      var nextNodes;
      try {
        currentNodes = document.querySelectorAll(selector);
        nextNodes = page.querySelectorAll(selector);
      } catch (error) {
        return;
      }
      var count = Math.min(currentNodes.length, nextNodes.length);
      for (var index = 0; index < count; index += 1) {
        var current = currentNodes[index];
        var next = nextNodes[index];
        if (current.matches('script, style, textarea')) current.textContent = next.textContent;
        else current.innerHTML = next.innerHTML;
        refreshed.push(current);
      }
    });
    if (refreshed.length) window.requestAnimationFrame(auditControlContrast);
    return refreshed;
  }

  function responseFormFor(form, page) {
    if (form.id) {
      try { return page.getElementById(form.id); } catch (error) {}
    }
    var action = form.querySelector('input[name="action"]');
    if (!action) return null;
    return Array.prototype.find.call(page.querySelectorAll('form'), function (candidate) {
      var candidateAction = candidate.querySelector('input[name="action"]');
      return candidateAction && candidateAction.value === action.value;
    }) || null;
  }

  function applyResponseValidation(form, page) {
    var nextForm = responseFormFor(form, page);
    if (!nextForm) return false;
    var marked = false;
    nextForm.querySelectorAll('[aria-invalid="true"], .is-invalid').forEach(function (nextField) {
      if (!nextField.name) return;
      var field = Array.prototype.find.call(form.elements || [], function (candidate) {
        return candidate.name === nextField.name && candidate.type !== 'hidden';
      });
      if (field) {
        markInvalid(field);
        marked = true;
      }
    });
    nextForm.querySelectorAll('.errorlist[id$="_error"], [data-field-error]').forEach(function (error) {
      var nextField = error.id ? page.getElementById(error.id.replace(/_error$/, '')) : null;
      if (nextField && !nextForm.contains(nextField)) nextField = null;
      if (!nextField) {
        var scope = error.closest('.form-group, .field, .np-field, .edit-field, .detail-box') || error.parentElement;
        nextField = scope && scope.querySelector('input:not([type="hidden"]), select, textarea');
      }
      if (!nextField || !nextField.name) return;
      var field = Array.prototype.find.call(form.elements || [], function (candidate) {
        return candidate.name === nextField.name && candidate.type !== 'hidden';
      });
      if (field) {
        markInvalid(field);
        marked = true;
      }
    });
    if (marked) {
      form.classList.add('ui-validation-attempted');
      var first = form.querySelector('.ui-invalid');
      if (first) {
        first.scrollIntoView({ behavior: 'smooth', block: 'center' });
        try { first.focus({ preventScroll: true }); } catch (error) { first.focus(); }
      }
    }
    return marked;
  }

  function dispatchSeamless(name, detail) {
    document.dispatchEvent(new CustomEvent(name, { detail: detail }));
  }

  function setSeamlessBusy(form, submitter, busy) {
    form.classList.toggle('is-seamless-saving', busy);
    form.setAttribute('aria-busy', busy ? 'true' : 'false');
    if (!busy) form.removeAttribute('aria-busy');
    if (!submitter) return;
    submitter.classList.toggle('is-submitting', busy);
    if (busy) submitter.setAttribute('aria-busy', 'true');
    else submitter.removeAttribute('aria-busy');
  }

  function submitSeamlessly(form, submitter) {
    if (!form || form.dataset.seamlessSaving === 'true') return Promise.resolve(null);
    if (typeof form.checkValidity === 'function' && !form.checkValidity()) {
      form.classList.add('ui-validation-attempted');
      form.reportValidity();
      return Promise.resolve(null);
    }

    var payload = new FormData(form);
    if (submitter && submitter.name) payload.append(submitter.name, submitter.value);
    var selectors = form.getAttribute('data-seamless-refresh') || '';
    var removeAfterSave = form.getAttribute('data-seamless-remove') === 'true';
    form.dataset.seamlessSaving = 'true';
    setSeamlessBusy(form, submitter, true);

    return fetch(form.action || window.location.href, {
      method: (form.method || 'POST').toUpperCase(),
      body: payload,
      credentials: 'same-origin',
      headers: { 'X-Requested-With': 'XMLHttpRequest' }
    }).then(function (response) {
      return response.text().then(function (html) {
        return { response: response, page: new DOMParser().parseFromString(html, 'text/html') };
      });
    }).then(function (result) {
      var notice = responseNotice(result.page);
      var rejected = !result.response.ok || !result.response.redirected ||
        (notice && notice.level === 'error');
      if (rejected) {
        applyResponseValidation(form, result.page);
        var errorMessage = notice && notice.text ? notice.text : 'That change could not be saved.';
        showSeamlessToast(errorMessage, 'error');
        dispatchSeamless('ui:seamless-error', {
          form: form, responseDocument: result.page, notice: notice, response: result.response
        });
        return result.page;
      }

      var refreshed = refreshRegions(result.page, selectors);
      if (selectors && !refreshed.length && result.response.redirected) {
        window.location.assign(result.response.url);
        return result.page;
      }
      if (form.getAttribute('data-seamless-reset') === 'true' && form.isConnected) form.reset();
      var focusSelector = form.getAttribute('data-seamless-focus');
      if (focusSelector) {
        var focusTarget = document.querySelector(focusSelector);
        if (focusTarget) focusTarget.focus();
      }
      var message = notice && notice.text ? notice.text : form.getAttribute('data-seamless-success');
      showSeamlessToast(message, notice ? notice.level : 'success');
      dispatchSeamless('ui:seamless-updated', {
        form: form,
        responseDocument: result.page,
        notice: notice,
        response: result.response,
        selectors: selectors,
        refreshed: refreshed
      });
      return result.page;
    }).catch(function (error) {
      var message = error && error.message ? error.message : 'That change could not be saved.';
      showSeamlessToast(message, 'error');
      dispatchSeamless('ui:seamless-error', { form: form, error: error });
      return null;
    }).finally(function () {
      delete form.dataset.seamlessSaving;
      setSeamlessBusy(form, submitter, false);
      if (removeAfterSave && form.isConnected) form.remove();
    });
  }

  function wireSeamlessForms() {
    document.addEventListener('submit', function (event) {
      var form = event.target;
      if (!form || !form.matches('form[data-seamless]') || event.defaultPrevented) return;
      event.preventDefault();
      submitSeamlessly(form, event.submitter || null);
    });
  }

  window.uiSeamlessSubmit = submitSeamlessly;
  window.uiSeamlessRefresh = function (selectors, url) {
    return fetch(url || window.location.href, {
      credentials: 'same-origin',
      headers: { 'X-Requested-With': 'XMLHttpRequest' }
    }).then(function (response) {
      if (!response.ok) throw new Error('The updated page could not be loaded.');
      return response.text().then(function (html) {
        return { response: response, page: new DOMParser().parseFromString(html, 'text/html') };
      });
    }).then(function (result) {
      var refreshed = refreshRegions(result.page, selectors);
      dispatchSeamless('ui:seamless-updated', {
        form: null,
        responseDocument: result.page,
        response: result.response,
        selectors: selectors,
        refreshed: refreshed
      });
      return result.page;
    });
  };

  function readJsonConfig(id, fallback) {
    var node = document.getElementById(id);
    if (!node) return fallback;
    try { return JSON.parse(node.textContent || 'null') || fallback; }
    catch (error) { return fallback; }
  }

  function dialogShell(title, options) {
    options = options || {};
    var previousFocus = document.activeElement;
    var overlay = document.createElement('div');
    overlay.className = 'ui-dialog-backdrop active';
    overlay.setAttribute('role', 'presentation');

    var dialog = document.createElement('section');
    dialog.className = 'ui-dialog' + (options.danger ? ' is-danger' : '');
    dialog.setAttribute('role', options.alert ? 'alertdialog' : 'dialog');
    dialog.setAttribute('aria-modal', 'true');
    var headingId = 'ui-dialog-title-' + Date.now();
    dialog.setAttribute('aria-labelledby', headingId);

    var header = document.createElement('header');
    var heading = document.createElement('h2');
    heading.id = headingId;
    heading.textContent = title;
    var closeButton = document.createElement('button');
    closeButton.type = 'button';
    closeButton.className = 'ui-dialog-close';
    closeButton.setAttribute('aria-label', 'Close');
    closeButton.textContent = '×';
    header.appendChild(heading);
    header.appendChild(closeButton);

    var body = document.createElement('div');
    body.className = 'ui-dialog-body';
    var footer = document.createElement('footer');
    footer.className = 'ui-dialog-actions';
    dialog.appendChild(header);
    dialog.appendChild(body);
    dialog.appendChild(footer);
    overlay.appendChild(dialog);
    document.body.appendChild(overlay);

    var closed = false;
    function close(reason) {
      if (closed) return;
      closed = true;
      overlay.remove();
      document.body.classList.remove('ui-dialog-open');
      document.removeEventListener('keydown', onKeyDown, true);
      if (previousFocus && previousFocus.focus) previousFocus.focus();
      if (typeof options.onClose === 'function') options.onClose(reason || 'dismiss');
    }
    function onKeyDown(event) {
      if (event.key === 'Escape') {
        event.preventDefault();
        close('cancel');
        return;
      }
      if (event.key !== 'Tab') return;
      var focusable = dialog.querySelectorAll('button:not([disabled]), a[href], input:not([disabled]), select:not([disabled]), textarea:not([disabled])');
      if (!focusable.length) return;
      var first = focusable[0];
      var last = focusable[focusable.length - 1];
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault(); last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault(); first.focus();
      }
    }
    closeButton.addEventListener('click', function () { close('cancel'); });
    overlay.addEventListener('mousedown', function (event) {
      if (event.target === overlay) close('cancel');
    });
    document.addEventListener('keydown', onKeyDown, true);
    document.body.classList.add('ui-dialog-open');
    window.requestAnimationFrame(function () { closeButton.focus(); });
    return { overlay: overlay, dialog: dialog, body: body, footer: footer, close: close };
  }

  function confirmAction(options) {
    if (typeof options === 'string') options = { message: options };
    options = options || {};
    return new Promise(function (resolve) {
      var settled = false;
      function finish(value) {
        if (settled) return;
        settled = true;
        resolve(value);
      }
      var shell = dialogShell(options.title || 'Confirm this action', {
        danger: options.tone !== 'neutral',
        alert: true,
        onClose: function () { finish(false); }
      });
      var message = document.createElement('p');
      message.className = 'ui-confirm-message';
      message.textContent = options.message || 'Do you want to continue?';
      shell.body.appendChild(message);
      if (options.detail) {
        var detail = document.createElement('p');
        detail.className = 'ui-confirm-detail';
        detail.textContent = options.detail;
        shell.body.appendChild(detail);
      }
      var cancel = document.createElement('button');
      cancel.type = 'button';
      cancel.className = 'ui-dialog-button secondary';
      cancel.textContent = options.cancelLabel || 'Cancel';
      var accept = document.createElement('button');
      accept.type = 'button';
      accept.className = 'ui-dialog-button ' + (options.tone === 'neutral' ? 'primary' : 'danger');
      accept.textContent = options.confirmLabel || 'Continue';
      cancel.addEventListener('click', function () { shell.close('cancel'); });
      accept.addEventListener('click', function () {
        finish(true);
        shell.close('accepted');
      });
      shell.footer.appendChild(cancel);
      shell.footer.appendChild(accept);
      window.requestAnimationFrame(function () { cancel.focus(); });
    });
  }

  function wireConfirmations() {
    document.addEventListener('submit', function (event) {
      var form = event.target;
      if (!form || !form.matches('form[data-confirm]')) return;
      if (form.dataset.uiConfirmed === 'true') {
        delete form.dataset.uiConfirmed;
        return;
      }
      event.preventDefault();
      event.stopImmediatePropagation();
      var submitter = event.submitter || null;
      confirmAction({
        title: form.getAttribute('data-confirm-title') || 'Confirm this action',
        message: form.getAttribute('data-confirm') || 'Do you want to continue?',
        confirmLabel: form.getAttribute('data-confirm-button') || 'Continue',
        tone: form.getAttribute('data-confirm-tone') || 'danger'
      }).then(function (accepted) {
        if (!accepted) return;
        form.dataset.uiConfirmed = 'true';
        if (typeof form.requestSubmit === 'function') form.requestSubmit(submitter);
        else form.submit();
      });
    }, true);

    document.addEventListener('click', function (event) {
      var link = event.target.closest && event.target.closest('a[data-confirm]');
      if (!link || event.defaultPrevented) return;
      event.preventDefault();
      confirmAction({
        title: link.getAttribute('data-confirm-title') || 'Confirm this action',
        message: link.getAttribute('data-confirm'),
        confirmLabel: link.getAttribute('data-confirm-button') || 'Continue',
        tone: link.getAttribute('data-confirm-tone') || 'danger'
      }).then(function (accepted) { if (accepted) window.location.assign(link.href); });
    });
  }

  window.uiConfirm = confirmAction;

  function wireAccessIndicators() {
    var canAdminister = document.body.dataset.canAdminister === 'true';
    function decorate(root) {
      var controls = [];
      if (root.matches && root.matches('[data-requires-admin]')) controls.push(root);
      if (root.querySelectorAll) {
        Array.prototype.push.apply(controls, root.querySelectorAll('[data-requires-admin]'));
      }
      controls.forEach(function (control) {
        var visual = control.matches('form')
          ? (control.querySelector('button[type="submit"], input[type="submit"]') || control)
          : control;
        if (visual.querySelector && visual.querySelector('.ui-access-marker')) return;
        if (canAdminister) {
          // The account chip already identifies a staff/admin session. Repeating
          // an Admin badge on every available action adds noise without adding
          // permission information.
          visual.classList.add('ui-admin-available');
          return;
        }
        visual.classList.add('ui-admin-locked');
        var marker = document.createElement('span');
        marker.className = 'ui-access-marker';
        marker.textContent = '🔒';
        marker.setAttribute('aria-label', 'Admin password required');
        if (visual.matches && visual.matches('input')) visual.insertAdjacentElement('afterend', marker);
        else visual.appendChild(marker);
        var title = visual.getAttribute('title');
        visual.setAttribute('title', (title ? title + ' — ' : '') + 'Admin password required');
      });
    }
    decorate(document);
    new MutationObserver(function (records) {
      records.forEach(function (record) {
        Array.prototype.forEach.call(record.addedNodes, function (node) {
          if (node.nodeType === 1) decorate(node);
        });
      });
    }).observe(document.body, { childList: true, subtree: true });

    var countdown = document.querySelector('[data-access-expires]');
    if (!countdown) return;
    var label = countdown.querySelector('.nav-label');
    var expires = Number(countdown.getAttribute('data-access-expires')) * 1000;
    function updateCountdown() {
      var seconds = Math.max(0, Math.ceil((expires - Date.now()) / 1000));
      if (!seconds) {
        if (label) label.textContent = 'Admin unlock expired';
        countdown.classList.add('is-expired');
        return;
      }
      var minutes = Math.ceil(seconds / 60);
      if (label) label.textContent = 'Admin unlocked · ' + minutes + 'm';
      window.setTimeout(updateCountdown, Math.min(60000, seconds * 1000));
    }
    updateCountdown();
  }

  function openShortcuts() {
    var shell = dialogShell('Keyboard shortcuts');
    var intro = document.createElement('p');
    intro.textContent = 'Hold Alt and press a key from any page.';
    shell.body.appendChild(intro);
    var shortcuts = [
      ['Alt + P', 'Purchase page'],
      ['Alt + O', 'Checkout sessions'],
      ['Alt + C', 'Check-in'],
      ['Alt + S', 'Product search'],
      ['Alt + D', 'Delivery'],
      ['Alt + X', 'Dashboard']
    ];
    var list = document.createElement('dl');
    list.className = 'ui-shortcut-list';
    shortcuts.forEach(function (item) {
      var term = document.createElement('dt');
      var key = document.createElement('kbd');
      key.textContent = item[0];
      term.appendChild(key);
      var description = document.createElement('dd');
      description.textContent = item[1];
      list.appendChild(term);
      list.appendChild(description);
    });
    shell.body.appendChild(list);
    var done = document.createElement('button');
    done.type = 'button';
    done.className = 'ui-dialog-button primary';
    done.textContent = 'Done';
    done.addEventListener('click', function () { shell.close('done'); });
    shell.footer.appendChild(done);
  }

  function openWorkflowGuide() {
    var guide = readJsonConfig('ui-workflow-help', null);
    if (!guide) return;
    var shell = dialogShell(guide.title || 'Page guide');
    var summary = document.createElement('p');
    summary.className = 'ui-guide-summary';
    summary.textContent = guide.summary || '';
    shell.body.appendChild(summary);
    if (Array.isArray(guide.steps) && guide.steps.length) {
      var stepsTitle = document.createElement('h3');
      stepsTitle.textContent = 'Recommended workflow';
      shell.body.appendChild(stepsTitle);
      var steps = document.createElement('ol');
      steps.className = 'ui-guide-steps';
      guide.steps.forEach(function (text) {
        var item = document.createElement('li');
        item.textContent = text;
        steps.appendChild(item);
      });
      shell.body.appendChild(steps);
    }
    if (guide.tip) {
      var tip = document.createElement('p');
      tip.className = 'ui-guide-tip';
      tip.textContent = guide.tip;
      shell.body.appendChild(tip);
    }
    var done = document.createElement('button');
    done.type = 'button';
    done.className = 'ui-dialog-button primary';
    done.textContent = 'Got it';
    done.addEventListener('click', function () { shell.close('done'); });
    shell.footer.appendChild(done);
  }

  function wireHelpButtons() {
    document.addEventListener('click', function (event) {
      var shortcut = event.target.closest && event.target.closest('[data-ui-open-shortcuts]');
      if (shortcut) { event.preventDefault(); openShortcuts(); return; }
      var guide = event.target.closest && event.target.closest('[data-ui-open-guide]');
      if (guide) { event.preventDefault(); openWorkflowGuide(); }
    });
  }

  function csrfToken() {
    var match = document.cookie.match(/(?:^|;\s*)csrftoken=([^;]+)/);
    return match ? decodeURIComponent(match[1]) : '';
  }

  function tableColumns(table) {
    if (!table.tHead || !table.tHead.rows.length) return [];
    var row = table.tHead.rows[table.tHead.rows.length - 1];
    var used = {};
    return Array.prototype.map.call(row.cells, function (cell, index) {
      var label = (cell.getAttribute('data-column-label') || cell.textContent || '').replace(/[▲▼↕]/g, '').trim() || ('Column ' + (index + 1));
      var base = cell.getAttribute('data-column-key') || label.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '') || ('column-' + (index + 1));
      used[base] = (used[base] || 0) + 1;
      return { index: index, key: used[base] > 1 ? base + '-' + used[base] : base, label: label };
    });
  }

  function applyTablePreference(table, columns, preference) {
    var hidden = Array.isArray(preference.hidden_columns) ? preference.hidden_columns : [];
    table.classList.toggle('ui-table-compact', preference.density === 'compact');
    columns.forEach(function (column) {
      var isHidden = hidden.indexOf(column.key) !== -1;
      Array.prototype.forEach.call(table.rows, function (row) {
        var cell = row.cells[column.index];
        if (!cell) return;
        cell.classList.toggle('ui-column-hidden', isHidden);
        cell.setAttribute('aria-hidden', isHidden ? 'true' : 'false');
        if (!isHidden) cell.removeAttribute('aria-hidden');
      });
    });
    if (typeof table._uiOverflowUpdate === 'function') {
      window.requestAnimationFrame(table._uiOverflowUpdate);
    }
  }

  function updateTableSummary(table, preference) {
    var summary = table._uiTableToolbar && table._uiTableToolbar.querySelector('.ui-table-view-summary');
    if (!summary) return;
    var hiddenCount = Array.isArray(preference.hidden_columns) ? preference.hidden_columns.length : 0;
    summary.textContent = (preference.density === 'compact' ? 'Compact' : 'Comfortable') +
      (hiddenCount ? ' · ' + hiddenCount + ' column' + (hiddenCount === 1 ? '' : 's') + ' hidden' : ' · all columns');
  }

  function saveTablePreference(payload) {
    return fetch(document.body.dataset.tablePreferenceUrl, {
      method: 'POST',
      credentials: 'same-origin',
      headers: { 'Content-Type': 'application/json', 'X-CSRFToken': csrfToken() },
      body: JSON.stringify(payload)
    }).then(function (response) {
      return response.json().then(function (data) {
        if (!response.ok || !data.ok) throw new Error(data.error || 'Table settings could not be saved.');
        return data;
      });
    });
  }

  function openTablePreferences(table, columns, preference) {
    var tableKey = table.getAttribute('data-table-key') || 'main';
    var pageKey = document.body.dataset.page || 'unknown';
    var defaultSize = Number(table.getAttribute('data-default-page-size')) || 50;
    var sizes = String(table.getAttribute('data-page-sizes') || '').split(',').map(Number).filter(Boolean);
    var shell = dialogShell('Personalize table');
    var form = document.createElement('form');
    form.className = 'ui-table-settings';

    var densityField = document.createElement('fieldset');
    var densityLegend = document.createElement('legend');
    densityLegend.textContent = 'Row spacing';
    densityField.appendChild(densityLegend);
    [['comfortable', 'Comfortable'], ['compact', 'Compact']].forEach(function (choice) {
      var label = document.createElement('label');
      var input = document.createElement('input');
      input.type = 'radio'; input.name = 'density'; input.value = choice[0];
      input.checked = (preference.density || 'comfortable') === choice[0];
      label.appendChild(input); label.appendChild(document.createTextNode(' ' + choice[1]));
      densityField.appendChild(label);
    });
    form.appendChild(densityField);

    if (sizes.length) {
      var sizeLabel = document.createElement('label');
      sizeLabel.className = 'ui-table-size-label';
      sizeLabel.appendChild(document.createTextNode('Rows per page'));
      var select = document.createElement('select');
      select.name = 'page_size';
      sizes.forEach(function (size) {
        var option = document.createElement('option');
        option.value = String(size); option.textContent = size + ' rows';
        option.selected = Number(preference.page_size || defaultSize) === size;
        select.appendChild(option);
      });
      sizeLabel.appendChild(select);
      form.appendChild(sizeLabel);
    }

    var columnField = document.createElement('fieldset');
    var columnLegend = document.createElement('legend');
    columnLegend.textContent = 'Visible columns';
    columnField.appendChild(columnLegend);
    var grid = document.createElement('div');
    grid.className = 'ui-column-choice-grid';
    columns.forEach(function (column) {
      var label = document.createElement('label');
      var input = document.createElement('input');
      input.type = 'checkbox'; input.value = column.key;
      input.checked = (preference.hidden_columns || []).indexOf(column.key) === -1;
      label.appendChild(input); label.appendChild(document.createTextNode(' ' + column.label));
      grid.appendChild(label);
    });
    columnField.appendChild(grid);
    form.appendChild(columnField);
    shell.body.appendChild(form);

    var reset = document.createElement('button');
    reset.type = 'button'; reset.className = 'ui-dialog-button secondary'; reset.textContent = 'Reset';
    var cancel = document.createElement('button');
    cancel.type = 'button'; cancel.className = 'ui-dialog-button secondary'; cancel.textContent = 'Cancel';
    var save = document.createElement('button');
    save.type = 'submit'; save.className = 'ui-dialog-button primary'; save.textContent = 'Save view';
    shell.footer.appendChild(reset); shell.footer.appendChild(cancel); shell.footer.appendChild(save);
    cancel.addEventListener('click', function () { shell.close('cancel'); });

    form.addEventListener('submit', function (event) {
      event.preventDefault();
      var visible = Array.prototype.filter.call(grid.querySelectorAll('input[type="checkbox"]'), function (input) { return input.checked; });
      if (!visible.length) { showSeamlessToast('Keep at least one table column visible.', 'warning'); return; }
      var hidden = Array.prototype.filter.call(grid.querySelectorAll('input[type="checkbox"]'), function (input) { return !input.checked; }).map(function (input) { return input.value; });
      var density = form.elements.density.value;
      var pageSize = form.elements.page_size ? Number(form.elements.page_size.value) : Number(preference.page_size || defaultSize);
      var reload = sizes.length && Number(preference.page_size || defaultSize) !== pageSize;
      save.disabled = true;
      saveTablePreference({ page_key: pageKey, table_key: tableKey, density: density, page_size: pageSize, hidden_columns: hidden })
        .then(function (data) {
          preference = data.preference;
          table._uiTablePreference = preference;
          applyTablePreference(table, columns, preference);
          updateTableSummary(table, preference);
          showSeamlessToast('Table view saved for your account.', 'success');
          shell.close('saved');
          if (reload) {
            var url = new URL(window.location.href);
            url.searchParams.delete(table.getAttribute('data-page-param') || 'page');
            window.location.assign(url.toString());
          }
        }).catch(function (error) { showSeamlessToast(error.message, 'error'); save.disabled = false; });
    });

    reset.addEventListener('click', function () {
      reset.disabled = true;
      saveTablePreference({ page_key: pageKey, table_key: tableKey, reset: true })
        .then(function () {
          var reload = sizes.length && Number(preference.page_size || defaultSize) !== defaultSize;
          preference = { density: 'comfortable', page_size: defaultSize, hidden_columns: [] };
          table._uiTablePreference = preference;
          applyTablePreference(table, columns, preference);
          updateTableSummary(table, preference);
          showSeamlessToast('Table view reset.', 'success');
          shell.close('reset');
          if (reload) {
            var url = new URL(window.location.href); url.searchParams.delete(table.getAttribute('data-page-param') || 'page'); window.location.assign(url.toString());
          }
        }).catch(function (error) { showSeamlessToast(error.message, 'error'); reset.disabled = false; });
    });
  }

  function initializePersonalizedTable(table, savedPreferences) {
    if (table.dataset.uiPersonalized === 'true') return;
    var columns = tableColumns(table);
    if (!columns.length) return;
    table.dataset.uiPersonalized = 'true';
    var key = table.getAttribute('data-table-key') || 'main';
    var preference = savedPreferences[key] || {
      density: 'comfortable',
      page_size: Number(table.getAttribute('data-default-page-size')) || 50,
      hidden_columns: []
    };
    table._uiTablePreference = preference;
    applyTablePreference(table, columns, preference);

    var toolbar = document.createElement('div');
    toolbar.className = 'ui-table-view-toolbar';
    var summary = document.createElement('span');
    summary.className = 'ui-table-view-summary';
    var button = document.createElement('button');
    button.type = 'button'; button.className = 'ui-table-view-button'; button.textContent = 'Table view';
    button.setAttribute('aria-label', 'Personalize this table');
    toolbar.appendChild(summary); toolbar.appendChild(button);
    table._uiTableToolbar = toolbar;
    var wrapper = table.closest('.table-responsive, [class*="table-wrap"], [class*="table-container"]');
    var anchor = wrapper || table;
    anchor.parentNode.insertBefore(toolbar, anchor);
    updateTableSummary(table, preference);
    button.addEventListener('click', function () { openTablePreferences(table, columns, table._uiTablePreference); });
  }

  function wireTablePersonalization() {
    var saved = readJsonConfig('ui-table-preferences', {});
    function scan() {
      document.querySelectorAll('table[data-personalize-table]').forEach(function (table) {
        initializePersonalizedTable(table, saved);
      });
    }
    scan();
    document.addEventListener('ui:seamless-updated', scan);
    var scanQueued = false;
    new MutationObserver(function (records) {
      var addedTable = records.some(function (record) {
        return Array.prototype.some.call(record.addedNodes, function (node) {
          return node.nodeType === 1 && (
            node.matches('table[data-personalize-table]')
            || node.querySelector('table[data-personalize-table]')
          );
        });
      });
      if (!addedTable) return;
      if (scanQueued) return;
      scanQueued = true;
      window.requestAnimationFrame(function () { scanQueued = false; scan(); });
    }).observe(document.body, { childList: true, subtree: true });
  }

  function wireTableOverflowScrollers() {
    var activeScrollers = [];
    var refreshQueued = false;

    function findScrollContainer(table) {
      var node = table;
      while (node && node !== document.body) {
        var overflowX = window.getComputedStyle(node).overflowX;
        if (overflowX === 'auto' || overflowX === 'scroll') return node;
        node = node.parentElement;
      }
      return null;
    }

    function initialize(table) {
      var scroller = findScrollContainer(table);
      if (!scroller) return;
      if (scroller._uiTopScrollUpdate) {
        table._uiOverflowUpdate = scroller._uiTopScrollUpdate;
        scroller._uiTopScrollUpdate();
        return;
      }

      var topScroll = document.createElement('div');
      topScroll.className = 'ui-table-top-scroll';
      topScroll.setAttribute('role', 'region');
      topScroll.setAttribute('aria-label', 'Horizontal table scroll');
      topScroll.tabIndex = -1;
      topScroll.hidden = true;
      var spacer = document.createElement('div');
      spacer.className = 'ui-table-top-scroll-spacer';
      topScroll.appendChild(spacer);
      scroller.parentNode.insertBefore(topScroll, scroller);

      var syncing = false;
      function update() {
        if (!scroller.isConnected) return;
        var hasOverflow = scroller.scrollWidth > scroller.clientWidth + 1;
        spacer.style.width = scroller.scrollWidth + 'px';
        topScroll.hidden = !hasOverflow;
        topScroll.tabIndex = hasOverflow ? 0 : -1;
        if (hasOverflow && Math.abs(topScroll.scrollLeft - scroller.scrollLeft) > 1) {
          topScroll.scrollLeft = scroller.scrollLeft;
        }
      }
      function releaseSync() {
        window.requestAnimationFrame(function () { syncing = false; });
      }
      topScroll.addEventListener('scroll', function () {
        if (syncing) return;
        syncing = true;
        scroller.scrollLeft = topScroll.scrollLeft;
        releaseSync();
      });
      scroller.addEventListener('scroll', function () {
        if (syncing) return;
        syncing = true;
        topScroll.scrollLeft = scroller.scrollLeft;
        releaseSync();
      });

      scroller._uiTopScrollUpdate = update;
      table._uiOverflowUpdate = update;
      activeScrollers.push(scroller);
      if (window.ResizeObserver) {
        var observer = new ResizeObserver(update);
        observer.observe(scroller);
        observer.observe(table);
        scroller._uiTopScrollObserver = observer;
      }
      update();
    }

    function scan() {
      document.querySelectorAll('table').forEach(initialize);
      activeScrollers = activeScrollers.filter(function (scroller) {
        if (!scroller.isConnected) return false;
        scroller._uiTopScrollUpdate();
        return true;
      });
    }
    function queueScan() {
      if (refreshQueued) return;
      refreshQueued = true;
      window.requestAnimationFrame(function () {
        refreshQueued = false;
        scan();
      });
    }

    scan();
    document.addEventListener('ui:seamless-updated', queueScan);
    window.addEventListener('resize', queueScan);
    new MutationObserver(queueScan).observe(document.body, { childList: true, subtree: true });
  }

  function ready() {
    document.body.classList.add('ui-ready');
    composeWorkflowHeader();
    refreshResponsiveLayout();
    auditControlContrast();
    wireValidation();
    wireConfirmations();
    wireAccessIndicators();
    wireHelpButtons();
    wireTablePersonalization();
    wireTableOverflowScrollers();
    wireSeamlessForms();

    var resizeFrame = null;
    window.addEventListener('resize', function () {
      if (resizeFrame) window.cancelAnimationFrame(resizeFrame);
      resizeFrame = window.requestAnimationFrame(function () {
        resizeFrame = null;
        refreshResponsiveLayout();
      });
    });

    /* Give users immediate feedback after a valid form is submitted. We do not
       disable or rename the submitter because its name/value may be required. */
    document.addEventListener('submit', function (event) {
      var form = event.target;
      if (!form || event.defaultPrevented || form.hasAttribute('data-no-submit-state') || form.hasAttribute('data-seamless')) return;
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
