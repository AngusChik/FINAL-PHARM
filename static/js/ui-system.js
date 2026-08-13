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

  function ready() {
    document.body.classList.add('ui-ready');
    composeWorkflowHeader();
    refreshResponsiveLayout();
    auditControlContrast();
    wireValidation();
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
