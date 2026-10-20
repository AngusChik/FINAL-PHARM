/* One expiry-date entry format, including dynamically inserted lot rows. */
(function () {
  'use strict';

  var selector = 'input.flatpickr-date, input[data-expiry-date]';
  var drafts = new WeakMap();

  function parseDate(value) {
    if (value instanceof Date) return value;
    var match = /^(\d{2})-(\d{2})-(\d{4})$/.exec(String(value));
    if (!match) return undefined;
    var date = new Date(0);
    date.setFullYear(Number(match[3]), Number(match[2]) - 1, Number(match[1]));
    date.setHours(0, 0, 0, 0);
    return date.getFullYear() === Number(match[3]) &&
      date.getMonth() === Number(match[2]) - 1 && date.getDate() === Number(match[1])
      ? date : undefined;
  }

  function format(value, deleting) {
    var iso = /^(\d{4})-(\d{2})-(\d{2})$/.exec(value.trim());
    if (iso) return iso[3] + '-' + iso[2] + '-' + iso[1];
    var digits = value.replace(/\D/g, '').slice(0, 8);
    var result = digits.slice(0, 2);
    if (digits.length > 2 || (digits.length === 2 && !deleting)) result += '-' + digits.slice(2, 4);
    if (digits.length > 4 || (digits.length === 4 && !deleting)) result += '-' + digits.slice(4, 8);
    return result;
  }

  function validate(input) {
    input.setCustomValidity(!input.value || parseDate(input.value)
      ? '' : 'Enter a valid expiry date as DD-MM-YYYY.');
  }

  function editable(input) {
    return input && input.matches && input.matches(selector) &&
      input.type !== 'hidden' && !input.disabled && !input.readOnly;
  }

  function formatInput(input, event) {
    var value = input.value;
    var caret = input.selectionStart;
    var deleting = event && (event.inputType || '').indexOf('delete') === 0;
    var formatted = format(value, deleting);
    if (formatted !== value) {
      var digitsBefore = value.slice(0, caret).replace(/\D/g, '').length;
      input.value = formatted;
      var position = 0;
      var seen = 0;
      while (position < formatted.length && seen < digitsBefore) {
        if (/\d/.test(formatted[position])) seen += 1;
        position += 1;
      }
      if (!deleting && formatted[position] === '-') position += 1;
      if (caret === value.length) position = formatted.length;
      input.setSelectionRange(position, position);
    }
    drafts.set(input, input.value);
    validate(input);
  }

  function init(input) {
    if (!input || input.type === 'hidden') return;
    input.placeholder = 'DD-MM-YYYY';
    input.inputMode = 'numeric';
    input.maxLength = 10;
    input.autocomplete = 'off';
    if (!input._flatpickr && window.flatpickr) {
      var initial = input.value;
      // Flatpickr normally rolls impossible dates forward or clears partial
      // entries on blur. Keep those entries visible for correction instead.
      if (initial && !parseDate(initial)) input.value = '';
      var picker = window.flatpickr(input, {
        dateFormat: 'd-m-Y', allowInput: true, disableMobile: true,
        monthSelectorType: 'static', parseDate: parseDate,
        onValueUpdate: function (dates, value) {
          var draft = drafts.get(input);
          if (!dates.length && draft && !parseDate(draft)) input.value = draft;
          else drafts.set(input, value);
          validate(input);
        },
        onClose: function () {
          var draft = drafts.get(input);
          if (draft && !parseDate(draft)) input.value = draft;
          validate(input);
          input.dispatchEvent(new Event('change', { bubbles: true }));
        }
      });
      // Reject invalid manual input before Flatpickr can clear it. Calendar
      // selections and callers supplying actual Date objects still work.
      var setDate = picker.setDate;
      picker.setDate = function (value, triggerChange, dateFormat) {
        if (typeof value === 'string' && value && !parseDate(value)) {
          input.value = value;
          drafts.set(input, value);
          validate(input);
          return;
        }
        if (typeof value === 'string') drafts.set(input, value);
        return setDate(value, triggerChange, dateFormat);
      };
      input.value = initial;
    }
    drafts.set(input, input.value);
    validate(input);
  }

  function scan(root) {
    if (root.matches && root.matches(selector)) init(root);
    if (root.querySelectorAll) root.querySelectorAll(selector).forEach(init);
  }

  // Capture runs before page-specific totals, draft saves, or form submits.
  document.addEventListener('input', function (event) {
    if (editable(event.target) && !event.isComposing) formatInput(event.target, event);
  }, true);
  document.addEventListener('compositionend', function (event) {
    if (editable(event.target)) formatInput(event.target, event);
  }, true);
  document.addEventListener('beforeinput', function (event) {
    var input = event.target;
    if (!editable(input) || input.selectionStart !== input.selectionEnd) return;
    var caret = input.selectionStart;
    // Backspace at an inserted dash should delete the preceding digit, so
    // deleting never gets stuck repeatedly removing and reinserting a dash.
    if (event.inputType === 'deleteContentBackward' && caret > 1 && input.value[caret - 1] === '-') {
      input.setSelectionRange(caret - 2, caret);
    } else if (event.inputType === 'deleteContentForward' && input.value[caret] === '-') {
      input.setSelectionRange(caret, caret + 2);
    }
  }, true);
  document.addEventListener('change', function (event) {
    if (editable(event.target)) validate(event.target);
  }, true);
  document.addEventListener('reset', function (event) {
    window.setTimeout(function () {
      event.target.querySelectorAll(selector).forEach(function (input) {
        if (input._flatpickr) input._flatpickr.setDate(input.value, false, 'd-m-Y');
        drafts.set(input, input.value);
        validate(input);
      });
    }, 0);
  });

  window.uiExpiryDates = { init: init };
  function ready() {
    scan(document);
    new MutationObserver(function (records) {
      records.forEach(function (record) { record.addedNodes.forEach(scan); });
    }).observe(document.body, { childList: true, subtree: true });
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', ready);
  else ready();
})();
