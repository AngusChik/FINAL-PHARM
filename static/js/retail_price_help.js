(function () {
  'use strict';
  var help = document.querySelector('.retail-price-help');
  if (!help) return;
  var button = help.querySelector('.retail-price-help-toggle');
  var panel = help.querySelector('.retail-price-help-panel');
  if (!button || !panel) return;

  function setOpen(open) {
    button.setAttribute('aria-expanded', String(open));
    panel.hidden = !open;
  }
  button.addEventListener('click', function () {
    setOpen(button.getAttribute('aria-expanded') !== 'true');
  });
  document.addEventListener('click', function (event) {
    if (!help.contains(event.target)) setOpen(false);
  });
  document.addEventListener('keydown', function (event) {
    if (event.key === 'Escape' && !panel.hidden) {
      setOpen(false);
      button.focus({ preventScroll: true });
    }
  });
  document.addEventListener('focusin', function (event) {
    if (!help.contains(event.target)) setOpen(false);
  });
})();
