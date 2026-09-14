// Ordinary links support keyboard navigation and opening records in new tabs.
var historyTypes = document.querySelector('.history-types');
var selectedHistoryType = historyTypes && historyTypes.querySelector('[aria-current="page"]');
if (selectedHistoryType && historyTypes.scrollWidth > historyTypes.clientWidth) {
  var typeBounds = selectedHistoryType.getBoundingClientRect();
  var navBounds = historyTypes.getBoundingClientRect();
  if (typeBounds.right > navBounds.right) historyTypes.scrollLeft += typeBounds.right - navBounds.right + 8;
  else if (typeBounds.left < navBounds.left) historyTypes.scrollLeft -= navBounds.left - typeBounds.left + 8;
}
document.addEventListener('click', function (event) {
  if (event.defaultPrevented || event.button !== 0 || event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) return;
  var row = event.target.closest('[data-history-row]');
  if (!row || event.target.closest('a, button, input, select, textarea, label')) return;
  if (window.getSelection && window.getSelection().toString()) return;
  var link = row.querySelector('[data-history-detail]');
  if (link) window.location.assign(link.href);
});
