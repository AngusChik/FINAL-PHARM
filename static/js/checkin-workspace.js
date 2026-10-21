document.addEventListener('DOMContentLoaded', function () {
  const trends = document.getElementById('checkinTrends');
  if (trends) {
    const key = 'checkin-trends:' + trends.dataset.checkinSessionId;
    try { const saved = sessionStorage.getItem(key); if (saved) trends.open = saved === 'open'; } catch (_) {}
    trends.addEventListener('toggle', function () {
      try { sessionStorage.setItem(key, trends.open ? 'open' : 'closed'); } catch (_) {}
      if (trends.open) window.checkinProductMovementChart?.resize();
    });
  }
  const history = document.getElementById('checkinActivityRail');
  const details = document.getElementById('productSecondaryDetails');
  const product = document.querySelector('.product-card');
  const search = document.getElementById('search-box');
  if (!history || !details || !product) return;
  let frame;
  function alignSessionPanel() {
    cancelAnimationFrame(frame);
    frame = requestAnimationFrame(function () {
      if (window.innerWidth < 1280) {
        history.style.removeProperty('--session-panel-height');
        return;
      }
      const height = details.getBoundingClientRect().top - history.getBoundingClientRect().top;
      history.style.setProperty('--session-panel-height', Math.max(220, height) + 'px');
    });
  }
  const observer = new ResizeObserver(alignSessionPanel);
  observer.observe(product);
  if (search) observer.observe(search);
  window.addEventListener('resize', alignSessionPanel);
  document.fonts.ready.then(alignSessionPanel);
  alignSessionPanel();
});
