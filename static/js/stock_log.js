/* Shared Stock Movement Log slider — used by the dashboard, the check-in
   dashboard and the check-in session page. Self-contained (no dependency on
   page-specific helpers). The endpoint is read from the panel's
   data-stocklog-url attribute. */
(function () {
  var panel = document.getElementById('slSliderPanel');
  if (!panel) return;

  var url = panel.getAttribute('data-stocklog-url') || '/stock-log/api/';
  var toggle = document.getElementById('slSliderToggle');
  var overlay = document.getElementById('slSliderOverlay');
  var closeBtn = document.getElementById('slSliderClose');
  var body = document.getElementById('slSliderBody');
  var pagination = document.getElementById('slPagination');
  var prevBtn = document.getElementById('slPrevBtn');
  var nextBtn = document.getElementById('slNextBtn');
  var pageInfo = document.getElementById('slPageInfo');
  var filterBtn = document.getElementById('slFilterBtn');
  var clearBtn = document.getElementById('slClearBtn');
  var exportBtn = document.getElementById('slExportBtn');
  var currentPage = 1;
  var loaded = false;
  var storeKey = 'sl_panel_open:' + window.location.pathname;
  var lockedScrollY = 0;

  function esc(s) {
    var d = document.createElement('div');
    d.textContent = s == null ? '' : s;
    return d.innerHTML;
  }
  function val(id) { var el = document.getElementById(id); return el ? el.value : ''; }

  function lockBody() {
    lockedScrollY = window.scrollY;
    document.body.style.position = 'fixed';
    document.body.style.top = '-' + lockedScrollY + 'px';
    document.body.style.left = '0';
    document.body.style.right = '0';
    document.body.style.overflowY = 'scroll';
  }
  function unlockBody() {
    document.body.style.position = '';
    document.body.style.top = '';
    document.body.style.left = '';
    document.body.style.right = '';
    document.body.style.overflowY = '';
    window.scrollTo(0, lockedScrollY);
  }

  function openPanel() {
    lockBody();
    panel.classList.add('open');
    if (overlay) overlay.classList.add('open');
    sessionStorage.setItem(storeKey, 'true');
    document.dispatchEvent(new CustomEvent('stocklog:open'));
    if (!loaded) { fetchLog(1); loaded = true; }
  }
  function closePanel() {
    unlockBody();
    panel.classList.remove('open');
    if (overlay) overlay.classList.remove('open');
    sessionStorage.removeItem(storeKey);
    document.dispatchEvent(new CustomEvent('stocklog:close'));
  }

  function buildQuery(page) {
    var params = new URLSearchParams();
    params.set('format', 'json');
    var product = val('slFilterProduct').trim();
    var type = val('slFilterType');
    var dateFrom = val('slFilterDateFrom');
    var dateTo = val('slFilterDateTo');
    if (product) params.set('log_product', product);
    if (type) params.set('log_type', type);
    if (dateFrom) params.set('log_date_from', dateFrom);
    if (dateTo) params.set('log_date_to', dateTo);
    if (page && page > 1) params.set('log_page', page);
    return params.toString();
  }

  function setKpi(data) {
    var c = document.getElementById('slKpiCheckins');
    var s = document.getElementById('slKpiSales');
    var a = document.getElementById('slKpiAdjust');
    if (c) c.textContent = data.kpi.checkins;
    if (s) s.textContent = data.kpi.sales;
    if (a) a.textContent = data.kpi.adjustments;
  }

  function fetchLog(page) {
    page = page || 1;
    currentPage = page;
    body.innerHTML = '<div class="sl-empty">Loading...</div>';
    fetch(url + '?' + buildQuery(page), { headers: { 'X-Requested-With': 'XMLHttpRequest' } })
      .then(function (r) { return r.json(); })
      .then(function (data) {
        if (data.kpi) setKpi(data);
        if (!data.entries || data.entries.length === 0) {
          body.innerHTML = '<div class="sl-empty">No stock changes found.</div>';
          if (pagination) pagination.style.display = 'none';
          return;
        }
        var html = '<div style="overflow-x:auto;"><table class="sl-table"><thead><tr>' +
          '<th>Time</th><th>Product</th><th>Barcode</th><th>Action</th><th style="text-align:right">Qty</th><th>Note</th>' +
          '</tr></thead><tbody>';
        data.entries.forEach(function (e) {
          var qtyHtml = e.positive
            ? '<span style="color:var(--success-text,#166534);">+' + e.qty + '</span>'
            : '<span style="color:var(--danger-text,#991b1b);">-' + e.qty + '</span>';
          html += '<tr>' +
            '<td style="white-space:nowrap;color:var(--text-secondary,#64748b);font-size:12px;">' + esc(e.time) + '</td>' +
            '<td><strong>' + esc(e.name) + '</strong></td>' +
            '<td style="color:var(--text-secondary,#64748b);font-family:monospace;font-size:12px;">' + esc(e.barcode) + '</td>' +
            '<td><span class="sl-badge ' + esc(e.badge_cls) + '">' + esc(e.action) + '</span></td>' +
            '<td style="text-align:right;font-weight:700;">' + qtyHtml + '</td>' +
            '<td style="color:var(--text-secondary,#64748b);font-size:12px;max-width:160px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">' + esc(e.note) + '</td>' +
            '</tr>';
        });
        html += '</tbody></table></div>';
        body.innerHTML = html;
        if (pagination) {
          if (data.num_pages > 1) {
            pagination.style.display = 'flex';
            if (pageInfo) pageInfo.textContent = 'Page ' + data.page + ' of ' + data.num_pages;
            if (prevBtn) { prevBtn.disabled = !data.has_prev; prevBtn.classList.toggle('disabled', !data.has_prev); }
            if (nextBtn) { nextBtn.disabled = !data.has_next; nextBtn.classList.toggle('disabled', !data.has_next); }
          } else {
            pagination.style.display = 'none';
          }
        }
      })
      .catch(function () {
        body.innerHTML = '<div class="sl-empty">Error loading data. Please try again.</div>';
      });
  }

  if (toggle) toggle.addEventListener('click', openPanel);
  if (closeBtn) closeBtn.addEventListener('click', closePanel);
  if (overlay) overlay.addEventListener('click', closePanel);
  document.addEventListener('click', function (e) {
    if (!panel.classList.contains('open')) return;
    if (panel.contains(e.target)) return;
    if (toggle && (e.target === toggle || toggle.contains(e.target))) return;
    closePanel();
  });
  if (prevBtn) prevBtn.addEventListener('click', function () { if (currentPage > 1) fetchLog(currentPage - 1); });
  if (nextBtn) nextBtn.addEventListener('click', function () { fetchLog(currentPage + 1); });
  if (filterBtn) filterBtn.addEventListener('click', function () { fetchLog(1); });
  if (clearBtn) clearBtn.addEventListener('click', function () {
    ['slFilterProduct', 'slFilterType', 'slFilterDateFrom', 'slFilterDateTo'].forEach(function (id) {
      var el = document.getElementById(id); if (el) el.value = '';
    });
    fetchLog(1);
  });
  if (exportBtn) exportBtn.addEventListener('click', function () {
    fetch(url + '?' + buildQuery(1) + '&export=csv', { headers: { 'X-Requested-With': 'XMLHttpRequest' } })
      .then(function (r) { return r.blob(); })
      .then(function (blob) {
        var u = URL.createObjectURL(blob);
        var a = document.createElement('a');
        a.href = u; a.download = 'stock_log.csv';
        document.body.appendChild(a); a.click(); document.body.removeChild(a);
        URL.revokeObjectURL(u);
      });
  });
  ['slFilterProduct', 'slFilterType', 'slFilterDateFrom', 'slFilterDateTo'].forEach(function (id) {
    var el = document.getElementById(id);
    if (el) el.addEventListener('keydown', function (e) {
      if (e.key === 'Enter') { e.preventDefault(); fetchLog(1); }
    });
  });
  if (sessionStorage.getItem(storeKey) === 'true') openPanel();
})();
