(function () {
  'use strict';

  const PALETTE = [
    '#4f46e5','#059669','#f59e0b','#0891b2','#7c3aed',
    '#dc2626','#16a34a','#d97706','#6366f1','#f43f5e',
    '#0284c7','#7d1d3f','#065f46','#92400e','#1e40af',
  ];

  function safeJSON(id) {
    const el = document.getElementById(id);
    if (!el) return [];
    try { return JSON.parse(el.textContent); } catch (e) { return []; }
  }

  function dollarTick(v) { return '$' + Number(v).toLocaleString('en-AU', {minimumFractionDigits:0, maximumFractionDigits:0}); }
  function dollarLabel(ctx) { return ctx.dataset.label + ': $' + Number(ctx.raw).toFixed(2); }

  const revenueData  = safeJSON('revenue-series-data');
  const productData  = safeJSON('top-products-data');
  const categoryData = safeJSON('category-data');

  const page = document.querySelector('.sa-page');
  page.querySelectorAll('[data-category-color]').forEach(dot => {
    dot.style.backgroundColor = PALETTE[Number(dot.dataset.categoryColor) % PALETTE.length];
  });
  if (typeof Chart === 'undefined') {
    page.querySelectorAll('canvas').forEach(canvas => {
      canvas.hidden = true;
      const message = document.createElement('p');
      message.className = 'sa-panel-note';
      message.textContent = 'Charts could not load. Use Top Products, Categories, or Period Breakdown for the detailed figures.';
      canvas.parentElement.appendChild(message);
    });
    return;
  }

  Chart.defaults.font.family = "'Inter', system-ui, sans-serif";
  window.pharmacyTypography.applyChartDefaults(Chart);

  function drawRevenue() {
    /* ── Chart 1: Revenue & Profit Over Time ── */
    const revenueCtx = document.getElementById('revenueChart');
    if (revenueCtx && revenueData.length) {
      new Chart(revenueCtx, {
        data: {
          labels: revenueData.map(d => d.label),
          datasets: [
            {
              type: 'bar',
              label: 'Revenue',
              data: revenueData.map(d => d.revenue),
              backgroundColor: 'rgba(79,70,229,0.75)',
              borderRadius: 4,
              order: 2,
            },
            {
              type: 'bar',
              label: 'Cost',
              data: revenueData.map(d => d.cost),
              backgroundColor: 'rgba(245,158,11,0.65)',
              borderRadius: 4,
              order: 3,
            },
            {
              type: 'line',
              label: 'Profit',
              data: revenueData.map(d => d.profit),
              borderColor: '#059669',
              backgroundColor: 'rgba(5,150,105,0.08)',
              fill: true,
              tension: 0.35,
              pointRadius: 4,
              pointBackgroundColor: '#059669',
              borderWidth: 2.5,
              order: 1,
            },
          ],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          interaction: { mode: 'index', intersect: false },
          plugins: {
            legend: { position: 'top', labels: { usePointStyle: true, padding: 16, font: { size: window.pharmacyTypography.size(13.5) } } },
            tooltip: {
              callbacks: {
                label: dollarLabel,
                afterBody: function(items) {
                  const d = revenueData[items[0].dataIndex];
                  return d.orders > 0 ? ['', `📦 ${d.orders} order${d.orders !== 1 ? 's' : ''}`] : [];
                },
              }
            }
          },
          scales: {
            y: { ticks: { callback: dollarTick }, grid: { color: 'rgba(0,0,0,0.04)' } },
            x: { grid: { display: false } },
          },
        },
      });
    }
  }

  function drawProducts() {
    /* ── Chart 2: Top Products (horizontal bar) ── */
    const productCtx = document.getElementById('topProductsChart');
    if (productCtx && productData.length) {
      const truncName = n => n.length > 32 ? n.slice(0, 30) + '…' : n;
      new Chart(productCtx, {
        type: 'bar',
        data: {
          labels: productData.map(d => truncName(d.name)),
          datasets: [{
            label: 'Revenue',
            data: productData.map(d => d.revenue),
            backgroundColor: productData.map((_, i) => PALETTE[i % PALETTE.length] + 'cc'),
            borderRadius: 4,
          }],
        },
        options: {
          indexAxis: 'y',
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: { display: false },
            tooltip: {
              callbacks: {
                label: ctx => {
                  const d = productData[ctx.dataIndex];
                  const lines = [`Revenue: $${d.revenue.toFixed(2)}`, `Units: ${d.units}`];
                  if (d.cost > 0) lines.push(`Profit: ${d.profit >= 0 ? '+' : ''}$${d.profit.toFixed(2)}`);
                  return lines;
                }
              }
            }
          },
          scales: {
            x: { ticks: { callback: dollarTick }, grid: { color: 'rgba(0,0,0,0.04)' } },
            y: { grid: { display: false }, ticks: { font: { size: window.pharmacyTypography.size(12.5) } } },
          },
        },
      });
    }
  }

  function drawCategories() {
    /* ── Chart 3: Category Donut ── */
    const donutCtx = document.getElementById('categoryDonutChart');
    if (donutCtx && categoryData.length) {
      const totalRev = categoryData.reduce((s, d) => s + d.revenue, 0);
      new Chart(donutCtx, {
        type: 'doughnut',
        data: {
          labels: categoryData.map(d => d.name),
          datasets: [{
            data: categoryData.map(d => d.revenue),
            backgroundColor: categoryData.map((_, i) => PALETTE[i % PALETTE.length]),
            borderWidth: 2,
            borderColor: '#ffffff',
            hoverOffset: 6,
          }],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: { display: false },
            tooltip: {
              callbacks: {
                label: ctx => {
                  const pct = totalRev > 0 ? (ctx.raw / totalRev * 100).toFixed(1) : 0;
                  return `${ctx.label}: $${ctx.raw.toFixed(2)} (${pct}%)`;
                }
              }
            }
          },
          cutout: '60%',
        },
      });

    }
  }

  function drawProfit() {
    /* ── Chart 4: Margins by Category ── */
    const marginsCtx = document.getElementById('marginsChart');
    if (marginsCtx && categoryData.length && categoryData.some(d => d.cost > 0)) {
      new Chart(marginsCtx, {
        type: 'bar',
        data: {
          labels: categoryData.map(d => d.name),
          datasets: [
            {
              label: 'Revenue',
              data: categoryData.map(d => d.revenue),
              backgroundColor: 'rgba(79,70,229,0.7)',
              borderRadius: 3,
            },
            {
              label: 'Cost',
              data: categoryData.map(d => d.cost),
              backgroundColor: 'rgba(245,158,11,0.65)',
              borderRadius: 3,
            },
            {
              label: 'Profit',
              data: categoryData.map(d => d.profit),
              backgroundColor: categoryData.map(d => d.profit >= 0 ? 'rgba(5,150,105,0.8)' : 'rgba(220,38,38,0.8)'),
              borderRadius: 3,
            },
          ],
        },
        options: {
          indexAxis: 'y',
          responsive: true,
          maintainAspectRatio: false,
          interaction: { mode: 'index', intersect: false },
          plugins: {
            legend: { position: 'top', labels: { usePointStyle: true, padding: 16 } },
            tooltip: { callbacks: { label: dollarLabel } },
          },
          scales: {
            x: { ticks: { callback: dollarTick }, grid: { color: 'rgba(0,0,0,0.04)' } },
            y: { grid: { display: false } },
          },
        },
      });
    }
  }

  const charts = {
    overview: ['revenueChart', drawRevenue],
    products: ['topProductsChart', drawProducts],
    categories: ['categoryDonutChart', drawCategories],
    profit: ['marginsChart', drawProfit],
  };
  function showChart(key) {
    const entry = charts[key];
    if (!entry) return;
    const canvas = document.getElementById(entry[0]);
    if (!canvas) return;
    const chart = Chart.getChart(canvas);
    if (chart) chart.resize();
    else entry[1]();
  }
  function showSelectedChart() {
    showChart(document.getElementById('sa-active-tab').value);
  }
  page.addEventListener('sales:tabchange', showSelectedChart);
  showSelectedChart();
  window.addEventListener('beforeprint', () => Object.keys(charts).forEach(showChart));
  window.addEventListener('afterprint', showSelectedChart);
})();
