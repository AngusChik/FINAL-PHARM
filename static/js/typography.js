(function () {
  'use strict';

  function size(base) {
    var scale = parseFloat(getComputedStyle(document.documentElement)
      .getPropertyValue('--ui-type-scale')) || 1;
    return base * scale;
  }

  window.pharmacyTypography = {
    size: size,
    applyChartDefaults: function (Chart) {
      Chart.defaults.font.size = size(12);
    }
  };
})();
