/* Shared resizer for the right-edge slide-out panels (Search, Ordering, Stock Log,
   Recent Sales, Sales Analytics, Expired Log, ...).

   - Adds a drag grip on each panel's LEFT edge so the user can widen/narrow it.
   - Persists the chosen width per panel (localStorage, keyed by element id).
   - Switches the show/hide from `right` to a `transform`, so a panel that has been
     widened past its original off-screen offset still hides completely when closed.
   - Disabled on phones (panels are full-width there).
   - Double-click the grip to reset to the stylesheet default. */
(function () {
  var MIN_W = 360;
  var MOBILE_BP = 768;
  function maxW() { return Math.min(1600, Math.round(window.innerWidth * 0.97)); }

  // CSS lives in base.html to prevent FOUC — no dynamic <style> injection needed.

  var panels = document.querySelectorAll('[class*="-slider-panel"], [class*="-history-panel"]');
  Array.prototype.forEach.call(panels, function (panel) {
    if (getComputedStyle(panel).position !== 'fixed') return;   // only fixed right-edge sliders
    if (panel.hasAttribute('data-panel-resizable')) return;
    panel.classList.add('panel-init');
    panel.setAttribute('data-panel-resizable', '');
    panel.offsetHeight;                                         // force reflow before re-enabling transitions
    panel.classList.remove('panel-init');

    var key = panel.id ? 'panelW:' + panel.id : null;

    function applyWidth(w, save) {
      if (window.innerWidth <= MOBILE_BP) {           // phones: leave the full-width CSS alone
        panel.style.width = ''; panel.style.maxWidth = ''; panel.style.minWidth = '';
        return;
      }
      w = Math.max(MIN_W, Math.min(maxW(), w));
      panel.style.width = w + 'px';
      panel.style.maxWidth = 'none';
      panel.style.minWidth = '0';
      if (save && key) { try { localStorage.setItem(key, String(w)); } catch (e) {} }
    }

    if (key) {
      var saved = parseInt(localStorage.getItem(key), 10);
      if (saved) applyWidth(saved, false);
    }

    var grip = document.createElement('div');
    grip.className = 'panel-resize-grip';
    grip.title = 'Drag to resize · double-click to reset';
    panel.appendChild(grip);

    var dragging = false;
    function pointX(e) { return e.touches ? e.touches[0].clientX : e.clientX; }
    function onMove(e) {
      if (!dragging) return;
      applyWidth(window.innerWidth - pointX(e), false);
      e.preventDefault();
    }
    function onUp() {
      if (!dragging) return;
      dragging = false;
      grip.classList.remove('dragging');
      document.body.classList.remove('panel-resizing');
      if (key) { try { localStorage.setItem(key, String(parseInt(panel.style.width, 10) || MIN_W)); } catch (e) {} }
      document.removeEventListener('mousemove', onMove);
      document.removeEventListener('mouseup', onUp);
      document.removeEventListener('touchmove', onMove);
      document.removeEventListener('touchend', onUp);
    }
    function onDown(e) {
      if (window.innerWidth <= MOBILE_BP) return;
      dragging = true;
      grip.classList.add('dragging');
      document.body.classList.add('panel-resizing');
      document.addEventListener('mousemove', onMove);
      document.addEventListener('mouseup', onUp);
      document.addEventListener('touchmove', onMove, { passive: false });
      document.addEventListener('touchend', onUp);
      e.preventDefault();
    }
    grip.addEventListener('mousedown', onDown);
    grip.addEventListener('touchstart', onDown, { passive: false });

    grip.addEventListener('dblclick', function () {         // reset to stylesheet default
      if (key) { try { localStorage.removeItem(key); } catch (e) {} }
      panel.style.width = ''; panel.style.maxWidth = ''; panel.style.minWidth = '';
    });

    window.addEventListener('resize', function () {
      if (window.innerWidth <= MOBILE_BP) {
        panel.style.width = ''; panel.style.maxWidth = ''; panel.style.minWidth = '';
      } else if (key) {
        var s = parseInt(localStorage.getItem(key), 10);
        if (s) applyWidth(s, false);
      }
    });
  });
})();
