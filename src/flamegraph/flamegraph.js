// zoom/hover controller injected into kvlm flamegraph SVGs by
// flamegraph::ToSVG. Written from scratch (not taken from
// flamegraph.pl). Geometry contract with the renderer: the svg root
// carries data-drawx/data-draww, every frame group carries
// data-x/data-w (fractions of the whole), data-n (full name),
// data-s (samples), data-p (percent); ~7 px per character matches
// truncateString in mod.rs.
var kfg = (function () {
  var frames, details, reset, W, X0, Y0, H, VERT, CHW = 7;

  function label(g) {
    return g.getAttribute('data-n') + ' (' + g.getAttribute('data-s') +
      ' samples, ' + g.getAttribute('data-p') + '%)';
  }

  function zoom(x0, w0) {
    var eps = 1e-9;
    frames.forEach(function (g) {
      var x = +g.getAttribute('data-x'), w = +g.getAttribute('data-w');
      var rect = g.querySelector('rect'), text = g.querySelector('text');
      var nx, nw;
      if (x >= x0 - eps && x + w <= x0 + w0 + eps) {
        // inside the zoomed subtree: rescale
        nx = (x - x0) / w0; nw = w / w0;
      } else if (x <= x0 + eps && x + w >= x0 + w0 - eps) {
        // ancestor of the zoomed frame: stretch to full width
        nx = 0; nw = 1;
      } else {
        g.style.display = 'none';
        return;
      }
      g.style.display = 'block';
      if (VERT) {
        // vertical partition: the stacked axis is y/height; the
        // column (x/width) never changes
        var py = Y0 + nx * H, ph = Math.max(nw * H, 0.3);
        rect.setAttribute('y', py.toFixed(1));
        rect.setAttribute('height', ph.toFixed(1));
        if (text) {
          if (ph >= 16) {
            text.setAttribute('y', (py + 14).toFixed(1));
            text.style.display = 'block';
          } else {
            text.style.display = 'none';
          }
        }
        return;
      }
      var px = X0 + nx * W, pw = Math.max(nw * W, 0.2);
      rect.setAttribute('x', px.toFixed(1));
      rect.setAttribute('width', pw.toFixed(1));
      if (text) {
        text.setAttribute('x', (px + 3).toFixed(1));
        if (pw > 40) {
          var name = g.getAttribute('data-n');
          var max = Math.floor((pw - 6) / CHW);
          text.textContent = name.length <= max ? name :
            (max <= 3 ? name.slice(0, max) : name.slice(0, max - 3) + '...');
          text.style.display = 'block';
        } else {
          text.style.display = 'none';
        }
      }
    });
    if (reset) reset.style.display = (w0 >= 1 - eps) ? 'none' : 'block';
  }

  function init() {
    var svg = document.documentElement;
    W = +svg.getAttribute('data-draww');
    X0 = +svg.getAttribute('data-drawx');
    Y0 = +svg.getAttribute('data-drawy');
    H = +svg.getAttribute('data-drawh');
    VERT = svg.getAttribute('data-orient') === 'v';
    frames = Array.prototype.slice.call(document.querySelectorAll('g.f'));
    details = document.getElementById('kfg-details');
    reset = document.getElementById('kfg-reset');
    frames.forEach(function (g) {
      g.addEventListener('click', function () {
        zoom(+g.getAttribute('data-x'), +g.getAttribute('data-w'));
      });
      g.addEventListener('mouseover', function () {
        if (details) details.textContent = label(g);
      });
      g.addEventListener('mouseout', function () {
        if (details) details.textContent = ' ';
      });
    });
    if (reset) reset.addEventListener('click', function () { zoom(0, 1); });
  }

  return { init: init };
})();
kfg.init();
