/**
 * InsightML Report — Interactive enhancements
 *
 * Used as external reference; html_renderer.py embeds a minified version inline.
 */

(function () {
  "use strict";

  // -------------------------------------------------------------------------
  // Sidebar active-link tracking (highlight the section currently in view)
  // -------------------------------------------------------------------------
  function initActiveLinks() {
    var sections = document.querySelectorAll("section[id]");
    var links = document.querySelectorAll(".sidebar a");

    if (!sections.length || !links.length) return;

    var observer = new IntersectionObserver(
      function (entries) {
        entries.forEach(function (entry) {
          if (entry.isIntersecting) {
            links.forEach(function (link) {
              link.classList.remove("active");
              if (link.getAttribute("href") === "#" + entry.target.id) {
                link.classList.add("active");
              }
            });
          }
        });
      },
      { rootMargin: "0px 0px -60% 0px", threshold: 0 }
    );

    sections.forEach(function (sec) {
      observer.observe(sec);
    });
  }

  // -------------------------------------------------------------------------
  // Smooth scroll for anchor links
  // -------------------------------------------------------------------------
  function initSmoothScroll() {
    document.querySelectorAll('a[href^="#"]').forEach(function (anchor) {
      anchor.addEventListener("click", function (e) {
        var target = document.querySelector(this.getAttribute("href"));
        if (target) {
          e.preventDefault();
          // Open <details> if target is inside one
          var details = target.closest("details");
          if (details && !details.open) {
            details.open = true;
          }
          target.scrollIntoView({ behavior: "smooth", block: "start" });
        }
      });
    });
  }

  // -------------------------------------------------------------------------
  // Expand all / collapse all buttons (injected dynamically)
  // -------------------------------------------------------------------------
  function initToggleAll() {
    var header = document.querySelector(".report-header");
    if (!header) return;

    var btn = document.createElement("button");
    btn.textContent = "Collapse all";
    btn.style.cssText =
      "margin-top:0.75rem;padding:0.3em 0.8em;border-radius:4px;" +
      "border:1px solid rgba(255,255,255,0.4);background:rgba(255,255,255,0.15);" +
      "color:white;cursor:pointer;font-size:0.8rem;";

    var expanded = true;
    btn.addEventListener("click", function () {
      expanded = !expanded;
      document.querySelectorAll("details").forEach(function (d) {
        d.open = expanded;
      });
      btn.textContent = expanded ? "Collapse all" : "Expand all";
    });

    header.appendChild(btn);
  }

  // -------------------------------------------------------------------------
  // Plotly chart resize on window resize (Plotly doesn't auto-resize in flex)
  // -------------------------------------------------------------------------
  function initPlotlyResize() {
    var resizeTimer;
    window.addEventListener("resize", function () {
      clearTimeout(resizeTimer);
      resizeTimer = setTimeout(function () {
        document.querySelectorAll(".plotly-graph-div").forEach(function (div) {
          if (window.Plotly) {
            window.Plotly.Plots.resize(div);
          }
        });
      }, 250);
    });
  }

  // -------------------------------------------------------------------------
  // Boot
  // -------------------------------------------------------------------------
  document.addEventListener("DOMContentLoaded", function () {
    initActiveLinks();
    initSmoothScroll();
    initToggleAll();
    initPlotlyResize();
  });
})();
