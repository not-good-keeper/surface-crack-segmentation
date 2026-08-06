/* Live screen: poll the current result, optionally advance the mock station.
 *
 * Threshold values, class rules and status wording are never computed here. This file
 * renders what /api/live returns and nothing else - the backend owns every decision,
 * so re-tuning a threshold in app/postprocess.py cannot leave the UI out of date.
 */
(function () {
  "use strict";

  var root = document.getElementById("live-root");
  var pollMs = root ? parseInt(root.getAttribute("data-poll-ms"), 10) || 2500 : 2500;
  var current = root ? parseInt(root.getAttribute("data-inspection-id"), 10) : null;

  var nextButton = document.getElementById("next-inspection");
  var autoToggle = document.getElementById("auto-advance");

  function text(id, value) {
    var el = document.getElementById(id);
    if (el) { el.textContent = value; }
  }

  function pxText(value, digits) {
    if (value === null || value === undefined) { return "\u2014"; }
    return Number(value).toLocaleString(undefined, {
      minimumFractionDigits: digits || 0, maximumFractionDigits: digits || 0
    }) + " px";
  }

  function renderBanner(summary) {
    var banner = document.getElementById("result-banner");
    if (!banner || !summary) { return; }
    banner.className = "banner banner--" + summary.state;
    banner.innerHTML = "";
    var headline = document.createElement("p");
    headline.className = "banner__headline";
    headline.textContent = summary.headline;
    banner.appendChild(headline);
    if (summary.detail) {
      var detail = document.createElement("p");
      detail.className = "banner__detail";
      detail.textContent = summary.detail;
      banner.appendChild(detail);
    }
    if (summary.reason) {
      var reason = document.createElement("p");
      reason.className = "banner__reason";
      reason.textContent = summary.reason;
      banner.appendChild(reason);
    }
  }

  function renderRegions(inspection) {
    var body = document.getElementById("region-rows");
    if (!body) { return; }
    body.innerHTML = "";
    inspection.regions.forEach(function (region) {
      var tr = document.createElement("tr");

      var th = document.createElement("th");
      th.setAttribute("scope", "row");
      th.textContent = region.region_index;
      tr.appendChild(th);

      [region.class_code,
       pxText(region.length_px),
       pxText(region.max_width_px),
       Number(region.area_px).toLocaleString()
      ].forEach(function (value, index) {
        var td = document.createElement("td");
        if (index > 0) { td.className = "num"; }
        td.textContent = value;
        tr.appendChild(td);
      });

      var actions = document.createElement("td");
      var link = document.createElement("a");
      link.className = "link-action";
      link.href = "/regions?inspection_id=" + inspection.inspection_id + "&region=" + region.region_index;
      link.textContent = "Open";
      actions.appendChild(link);
      tr.appendChild(actions);

      body.appendChild(tr);
    });
  }

  function render(payload) {
    if (!payload || !payload.inspection) { return; }
    var inspection = payload.inspection;

    if (inspection.inspection_id === current) { return; }
    current = inspection.inspection_id;
    if (root) { root.setAttribute("data-inspection-id", String(current)); }

    renderBanner(inspection.summary);
    renderRegions(inspection);

    var image = document.getElementById("live-overlay");
    if (image && inspection.overlay_image_url) {
      image.src = inspection.overlay_image_url + "?v=" + inspection.inspection_id;
      image.alt = "Inspection overlay for " + (inspection.product_id || "the current part")
        + " at source resolution, defect regions outlined and numbered";
    } else if (image && !inspection.overlay_image_url) {
      // A failure has no overlay: reload so the correct empty state renders server-side.
      window.location.reload();
      return;
    }

    var caption = document.getElementById("live-caption");
    if (caption) {
      caption.firstChild.nodeValue = " Product " + (inspection.product_id || "\u2014")
        + " \u00b7 " + inspection.captured_at
        + " \u00b7 " + (inspection.material || "unknown material")
        + " \u00b7 " + inspection.station + " ";
    }

    var openRegions = document.querySelector('a[href^="/regions?inspection_id"]');
    if (openRegions && inspection.regions.length) {
      openRegions.href = "/regions?inspection_id=" + inspection.inspection_id
        + "&region=" + inspection.regions[0].region_index;
    }
  }

  function refresh() {
    return fetch("/api/live", { headers: { "Accept": "application/json" } })
      .then(function (response) { return response.ok ? response.json() : null; })
      .then(function (payload) {
        if (!payload) { return; }
        // A failing station check changes the whole screen, not just the result.
        if (!payload.station_ok) { window.location.reload(); return; }
        render(payload);
      })
      .catch(function () { /* offline by design: keep the last good result on screen */ });
  }

  function advance() {
    if (nextButton) { nextButton.disabled = true; }
    return fetch("/api/demo/next", { method: "POST" })
      .then(function (response) {
        if (!response.ok) { window.location.reload(); return null; }
        return response.json();
      })
      .then(function (payload) { if (payload) { render(payload); } })
      .catch(function () { })
      .then(function () { if (nextButton) { nextButton.disabled = false; } });
  }

  if (nextButton) {
    nextButton.addEventListener("click", advance);
  }

  window.setInterval(function () {
    if (document.hidden) { return; }
    if (autoToggle && autoToggle.checked) { advance(); } else { refresh(); }
  }, pollMs);
})();
