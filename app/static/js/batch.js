/* Batch screen: start a run and poll its progress.
 *
 * Totals are never accumulated in the browser. Every number shown comes from
 * /api/batches/{id}, which computes it from the same rows the export produces.
 */
(function () {
  "use strict";

  var form = document.getElementById("batch-form");
  var errorBox = document.getElementById("batch-error");
  var dryRunButton = document.getElementById("dry-run");
  var startButton = document.getElementById("start-run");

  function showError(message) {
    if (!errorBox) { return; }
    errorBox.textContent = message;
    errorBox.hidden = false;
  }

  function payload(dryRun) {
    return {
      source_folder: (document.getElementById("source_folder") || {}).value || "",
      material: (document.getElementById("material") || {}).value || "",
      product_prefix: (document.getElementById("product_prefix") || {}).value || "",
      dry_run: !!dryRun
    };
  }

  function start(dryRun) {
    if (errorBox) { errorBox.hidden = true; }
    if (startButton) { startButton.disabled = true; }

    fetch("/api/batches", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload(dryRun))
    })
      .then(function (response) {
        return response.json().then(function (body) {
          if (!response.ok) { throw new Error(body.detail || body.error || "The run could not be started."); }
          return body;
        });
      })
      .then(function (body) {
        window.location.href = "/batch?batch_run_id=" + body.batch_run_id;
      })
      .catch(function (error) {
        showError(error.message);
        if (startButton) { startButton.disabled = false; }
      });
  }

  if (form) {
    form.addEventListener("submit", function (event) { event.preventDefault(); start(false); });
  }
  if (dryRunButton) {
    dryRunButton.addEventListener("click", function () { start(true); });
  }

  // Poll while a run is still going.
  var section = document.getElementById("progress-section");
  if (!section) { return; }
  var params = new URLSearchParams(window.location.search);
  var runId = params.get("batch_run_id");
  if (!runId) { return; }

  var timer = window.setInterval(function () {
    fetch("/api/batches/" + runId)
      .then(function (response) { return response.ok ? response.json() : null; })
      .then(function (body) {
        if (!body) { return; }
        var progress = body.progress || {};
        var total = progress.total || body.run.image_count || 0;
        var processed = progress.processed || body.totals.processed || 0;
        var fill = document.getElementById("progress-fill");
        if (fill && total) { fill.style.width = Math.round(processed / total * 100) + "%"; }

        var label = document.getElementById("progress-label");
        if (label) {
          label.textContent = processed + " / " + total + " images \u00b7 run " + runId
            + " \u00b7 " + (progress.status || body.run.status)
            + (progress.eta_seconds ? " \u00b7 est. " + progress.eta_seconds + " s remaining" : "");
        }

        var cards = {
          "card-processed": body.totals.processed,
          "card-regions": body.totals.regions_found,
          "card-clean": body.totals.clean,
          "card-failed": body.totals.failed,
          "card-region-total": body.totals.regions_total
        };
        Object.keys(cards).forEach(function (id) {
          var el = document.getElementById(id);
          if (el) { el.textContent = Number(cards[id]).toLocaleString(); }
        });

        if (progress.status && progress.status !== "running") {
          window.clearInterval(timer);
          window.location.reload();
        }
      })
      .catch(function () { window.clearInterval(timer); });
  }, 1200);
})();
