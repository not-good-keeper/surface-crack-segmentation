/* Status: re-run the checks, and drive the mock-mode fault simulation control. */
(function () {
  "use strict";

  var recheck = document.getElementById("recheck-form");
  if (recheck) {
    recheck.addEventListener("submit", function (event) {
      event.preventDefault();
      var button = document.getElementById("run-checks");
      if (button) { button.disabled = true; }
      fetch("/api/status/check", { method: "POST" })
        .then(function () { window.location.reload(); })
        .catch(function () { if (button) { button.disabled = false; } });
    });
  }

  var simulate = document.getElementById("simulate-form");
  if (simulate) {
    simulate.addEventListener("submit", function (event) {
      event.preventDefault();
      var value = (document.getElementById("simulate") || {}).value || "none";
      fetch("/api/status/check?simulate=" + encodeURIComponent(value), { method: "POST" })
        .then(function () { window.location.reload(); })
        .catch(function () { });
    });
  }
})();
