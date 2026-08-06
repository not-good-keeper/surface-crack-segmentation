/* History: keep the export links pointing at the filter currently in the form. */
(function () {
  "use strict";

  var form = document.getElementById("history-filters");
  if (!form) { return; }

  form.addEventListener("submit", function () {
    // Drop empty fields so the query string stays readable and shareable.
    Array.prototype.forEach.call(form.elements, function (element) {
      if (element.name && !element.value) { element.disabled = true; }
    });
  });
})();
