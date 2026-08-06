/* Region detail: keyboard navigation between regions.
 *
 * Every control on this screen is a real link that works without JavaScript; this file
 * only adds arrow-key shortcuts for an operator who is stepping through regions.
 */
(function () {
  "use strict";

  function go(selector) {
    var link = document.getElementById(selector);
    if (link && link.tagName === "A") { window.location.href = link.href; }
  }

  document.addEventListener("keydown", function (event) {
    if (event.altKey || event.ctrlKey || event.metaKey) { return; }
    var tag = (event.target.tagName || "").toLowerCase();
    if (tag === "input" || tag === "select" || tag === "textarea") { return; }

    if (event.key === "ArrowLeft") { go("prev-region"); }
    if (event.key === "ArrowRight") { go("next-region"); }
  });
})();
