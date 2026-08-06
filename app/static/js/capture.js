/* Capture screen: take a frame from the device camera, or pick a photo, and inspect it.
 *
 * Both paths end in the same place - a Blob POSTed to /api/inspections/capture as
 * multipart form data. Nothing about the result is decided here: no threshold, no class
 * rule, no status wording. This file moves bytes and renders what the API returns, so a
 * re-tuned threshold or a changed status cannot leave the screen stating something the
 * backend no longer believes.
 *
 * The camera is deliberately not started on load. getUserMedia triggers a permission
 * prompt, and a page that demands the camera before the operator has asked for it gets
 * denied once and then stays denied for the origin.
 */
(function () {
  "use strict";

  var root = document.getElementById("capture-root");
  if (!root) { return; }

  var maxBytes = parseInt(root.getAttribute("data-max-bytes"), 10) || 16777216;
  var blocked = root.getAttribute("data-blocked") === "true";

  var video = document.getElementById("capture-video");
  var canvas = document.getElementById("capture-canvas");
  var placeholder = document.getElementById("camera-placeholder");
  var startBtn = document.getElementById("camera-start");
  var shootBtn = document.getElementById("camera-shoot");
  var flipBtn = document.getElementById("camera-flip");
  var stopBtn = document.getElementById("camera-stop");
  var fileInput = document.getElementById("capture-file");
  var drop = document.getElementById("capture-drop");
  var statusEl = document.getElementById("capture-status");
  var errorEl = document.getElementById("capture-error");
  var resultEl = document.getElementById("capture-result");

  var stream = null;
  var facing = "environment";   // rear camera on a phone; ignored by most laptops
  var busy = false;

  // -- small helpers ---------------------------------------------------------
  function setStatus(message) {
    if (statusEl) { statusEl.textContent = message || ""; }
  }

  function setError(message) {
    if (!errorEl) { return; }
    if (message) {
      errorEl.textContent = message;
      errorEl.hidden = false;
    } else {
      errorEl.textContent = "";
      errorEl.hidden = true;
    }
  }

  function pxText(value) {
    if (value === null || value === undefined) { return "—"; }
    return Number(value).toLocaleString(undefined, { maximumFractionDigits: 0 }) + " px";
  }

  function numberText(value) {
    if (value === null || value === undefined) { return "—"; }
    return Number(value).toLocaleString(undefined, { maximumFractionDigits: 0 });
  }

  function cameraSupported() {
    return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
  }

  // -- camera ----------------------------------------------------------------
  function stopCamera() {
    if (stream) {
      stream.getTracks().forEach(function (track) { track.stop(); });
      stream = null;
    }
    if (video) {
      video.srcObject = null;
      video.hidden = true;
    }
    if (placeholder) { placeholder.hidden = false; }
    shootBtn.disabled = true;
    stopBtn.disabled = true;
    flipBtn.disabled = true;
    startBtn.disabled = false;
    startBtn.textContent = "Start camera";
  }

  function startCamera() {
    if (!cameraSupported()) {
      setError(
        "This browser exposes no camera API. Camera access needs HTTPS or localhost; " +
        "over plain HTTP on another host the browser removes it entirely. " +
        "Use the photo picker below instead."
      );
      return;
    }
    setError(null);
    setStatus("Requesting camera permission…");
    startBtn.disabled = true;

    navigator.mediaDevices.getUserMedia({
      video: { facingMode: facing, width: { ideal: 1280 }, height: { ideal: 960 } },
      audio: false
    }).then(function (granted) {
      stream = granted;
      video.srcObject = stream;
      video.hidden = false;
      if (placeholder) { placeholder.hidden = true; }
      return video.play();
    }).then(function () {
      setStatus("Camera running.");
      shootBtn.disabled = blocked;
      stopBtn.disabled = false;
      flipBtn.disabled = false;
      startBtn.textContent = "Camera running";
    }).catch(function (err) {
      startBtn.disabled = false;
      setStatus("");
      // The distinction matters: a refused permission is the operator's decision and
      // is recoverable from the browser's site settings; no device at all is not.
      var name = err && err.name ? err.name : "";
      if (name === "NotAllowedError" || name === "SecurityError") {
        setError("Camera permission was refused. Allow it in the browser's site settings, or use the photo picker.");
      } else if (name === "NotFoundError" || name === "OverconstrainedError") {
        setError("No camera matched the request on this device. Use the photo picker instead.");
      } else if (name === "NotReadableError") {
        setError("The camera is in use by another application.");
      } else {
        setError("The camera could not be started: " + (err && err.message ? err.message : name || "unknown error"));
      }
    });
  }

  function flipCamera() {
    facing = facing === "environment" ? "user" : "environment";
    stopCamera();
    startCamera();
  }

  function shoot() {
    if (!stream || !video.videoWidth) {
      setError("The camera has not produced a frame yet.");
      return;
    }
    // Capture at the sensor's own frame size, not the CSS size of the preview. The
    // overlay and every geometry the operator is shown are in source pixels, so
    // sending a frame scaled to the width of a page column would quietly change the
    // units of every measurement on the result.
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height);
    canvas.toBlob(function (blob) {
      if (!blob) {
        setError("The frame could not be encoded.");
        return;
      }
      submit(blob, "frame.png", "camera");
    }, "image/png");
  }

  // -- upload ----------------------------------------------------------------
  function chooseFile(file) {
    if (!file) { return; }
    if (file.size > maxBytes) {
      setError(
        "That file is " + (file.size / 1048576).toFixed(1) + " MB; the limit is " +
        Math.round(maxBytes / 1048576) + " MB."
      );
      return;
    }
    submit(file, file.name, "upload");
  }

  // -- submit ----------------------------------------------------------------
  function submit(blob, filename, source) {
    if (busy) { return; }
    if (blocked) {
      setError("A station check is failing, so inspection is stopped. Open Status to see which.");
      return;
    }
    busy = true;
    setError(null);
    setStatus("Inspecting…");
    shootBtn.disabled = true;

    var form = new FormData();
    form.append("file", blob, filename);
    form.append("material", document.getElementById("capture-material").value);
    form.append("source", source);
    var productId = document.getElementById("capture-product").value.trim();
    if (productId) { form.append("product_id", productId); }

    fetch("/api/inspections/capture", { method: "POST", body: form })
      .then(function (response) {
        return response.json().then(function (payload) {
          if (!response.ok) {
            throw new Error(payload.detail || payload.error || ("Request failed: " + response.status));
          }
          return payload;
        });
      })
      .then(render)
      .catch(function (err) {
        setStatus("");
        setError(err.message || "The capture could not be inspected.");
      })
      .then(function () {
        busy = false;
        shootBtn.disabled = !stream || blocked;
      });
  }

  // -- render ----------------------------------------------------------------
  function renderBanner(summary) {
    var host = document.getElementById("capture-banner");
    host.innerHTML = "";
    if (!summary) { return; }
    var banner = document.createElement("div");
    banner.className = "banner banner--" + summary.state;
    banner.setAttribute("role", "status");
    banner.setAttribute("aria-live", "polite");

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
    host.appendChild(banner);
  }

  function renderRegions(regions) {
    var body = document.getElementById("capture-region-rows");
    var empty = document.getElementById("capture-no-regions");
    body.innerHTML = "";
    var list = regions || [];
    empty.hidden = list.length > 0;

    list.forEach(function (region) {
      var row = document.createElement("tr");
      var index = document.createElement("th");
      index.setAttribute("scope", "row");
      index.textContent = region.region_index;
      row.appendChild(index);
      [
        region.class_code,
        pxText(region.length_px),
        pxText(region.max_width_px),
        numberText(region.area_px)
      ].forEach(function (value, position) {
        var cell = document.createElement("td");
        if (position > 0) { cell.className = "num"; }
        cell.textContent = value;
        row.appendChild(cell);
      });
      body.appendChild(row);
    });
  }

  function render(payload) {
    var inspection = payload.inspection;
    if (!inspection) {
      setError("The capture was stored but returned no result to display.");
      return;
    }

    renderBanner(inspection.summary);
    renderRegions(inspection.regions);

    var image = document.getElementById("capture-image");
    // Prefer the overlay; fall back to the plain frame. A failed capture has neither,
    // and showing the raw photograph as though it had been inspected would be worse
    // than showing nothing.
    var url = inspection.overlay_image_url || inspection.source_image_url;
    if (url) {
      image.src = url;
      image.hidden = false;
      image.alt = inspection.overlay_image_url
        ? "Inspection overlay at source resolution, defect regions outlined and numbered"
        : "The submitted frame; no overlay was produced";
    } else {
      image.hidden = true;
      image.removeAttribute("src");
      image.alt = "";
    }

    var parts = [];
    if (inspection.product_id) { parts.push("Product " + inspection.product_id); }
    if (inspection.material) { parts.push(inspection.material); }
    if (inspection.latency_ms !== null && inspection.latency_ms !== undefined) {
      parts.push(Math.round(inspection.latency_ms) + " ms");
    }
    parts.push("inspection " + payload.inspection_id);
    if (payload.generated) {
      parts.push("regions generated by the mock provider, not measured");
    }
    document.getElementById("capture-caption").textContent = parts.join(" · ");

    var first = (inspection.regions && inspection.regions.length)
      ? inspection.regions[0].region_index : 1;
    document.getElementById("capture-open-region").href =
      "/regions?inspection_id=" + payload.inspection_id + "&region=" + first;

    resultEl.hidden = false;
    setStatus("Stored as inspection " + payload.inspection_id + ".");
    resultEl.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  // -- wiring ----------------------------------------------------------------
  startBtn.addEventListener("click", startCamera);
  stopBtn.addEventListener("click", function () {
    stopCamera();
    setStatus("Camera stopped.");
  });
  flipBtn.addEventListener("click", flipCamera);
  shootBtn.addEventListener("click", shoot);

  fileInput.addEventListener("change", function () {
    chooseFile(fileInput.files && fileInput.files[0]);
    fileInput.value = "";      // so the same file can be re-submitted
  });

  ["dragenter", "dragover"].forEach(function (name) {
    drop.addEventListener(name, function (event) {
      event.preventDefault();
      drop.classList.add("capture__drop--over");
    });
  });
  ["dragleave", "drop"].forEach(function (name) {
    drop.addEventListener(name, function (event) {
      event.preventDefault();
      drop.classList.remove("capture__drop--over");
    });
  });
  drop.addEventListener("drop", function (event) {
    var files = event.dataTransfer && event.dataTransfer.files;
    chooseFile(files && files[0]);
  });

  if (!cameraSupported()) {
    startBtn.disabled = true;
    document.getElementById("camera-note").textContent =
      "This browser exposes no camera API here. Camera access needs HTTPS or localhost. " +
      "Use the photo picker below.";
  }

  // Releasing the device on navigation stops the camera light staying on.
  window.addEventListener("pagehide", stopCamera);
})();
