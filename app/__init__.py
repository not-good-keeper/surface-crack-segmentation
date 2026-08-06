"""Surface-defect inspection: pipeline (inference, postprocess, batch) and the Phase 2
operator interface (main, routes, services, repositories, providers).

Deliberately empty of imports. `app.main` must be importable in a deployment where
numpy, cv2 and onnxruntime are not installed — anything imported here would be pulled
into every one of those environments.
"""
