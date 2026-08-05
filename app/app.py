"""Local inspection prototype.

    streamlit run app/app.py

Deliberately one page and one job: show what the model found and let an operator judge
it. Per docs/ARCHITECTURE.md section 9 the system does the searching and a human does
the deciding, so there is no pass/fail verdict here and no confidence percentage --
the model's scores are not calibrated, and a made-up "97 % sure" would be read as one.

All computation goes through app/inference.py and app/postprocess.py, which the batch
CLI also uses. Nothing is measured in this file.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from inference import Inspector                       # noqa: E402
from postprocess import Profile, record_to_rows       # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
EXPORT = ROOT / "data/export"

st.set_page_config(page_title="Surface Defect Inspection", layout="wide")


@st.cache_resource(show_spinner=False)
def get_inspector(model_path, ct, st_, min_area,
                  min_skel: int, station: str) -> Inspector:
    """Cached per configuration: reloading the ONNX session per image would dominate
    the latency the app is meant to demonstrate."""
    return Inspector(model_path,
                     Profile(crack_thresh=ct, scratch_thresh=st_,
                             min_area_px=min_area, min_skeleton_px=min_skel),
                     station_id=station)


def main():
    st.title("Surface Defect Inspection")
    st.caption("Conveyor-line end-of-line QC · segmentation + defect type + geometry · "
               "runs offline on CPU")

    models = sorted(EXPORT.glob("*.onnx"))
    if not models:
        st.error(f"No ONNX model in {EXPORT}. Export one first:\n\n"
                 "`python bench/export_onnx.py --models smpslim_timm-mobilenetv3_small_100 "
                 "--classes 3 --tag c3 --weights <checkpoint.pt> --allow-busy`")
        return

    with st.sidebar:
        st.header("Station")
        model_path = st.selectbox("Model", models, format_func=lambda p: p.name)
        station = st.text_input("Station ID", "LINE-1")
        st.header("Processing profile")
        st.caption("Recorded with every result, so a record can be reproduced.")
        ct = st.slider("Crack threshold", 0.05, 0.95, 0.50, 0.05)
        sct = st.slider("Scratch threshold", 0.05, 0.95, 0.50, 0.05)
        min_area = st.number_input("Min region area (px)", 0, 5000, 24, 4)
        min_skel = st.number_input("Min skeleton length (px)", 0, 500, 6, 1)

    insp = get_inspector(str(model_path), ct, sct, int(min_area), int(min_skel),
                         station)
    if insp.sess.get_outputs()[0].shape[1] == 1:
        st.warning("This checkpoint has a **binary** head: it can only report cracks. "
                   "Every scratch it finds will be labelled `crack`. Load a 3-class "
                   "export for crack/scratch discrimination.")

    files = st.file_uploader("Product images", type=["png", "jpg", "jpeg", "bmp"],
                             accept_multiple_files=True)
    if not files:
        st.info("Upload one or more captures. Multiple files produce a batch CSV.")
        return

    records, all_rows = [], []
    for f in files:
        raw = f.getvalue()
        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            st.error(f"{f.name}: not a readable image — recorded as an acquisition "
                     f"failure, not as a clean product.")
            continue
        rec, ov, _ = insp.inspect(img, raw, product_id=f.name)
        records.append(rec)
        all_rows += record_to_rows(rec)

        st.subheader(f.name)
        c1, c2 = st.columns(2)
        c1.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="capture",
                 use_container_width=True)
        c2.image(cv2.cvtColor(ov, cv2.COLOR_BGR2RGB), caption="detected regions",
                 use_container_width=True)

        if rec["empty"]:
            st.success("No defect regions found. Recorded explicitly as clean.")
        else:
            st.dataframe(pd.DataFrame(rec["regions"]), use_container_width=True,
                         hide_index=True)
            st.caption("Geometry is in pixels. Millimetres require a calibrated "
                       "scale reference at the inspection plane (architecture §5).")
        with st.expander("Inspection record (JSON)"):
            st.code(json.dumps(rec, indent=2), language="json")

    if all_rows:
        st.divider()
        csv = pd.DataFrame(all_rows).to_csv(index=False).encode()
        st.download_button("Download batch CSV", csv, "inspection_report.csv",
                           "text/csv")
        st.download_button("Download records (JSON)",
                           json.dumps(records, indent=2).encode(),
                           "inspection_records.json", "application/json")


if __name__ == "__main__":
    main()
