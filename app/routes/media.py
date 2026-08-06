"""Image serving.

No request ever names a filesystem path.  A request names an inspection and a role
(source / overlay / region crop); the path comes from the database and is then checked
to resolve inside the configured data roots before anything is read.  There is no
endpoint that accepts a path, so there is no traversal to defend against - and the
resolve-and-check in inspection_service.image_path_for is the second line anyway.
"""

from __future__ import annotations

import io
import sqlite3

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse, Response

from app.config import Settings
from app.dependencies import get_db, settings_dep
from app.providers import mock_assets
from app.repositories import inspection_repository as inspections
from app.services import inspection_service

router = APIRouter(prefix="/media", tags=["media"])

CACHE = {"Cache-Control": "public, max-age=300"}


def _row(conn: sqlite3.Connection, inspection_id: int) -> dict:
    row = inspections.get(conn, inspection_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"No inspection {inspection_id}")
    return row


@router.get("/inspection/{inspection_id}/{kind}")
def inspection_image(
    inspection_id: int,
    kind: str,
    conn: sqlite3.Connection = Depends(get_db),
    settings: Settings = Depends(settings_dep),
) -> FileResponse:
    if kind not in ("source", "overlay"):
        raise HTTPException(status_code=404, detail="Unknown image kind")
    row = _row(conn, inspection_id)
    path = inspection_service.image_path_for(row, kind, settings)
    if path is None:
        raise HTTPException(
            status_code=404,
            detail=f"No {kind} image stored for inspection {inspection_id}",
        )
    return FileResponse(path, media_type="image/png", headers=CACHE)


@router.get("/inspection/{inspection_id}/region/{region_index}")
def region_crop(
    inspection_id: int,
    region_index: int,
    mode: str = Query("overlay", pattern="^(overlay|source)$"),
    zoom: int = Query(3, ge=1, le=6),
    conn: sqlite3.Connection = Depends(get_db),
    settings: Settings = Depends(settings_dep),
) -> Response:
    """Crop around one region and enlarge it - the region-detail centre pane.

    Cropping happens here rather than in the browser so the operator downloads one
    small tile instead of a full-resolution frame per region.
    """
    row = _row(conn, inspection_id)
    region = inspections.region(conn, inspection_id, region_index)
    if region is None:
        raise HTTPException(status_code=404, detail=f"No region {region_index}")

    path = inspection_service.image_path_for(row, mode, settings)
    if path is None:
        path = inspection_service.image_path_for(row, "source", settings)
    if path is None:
        raise HTTPException(status_code=404, detail="No image stored for this inspection")

    from PIL import Image

    image = Image.open(path).convert("RGB")
    bbox = (region["bbox_x"], region["bbox_y"], region["bbox_width"], region["bbox_height"])
    crop = mock_assets.crop_region(image, bbox, zoom=zoom)
    buffer = io.BytesIO()
    crop.save(buffer, format="PNG")
    return Response(content=buffer.getvalue(), media_type="image/png", headers=CACHE)
