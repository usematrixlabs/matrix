"""S4-internal IO helpers: load PLY from S3 artifacts, write georeferenced PLY.

These helpers are internal to S4 and must not import from any other
subsystem. They read S3's wire-format contract via duck-typed field
access (artifact_paths, metadata, etc.) and operate on the canonical
PLY format S3 writes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .georeferencer import GeoreferencedResult
from .models_s3_pcd import PointCloudData


def load_point_cloud_from_s3_artifacts(
    s3_contract: Any,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict[str, Any]]:
    """Load a (N,3) point cloud + optional colors + metadata from S3 artifacts.

    Returns ``(points, colors, metadata)``. ``points`` is ``None`` when
    the artifacts cannot be read; ``metadata`` is whatever the S3
    contract provided (may be empty).
    """
    artifact_paths = getattr(s3_contract, "artifact_paths", None) or {}
    metadata: Dict[str, Any] = dict(getattr(s3_contract, "metadata", {}) or {})

    ply_path_str = artifact_paths.get("ply")
    if not ply_path_str:
        return None, None, metadata

    ply_path = Path(ply_path_str)
    if not ply_path.is_file():
        return None, None, metadata

    pcd = PointCloudData.read_ply(ply_path)
    points = np.asarray(pcd.points, dtype=np.float64)
    colors = (
        np.asarray(pcd.colors, dtype=np.uint8)
        if pcd.colors is not None
        else None
    )

    # Augment metadata with the S3 contract's quality info when available.
    quality = getattr(s3_contract, "quality", None)
    if quality:
        metadata.setdefault("s3_quality", dict(quality))
    spatial_reference = getattr(s3_contract, "spatial_reference", None)
    if spatial_reference:
        metadata.setdefault("s3_spatial_reference", dict(spatial_reference))

    return points, colors, metadata


def write_georeferenced_ply(path: Path, result: GeoreferencedResult) -> None:
    """Write the georeferenced point cloud as a binary PLY."""
    pcd = PointCloudData(
        points=np.asarray(result.points, dtype=np.float64),
        colors=(
            np.asarray(result.colors, dtype=np.uint8)
            if result.colors is not None
            else None
        ),
    )
    pcd.write_ply(path, binary=True)


__all__ = ["load_point_cloud_from_s3_artifacts", "write_georeferenced_ply"]
