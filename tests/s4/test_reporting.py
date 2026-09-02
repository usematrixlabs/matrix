"""Unit tests for S4 reporting (JSON, HTML, and S4->S5 contract payload export)."""

import json
from pathlib import Path
import numpy as np
import pytest

from src.georeferencing_validation._internal.control_points import ControlPoints
from src.georeferencing_validation._internal.crs import CoordinateReference
from src.georeferencing_validation._internal.georeferencer import Georeferencer
from src.georeferencing_validation._internal.helmert import HelmertTransform
from src.georeferencing_validation._internal.input import ReconstructionInput
from src.georeferencing_validation._internal.validator import GeoreferencingValidator


def test_validator_html_and_json_export(tmp_path: Path):
    src = np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]], dtype=np.float64)
    cp = ControlPoints(source=src, target=src)
    transform = HelmertTransform(rotation=np.eye(3), scale=1.0, translation=np.zeros(3))

    validator = GeoreferencingValidator(control_points=cp, transformation=transform, tolerance=0.5)
    res = validator.validate()

    json_path = tmp_path / "report.json"
    json_str = res.to_json(filepath=json_path)
    assert json_path.is_file()
    parsed = json.loads(json_str)
    assert parsed["num_points"] == 4
    assert parsed["pass_status"]["passed_3d"] is True

    html_path = tmp_path / "report.html"
    html_str = res.to_html(filepath=html_path)
    assert html_path.is_file()
    assert "PASSED" in html_str
    assert "Matrix S4" in html_str


def test_georeferencer_contract_payload_export():
    recon_pts = np.array([[0,0,0],[2,0,0],[0,2,0],[0,0,2]], dtype=np.float64)
    recon = ReconstructionInput(points=recon_pts)

    src_gcp = recon_pts
    tgt_gcp = src_gcp + np.array([500000.0, 3000000.0, 100.0])
    control = ControlPoints(source=src_gcp, target=tgt_gcp)

    source_crs = CoordinateReference.local(allow_local_to_world=True)
    target_crs = CoordinateReference.utm(zone=43)

    georeferencer = Georeferencer(
        reconstruction_data=recon,
        control_points=control,
        source_crs=source_crs,
        target_crs=target_crs,
    )
    result = georeferencer.georeference(validate_accuracy=True, tolerance=0.5)

    payload = result.export_contract_payload()

    assert "geo_referenced_scene" in payload
    assert "validation_metrics" in payload
    assert "coordinate_reference" in payload
    assert "quality_status" in payload
    assert "known_limitations" in payload

    geo_scene = payload["geo_referenced_scene"]
    assert geo_scene["point_cloud"]["num_points"] == 4
    assert len(geo_scene["scene_origin"]) == 3
    assert len(geo_scene["scene_orientation"]) == 3

    val_metrics = payload["validation_metrics"]
    assert "geometric_accuracy" in val_metrics
    assert "horizontal_accuracy" in val_metrics
    assert "vertical_accuracy" in val_metrics

    quality = payload["quality_status"]
    assert "confidence_level" in quality
    assert quality["confidence_level"] in ["high", "medium", "low"]

