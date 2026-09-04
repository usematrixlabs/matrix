"""Unit tests for high-level S4 Georeferencer pipeline and automated limitation reporting."""

import numpy as np
import pytest

from georeferencing_validation._internal.control_points import ControlPoints
from georeferencing_validation._internal.crs import CoordinateReference
from georeferencing_validation._internal.georeferencer import Georeferencer
from georeferencing_validation._internal.input import ReconstructionInput


def test_georeferencer_end_to_end_with_colors():
    pts = np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]], dtype=np.float64)
    colors = np.array([[255,0,0],[0,255,0],[0,0,255],[255,255,255]], dtype=np.uint8)
    recon = ReconstructionInput(points=pts, colors=colors, metadata={"job": "test_job"})

    gcp_tgt = pts * 2.0 + np.array([100.0, 200.0, 300.0])
    control = ControlPoints(source=pts, target=gcp_tgt)

    src_crs = CoordinateReference.local(allow_local_to_world=True)
    tgt_crs = CoordinateReference.utm(zone=43)

    georef = Georeferencer(
        reconstruction_data=recon,
        control_points=control,
        source_crs=src_crs,
        target_crs=tgt_crs,
    )
    result = georef.georeference(validate_accuracy=True, tolerance=0.1)

    assert result.points.shape == (4, 3)
    np.testing.assert_allclose(result.colors, colors)
    assert result.validation_result is not None
    assert result.validation_result.passed is True
    assert result.transformation.scale == pytest.approx(2.0, rel=1e-4)


def test_georeferencer_detects_known_limitations():
    pts = np.array([[0,0,0],[10,0,0],[0,10,0],[10,10,0]], dtype=np.float64)
    recon = ReconstructionInput(points=pts)

    gcp_tgt = pts + np.array([500000.0, 3000000.0, 100.0])
    control = ControlPoints(source=pts, target=gcp_tgt)

    src_crs = CoordinateReference.local(allow_local_to_world=True)
    tgt_crs = CoordinateReference.utm(zone=43)

    georef = Georeferencer(
        reconstruction_data=recon,
        control_points=control,
        source_crs=src_crs,
        target_crs=tgt_crs,
    )
    result = georef.georeference()

    assert len(result.known_limitations) >= 2
    assert any("Small number of Ground Control Points" in lim for lim in result.known_limitations)
    assert any("Low vertical GCP distribution" in lim for lim in result.known_limitations)
    assert result.quality_status["confidence_level"] in ["medium", "high"]

