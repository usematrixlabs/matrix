"""Unit and integration tests for S5 Orchestrator."""

import json
from pathlib import Path
import numpy as np
import pytest

from src.application_deployment.orchestrator import Orchestrator
from src.georeferencing_validation.control_points import ControlPoints
from src.georeferencing_validation.crs import CoordinateReference
from tests.fixtures.synthetic_scene import generate_synthetic_uav_dataset


def test_orchestrator_initial_status():
    orchestrator = Orchestrator()
    status = orchestrator.get_status()
    assert status["status"] == "initialized"
    assert status["job_id"] is None
    assert "s1_perception" in status["stages"]
    assert "s5_presentation" in status["stages"]


def test_orchestrator_with_s2_payload_end_to_end(tmp_path: Path):
    payload, _, _ = generate_synthetic_uav_dataset(
        num_frames=6, num_points=30, noise_std_px=0.05, seed=42
    )

    out_dir = tmp_path / "matrix_job_001"
    orchestrator = Orchestrator(config={"tolerance": 0.5})

    manifest = orchestrator.run_pipeline(
        s2_payload=payload,
        output_dir=out_dir,
        job_id="test_job_001",
    )

    assert manifest["status"] == "complete"
    assert manifest["job_id"] == "test_job_001"
    assert manifest["error"] is None

    # Check stage metrics
    stages = manifest["stage_metrics"]
    assert stages["s1_perception"]["status"] == "skipped"
    assert stages["s2_localization"]["status"] == "success"
    assert stages["s3_reconstruction"]["status"] in ["SUCCESS", "WARNING", "success"]
    assert stages["s3_reconstruction"]["points_reconstructed"] >= 15
    assert stages["s4_georeferencing"]["status"] == "success"
    assert stages["s5_presentation"]["status"] == "success"

    # Check deliverables
    deliv = manifest["deliverables"]
    assert "point_cloud_ply" in deliv
    assert Path(deliv["point_cloud_ply"]).is_file()
    assert "s3_metadata_json" in deliv
    assert Path(deliv["s3_metadata_json"]).is_file()
    assert "s4_contract_payload" in deliv
    assert Path(deliv["s4_contract_payload"]).is_file()
    assert "manifest_json" in deliv
    assert Path(deliv["manifest_json"]).is_file()


def test_orchestrator_custom_gcp_data(tmp_path: Path):
    payload, gt_points, _ = generate_synthetic_uav_dataset(
        num_frames=5, num_points=25, seed=123
    )

    out_dir = tmp_path / "matrix_job_gcp"
    orchestrator = Orchestrator()

    # Custom GCPs in UTM 43N
    gcp_src = np.array([
        [0.0, 0.0, 0.0],
        [10.0, 0.0, 0.0],
        [0.0, 10.0, 0.0],
        [10.0, 10.0, 2.0],
    ], dtype=np.float64)
    gcp_tgt = gcp_src + np.array([500000.0, 3000000.0, 100.0])
    cp = ControlPoints(source=gcp_src, target=gcp_tgt)

    manifest = orchestrator.run_pipeline(
        s2_payload=payload,
        gcp_data=cp,
        target_crs=CoordinateReference.utm(zone=43),
        output_dir=out_dir,
    )

    assert manifest["status"] == "complete"
    assert "validation_report_html" in manifest["deliverables"]
    assert Path(manifest["deliverables"]["validation_report_html"]).is_file()


def test_orchestrator_error_handling(tmp_path: Path):
    orchestrator = Orchestrator()
    # Pass invalid payload dict with no observations
    invalid_s2 = {"observations": "not_a_list"}

    manifest = orchestrator.run_pipeline(
        s2_payload=invalid_s2,
        output_dir=tmp_path / "error_job",
    )

    assert manifest["status"] == "failed"
    assert manifest["error"] is not None

