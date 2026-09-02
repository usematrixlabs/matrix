"""
georeferencing.validator

S4 — Georeferencing & Validation subsystem
Component 5: Accuracy & Spatial Consistency Validator

Evaluates 3D Helmert transformation accuracy (3D RMSE, horizontal RMSE, vertical RMSE),
performs independent tolerance pass/fail checks, spatial consistency analyses
(neighbor distance dispersion, dominant terrain plane fit, relative scale preservation),
and generates structured JSON & HTML validation reports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np

from .control_points import ControlPoints
from .helmert import HelmertTransform


@dataclass
class SpatialConsistencyReport:
    """Detailed spatial consistency assessment metrics."""
    mean_neighbor_distance: float = 0.0
    min_neighbor_distance: float = 0.0
    max_neighbor_distance: float = 0.0
    neighbor_distance_std: float = 0.0
    plane_fit_residual_rmse: float = 0.0
    scale_preservation_max_error: float = 0.0
    spatial_consistency_score: float = 1.0
    passed: bool = True
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mean_neighbor_distance": float(self.mean_neighbor_distance),
            "min_neighbor_distance": float(self.min_neighbor_distance),
            "max_neighbor_distance": float(self.max_neighbor_distance),
            "neighbor_distance_std": float(self.neighbor_distance_std),
            "plane_fit_residual_rmse": float(self.plane_fit_residual_rmse),
            "scale_preservation_max_error": float(self.scale_preservation_max_error),
            "spatial_consistency_score": float(self.spatial_consistency_score),
            "passed": self.passed,
            "warnings": list(self.warnings),
        }


@dataclass
class ValidationResult:
    """Results of S4 georeferencing accuracy and spatial quality validation."""

    residuals: np.ndarray
    point_errors: np.ndarray
    rmse: float
    mean_error: float
    max_error: float
    min_error: float
    num_points: int
    tolerance: Optional[float] = None
    horizontal_tolerance: Optional[float] = None
    vertical_tolerance: Optional[float] = None
    passed: Optional[bool] = None
    passed_3d: Optional[bool] = None
    passed_horizontal: Optional[bool] = None
    passed_vertical: Optional[bool] = None
    horizontal_rmse: float = 0.0
    vertical_rmse: float = 0.0
    rmse_x: float = 0.0
    rmse_y: float = 0.0
    rmse_z: float = 0.0
    median_error: float = 0.0
    std_error: float = 0.0
    inlier_count: int = 0
    inlier_mask: Optional[np.ndarray] = None
    spatial_consistency: Optional[SpatialConsistencyReport] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert validation result summary to a dictionary."""
        return {
            "num_points": int(self.num_points),
            "inlier_count": int(self.inlier_count),
            "metrics": {
                "rmse_3d": float(self.rmse),
                "horizontal_rmse": float(self.horizontal_rmse),
                "vertical_rmse": float(self.vertical_rmse),
                "rmse_x": float(self.rmse_x),
                "rmse_y": float(self.rmse_y),
                "rmse_z": float(self.rmse_z),
                "mean_error": float(self.mean_error),
                "median_error": float(self.median_error),
                "std_error": float(self.std_error),
                "min_error": float(self.min_error),
                "max_error": float(self.max_error),
            },
            "tolerances": {
                "tolerance_3d": self.tolerance,
                "horizontal_tolerance": self.horizontal_tolerance,
                "vertical_tolerance": self.vertical_tolerance,
            },
            "pass_status": {
                "overall_passed": self.passed,
                "passed_3d": self.passed_3d,
                "passed_horizontal": self.passed_horizontal,
                "passed_vertical": self.passed_vertical,
            },
            "spatial_consistency": (
                self.spatial_consistency.to_dict() if self.spatial_consistency else None
            ),
            "metadata": dict(self.metadata),
        }

    def to_json(
        self,
        filepath: Optional[Union[str, Path]] = None,
        indent: int = 2,
    ) -> str:
        """Export validation result to formatted JSON."""
        json_str = json.dumps(self.to_dict(), indent=indent)
        if filepath is not None:
            path = Path(filepath)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(json_str)
        return json_str

    def to_html(self, filepath: Optional[Union[str, Path]] = None) -> str:
        """Generate a self-contained HTML accuracy validation report."""
        status_color = "#28a745" if self.passed is not False else "#dc3545"
        status_text = "PASSED" if self.passed else ("FAILED" if self.passed is False else "EVALUATED")

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Matrix S4 — Georeferencing Validation Report</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 30px; background: #f8f9fa; color: #333; }}
        .card {{ background: white; padding: 25px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.08); margin-bottom: 20px; }}
        h1, h2 {{ margin-top: 0; }}
        .badge {{ display: inline-block; padding: 6px 14px; border-radius: 20px; font-weight: bold; color: white; background: {status_color}; font-size: 1.1em; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 15px; }}
        th, td {{ padding: 10px 14px; text-align: left; border-bottom: 1px solid #e9ecef; }}
        th {{ background: #f1f3f5; font-weight: 600; }}
        .metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-top: 15px; }}
        .metric-box {{ background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 6px; padding: 15px; text-align: center; }}
        .metric-val {{ font-size: 1.6em; font-weight: bold; color: #0d6efd; margin-top: 5px; }}
    </style>
</head>
<body>
    <div class="card">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <h1>Matrix S4 — Georeferencing & Validation Report</h1>
            <span class="badge">{status_text}</span>
        </div>
        <p>Evaluated <strong>{self.num_points}</strong> Ground Control Points (GCPs) with <strong>{self.inlier_count}</strong> inliers.</p>
    </div>

    <div class="card">
        <h2>Accuracy Metrics (RMSE)</h2>
        <div class="metric-grid">
            <div class="metric-box">
                <div>3D RMSE</div>
                <div class="metric-val">{self.rmse:.4f} m</div>
                <small>Tolerance: {self.tolerance if self.tolerance is not None else 'N/A'}</small>
            </div>
            <div class="metric-box">
                <div>Horizontal (XY) RMSE</div>
                <div class="metric-val">{self.horizontal_rmse:.4f} m</div>
                <small>Tolerance: {self.horizontal_tolerance if self.horizontal_tolerance is not None else 'N/A'}</small>
            </div>
            <div class="metric-box">
                <div>Vertical (Z) RMSE</div>
                <div class="metric-val">{self.vertical_rmse:.4f} m</div>
                <small>Tolerance: {self.vertical_tolerance if self.vertical_tolerance is not None else 'N/A'}</small>
            </div>
            <div class="metric-box">
                <div>Mean Residual Error</div>
                <div class="metric-val">{self.mean_error:.4f} m</div>
                <small>Max: {self.max_error:.4f} m</small>
            </div>
        </div>
    </div>
</body>
</html>"""
        if filepath is not None:
            path = Path(filepath)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(html_content)
        return html_content


class GeoreferencingValidator:
    """S4 Accuracy and Spatial Quality Validator."""

    def __init__(
        self,
        control_points: ControlPoints,
        transformation: HelmertTransform,
        tolerance: Optional[float] = None,
        horizontal_tolerance: Optional[float] = None,
        vertical_tolerance: Optional[float] = None,
    ) -> None:
        """Initialize the validator."""
        if not isinstance(control_points, ControlPoints):
            raise TypeError("control_points must be a ControlPoints instance.")
        if not isinstance(transformation, HelmertTransform):
            raise TypeError("transformation must be a HelmertTransform instance.")

        self._control_points = control_points
        self._transformation = transformation
        self._tolerance = self._validate_tolerance(tolerance, "tolerance")
        self._horizontal_tolerance = self._validate_tolerance(horizontal_tolerance, "horizontal_tolerance")
        self._vertical_tolerance = self._validate_tolerance(vertical_tolerance, "vertical_tolerance")

    @property
    def tolerance(self) -> Optional[float]:
        return self._tolerance

    @property
    def horizontal_tolerance(self) -> Optional[float]:
        return self._horizontal_tolerance

    @property
    def vertical_tolerance(self) -> Optional[float]:
        return self._vertical_tolerance

    @staticmethod
    def _validate_tolerance(val: Optional[float], name: str) -> Optional[float]:
        if val is None:
            return None
        if not isinstance(val, (int, float)) or isinstance(val, bool):
            raise TypeError(f"{name} must be a positive number.")
        val_f = float(val)
        if not np.isfinite(val_f) or val_f <= 0.0:
            raise ValueError(f"{name} must be a positive finite number.")
        return val_f

    def validate(
        self,
        reconstruction_points: Optional[np.ndarray] = None,
        source_points: Optional[np.ndarray] = None,
    ) -> ValidationResult:
        """Validate the fitted Helmert transformation against GCPs and spatial consistency."""
        source = np.asarray(self._control_points.source_array, dtype=np.float64)
        target = np.asarray(self._control_points.target_array, dtype=np.float64)

        transformed = self._transformation.transform_points(source)
        residuals = transformed - target

        dx = residuals[:, 0]
        dy = residuals[:, 1]
        dz = residuals[:, 2]

        point_errors = np.linalg.norm(residuals, axis=1)
        horizontal_errors = np.sqrt(dx**2 + dy**2)
        vertical_errors = np.abs(dz)

        num_points = int(source.shape[0])
        rmse_3d = float(np.sqrt(np.mean(point_errors**2)))
        horizontal_rmse = float(np.sqrt(np.mean(horizontal_errors**2)))
        vertical_rmse = float(np.sqrt(np.mean(vertical_errors**2)))

        rmse_x = float(np.sqrt(np.mean(dx**2)))
        rmse_y = float(np.sqrt(np.mean(dy**2)))
        rmse_z = float(np.sqrt(np.mean(dz**2)))

        mean_err = float(np.mean(point_errors))
        median_err = float(np.median(point_errors))
        std_err = float(np.std(point_errors))
        min_err = float(np.min(point_errors))
        max_err = float(np.max(point_errors))

        passed_3d = (rmse_3d <= self._tolerance) if self._tolerance is not None else None
        passed_horiz = (horizontal_rmse <= self._horizontal_tolerance) if self._horizontal_tolerance is not None else None
        passed_vert = (vertical_rmse <= self._vertical_tolerance) if self._vertical_tolerance is not None else None

        active_checks = [p for p in [passed_3d, passed_horiz, passed_vert] if p is not None]
        overall_passed = all(active_checks) if active_checks else None

        inlier_mask = getattr(self._transformation, "inlier_mask", None)
        inlier_cnt = int(np.sum(inlier_mask)) if inlier_mask is not None else num_points

        # Spatial consistency check
        spatial_report = None
        if reconstruction_points is not None:
            spatial_report = self.check_spatial_consistency(
                points=reconstruction_points,
                source_points=source_points,
            )

        return ValidationResult(
            residuals=residuals,
            point_errors=point_errors,
            rmse=rmse_3d,
            mean_error=mean_err,
            max_error=max_err,
            min_error=min_err,
            num_points=num_points,
            tolerance=self._tolerance,
            horizontal_tolerance=self._horizontal_tolerance,
            vertical_tolerance=self._vertical_tolerance,
            passed=overall_passed,
            passed_3d=passed_3d,
            passed_horizontal=passed_horiz,
            passed_vertical=passed_vert,
            horizontal_rmse=horizontal_rmse,
            vertical_rmse=vertical_rmse,
            rmse_x=rmse_x,
            rmse_y=rmse_y,
            rmse_z=rmse_z,
            median_error=median_err,
            std_error=std_err,
            inlier_count=inlier_cnt,
            inlier_mask=inlier_mask,
            spatial_consistency=spatial_report,
        )

    def check_spatial_consistency(
        self,
        points: np.ndarray,
        source_points: Optional[np.ndarray] = None,
        sample_size: int = 500,
    ) -> SpatialConsistencyReport:
        """Analyze spatial consistency, neighbor distance distributions, and plane fitting."""
        pts = np.asarray(points, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError(f"points must be (N, 3), got {pts.shape}")

        n_pts = pts.shape[0]
        if n_pts < 3:
            return SpatialConsistencyReport(spatial_consistency_score=1.0, passed=True)

        # 1. Neighbor distance distribution
        sub_pts = pts
        if n_pts > sample_size:
            idx = np.random.choice(n_pts, size=sample_size, replace=False)
            sub_pts = pts[idx]

        # Compute pairwise distance to nearest neighbor
        diff = sub_pts[:, np.newaxis, :] - sub_pts[np.newaxis, :, :]
        dist_mat = np.linalg.norm(diff, axis=-1)
        np.fill_diagonal(dist_mat, np.inf)
        nn_dists = np.min(dist_mat, axis=1)

        mean_nn = float(np.mean(nn_dists))
        min_nn = float(np.min(nn_dists))
        max_nn = float(np.max(nn_dists))
        std_nn = float(np.std(nn_dists))

        # 2. Dominant terrain plane fit residual RMSE
        centroid = np.mean(sub_pts, axis=0)
        centered = sub_pts - centroid
        _, _, vh = np.linalg.svd(centered)
        normal = vh[-1]
        plane_dists = np.abs(centered @ normal)
        plane_rmse = float(np.sqrt(np.mean(plane_dists**2)))

        # 3. Relative scale and distance preservation between source and transformed points
        scale_err = 0.0
        warnings = []
        if source_points is not None and len(source_points) == len(pts):
            src_sub = source_points[idx] if n_pts > sample_size else source_points
            src_diff = src_sub[:, np.newaxis, :] - src_sub[np.newaxis, :, :]
            src_dist = np.linalg.norm(src_diff, axis=-1)

            # Ratio of transformed distance to scaled source distance
            s = self._transformation.scale
            expected_dist = s * src_dist
            valid_mask = expected_dist > 1e-6
            if np.any(valid_mask):
                abs_diff = np.abs(dist_mat[valid_mask] - expected_dist[valid_mask])
                scale_err = float(np.max(abs_diff / expected_dist[valid_mask]))
                if scale_err > 1e-3:
                    warnings.append(f"Non-rigid distortion detected in point cloud (max relative scale error {scale_err:.2e})")

        # 4. Consistency score computation (0.0 to 1.0)
        score = 1.0
        if std_nn > 0 and mean_nn > 0:
            cv = std_nn / mean_nn  # coefficient of variation
            score = max(0.0, min(1.0, 1.0 - (cv * 0.2)))

        return SpatialConsistencyReport(
            mean_neighbor_distance=mean_nn,
            min_neighbor_distance=min_nn,
            max_neighbor_distance=max_nn,
            neighbor_distance_std=std_nn,
            plane_fit_residual_rmse=plane_rmse,
            scale_preservation_max_error=scale_err,
            spatial_consistency_score=score,
            passed=len(warnings) == 0,
            warnings=warnings,
        )
