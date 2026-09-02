"""
Synthetic UAV Scene Generator for S3 Reconstruction Testing

Generates mathematically controlled synthetic UAV flight trajectories, ground truth
3D landmark points, and corresponding pinhole camera projections.
"""

from typing import Any, Dict, List, Tuple
import numpy as np

from src.reconstruction._internal.models.schema import (
    CameraIntrinsics,
    CameraPose,
    FeatureObservation,
    LocalizationInfo,
    S2Observation,
    S2Payload,
)


def generate_synthetic_uav_dataset(
    num_frames: int = 6,
    num_points: int = 40,
    noise_std_px: float = 0.0,
    seed: int = 42,
) -> Tuple[S2Payload, np.ndarray, Dict[str, Any]]:
    """
    Generate a synthetic UAV dataset with ground truth 3D points and projections.

    Parameters:
        num_frames: Number of camera observation frames along trajectory.
        num_points: Number of 3D landmark points to generate.
        noise_std_px: Standard deviation of Gaussian noise added to 2D projections.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (S2Payload, ground_truth_points_array, ground_truth_metadata).
    """
    rng = np.random.RandomState(seed)

    # 1. Camera Intrinsics (1920x1080 Full HD UAV camera)
    width, height = 1920, 1080
    fx = 1200.0
    fy = 1200.0
    cx = 960.0
    cy = 540.0
    intrinsics = CameraIntrinsics(
        fx=fx, fy=fy, cx=cx, cy=cy,
        model="PINHOLE",
        image_width=width, image_height=height,
    )
    k_mat = intrinsics.k_matrix

    # 2. Ground truth 3D landmarks in local frame [X: -15 to 15, Y: -15 to 15, Z: 0 to 4]
    gt_points = np.zeros((num_points, 3), dtype=np.float64)
    gt_points[:, 0] = rng.uniform(-15.0, 15.0, size=num_points)
    gt_points[:, 1] = rng.uniform(-15.0, 15.0, size=num_points)
    gt_points[:, 2] = rng.uniform(0.0, 4.0, size=num_points)

    # RGB colors for points
    gt_colors = rng.randint(30, 240, size=(num_points, 3), dtype=np.uint8)

    # 3. UAV flight trajectory (straight flight along X axis at altitude Z=25m)
    # Looking down and slightly forward
    observations: List[S2Observation] = []
    
    # Camera orientation looking down (pitch down by ~75 degrees)
    # Rotation: Camera Z points towards ground (optical axis forward into scene)
    # We define world-to-camera or camera-in-world pose
    # Let standard camera frame: X right, Y down, Z forward
    # For a drone looking straight down: R_world_to_cam rotates [0, 0, -1] into [0, 0, 1]
    rot_down = np.array([
        [1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0]
    ], dtype=np.float64)

    start_x = -20.0
    end_x = 20.0
    x_positions = np.linspace(start_x, end_x, num_frames)

    for i, x_pos in enumerate(x_positions):
        timestamp = 100.0 + i * 1.5
        obs_id = f"frame_{i+1:04d}"
        img_path = f"frames/{obs_id}.jpg"

        # Camera optical center in world coordinates
        cam_pos_world = np.array([x_pos, 0.0, 25.0], dtype=np.float64)

        # Extrinsic matrix [R | t] where X_cam = R @ (X_world - cam_pos) = R @ X_world - R @ cam_pos
        r_mat = rot_down
        t_vec = -r_mat @ cam_pos_world

        # Pose stores R and t such that X_cam = R @ X_world + t
        pose = CameraPose(
            position=t_vec.tolist(),
            orientation=r_mat.tolist(),
            orientation_format="ROTATION_MATRIX",
        )

        # Project 3D points into this camera
        features: List[FeatureObservation] = []
        p_proj = k_mat @ np.hstack([r_mat, t_vec.reshape(3, 1)])

        for pt_idx in range(num_points):
            pt_3d = gt_points[pt_idx]
            pt_homog = np.append(pt_3d, 1.0)
            p_cam = p_proj @ pt_homog

            # Check point is in front of camera (Z > 0)
            if p_cam[2] <= 0.1:
                continue

            u = p_cam[0] / p_cam[2]
            v = p_cam[1] / p_cam[2]

            # Add noise if requested
            if noise_std_px > 0:
                u += rng.normal(0.0, noise_std_px)
                v += rng.normal(0.0, noise_std_px)

            # Check if within image bounds
            if 0 <= u < width and 0 <= v < height:
                features.append(FeatureObservation(
                    feature_id=f"feat_{obs_id}_{pt_idx:04d}",
                    track_id=f"trk_{pt_idx:04d}",
                    xy=[float(u), float(v)],
                    rgb=gt_colors[pt_idx].tolist(),
                ))

        observations.append(S2Observation(
            observation_id=obs_id,
            timestamp=timestamp,
            image_path=img_path,
            image_width=width,
            image_height=height,
            camera=intrinsics,
            pose=pose,
            localization=LocalizationInfo(
                status="ESTIMATED",
                confidence=0.95,
                source="VISUAL_GPS_FUSED",
            ),
            features=features,
            metadata={"drone_altitude": 25.0, "x_pos": x_pos},
        ))

    payload = S2Payload(
        schema_version="1.0.0",
        job_id="job_synthetic_uav_test",
        source_system="S2_LOCALIZATION_SENSOR_FUSION",
        timestamp_created=1772456000.0,
        coordinate_frame="S2_LOCAL",
        units="meters",
        observations=observations,
    )

    metadata = {
        "num_frames": num_frames,
        "num_gt_points": num_points,
        "gt_colors": gt_colors,
        "intrinsics": intrinsics,
    }

    return payload, gt_points, metadata

