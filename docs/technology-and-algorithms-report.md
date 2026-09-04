# Matrix Technology, Algorithms, Models, and Mathematics Report

## Scope and Status

This report describes the technologies and mathematical methods present in the Matrix repository. Status labels are used throughout:

- **Active**: used by the current public pipeline path.
- **Implemented, not active**: present and tested, but not currently called by the main path.
- **Optional**: selected through configuration or available only when an extra dependency is installed.
- **Planned/documented**: described by architecture or contracts, but not currently emitted or wired end to end.

Matrix is a Python 3.12–3.14 UAV video-to-3D geospatial reconstruction system. The pipeline is S1 → S2 → S3 → S4 → S5; the pipeline package owns composition while each subsystem owns its algorithms.

## Shared Foundation

| Technology | Used for | Status |
|---|---|---|
| Python | Subsystem implementation, CLI entry points, data processing, and orchestration | Active |
| NumPy | Dense arrays, vector/matrix operations, linear algebra, statistics, covariance, geometry, and quaternion calculations | Active |
| OpenCV | Video decoding, image conversion, histograms, feature detection/matching, camera calibration, distortion correction, PnP, and image encoding | Active |
| Pydantic v2 | Validation and serialization of the S1–S5 boundary contracts | Active |
| PyYAML | S1 configuration and OpenCV calibration YAML parsing | Active |
| pyproj | WGS84, ECEF, ENU, UTM, and other CRS transformations | Active in supporting utilities; S4 runtime georeferencing is currently placeholder-based |
| pytest | Unit, contract, subsystem, integration, and benchmark-marked tests | Active |
| JSON, CSV, YAML | Portable contracts, GPS input, configuration, and calibration interchange | Active |
| PLY | Point-cloud input/output between S3 and S4 | Active |

The root project does not require PyTorch, TorchVision, Kornia, or LightGlue. Those are isolated under `models/LightGlue/` and are optional for the LightGlue matcher.

## S1 — Visual Perception

S1 converts UAV video into ordered, quality-scored, portable visual observations. It does not interpret GPS or IMU data as a localization solution; it carries available telemetry to S2.

### Technologies and implementation

- **OpenCV `VideoCapture`** validates and decodes video. Validation covers path existence, zero-byte files, supported containers, decoder availability, FourCC, FPS, resolution, and frame count.
- **Deterministic frame sampling** supports fixed interval, target-FPS, and all-frame extraction. Frames are encoded as JPEG or PNG.
- **Stable identifiers and timestamps** use deterministic IDs such as `frame_000123` and monotonic video-capture timestamps in seconds.
- **Pydantic contracts and JSON packaging** write `observations.json` plus a `frames/` directory. Relative paths, camera metadata, quality fields, and keyframe flags are preserved.
- **Camera metadata and calibration preservation** accepts JSON, YAML, or dictionaries and keeps intrinsics and distortion explicit, including `null` when calibration is unavailable.
- **Optional telemetry transport** preserves GPS/GNSS, IMU, altitude, RTK/PPK, and flight metadata without making S1 responsible for fusion.

### Algorithms and mathematical methods

1. **Histogram-based keyframe selection**
   - Grayscale histograms are normalized with OpenCV.
   - Content change is measured with the Bhattacharyya distance:
     $$D_B(p,q)=\sqrt{1-\sum_i\sqrt{p_iq_i}}.$$
   - A minimum interval prevents redundant selections; a maximum interval guarantees temporal coverage.
   - Alternative selectors are uniform sampling and local quality maxima.

2. **Laplacian-variance sharpness test**
   - The grayscale Laplacian is computed and its variance is used as a blur score:
     $$B=\operatorname{Var}(\nabla^2 I).$$
   - Low variance indicates weak high-frequency detail and possible blur.

3. **Exposure assessment**
   - Mean grayscale intensity is compared with underexposure and overexposure thresholds.

4. **Shannon entropy**
   - The grayscale histogram is converted into probabilities and texture richness is estimated as:
     $$H(I)=-\sum_i p_i\log_2 p_i.$$

5. **Feature-density assessment**
   - OpenCV FAST keypoint detection counts usable local features.
   - `goodFeaturesToTrack` is the fallback if FAST fails.

6. **Composite quality score**
   - Sharpness contributes up to 40 points, exposure up to 30, and feature count up to 30, producing a normalized 0–100 score.
   - Quality states include `GOOD`, `BLURRY`, `OVEREXPOSED`, `UNDEREXPOSED`, `LOW_FEATURE`, and `CORRUPTED`.

### Current boundary behavior

S1 retains candidate frames even when they are not keyframes. Downstream consumers can choose whether to use all observations or only marked keyframes. Quality and degradation diagnostics are explicit rather than silently dropping information.

## S2 — Localization and Sensor Fusion

S2 estimates camera position, orientation, trajectory, and confidence from S1 observations and optional location/sensor data.

### Technologies and implementation

- **CSV GPS ingestion** reads `drone_lat`, `drone_lon`, `drone_altitude_m`, and a video-time/timestamp field.
- **Pydantic pose schemas** represent position, camera pose, quaternion orientation, covariance, confidence, and observation associations.
- **NumPy** provides state vectors, covariance matrices, coordinate rotations, and quaternion operations.
- **pyproj** supports geodetic-to-ECEF and ECEF-to-ENU conversion in the reusable coordinate transformer.
- **OpenCV ORB and BFMatcher** provide the classical visual feature backend.
- **LightGlue** is an optional learned matcher backend using SuperPoint features and the vendored LightGlue model.

### Algorithms and mathematical methods

1. **Timestamp-nearest GPS association**
   - Each visual observation is associated with the GPS record nearest in time.
   - The current public integration path derives a local ENU-like position from latitude, longitude, and altitude differences using metres-per-degree approximations.

2. **Coordinate reference transformations**
   - WGS84 geodetic coordinates $(\phi,\lambda,h)$ are converted through ECEF and then to a local East-North-Up frame.
   - The ENU frame is a local tangent frame represented by a rotation matrix; this avoids treating latitude/longitude degrees as Euclidean metres in the general coordinate utility.

3. **Trajectory smoothing**
   - Positions use centered moving-window averaging.
   - Orientations use normalized quaternions, hemisphere sign alignment, and SLERP interpolation. Sign alignment avoids interpolating between equivalent quaternions on opposite sides of the unit sphere.

4. **Extended Kalman Filter** — implemented and tested
   - The state has 16 components:
     $$x=[p,v,a,q,\omega],$$
     where $p,v,a\in\mathbb{R}^3$, $q\in\mathbb{R}^4$, and angular velocity $\omega\in\mathbb{R}^3$.
   - The constant-acceleration prediction includes:
     $$p_{k+1}=p_k+v_k\Delta t+\tfrac12a_k\Delta t^2,$$
     $$v_{k+1}=v_k+a_k\Delta t.$$
   - Covariance propagation uses the standard linearized form:
     $$P_{k+1}=F P_k F^T+Q\Delta t.$$
   - Measurement updates use innovation covariance and Kalman gain:
     $$S=HPH^T+R,\qquad K=PH^TS^{-1}.$$
   - GPS, accelerometer, gyroscope, orientation, custom covariance, and confidence scoring are supported by the engine.
   - The public `run_s2` path currently uses GPS-derived poses and trajectory processing; the full sensor-fusion engine is available as an internal component.

5. **Visual pose estimation** — implemented, tested, but not active in `run_s2`
   - ORB descriptors are matched with Hamming distance.
   - OpenCV `solvePnPRansac` estimates camera pose from 2D–3D correspondences while rejecting outliers.
   - Rodrigues vectors/matrices represent rotation, then rotation matrices are converted to quaternions for the contract.

6. **Matcher backends**
   - **Classical**: ORB + cross-checked brute-force Hamming matching; no learned weights.
   - **LightGlue**: SuperPoint keypoint/descriptor extraction followed by the LightGlue learned matcher. It is optional, lazily imported, and configured as `backend: lightglue`.
   - The backend abstraction is wired and smoke-tested, but current `run_s2` records the selected backend without using it to compute the final pose or feature tracks.

## S3 — 3D Reconstruction

S3 generates the current sparse 3D representation from observations and camera geometry.

### Technologies and implementation

- **OpenCV and ORB** extract features from S1 images and match consecutive frames.
- **Brute-force Hamming matching** creates pairwise matches.
- **Track building** merges pairwise matches into multi-view feature-track IDs.
- **OpenCV calibration YAML** is parsed into a pinhole camera model and distortion coefficients.
- **`cv2.undistortPoints(P=K)`** produces undistorted pixel coordinates in the same intrinsic-matrix pixel space used by triangulation.
- **NumPy SVD** solves the multi-view geometry problem.
- **PLY I/O** writes the reconstructed point cloud and RGB values in a standard interchange format.
- **Pydantic contracts** validate the S2-to-S3 input and S3-to-S4 output.

### Algorithms and mathematical methods

1. **Pinhole camera projection**
   - Each camera uses:
     $$P=K[R\mid t],$$
     where $K$ is the intrinsic matrix and $[R\mid t]$ is the camera pose.
   - If calibration is absent, the active bridge uses a heuristic fallback with $f_x=f_y=\text{image width}$ and the principal point at the image center.

2. **Radial/tangential distortion correction**
   - The supported `plumb_bob`/`radtan` model uses the coefficient order $[k_1,k_2,p_1,p_2,k_3]$.
   - Distortion is corrected before triangulation; raw coordinates remain available for audit.

3. **N-view Direct Linear Transform triangulation**
   - For each observation $(u_i,v_i)$, the engine builds:
     $$u_iP_i^{(3)}-P_i^{(1)},\qquad v_iP_i^{(3)}-P_i^{(2)}.$$
   - Stacking these rows gives an overdetermined homogeneous system:
     $$AX=0.$$
   - SVD supplies the homogeneous solution from the right singular vector associated with the smallest singular value. The 3D point is obtained by dehomogenizing $X$.

4. **Geometric validity checks**
   - **Cheirality** rejects points with insufficient positive camera depth.
   - **Reprojection error** is computed as the mean pixel distance between observed and projected points:
     $$e=\frac1N\sum_i\sqrt{(u_i-\hat u_i)^2+(v_i-\hat v_i)^2}.$$
   - **Parallax** is the largest angle between camera-to-point rays; small parallax is rejected because depth becomes poorly conditioned.

5. **Point-cloud processing**
   - RGB samples are averaged for reconstructed points.
   - Statistical outlier removal and point-density calculation are available.
   - Quality metrics include reprojection error, coverage ratio, and triangulation success ratio.

### Current reconstruction scope

The active S3 output is a sparse point cloud in PLY form. Dense reconstruction, mesh generation, texture generation, normals, and bundle adjustment are described as optional/future capabilities but are not part of the current emitted scene.

## S4 — Georeferencing and Validation

S4 transforms local reconstruction coordinates into a target spatial reference and evaluates accuracy and spatial consistency.

### Technologies and implementation

- **NumPy** handles 3D arrays, rotations, residuals, covariance-like statistics, and SVD.
- **pyproj** provides CRS metadata and cross-CRS transformation support.
- **PLY reader/writer** ingests S3 point clouds and writes transformed point clouds.
- **JSON and HTML reports** expose georeferencing parameters, metrics, quality state, and limitations.

### Algorithms and mathematical methods

1. **Control-point validation**
   - Source and target control points must be finite, have shape $N\times3$, have equal counts, contain at least three points, avoid duplicates, and have sufficient geometric rank.

2. **Seven-parameter Helmert similarity transform**
   - The model is:
     $$y=sRx+t,$$
     where $s>0$ is uniform scale, $R$ is a proper 3D rotation, and $t$ is translation.
   - Rotation, scale, and translation are estimated with the SVD-based Umeyama method.
   - Rotation validity is checked with:
     $$R^TR\approx I,\qquad \det(R)\approx+1.$$
   - A bounded robust refinement evaluates small control-point subsets and rejects points with unusually large residuals. This is a conservative RANSAC-like step, not a full general-purpose RANSAC implementation.

3. **Accuracy metrics**
   - Residuals are split into 3D, horizontal, vertical, and per-axis components.
   - Root mean square error is:
     $$\operatorname{RMSE}(e)=\sqrt{\frac1N\sum_{i=1}^N e_i^2}.$$
   - Mean, median, minimum, maximum, standard deviation, inlier count, and tolerance-based pass/fail are reported.

4. **Spatial consistency**
   - Nearest-neighbour distance distributions detect irregular point spacing.
   - SVD plane fitting estimates a dominant terrain plane and its residual RMSE.
   - Relative scale preservation checks whether transformation changes spatial distances beyond configured limits.

### Current execution-path limitation

The reusable S4 georeferencing and validation components are implemented and tested. However, the public `run_s4` path currently selects points from the S3 cloud, copies them as both source and target points, and fits an identity-like transform using placeholder local/geographic metadata. Real geographic alignment requires actual ground-control point correspondences and a target CRS wired into the runtime configuration.

## S5 — Application and Deployment

S5 is currently a portable result-finalization layer rather than a web application.

### Technologies and implementation

- **Python CLI-oriented integration** receives per-stage contracts from the pipeline.
- **Dataclasses** represent the final output bundle in memory.
- **JSON manifest generation** writes `final_output.json` containing success, scene ID, stage statuses, artifact paths, summary metrics, creation time, and schema version.
- **Pydantic S5 contract** validates the application-facing manifest.

### Algorithms and mathematical methods

S5 does not implement new perception, localization, reconstruction, or georeferencing algorithms. Its main logic is deterministic artifact aggregation and status/limitation propagation from S1–S4.

### Current scope

The repository does not currently implement the documented web UI, API service, upload workflow, job queue, process manager, interactive 3D viewer, or deployment infrastructure. Those are S5 responsibilities in the architecture, but the implemented path is CLI execution plus manifest finalization.

## Pipeline Orchestration

The pipeline package is a thin integration layer and deliberately does not duplicate subsystem algorithms.

- `run_pipeline` invokes S1, S2, S3, S4, and S5 in order.
- Public interfaces and Pydantic contracts are used at subsystem boundaries.
- Per-stage output directories preserve intermediate artifacts.
- Structural S3 validation checks that the point count is non-zero and the PLY artifact exists before S4 runs.
- Exceptions are converted into a `PipelineResult` with the failing stage while successful artifacts remain available for inspection.
- The CLI accepts video, GPS CSV, output directory, and optional OpenCV calibration YAML.

## Overall Technical Character

Matrix currently combines classical computer vision and geometric estimation rather than an end-to-end learned reconstruction model:

- S1 relies on deterministic image statistics and OpenCV feature detection.
- S2 has classical ORB/PnP and an optional learned LightGlue/SuperPoint matcher, but the active integration path is GPS-derived localization.
- S3 uses sparse ORB tracks and calibrated multi-view DLT triangulation.
- S4 uses explicit CRS metadata, similarity-transform geometry, and quantitative residual validation.
- S5 packages results and preserves quality/limitation information.

The main mathematical foundations are image statistics, projective camera geometry, coordinate-frame transformations, quaternion rotation algebra, Kalman filtering, SVD-based estimation, similarity transforms, and error/residual statistics.

## Primary Source Locations

- [System architecture](architecture/system-architecture.md)
- [S1 implementation](../src/visual_perception/)
- [S2 implementation](../src/localization_sensor_fusion/)
- [S3 implementation](../src/reconstruction/)
- [S4 implementation](../src/georeferencing_validation/)
- [S5 implementation](../src/application_deployment/)
- [Pipeline orchestrator](../src/pipeline/orchestrator.py)
- [S3 reconstruction decision](decisions/ADR-001-s3-reconstruction-approach.md)
- [Subsystem contracts](architecture/contracts/)
