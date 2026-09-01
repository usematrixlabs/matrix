# Subsystem 1 (S1) — Visual Perception

**Matrix UAV Video-to-3D Reconstruction System (SIH26158)**

The **Visual Perception (S1)** subsystem ingests single-pass UAV video streams and optional flight telemetry, validates stream integrity, extracts candidate observations using configurable temporal sampling, generates deterministic stable identifiers and monotonic capture timestamps, assesses visual quality conditions (sharpness, exposure, texture richness), marks informative keyframes, preserves camera calibration, and packages portable outputs for **Subsystem 2 (Localization & Sensor Fusion)** and **Subsystem 3 (3D Reconstruction)**.

---

## 1. Architectural Responsibility & Boundaries

* **What S1 Owns:**
  * UAV video ingestion, format validation, and decoding.
  * Frame sampling (fixed-interval, target frame rate, or all-frames).
  * Stable deterministic observation identifiers (`frame_000001`, `frame_000002`, ...).
  * Monotonic capture timestamp calculation (in seconds).
  * Visual quality condition assessment (Laplacian blur, exposure mean, FAST corners, Shannon entropy, corruption detection).
  * Keyframe detection via Bhattacharyya histogram content-change distance without deleting non-keyframes.
  * Ingestion and preservation of camera calibration and optional UAV sensor telemetry.
  * Packaging canonical `s1_output/` artifact directory with portable relative image paths (`frames/frame_000001.jpg`) and `observations.json`.
  * Failure diagnostics and graceful degradation reporting.
* **What S1 Does NOT Own:**
  * Camera pose estimation or trajectory optimization (owned by **S2**).
  * Sensor fusion or GPS/IMU interpretation as a navigation state (owned by **S2**).
  * 3D point cloud generation or mesh reconstruction (owned by **S3**).
  * World geographic coordinate alignment (owned by **S4**).
  * Application UI and visualization orchestration (owned by **S5**).

---

## 2. Directory & Component Layout

```text
src/visual_perception/
├── __init__.py                 # Clean package exports
├── benchmark.py                # Performance benchmarking runner & reporting
├── camera_calibrator.py        # Camera intrinsics loader & distortion parser
├── config.py                   # Dataclass configuration loader (S1Config)
├── diagnostics.py              # Health evaluator & degradation diagnostics
├── downstream_validator.py     # Downstream S2/S3 contract validation harness
├── exceptions.py               # Standardized S1 exception hierarchy
├── frame_extractor.py          # Sequential OpenCV frame decoder & sampler
├── identifier.py               # Deterministic observation ID generator & validator
├── keyframe_selector.py        # Keyframe detector (Bhattacharyya, uniform, quality)
├── logger.py                   # Structured subsystem logging
├── metadata_extractor.py       # Stream geometry & sidecar metadata parser
├── packager.py                 # Canonical s1_output/ packaging engine & JSON Schema
├── pipeline.py                 # End-to-end pipeline runner & CLI entrypoint
├── quality_assessor.py         # Multi-metric visual quality analyzer
├── timestamp_handler.py        # Monotonic capture timestamp engine
├── types.py                    # S1 dataclass models conforming to S1->S2 contract
├── video_validator.py          # 6-stage video integrity & format validator
└── configs/
    └── default_s1_config.yaml  # Default YAML configuration
```

---

## 3. Installation & Setup

### Prerequisites
* Python 3.10+
* OpenCV Headless (`opencv-python-headless`)
* NumPy, PyYAML

### Install Dependencies
```bash
pip install -r requirements-s1.txt
```

---

## 4. Quickstart & CLI Usage

### Basic Frame Extraction
```bash
python -m src.visual_perception.pipeline \
  --video data/raw/uav_flight.mp4 \
  --output-dir data/output/s1_observations \
  --sampling-interval 10
```

### Full Pipeline with Calibration and Keyframing
```bash
python -m src.visual_perception.pipeline \
  --video data/raw/uav_flight.mp4 \
  --telemetry data/raw/telemetry.json \
  --calibration data/raw/camera_calibration.json \
  --output-dir data/output/s1_observations \
  --sampling-mode fixed \
  --sampling-interval 5 \
  --keyframe-method content_change \
  --keyframe-change-threshold 0.15 \
  --save-output data/output/s1_observations/s1_contract.json
```

### Run Performance Benchmark
```bash
python -m src.visual_perception.benchmark --video data/raw/uav_flight.mp4 --interval 5
```

---

## 5. Programmatic Python API

```python
from src.visual_perception import S1Config, S1Pipeline, ObservationPackager

# 1. Configure pipeline
config = S1Config(
    video_path="data/raw/flight_01.mp4",
    calibration_path="data/raw/calib.json",
    output_dir="data/output/s1_observations",
    sampling_mode="fixed",
    sampling_interval=5,
    keyframe_method="content_change",
    keyframe_change_threshold=0.15,
)

# 2. Execute pipeline
pipeline = S1Pipeline(config=config)
s1_output = pipeline.run()

print(f"Status: {s1_output.status}")
print(f"Extracted: {len(s1_output.visual_observations.frames)} frames")
print(f"Keyframes: {len(s1_output.visual_observations.keyframes)}")
print(f"Diagnostics: {s1_output.diagnostics['observations_summary']}")

# 3. Load packaged bundle programmatically
package = ObservationPackager.load_package("data/output/s1_observations")
```

---

## 6. Input & Output Contracts

### Input Requirements
* **Video:** MP4, MOV, AVI, MKV, WebM container format. Supported codecs: H.264, HEVC/H.265, MJPEG, ProRes.
* **Optional Sidecar Telemetry:** JSON file with GPS, IMU, altitude, and RTK coordinates.
* **Optional Camera Calibration:** JSON/YAML file with `fx`, `fy`, `cx`, `cy`, and `distortion_coefficients`.

### Output Structure (`s1_output/`)
```text
s1_output/
│
├── frames/
│   ├── frame_000001.jpg
│   ├── frame_000002.jpg
│   └── ...
│
└── observations.json
```

### Observation Item Schema (`observations.json`)
```json
{
  "observation_id": "frame_000123",
  "timestamp": 12.34,
  "image": "frames/frame_000123.jpg",
  "camera": {
    "width": 1920,
    "height": 1080,
    "intrinsics": {
      "fx": 1450.0,
      "fy": 1452.0,
      "cx": 960.0,
      "cy": 540.0,
      "camera_matrix": [[1450.0, 0.0, 960.0], [0.0, 1452.0, 540.0], [0.0, 0.0, 1.0]]
    },
    "distortion": {
      "coefficients": [-0.12, 0.05, 0.0, 0.0, 0.0],
      "model": "radtan"
    }
  },
  "quality": {
    "status": "GOOD",
    "blur_score": 245.3,
    "quality_score": 88.5,
    "flags": []
  },
  "keyframe": true
}
```

---

## 7. Failure & Degradation Handling (Phase 11)

| Status | Trigger Condition | System Behavior |
| :--- | :--- | :--- |
| **`"completed"`** | Valid video, healthy observations, valid metadata | Full packaging, complete health report |
| **`"degraded"`** | $< \text{min\_valid\_observations}$ frames, or $> 80\%$ blurry/featureless frames | Completes packaging with diagnostic warning flags |
| **`"failed"`** | Missing video, corrupt 0-byte header, unsupported codec | Halts execution, records actionable error in `S1Output.errors` |

---

## 8. Performance Benchmarks (Phase 13 Baseline)

*Measured on 60-frame 640x480 video with sampling interval = 5 frames:*

| Mode | Total Time (s) | Throughput (FPS) | Observations | Keyframes | Peak RAM (MB) | Storage (KB) | Overhead vs Baseline |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Sampling Only** | 0.042s | 1428.5 FPS | 12 | 12 | 0.18 MB | 145.2 KB | 0.0% (Baseline) |
| **Sampling + Quality** | 0.075s | 800.0 FPS | 12 | 12 | 0.24 MB | 145.2 KB | +78.5% |
| **Sampling + Quality + Keyframes** | 0.088s | 681.8 FPS | 12 | 6 | 0.29 MB | 145.2 KB | +109.5% |

**Keyframe Computational Overhead:** ~17.3% additional runtime over Quality Assessment mode.

---

## 9. Downstream Interface Validation (S2 & S3)

* **S2 Localization Consumer:**
  * Uses `frame_ordering` and monotonic timestamps for temporal trajectory estimation.
  * Allows querying `all_observations` for tracking vs `keyframes_only` for loop closure.
* **S3 3D Reconstruction Consumer:**
  * Direct ingestion of `frames/` image files using relative paths.
  * Seamless support for calibrated or uncalibrated camera models.

---

## 10. Automated Test Suites

Run the complete 12-suite test suite:
```bash
python -m unittest discover tests -v
```
All 70+ unit and integration tests are verified with 100% pass rate.
