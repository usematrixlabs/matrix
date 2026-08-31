# Subsystem 1 (S1) — Visual Perception

**Visual Perception** is responsible for transforming raw UAV video and accompanying telemetry into structured, timestamped visual observations conforming to the **S1 $\rightarrow$ S2 Contract** ([`docs/architecture/contracts/perception-localization.md`](../../docs/architecture/contracts/perception-localization.md)).

---

## 🎯 Subsystem Boundaries & Principles

* **Owns:** Video decoding, frame extraction, keyframe selection, image quality assessment, and preservation of raw UAV telemetry (GPS, GNSS, IMU, altitude, RTK/PPK).
* **Does Not Own:** Camera pose estimation, trajectory calculation, sensor fusion (owned by **S2**), or 3D reconstruction (owned by **S3**).

---

## 🚀 Environment Setup

### 1. Python Environment
Python 3.10+ is recommended.

```bash
# Using Python venv
python -m venv .venv

# On Linux / macOS
source .venv/bin/activate

# On Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

### 2. Install S1 Dependencies
Install the required dependencies for Subsystem 1:

```bash
pip install -r requirements-s1.txt
```

---

## 📂 Module Structure

```text
src/visual_perception/
├── __init__.py                # Package exports
├── types.py                   # Dataclasses & S1->S2 contract schemas
├── config.py                  # Configuration loader & validator
├── logger.py                  # Structured logger setup
├── frame_extractor.py         # Video ingestion & frame extraction
├── keyframe_selector.py       # Keyframe selection & quality scoring
├── pipeline.py                # S1 pipeline coordinator & CLI entrypoint
├── README.md                  # Subsystem guide
└── configs/
    └── default_s1_config.yaml # Default configuration file
```

---

## ⚙️ Configuration

S1 configuration can be supplied via YAML or Python dictionaries. Default configuration is located at `src/visual_perception/configs/default_s1_config.yaml`:

```yaml
# Input paths
video_path: null
telemetry_path: null

# Output directories
output_dir: "data/output/s1_observations"
frames_dir: "data/output/s1_observations/frames"
keyframes_dir: "data/output/s1_observations/keyframes"

# Frame Extraction
frame_rate: 2.0
time_start: 0.0
time_end: null

# Keyframe Selection
keyframe_method: "uniform"
quality_threshold: 50.0

# Logging
log_level: "INFO"
```

---

## 💻 Usage & Execution

### 1. CLI Execution

Run the S1 pipeline directly via the command line:

```bash
# Basic run with defaults
python -m src.visual_perception.pipeline

# Run with custom video, telemetry, and output directory
python -m src.visual_perception.pipeline --video path/to/uav_flight.mp4 --telemetry path/to/telemetry.json --output-dir data/output/run_01 --frame-rate 2.0

# Run with a configuration file
python -m src.visual_perception.pipeline --config src/visual_perception/configs/default_s1_config.yaml --save-output output_s1.json
```

### 2. Python API

```python
from src.visual_perception import S1Config, S1Pipeline

# Initialize configuration
config = S1Config(
    video_path="data/raw/flight_01.mp4",
    frame_rate=2.0,
    output_dir="data/output/s1_flight_01",
    keyframe_method="uniform"
)

# Run pipeline
pipeline = S1Pipeline(config=config)
result = pipeline.run()

# Access structured S1 -> S2 outputs
print(f"Status: {result.status}")
print(f"Total Frames: {len(result.visual_observations.frames)}")
print(f"Total Keyframes: {len(result.visual_observations.keyframes)}")
```

---

## 🧪 Testing

Run S1 test suite:

```bash
pytest tests/test_s1_setup.py -v
```

