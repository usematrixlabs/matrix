"""S1 Configuration Management.

Defines configuration data models and loading mechanisms for Subsystem 1.
"""

from dataclasses import asdict, dataclass, field
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class S1Config:
    """Configuration settings for S1 Visual Perception subsystem."""

    # Input and Output Paths
    video_path: Optional[str] = None
    telemetry_path: Optional[str] = None
    output_dir: str = "data/output/s1_observations"
    frames_dir: str = "data/output/s1_observations/frames"
    keyframes_dir: str = "data/output/s1_observations/keyframes"

    # Frame Extraction & Sampling Parameters (Phase 4)
    sampling_mode: str = "fixed"  # "fixed", "fps", "all"
    sampling_interval: int = 10  # Sample every N frames when sampling_mode="fixed"
    frame_rate: float = 2.0  # Frames extracted per second when sampling_mode="fps"
    time_start: float = 0.0  # Start time offset in seconds
    time_end: Optional[float] = None  # End time offset in seconds
    target_width: Optional[int] = None
    target_height: Optional[int] = None
    image_format: str = "jpg"  # "jpg", "jpeg", "png"
    jpeg_quality: int = 95  # JPEG quality (1-100)

    # Keyframe Selection Parameters
    keyframe_method: str = "uniform"  # "uniform", "laplacian_variance", "feature_diff"
    quality_threshold: float = 50.0  # Minimum sharpness / quality score
    max_keyframes: Optional[int] = None

    # Logging Parameters
    log_level: str = "INFO"
    log_file: Optional[str] = None

    # Extra parameters
    extra_params: Dict[str, Any] = field(default_factory=dict)

    def ensure_directories(self) -> None:
        """Create configured output directories if they do not exist."""
        for path_str in [self.output_dir, self.frames_dir, self.keyframes_dir]:
            if path_str:
                Path(path_str).mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "S1Config":
        """Instantiate configuration from a dictionary."""
        valid_fields = cls.__dataclass_fields__.keys()
        init_args = {k: v for k, v in data.items() if k in valid_fields and k != "extra_params"}
        extra = {k: v for k, v in data.items() if k not in valid_fields}
        return cls(**init_args, extra_params=extra)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "S1Config":
        """Instantiate configuration from a YAML file."""
        path = Path(yaml_path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")

        try:
            import yaml
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        except ImportError:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

        return cls.from_dict(data)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)

    def save_yaml(self, yaml_path: str) -> None:
        """Save configuration to a YAML file."""
        Path(yaml_path).parent.mkdir(parents=True, exist_ok=True)
        try:
            import yaml
            with open(yaml_path, "w", encoding="utf-8") as f:
                yaml.dump(self.to_dict(), f, default_flow_style=False)
        except ImportError:
            with open(yaml_path, "w", encoding="utf-8") as f:
                json.dump(self.to_dict(), f, indent=2)
