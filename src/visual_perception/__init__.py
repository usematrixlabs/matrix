"""S1 — Visual Perception

Transforms UAV video into usable visual observations and preserves
input information required by downstream subsystems.
"""
from .frame_extractor import FrameExtractor
from .keyframe_selector import KeyframeSelector

__all__ = ["FrameExtractor", "KeyframeSelector"]