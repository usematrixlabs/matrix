"""
Abstract Base Class for S3 Reconstruction Engines
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple
import numpy as np

from ..preprocessing.prepare import PreparedReconstructionData


class ReconstructionEngineBase(ABC):
    """Abstract base class for 3D reconstruction engines."""

    @abstractmethod
    def reconstruct(
        self,
        prepared_data: PreparedReconstructionData,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray, Dict[str, Any]]:
        """
        Execute 3D reconstruction.

        Parameters:
            prepared_data: Normalized input data with multi-view tracks and camera geometry.

        Returns:
            Tuple of:
                - points_3d: (N, 3) float64 array of reconstructed points.
                - colors: Optional (N, 3) uint8 array of point colors.
                - reprojection_errors: (N,) float64 array of mean reprojection errors in pixels.
                - engine_stats: Dictionary containing runtime metrics and counts.
        """
        raise NotImplementedError

