"""
S3 Output Packaging

Packages reconstructed 3D point cloud and metadata artifacts into disk deliverables
(scene.ply + metadata.json).
"""

import json
from pathlib import Path
from typing import Dict, Union

from ..geometry.ply_io import PlyIO
from ..models.s3_output import S3ReconstructionResult


class S3OutputPackager:
    """Packages S3 results for disk export."""

    @staticmethod
    def package_to_directory(
        result: S3ReconstructionResult,
        output_directory: Union[str, Path],
        binary_ply: bool = True,
    ) -> Dict[str, Path]:
        """
        Export S3 output artifacts (scene.ply and metadata.json) to an output directory.

        Parameters:
            result: S3ReconstructionResult object.
            output_directory: Directory where artifacts will be written.
            binary_ply: If True, exports binary PLY; else ASCII PLY.

        Returns:
            Dictionary mapping artifact keys to their written Path locations:
            {"ply": Path(.../scene.ply), "metadata": Path(.../metadata.json)}
        """
        out_dir = Path(output_directory)
        out_dir.mkdir(parents=True, exist_ok=True)

        ply_path = out_dir / "scene.ply"
        meta_path = out_dir / "metadata.json"

        # 1. Write PLY geometry
        PlyIO.write_ply(ply_path, result.point_cloud, binary=binary_ply)

        # 2. Write JSON metadata
        meta_dict = result.to_metadata_dict()
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_dict, f, indent=2)

        return {
            "ply": ply_path,
            "metadata": meta_path,
        }

