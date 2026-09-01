"""
S3 PLY File I/O

High-performance read/write operations for standard Stanford PLY point clouds
in both ASCII and binary little-endian formats.
"""

from pathlib import Path
from typing import Optional, Tuple, Union
import numpy as np

from ..models.s3_output import PointCloudData


class PlyIO:
    """Read and write Stanford PLY files for 3D point clouds."""

    @staticmethod
    def write_ply(
        file_path: Union[str, Path],
        point_cloud: PointCloudData,
        binary: bool = True,
    ) -> None:
        """
        Write PointCloudData to a .ply file.

        Parameters:
            file_path: Destination path for .ply file.
            point_cloud: PointCloudData instance containing points and optional attributes.
            binary: If True, writes binary_little_endian 1.0; else writes ASCII.
        """
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        points = point_cloud.points
        n_points = points.shape[0]
        has_colors = point_cloud.colors is not None
        has_normals = point_cloud.normals is not None
        has_conf = point_cloud.confidences is not None

        # Build PLY Header
        header_lines = [
            "ply",
            "format binary_little_endian 1.0" if binary else "format ascii 1.0",
            "comment Matrix SIH2026 S3 Reconstruction Point Cloud",
            f"element vertex {n_points}",
            "property float x",
            "property float y",
            "property float z",
        ]

        if has_normals:
            header_lines.extend([
                "property float nx",
                "property float ny",
                "property float nz",
            ])

        if has_colors:
            header_lines.extend([
                "property uchar red",
                "property uchar green",
                "property uchar blue",
            ])

        if has_conf:
            header_lines.append("property float confidence")

        header_lines.append("end_header\n")
        header_text = "\n".join(header_lines)

        if binary:
            dtype_fields = [("x", "<f4"), ("y", "<f4"), ("z", "<f4")]
            if has_normals:
                dtype_fields.extend([("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4")])
            if has_colors:
                dtype_fields.extend([("red", "u1"), ("green", "u1"), ("blue", "u1")])
            if has_conf:
                dtype_fields.append(("confidence", "<f4"))

            structured_arr = np.empty(n_points, dtype=dtype_fields)
            structured_arr["x"] = points[:, 0].astype(np.float32)
            structured_arr["y"] = points[:, 1].astype(np.float32)
            structured_arr["z"] = points[:, 2].astype(np.float32)

            if has_normals:
                structured_arr["nx"] = point_cloud.normals[:, 0].astype(np.float32)
                structured_arr["ny"] = point_cloud.normals[:, 1].astype(np.float32)
                structured_arr["nz"] = point_cloud.normals[:, 2].astype(np.float32)

            if has_colors:
                structured_arr["red"] = point_cloud.colors[:, 0].astype(np.uint8)
                structured_arr["green"] = point_cloud.colors[:, 1].astype(np.uint8)
                structured_arr["blue"] = point_cloud.colors[:, 2].astype(np.uint8)

            if has_conf:
                structured_arr["confidence"] = point_cloud.confidences.astype(np.float32)

            with open(path, "wb") as f:
                f.write(header_text.encode("ascii"))
                structured_arr.tofile(f)
        else:
            with open(path, "w", encoding="ascii") as f:
                f.write(header_text)
                for i in range(n_points):
                    line_parts = [
                        f"{points[i, 0]:.6f}",
                        f"{points[i, 1]:.6f}",
                        f"{points[i, 2]:.6f}",
                    ]
                    if has_normals:
                        line_parts.extend([
                            f"{point_cloud.normals[i, 0]:.6f}",
                            f"{point_cloud.normals[i, 1]:.6f}",
                            f"{point_cloud.normals[i, 2]:.6f}",
                        ])
                    if has_colors:
                        line_parts.extend([
                            str(int(point_cloud.colors[i, 0])),
                            str(int(point_cloud.colors[i, 1])),
                            str(int(point_cloud.colors[i, 2])),
                        ])
                    if has_conf:
                        line_parts.append(f"{point_cloud.confidences[i]:.4f}")
                    f.write(" ".join(line_parts) + "\n")

    @staticmethod
    def read_ply(file_path: Union[str, Path]) -> PointCloudData:
        """
        Read a .ply point cloud file into PointCloudData.

        Parameters:
            file_path: Path to the .ply file.

        Returns:
            PointCloudData instance.
        """
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"PLY file not found: {path}")

        with open(path, "rb") as f:
            # Parse header
            header_lines = []
            is_binary = False
            num_vertices = 0
            properties = []

            while True:
                line = f.readline().decode("ascii", errors="ignore").strip()
                header_lines.append(line)
                if line.startswith("format binary_little_endian"):
                    is_binary = True
                elif line.startswith("element vertex"):
                    num_vertices = int(line.split()[-1])
                elif line.startswith("property"):
                    parts = line.split()
                    properties.append((parts[1], parts[2]))
                elif line == "end_header":
                    break

            if num_vertices == 0:
                return PointCloudData(points=np.empty((0, 3), dtype=np.float64))

            prop_names = [p[1] for p in properties]

            if is_binary:
                type_map = {
                    "float": "<f4",
                    "float32": "<f4",
                    "double": "<f8",
                    "float64": "<f8",
                    "uchar": "u1",
                    "uint8": "u1",
                    "int": "<i4",
                }
                dtype_list = [(p[1], type_map.get(p[0], "<f4")) for p in properties]
                structured_data = np.fromfile(f, dtype=dtype_list, count=num_vertices)

                points = np.column_stack([
                    structured_data["x"],
                    structured_data["y"],
                    structured_data["z"]
                ]).astype(np.float64)

                colors = None
                if "red" in prop_names and "green" in prop_names and "blue" in prop_names:
                    colors = np.column_stack([
                        structured_data["red"],
                        structured_data["green"],
                        structured_data["blue"]
                    ]).astype(np.uint8)

                normals = None
                if "nx" in prop_names and "ny" in prop_names and "nz" in prop_names:
                    normals = np.column_stack([
                        structured_data["nx"],
                        structured_data["ny"],
                        structured_data["nz"]
                    ]).astype(np.float64)

                confidences = None
                if "confidence" in prop_names:
                    confidences = structured_data["confidence"].astype(np.float32)

                return PointCloudData(
                    points=points,
                    colors=colors,
                    normals=normals,
                    confidences=confidences,
                )
            else:
                # ASCII mode
                raw_lines = [line.strip().split() for line in f.read().decode("ascii").splitlines() if line.strip()]
                data_matrix = np.array(raw_lines[:num_vertices], dtype=np.float64)

                x_idx = prop_names.index("x")
                y_idx = prop_names.index("y")
                z_idx = prop_names.index("z")
                points = data_matrix[:, [x_idx, y_idx, z_idx]]

                colors = None
                if "red" in prop_names and "green" in prop_names and "blue" in prop_names:
                    r_idx = prop_names.index("red")
                    g_idx = prop_names.index("green")
                    b_idx = prop_names.index("blue")
                    colors = data_matrix[:, [r_idx, g_idx, b_idx]].astype(np.uint8)

                return PointCloudData(points=points, colors=colors)

