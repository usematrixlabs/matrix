"""S4-internal minimal PLY reader/writer.

S4 needs to read the PLY files S3 writes and write a transformed PLY of
its own. To remain isolated from S3, S4 carries a tiny self-contained
PLY helper instead of importing ``src.reconstruction.geometry.ply_io``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np


@dataclass
class PointCloudData:
    """Minimal N x 3 (+ optional attributes) point cloud."""

    points: np.ndarray
    colors: Optional[np.ndarray] = None

    @staticmethod
    def read_ply(path: Path) -> "PointCloudData":
        """Read a binary or ASCII PLY file and return a :class:`PointCloudData`."""
        path = Path(path)
        with open(path, "rb") as f:
            data = f.read()

        header_end = data.find(b"end_header\n")
        if header_end < 0:
            raise ValueError(f"PLY header terminator 'end_header\\n' not found in {path}")
        header = data[:header_end].decode("ascii", errors="ignore")
        body = data[header_end + len(b"end_header\n"):]

        is_binary = "binary_little_endian" in header or "binary_big_endian" in header
        little_endian = "binary_little_endian" in header

        vertex_count = 0
        properties: List[str] = []
        for line in header.splitlines():
            line = line.strip()
            if line.startswith("element vertex"):
                _, _, count_str = line.partition(" ")
                # count is the last token
                tokens = line.split()
                vertex_count = int(tokens[-1])
            elif line.startswith("property"):
                tokens = line.split()
                # property <type> <name>
                properties.append(tokens[-1])

        if not properties or vertex_count == 0:
            return PointCloudData(points=np.zeros((0, 3), dtype=np.float64))

        has_color = "red" in properties and "green" in properties and "blue" in properties
        color_offsets = (
            [properties.index("red"), properties.index("green"), properties.index("blue")]
            if has_color
            else None
        )
        # PLY `property float` is 32-bit (4 bytes). S3 writes ``float`` /
        # ``property float`` in its binary_little_endian PLY output, so we
        # must match that width here. Earlier versions used float64 for
        # xyz which overran the buffer.
        per_point_dtypes = []
        for name in properties:
            if name in ("x", "y", "z"):
                per_point_dtypes.append((name, "<f4"))
            elif has_color and name in ("red", "green", "blue"):
                per_point_dtypes.append((name, "u1"))
            else:
                per_point_dtypes.append((name, "<f4"))

        if is_binary:
            arr = np.frombuffer(body, dtype=per_point_dtypes, count=vertex_count)
        else:
            text = body.decode("ascii", errors="ignore")
            tokens = text.split()
            arr = np.array(tokens, dtype=np.float64).reshape(-1, len(properties))

        points = np.stack([arr["x"], arr["y"], arr["z"]], axis=-1).astype(np.float64)
        if has_color:
            colors = np.stack([arr["red"], arr["green"], arr["blue"]], axis=-1).astype(np.uint8)
        else:
            colors = None

        return PointCloudData(points=points, colors=colors)

    def write_ply(self, path: Path, binary: bool = True) -> None:
        """Write the point cloud as a binary (or ASCII) PLY file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        n = int(self.points.shape[0])
        has_color = self.colors is not None

        if binary:
            header_lines = [
                "ply",
                "format binary_little_endian 1.0",
                f"element vertex {n}",
                "property float x",
                "property float y",
                "property float z",
            ]
            if has_color:
                header_lines += [
                    "property uchar red",
                    "property uchar green",
                    "property uchar blue",
                ]
            header_lines.append("end_header")
            header = ("\n".join(header_lines) + "\n").encode("ascii")

            dtype = [("x", "<f4"), ("y", "<f4"), ("z", "<f4")]
            if has_color:
                dtype += [("red", "u1"), ("green", "u1"), ("blue", "u1")]

            arr = np.empty(n, dtype=dtype)
            arr["x"] = self.points[:, 0].astype("<f4")
            arr["y"] = self.points[:, 1].astype("<f4")
            arr["z"] = self.points[:, 2].astype("<f4")
            if has_color:
                arr["red"] = self.colors[:, 0].astype(np.uint8)
                arr["green"] = self.colors[:, 1].astype(np.uint8)
                arr["blue"] = self.colors[:, 2].astype(np.uint8)

            with open(path, "wb") as f:
                f.write(header)
                f.write(arr.tobytes())
        else:
            lines = [
                "ply",
                "format ascii 1.0",
                f"element vertex {n}",
                "property float x",
                "property float y",
                "property float z",
            ]
            if has_color:
                lines += [
                    "property uchar red",
                    "property uchar green",
                    "property uchar blue",
                ]
            lines.append("end_header")
            with open(path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
                for i in range(n):
                    if has_color:
                        f.write(
                            f"{self.points[i, 0]:.6f} {self.points[i, 1]:.6f} "
                            f"{self.points[i, 2]:.6f} "
                            f"{int(self.colors[i, 0])} {int(self.colors[i, 1])} "
                            f"{int(self.colors[i, 2])}\n"
                        )
                    else:
                        f.write(
                            f"{self.points[i, 0]:.6f} {self.points[i, 1]:.6f} "
                            f"{self.points[i, 2]:.6f}\n"
                        )


__all__ = ["PointCloudData"]
