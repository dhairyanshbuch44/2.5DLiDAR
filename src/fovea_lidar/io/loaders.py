"""Point cloud loaders for PCD and PLY files in Fovea-LiDAR."""

from __future__ import annotations

import os
from pathlib import Path
import re
import struct
from typing import Any, Dict, Optional, Union
import numpy as np

from fovea_lidar.contracts.point_cloud import FrameSource, PointCloudFrame

# Optional Open3D import
try:
    import open3d as o3d
except ImportError:
    o3d = None


class PointCloudIOError(Exception):
    """Base exception for point cloud I/O errors."""


class PointCloudFileNotFoundError(PointCloudIOError, FileNotFoundError):
    """Raised when a requested point cloud file does not exist."""


class CorruptFileError(PointCloudIOError, ValueError):
    """Raised when a point cloud file has invalid syntax or truncated data."""


class EmptyPointCloudError(PointCloudIOError, ValueError):
    """Raised when a point cloud file contains zero points."""


# ==============================================================================
# PCD Loader & Writer
# ==============================================================================

def load_pcd(path: Union[str, Path]) -> PointCloudFrame:
    """Load a PointCloudFrame from a PCD (Point Cloud Data) file.

    Parameters
    ----------
    path : str or Path
        Path to the .pcd file.

    Returns
    -------
    PointCloudFrame
        Instantiated frame with points in meters.

    Raises
    ------
    PointCloudFileNotFoundError
        If the file does not exist.
    CorruptFileError
        If the file header or data is malformed.
    EmptyPointCloudError
        If the point cloud contains zero points.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise PointCloudFileNotFoundError(f"PCD file not found: {file_path}")

    try:
        with open(file_path, "rb") as f:
            header_lines: list[str] = []
            while True:
                line = f.readline()
                if not line:
                    raise CorruptFileError(f"Truncated header in PCD file: {file_path}")
                decoded_line = line.decode("ascii", errors="replace").strip()
                header_lines.append(decoded_line)
                if decoded_line.startswith("DATA"):
                    break

            # Parse header fields
            fields: list[str] = []
            size: list[int] = []
            type_code: list[str] = []
            count: list[int] = []
            num_points = 0
            data_type = "ascii"

            for hline in header_lines:
                if hline.startswith("#") or not hline:
                    continue
                tokens = hline.split()
                key = tokens[0].upper()
                if key == "FIELDS":
                    fields = tokens[1:]
                elif key == "SIZE":
                    size = [int(x) for x in tokens[1:]]
                elif key == "TYPE":
                    type_code = tokens[1:]
                elif key == "COUNT":
                    count = [int(x) for x in tokens[1:]]
                elif key == "POINTS":
                    num_points = int(tokens[1])
                elif key == "DATA":
                    data_type = tokens[1].lower()

            if not fields:
                raise CorruptFileError(f"Missing FIELDS header in PCD file: {file_path}")
            if num_points == 0:
                raise EmptyPointCloudError(f"PCD file '{file_path}' contains zero points.")

            # Identify coordinate indices
            fields_lower = [f.lower() for f in fields]
            if "x" not in fields_lower or "y" not in fields_lower or "z" not in fields_lower:
                raise CorruptFileError(f"PCD file '{file_path}' must contain x, y, and z fields.")

            x_idx = fields_lower.index("x")
            y_idx = fields_lower.index("y")
            z_idx = fields_lower.index("z")
            has_intensity = "intensity" in fields_lower or "i" in fields_lower
            i_idx = fields_lower.index("intensity") if "intensity" in fields_lower else (
                fields_lower.index("i") if "i" in fields_lower else -1
            )

            # Read data body
            if data_type == "ascii":
                content = f.read().decode("ascii", errors="replace").strip()
                if not content:
                    raise EmptyPointCloudError(f"PCD file '{file_path}' has empty data section.")
                lines = content.splitlines()
                parsed_rows = []
                for line_idx, l in enumerate(lines):
                    tokens = l.strip().split()
                    if not tokens:
                        continue
                    try:
                        vals = [float(t) for t in tokens]
                    except ValueError as e:
                        raise CorruptFileError(f"Corrupt number on data line {line_idx+1} in {file_path}: {e}") from e
                    if len(vals) < len(fields):
                        raise CorruptFileError(
                            f"Line {line_idx+1} in {file_path} has {len(vals)} values, expected at least {len(fields)}"
                        )

                    x, y, z = vals[x_idx], vals[y_idx], vals[z_idx]
                    if has_intensity and i_idx >= 0 and i_idx < len(vals):
                        parsed_rows.append([x, y, z, vals[i_idx]])
                    else:
                        parsed_rows.append([x, y, z])

                if not parsed_rows:
                    raise EmptyPointCloudError(f"PCD file '{file_path}' contains zero valid points.")
                points = np.array(parsed_rows, dtype=np.float32)

            elif data_type == "binary":
                # Assuming 4-byte floats for simplicity
                num_fields = len(fields)
                point_dtype = np.dtype([("f" + str(i), np.float32) for i in range(num_fields)])
                raw_bytes = f.read()
                expected_bytes = num_points * num_fields * 4
                if len(raw_bytes) < expected_bytes:
                    raise CorruptFileError(
                        f"Binary PCD data truncated: expected {expected_bytes} bytes, found {len(raw_bytes)}"
                    )
                data_arr = np.frombuffer(raw_bytes[:expected_bytes], dtype=np.float32).reshape((num_points, num_fields))
                if has_intensity and i_idx >= 0:
                    points = np.column_stack([data_arr[:, x_idx], data_arr[:, y_idx], data_arr[:, z_idx], data_arr[:, i_idx]]).astype(np.float32)
                else:
                    points = np.column_stack([data_arr[:, x_idx], data_arr[:, y_idx], data_arr[:, z_idx]]).astype(np.float32)
            else:
                raise CorruptFileError(f"Unsupported PCD DATA type: '{data_type}' in {file_path}")

    except (PointCloudIOError, FileNotFoundError):
        raise
    except Exception as exc:
        raise CorruptFileError(f"Failed to parse PCD file '{file_path}': {exc}") from exc

    if points.shape[0] == 0:
        raise EmptyPointCloudError(f"PCD file '{file_path}' contains zero points.")

    return PointCloudFrame(
        points=points,
        timestamp=0.0,
        frame_id=file_path.stem,
        source=FrameSource.REPLAY,
        metadata={"file_path": str(file_path), "format": "pcd"},
    )


def save_pcd(frame: PointCloudFrame, path: Union[str, Path]) -> None:
    """Save a PointCloudFrame to an ASCII PCD file.

    Parameters
    ----------
    frame : PointCloudFrame
        Frame containing points (Nx3 or Nx4).
    path : str or Path
        Destination path.
    """
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    points = frame.points
    num_points = points.shape[0]
    has_intensity = points.shape[1] == 4

    fields_str = "x y z intensity" if has_intensity else "x y z"
    size_str = "4 4 4 4" if has_intensity else "4 4 4"
    type_str = "F F F F" if has_intensity else "F F F"
    count_str = "1 1 1 1" if has_intensity else "1 1 1"

    header = (
        "# .PCD v0.7 - Point Cloud Data\n"
        "VERSION 0.7\n"
        f"FIELDS {fields_str}\n"
        f"SIZE {size_str}\n"
        f"TYPE {type_str}\n"
        f"COUNT {count_str}\n"
        f"WIDTH {num_points}\n"
        "HEIGHT 1\n"
        "VIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {num_points}\n"
        "DATA ascii\n"
    )

    with open(file_path, "w", encoding="ascii") as f:
        f.write(header)
        for row in points:
            if has_intensity:
                f.write(f"{row[0]:.4f} {row[1]:.4f} {row[2]:.4f} {row[3]:.4f}\n")
            else:
                f.write(f"{row[0]:.4f} {row[1]:.4f} {row[2]:.4f}\n")


# ==============================================================================
# PLY Loader & Writer
# ==============================================================================

def load_ply(path: Union[str, Path]) -> PointCloudFrame:
    """Load a PointCloudFrame from a PLY (Polygon File Format) file.

    Parameters
    ----------
    path : str or Path
        Path to the .ply file.

    Returns
    -------
    PointCloudFrame
        Instantiated frame with points in meters.

    Raises
    ------
    PointCloudFileNotFoundError
        If the file does not exist.
    CorruptFileError
        If the file header or data is malformed.
    EmptyPointCloudError
        If the point cloud contains zero points.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise PointCloudFileNotFoundError(f"PLY file not found: {file_path}")

    try:
        with open(file_path, "rb") as f:
            first_line = f.readline().decode("ascii", errors="replace").strip()
            if first_line != "ply":
                raise CorruptFileError(f"Invalid PLY header magic in '{file_path}': '{first_line}'")

            format_type = "ascii"
            num_vertices = 0
            properties: list[tuple[str, str]] = []
            in_vertex_element = False

            while True:
                line = f.readline()
                if not line:
                    raise CorruptFileError(f"Truncated PLY header in '{file_path}'")
                sline = line.decode("ascii", errors="replace").strip()
                if sline == "end_header":
                    break
                tokens = sline.split()
                if not tokens:
                    continue
                if tokens[0] == "format":
                    format_type = tokens[1].lower()
                elif tokens[0] == "element":
                    if tokens[1] == "vertex":
                        in_vertex_element = True
                        num_vertices = int(tokens[2])
                    else:
                        in_vertex_element = False
                elif tokens[0] == "property" and in_vertex_element:
                    properties.append((tokens[1], tokens[2]))

            if num_vertices == 0:
                raise EmptyPointCloudError(f"PLY file '{file_path}' contains zero vertices.")

            prop_names = [p[1].lower() for p in properties]
            if "x" not in prop_names or "y" not in prop_names or "z" not in prop_names:
                raise CorruptFileError(f"PLY vertex definition in '{file_path}' must contain x, y, and z.")

            x_idx = prop_names.index("x")
            y_idx = prop_names.index("y")
            z_idx = prop_names.index("z")
            has_intensity = "intensity" in prop_names or "i" in prop_names
            i_idx = prop_names.index("intensity") if "intensity" in prop_names else (
                prop_names.index("i") if "i" in prop_names else -1
            )

            if format_type == "ascii":
                content = f.read().decode("ascii", errors="replace").strip()
                if not content:
                    raise EmptyPointCloudError(f"PLY file '{file_path}' has empty data section.")
                lines = content.splitlines()
                parsed_rows = []
                for line_idx, l in enumerate(lines[:num_vertices]):
                    tokens = l.strip().split()
                    if not tokens:
                        continue
                    try:
                        vals = [float(t) for t in tokens]
                    except ValueError as e:
                        raise CorruptFileError(f"Corrupt number on data line {line_idx+1} in {file_path}: {e}") from e
                    if len(vals) < len(properties):
                        raise CorruptFileError(f"Line {line_idx+1} has {len(vals)} properties, expected {len(properties)}")
                    x, y, z = vals[x_idx], vals[y_idx], vals[z_idx]
                    if has_intensity and i_idx >= 0 and i_idx < len(vals):
                        parsed_rows.append([x, y, z, vals[i_idx]])
                    else:
                        parsed_rows.append([x, y, z])

                if not parsed_rows:
                    raise EmptyPointCloudError(f"PLY file '{file_path}' contains zero points.")
                points = np.array(parsed_rows, dtype=np.float32)

            elif format_type in ("binary_little_endian", "binary_big_endian"):
                endian = "<" if format_type == "binary_little_endian" else ">"
                type_map = {
                    "float": "f",
                    "float32": "f",
                    "float64": "d",
                    "double": "d",
                    "int": "i",
                    "int32": "i",
                    "uchar": "B",
                    "uint8": "B",
                    "char": "b",
                    "int8": "b",
                    "ushort": "H",
                    "uint16": "H",
                    "short": "h",
                    "int16": "h",
                    "uint": "I",
                    "uint32": "I",
                }
                struct_fmt = endian + "".join(type_map.get(p[0].lower(), "f") for p in properties)
                record_size = struct.calcsize(struct_fmt)
                raw_bytes = f.read(record_size * num_vertices)
                if len(raw_bytes) < record_size * num_vertices:
                    raise CorruptFileError(f"Truncated binary PLY data in '{file_path}'")

                parsed_rows = []
                for i in range(num_vertices):
                    record = struct.unpack_from(struct_fmt, raw_bytes, i * record_size)
                    x, y, z = float(record[x_idx]), float(record[y_idx]), float(record[z_idx])
                    if has_intensity and i_idx >= 0:
                        parsed_rows.append([x, y, z, float(record[i_idx])])
                    else:
                        parsed_rows.append([x, y, z])
                points = np.array(parsed_rows, dtype=np.float32)
            else:
                raise CorruptFileError(f"Unsupported PLY format '{format_type}' in '{file_path}'")

    except (PointCloudIOError, FileNotFoundError):
        raise
    except Exception as exc:
        raise CorruptFileError(f"Failed to parse PLY file '{file_path}': {exc}") from exc

    if points.shape[0] == 0:
        raise EmptyPointCloudError(f"PLY file '{file_path}' contains zero points.")

    return PointCloudFrame(
        points=points,
        timestamp=0.0,
        frame_id=file_path.stem,
        source=FrameSource.REPLAY,
        metadata={"file_path": str(file_path), "format": "ply"},
    )


def save_ply(frame: PointCloudFrame, path: Union[str, Path]) -> None:
    """Save a PointCloudFrame to an ASCII PLY file.

    Parameters
    ----------
    frame : PointCloudFrame
        Frame containing points (Nx3 or Nx4).
    path : str or Path
        Destination path.
    """
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    points = frame.points
    num_points = points.shape[0]
    has_intensity = points.shape[1] == 4

    header = (
        "ply\n"
        "format ascii 1.0\n"
        "comment Fovea-LiDAR generated\n"
        f"element vertex {num_points}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
    )
    if has_intensity:
        header += "property float intensity\n"
    header += "end_header\n"

    with open(file_path, "w", encoding="ascii") as f:
        f.write(header)
        for row in points:
            if has_intensity:
                f.write(f"{row[0]:.4f} {row[1]:.4f} {row[2]:.4f} {row[3]:.4f}\n")
            else:
                f.write(f"{row[0]:.4f} {row[1]:.4f} {row[2]:.4f}\n")


# ==============================================================================
# Generic Dispatch Loader
# ==============================================================================

def load(path: Union[str, Path]) -> PointCloudFrame:
    """Load a PointCloudFrame by dispatching on file extension (.pcd or .ply).

    Parameters
    ----------
    path : str or Path
        Path to point cloud file.

    Returns
    -------
    PointCloudFrame
        Loaded frame.

    Raises
    ------
    ValueError
        If the file format is not supported.
    """
    file_path = Path(path)
    ext = file_path.suffix.lower()
    if ext == ".pcd":
        return load_pcd(file_path)
    elif ext == ".ply":
        return load_ply(file_path)
    else:
        raise ValueError(
            f"Unsupported point cloud format: '{ext}'. Supported formats: .pcd, .ply"
        )
