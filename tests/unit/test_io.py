"""Unit tests for point cloud loaders and file I/O."""

from pathlib import Path
import numpy as np
import pytest

from fovea_lidar.contracts.point_cloud import FrameSource, PointCloudFrame
from fovea_lidar.io.loaders import (
    CorruptFileError,
    EmptyPointCloudError,
    PointCloudFileNotFoundError,
    load,
    load_pcd,
    load_ply,
    save_pcd,
    save_ply,
)


@pytest.fixture
def sample_frame() -> PointCloudFrame:
    """Fixture providing a standard 4-point cloud with intensity."""
    points = np.array(
        [
            [1.0, 2.0, 3.0, 0.5],
            [4.0, 5.0, 6.0, 0.75],
            [7.0, 8.0, 9.0, 0.1],
            [10.0, 11.0, 12.0, 0.9],
        ],
        dtype=np.float32,
    )
    return PointCloudFrame(
        points=points,
        timestamp=12.34,
        frame_id="test_sensor",
        source=FrameSource.MOCK,
    )


def test_pcd_save_and_load_roundtrip(tmp_path: Path, sample_frame: PointCloudFrame):
    """Test saving and loading a valid PCD file."""
    pcd_path = tmp_path / "sample.pcd"
    save_pcd(sample_frame, pcd_path)
    assert pcd_path.is_file()

    loaded = load_pcd(pcd_path)
    assert isinstance(loaded, PointCloudFrame)
    assert loaded.points.shape == sample_frame.points.shape
    assert np.allclose(loaded.points, sample_frame.points, atol=1e-3)
    assert loaded.source == FrameSource.REPLAY


def test_ply_save_and_load_roundtrip(tmp_path: Path, sample_frame: PointCloudFrame):
    """Test saving and loading a valid PLY file."""
    ply_path = tmp_path / "sample.ply"
    save_ply(sample_frame, ply_path)
    assert ply_path.is_file()

    loaded = load_ply(ply_path)
    assert isinstance(loaded, PointCloudFrame)
    assert loaded.points.shape == sample_frame.points.shape
    assert np.allclose(loaded.points, sample_frame.points, atol=1e-3)
    assert loaded.source == FrameSource.REPLAY


def test_generic_load_dispatcher(tmp_path: Path, sample_frame: PointCloudFrame):
    """Test generic load() routing between .pcd and .ply."""
    pcd_path = tmp_path / "cloud.pcd"
    ply_path = tmp_path / "cloud.ply"
    save_pcd(sample_frame, pcd_path)
    save_ply(sample_frame, ply_path)

    pcd_loaded = load(pcd_path)
    ply_loaded = load(ply_path)

    assert pcd_loaded.points.shape == (4, 4)
    assert ply_loaded.points.shape == (4, 4)

    # Test unsupported extension
    bad_path = tmp_path / "cloud.xyz"
    with pytest.raises(ValueError) as exc:
        load(bad_path)
    assert "unsupported" in str(exc.value).lower()


def test_missing_file_raises_not_found(tmp_path: Path):
    """Test that missing files raise PointCloudFileNotFoundError."""
    missing_pcd = tmp_path / "non_existent.pcd"
    missing_ply = tmp_path / "non_existent.ply"

    with pytest.raises(PointCloudFileNotFoundError):
        load_pcd(missing_pcd)

    with pytest.raises(PointCloudFileNotFoundError):
        load_ply(missing_ply)

    with pytest.raises(PointCloudFileNotFoundError):
        load(missing_pcd)


def test_corrupt_pcd_raises_corrupt_error(tmp_path: Path):
    """Test that malformed/corrupt PCD files raise CorruptFileError."""
    # 1. Truncated header
    truncated_pcd = tmp_path / "truncated.pcd"
    truncated_pcd.write_text("VERSION 0.7\nFIELDS x y z\n", encoding="ascii")
    with pytest.raises(CorruptFileError):
        load_pcd(truncated_pcd)

    # 2. Corrupt floating point data
    corrupt_data_pcd = tmp_path / "corrupt_data.pcd"
    corrupt_data_pcd.write_text(
        "VERSION 0.7\n"
        "FIELDS x y z\n"
        "SIZE 4 4 4\n"
        "TYPE F F F\n"
        "COUNT 1 1 1\n"
        "WIDTH 1\n"
        "HEIGHT 1\n"
        "POINTS 1\n"
        "DATA ascii\n"
        "1.0 NOT_A_NUMBER 3.0\n",
        encoding="ascii",
    )
    with pytest.raises(CorruptFileError):
        load_pcd(corrupt_data_pcd)


def test_corrupt_ply_raises_corrupt_error(tmp_path: Path):
    """Test that malformed/corrupt PLY files raise CorruptFileError."""
    # 1. Bad magic header
    bad_magic = tmp_path / "bad_magic.ply"
    bad_magic.write_text("invalid_header\nformat ascii 1.0\nend_header\n", encoding="ascii")
    with pytest.raises(CorruptFileError):
        load_ply(bad_magic)

    # 2. Corrupt numeric data
    corrupt_ply = tmp_path / "corrupt.ply"
    corrupt_ply.write_text(
        "ply\n"
        "format ascii 1.0\n"
        "element vertex 1\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "end_header\n"
        "1.0 2.0 NOT_A_FLOAT\n",
        encoding="ascii",
    )
    with pytest.raises(CorruptFileError):
        load_ply(corrupt_ply)


def test_empty_point_cloud_raises_empty_error(tmp_path: Path):
    """Test that zero-point PCD and PLY files raise EmptyPointCloudError."""
    # Empty PCD (POINTS 0)
    empty_pcd = tmp_path / "empty.pcd"
    empty_pcd.write_text(
        "VERSION 0.7\n"
        "FIELDS x y z\n"
        "SIZE 4 4 4\n"
        "TYPE F F F\n"
        "COUNT 1 1 1\n"
        "WIDTH 0\n"
        "HEIGHT 1\n"
        "POINTS 0\n"
        "DATA ascii\n",
        encoding="ascii",
    )
    with pytest.raises(EmptyPointCloudError):
        load_pcd(empty_pcd)

    # Empty PLY (element vertex 0)
    empty_ply = tmp_path / "empty.ply"
    empty_ply.write_text(
        "ply\n"
        "format ascii 1.0\n"
        "element vertex 0\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "end_header\n",
        encoding="ascii",
    )
    with pytest.raises(EmptyPointCloudError):
        load_ply(empty_ply)
