"""Unit tests for ReplayController playback, seek, speed, and pause behavior."""

from pathlib import Path
import numpy as np
import pytest

from fovea_lidar.contracts.point_cloud import FrameSource, PointCloudFrame
from fovea_lidar.io.replay import ReplayController


@pytest.fixture
def sample_frames() -> list[PointCloudFrame]:
    """Create 5 dummy PointCloudFrame instances."""
    frames = []
    for i in range(5):
        pts = np.array([[float(i), 0.0, 0.0], [float(i), 1.0, 0.0]], dtype=np.float32)
        frames.append(
            PointCloudFrame(
                points=pts,
                timestamp=i * 0.1,
                frame_id=f"frame_{i}",
                source=FrameSource.MOCK,
            )
        )
    return frames


def test_replay_step_and_iteration(sample_frames: list[PointCloudFrame]):
    """Test step() and iterator interfaces."""
    replay = ReplayController(sample_frames)
    assert replay.total_frames == 5
    assert replay.current_index == 0

    # Step through
    f0 = replay.step()
    assert f0 is not None
    assert f0.frame_id == "frame_0"
    assert replay.current_index == 1

    # Remaining frames via iteration
    remaining = list(replay)
    assert len(remaining) == 4
    assert replay.current_index == 5
    assert replay.step() is None


def test_replay_seek_and_clamp(sample_frames: list[PointCloudFrame]):
    """Test seek navigation with bounds clamping."""
    replay = ReplayController(sample_frames)

    # Valid seek
    idx = replay.seek(3)
    assert idx == 3
    assert replay.current_index == 3
    frame = replay.step()
    assert frame is not None
    assert frame.frame_id == "frame_3"

    # Clamp negative seek
    idx_neg = replay.seek(-10)
    assert idx_neg == 0
    assert replay.current_index == 0

    # Clamp overflow seek
    idx_over = replay.seek(9999)
    assert idx_over == 4
    assert replay.current_index == 4


def test_replay_speed_configuration(sample_frames: list[PointCloudFrame]):
    """Test setting speed multipliers and error on invalid values."""
    replay = ReplayController(sample_frames, speed=1.0)
    assert replay.speed == 1.0

    replay.set_speed(2.0)
    assert replay.speed == 2.0

    replay.set_speed(0.5)
    assert replay.speed == 0.5

    with pytest.raises(ValueError):
        replay.set_speed(0.0)

    with pytest.raises(ValueError):
        replay.set_speed(-1.5)


def test_replay_play_fast_mode(sample_frames: list[PointCloudFrame]):
    """Test playing all frames as-fast-as-possible via callback."""
    replay = ReplayController(sample_frames)
    delivered: list[tuple[str, int]] = []

    def on_frame(frame: PointCloudFrame, idx: int):
        delivered.append((frame.frame_id, idx))

    replay.play(callback=on_frame, real_time=False)

    assert len(delivered) == 5
    assert delivered[0] == ("frame_0", 0)
    assert delivered[-1] == ("frame_4", 4)
    assert replay.current_index == 5
    assert not replay.is_playing


def test_replay_pause_and_resume(sample_frames: list[PointCloudFrame]):
    """Test pausing and resuming replay state."""
    replay = ReplayController(sample_frames)
    replay.play()
    assert replay.is_playing is True

    replay.pause()
    assert replay.is_paused is True
    assert replay.is_playing is False

    replay.resume()
    assert replay.is_paused is False
    assert replay.is_playing is True


def test_replay_from_directory(tmp_path: Path, sample_frames: list[PointCloudFrame]):
    """Test initializing ReplayController directly from a directory of PCD files."""
    from fovea_lidar.io.loaders import save_pcd

    for idx, f in enumerate(sample_frames):
        save_pcd(f, tmp_path / f"frame_{idx:03d}.pcd")

    replay = ReplayController(tmp_path)
    assert replay.total_frames == 5
    first = replay.step()
    assert first is not None
    assert first.frame_id == "frame_000"
