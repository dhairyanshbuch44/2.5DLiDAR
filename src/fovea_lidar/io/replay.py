"""Frame replay controller for point cloud sequences in Fovea-LiDAR."""

from __future__ import annotations

from pathlib import Path
import time
from typing import Callable, Iterable, Iterator, List, Optional, Sequence, Union

from fovea_lidar.contracts.point_cloud import PointCloudFrame
from fovea_lidar.io.loaders import load


class ReplayController:
    """Controls sequential playback of PointCloudFrame sequences.

    Supports interactive control (play, pause, resume, seek), speed adjustments
    (0.5x, 1x, 2x), and dual operational modes:
    - As-fast-as-possible execution (ideal for batch benchmarking).
    - Paced real-time simulation (ideal for live dashboard streaming).

    Attributes
    ----------
    frames : List[PointCloudFrame]
        Sequence of point cloud frames to replay.
    speed : float
        Playback speed multiplier (> 0.0), default 1.0.
    loop : bool
        Whether to loop back to frame 0 upon reaching the end.
    """

    def __init__(
        self,
        frames: Union[Sequence[PointCloudFrame], str, Path],
        speed: float = 1.0,
        loop: bool = False,
    ) -> None:
        if isinstance(frames, (str, Path)):
            self.frames = self._load_from_directory(Path(frames))
        else:
            self.frames = list(frames)

        self.set_speed(speed)
        self.loop = loop
        self._current_index: int = 0
        self._is_playing: bool = False
        self._is_paused: bool = False

    @staticmethod
    def _load_from_directory(directory: Path) -> List[PointCloudFrame]:
        """Load and sort all PCD/PLY point cloud files in a directory."""
        if not directory.is_dir():
            raise FileNotFoundError(f"Replay source directory not found: {directory}")

        files = sorted(
            [f for f in directory.iterdir() if f.suffix.lower() in (".pcd", ".ply")],
            key=lambda x: x.name,
        )
        if not files:
            raise ValueError(f"No .pcd or .ply files found in directory: {directory}")

        loaded_frames: List[PointCloudFrame] = []
        for idx, file in enumerate(files):
            fr = load(file)
            if fr.timestamp == 0.0 and idx > 0:
                fr.timestamp = round(idx * 0.1, 3)
            loaded_frames.append(fr)
        return loaded_frames

    @property
    def current_index(self) -> int:
        """Current frame index (0-indexed)."""
        return self._current_index

    @property
    def total_frames(self) -> int:
        """Total number of frames available."""
        return len(self.frames)

    @property
    def is_playing(self) -> bool:
        """True if playback is active and not paused."""
        return self._is_playing and not self._is_paused

    @property
    def is_paused(self) -> bool:
        """True if playback is currently paused."""
        return self._is_paused

    def set_speed(self, speed: float) -> None:
        """Configure playback speed multiplier."""
        speed_val = float(speed)
        if speed_val <= 0.0:
            raise ValueError(f"Replay speed must be strictly positive, got {speed_val}")
        self.speed = speed_val

    def play(
        self,
        callback: Optional[Callable[[PointCloudFrame, int], None]] = None,
        real_time: bool = False,
    ) -> None:
        """Start or resume playback.

        Parameters
        ----------
        callback : Callable[[PointCloudFrame, int], None], optional
            Callback function invoked on each frame (frame, frame_idx).
        real_time : bool
            If True, pauses between frames to match timestamp delta scaled by speed.
            If False, delivers frames as fast as possible.
        """
        self._is_playing = True
        self._is_paused = False

        if callback is not None:
            prev_timestamp: Optional[float] = None
            while self.is_playing and self._current_index < len(self.frames):
                frame = self.frames[self._current_index]

                if real_time and prev_timestamp is not None:
                    delta_t = (frame.timestamp - prev_timestamp) / self.speed
                    if delta_t > 0:
                        time.sleep(delta_t)

                callback(frame, self._current_index)
                prev_timestamp = frame.timestamp
                self._current_index += 1

                if self._current_index >= len(self.frames) and self.loop:
                    self._current_index = 0
                    prev_timestamp = None

            if self._current_index >= len(self.frames):
                self._is_playing = False

    def pause(self) -> None:
        """Pause playback at the current frame index."""
        if self._is_playing:
            self._is_paused = True

    def resume(self) -> None:
        """Resume playback from paused state."""
        if self._is_paused:
            self._is_paused = False
            self._is_playing = True

    def seek(self, frame_index: int) -> int:
        """Jump to a specific frame index, clamping within valid range [0, total_frames - 1]."""
        if self.total_frames == 0:
            self._current_index = 0
            return 0
        clamped = max(0, min(int(frame_index), self.total_frames - 1))
        self._current_index = clamped
        return self._current_index

    def step(self) -> Optional[PointCloudFrame]:
        """Advance by exactly one frame and return it (None if EOF)."""
        if self.total_frames == 0:
            return None

        if self._current_index >= self.total_frames:
            if self.loop:
                self._current_index = 0
            else:
                self._is_playing = False
                return None

        frame = self.frames[self._current_index]
        self._current_index += 1

        if self._current_index >= self.total_frames and not self.loop:
            self._is_playing = False

        return frame

    def rewind(self) -> None:
        """Rewind to the first frame."""
        self.seek(0)

    def __iter__(self) -> Iterator[PointCloudFrame]:
        """Iterate through remaining frames."""
        while True:
            frame = self.step()
            if frame is None:
                break
            yield frame
