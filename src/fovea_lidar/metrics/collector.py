"""Metrics collector for stage timings, throughput, counts, and rolling averages."""

from __future__ import annotations

from collections import deque
import time
from typing import Any, Deque, Dict, List, Optional, Union
import numpy as np


class MetricsCollector:
    """Collects and aggregates per-frame execution metrics across the pipeline.

    Maintains bounded temporal history to provide both single-frame telemetry
    and windowed rolling averages for latency, FPS, and data counts.

    Attributes
    ----------
    max_history : int
        Maximum number of historic frame records retained in the rolling buffer.
    """

    def __init__(self, max_history: int = 1000) -> None:
        self.max_history = int(max_history)
        self._history: Deque[Dict[str, Any]] = deque(maxlen=self.max_history)
        self._frame_counter: int = 0

    @property
    def total_frames(self) -> int:
        """Total number of frames recorded since initialization or last reset."""
        return self._frame_counter

    def reset(self) -> None:
        """Clear all metric records and reset the frame counter."""
        self._history.clear()
        self._frame_counter = 0

    def record_frame(
        self,
        stage_timings: Dict[str, float],
        counts: Dict[str, Union[int, float]],
    ) -> None:
        """Record telemetry for a completed frame processing iteration.

        Parameters
        ----------
        stage_timings : Dict[str, float]
            Execution durations in milliseconds (ms) for each pipeline stage.
            Expected keys: 'preprocessing_ms', 'perception_ms', 'tracking_ms',
            'resolution_ms', 'mapping_ms', 'total_ms', etc.
        counts : Dict[str, Union[int, float]]
            Data item counts for the frame (e.g. 'raw_points', 'cleaned_points',
            'tracked_objects', 'active_foveas', 'total_cells').
        """
        self._frame_counter += 1
        total_ms = float(stage_timings.get("total_ms", 0.0))
        fps = (1000.0 / total_ms) if total_ms > 0.0 else 0.0

        record: Dict[str, Any] = {
            "frame_index": self._frame_counter,
            "recorded_at": time.perf_counter(),
            "stage_timings": {k: float(v) for k, v in stage_timings.items()},
            "counts": {k: float(v) for k, v in counts.items()},
            "fps": round(fps, 2),
        }
        self._history.append(record)

    def get_current(self) -> Dict[str, Any]:
        """Retrieve telemetry for the most recently recorded frame.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing stage timings, counts, and instantaneous FPS.
            Returns empty template dictionary if no frames have been recorded yet.
        """
        if not self._history:
            return {
                "frame_index": 0,
                "fps": 0.0,
                "stage_timings": {},
                "counts": {},
                "total_ms": 0.0,
            }

        latest = self._history[-1]
        timings = latest["stage_timings"]
        counts = latest["counts"]

        out: Dict[str, Any] = {
            "frame_index": latest["frame_index"],
            "fps": latest["fps"],
            "stage_timings": dict(timings),
            "counts": dict(counts),
        }
        # Flat accessors for convenience
        out.update(timings)
        out.update(counts)
        return out

    def get_rolling_average(self, window: int = 10) -> Dict[str, Any]:
        """Compute rolling averages over the last N recorded frames.

        Parameters
        ----------
        window : int
            Number of recent frames to average over (defaults to 10).

        Returns
        -------
        Dict[str, Any]
            Dictionary containing windowed average stage timings, counts, and FPS.
        """
        if not self._history:
            return {
                "window": 0,
                "total_frames": 0,
                "fps": 0.0,
                "stage_timings": {},
                "counts": {},
                "total_ms": 0.0,
            }

        w = max(1, int(window))
        subset = list(self._history)[-w:]
        actual_window = len(subset)

        # Average stage timings
        all_timing_keys = set()
        for r in subset:
            all_timing_keys.update(r["stage_timings"].keys())

        avg_timings: Dict[str, float] = {}
        for k in all_timing_keys:
            vals = [r["stage_timings"].get(k, 0.0) for r in subset]
            avg_timings[k] = round(float(np.mean(vals)), 3)

        # Average counts
        all_count_keys = set()
        for r in subset:
            all_count_keys.update(r["counts"].keys())

        avg_counts: Dict[str, float] = {}
        for k in all_count_keys:
            vals = [r["counts"].get(k, 0.0) for r in subset]
            avg_counts[k] = round(float(np.mean(vals)), 2)

        mean_total_ms = avg_timings.get("total_ms", 0.0)
        rolling_fps = (1000.0 / mean_total_ms) if mean_total_ms > 0.0 else 0.0

        out: Dict[str, Any] = {
            "window": actual_window,
            "total_frames": self._frame_counter,
            "fps": round(rolling_fps, 2),
            "stage_timings": avg_timings,
            "counts": avg_counts,
        }
        # Flat accessors for easy inspection
        out.update(avg_timings)
        out.update(avg_counts)
        return out
