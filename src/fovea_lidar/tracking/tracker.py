"""Multi-object tracking engine with kinematic Kalman filtering and stale track decay."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union
import numpy as np
from scipy.optimize import linear_sum_assignment

from fovea_lidar.contracts.objects import DetectedObject
from fovea_lidar.contracts.tracking import TrackedObject
from fovea_lidar.tracking.kalman import ConstantVelocityKalmanFilter
from fovea_lidar.utils.config import Config, load_config


class TrackState(str, Enum):
    """Lifecycle state of a tracked object in the scene."""

    OBSERVED = "OBSERVED"  # Actively detected in current scan frame
    STALE = "STALE"        # Coasting without observation (occluded/missed)
    REMOVED = "REMOVED"    # Exceeded missed frame budget or confidence threshold


class _Track:
    """Internal track representation managing state, covariance, and lifecycle."""

    def __init__(
        self,
        track_id: str,
        initial_det: DetectedObject,
        timestamp: float,
        process_noise_std: float = 1.0,
        measurement_noise_std: float = 0.15,
        max_history_len: int = 50,
    ) -> None:
        self.track_id = str(track_id)
        self.semantic_class = str(initial_det.semantic_class)
        self.bounding_region = initial_det.bounding_region
        self.confidence = float(initial_det.confidence)
        self.track_confidence = 1.0
        self.last_seen = float(timestamp)
        self.missed_frames = 0
        self.hits = 1
        self.state = TrackState.OBSERVED
        self.is_dynamic = bool(initial_det.dynamic)
        self.consecutive_moving_frames = 0
        self.consecutive_static_frames = 0

        # Initialize Kalman filter with initial detection position
        self.kf = ConstantVelocityKalmanFilter(
            initial_position=initial_det.position,
            initial_velocity=(0.0, 0.0, 0.0),
            process_noise_std=process_noise_std,
            measurement_noise_std=measurement_noise_std,
        )

        self.position_history: List[Tuple[float, float, float]] = [self.kf.position]

    def add_position(self, pos: Tuple[float, float, float], max_history_len: int) -> None:
        """Append position to history, keeping bounded length."""
        self.position_history.append((float(pos[0]), float(pos[1]), float(pos[2])))
        if len(self.position_history) > max_history_len:
            self.position_history.pop(0)

    def predict(self, dt: float) -> None:
        """Advance track state under constant-velocity dynamics."""
        self.kf.predict(dt)

    def update_matched(
        self,
        det: DetectedObject,
        timestamp: float,
        static_speed_threshold: float,
        min_moving_frames: int,
        max_history_len: int,
    ) -> None:
        """Incorporate new detection measurement into the track."""
        self.kf.update(det.position)
        self.last_seen = float(timestamp)
        self.missed_frames = 0
        self.hits += 1
        self.state = TrackState.OBSERVED
        self.bounding_region = det.bounding_region
        self.semantic_class = det.semantic_class
        self.confidence = float(det.confidence)
        # Restore active confidence when matched
        self.track_confidence = max(self.track_confidence, float(det.confidence), 0.90)

        self.add_position(self.kf.position, max_history_len)

        # Dynamic vs static verification based on sustained speed
        speed = self.kf.speed
        if speed >= static_speed_threshold:
            self.consecutive_moving_frames += 1
            self.consecutive_static_frames = 0
            if self.consecutive_moving_frames >= min_moving_frames:
                self.is_dynamic = True
        else:
            self.consecutive_static_frames += 1
            self.consecutive_moving_frames = 0
            if self.consecutive_static_frames >= min_moving_frames:
                self.is_dynamic = False

    def update_unmatched(
        self,
        confidence_decay_rate: float,
        max_missed_frames: int,
        min_track_confidence: float,
        max_history_len: int,
    ) -> None:
        """Advance track lifecycle when no detection was matched in this frame."""
        self.missed_frames += 1
        self.consecutive_moving_frames = 0
        self.consecutive_static_frames = 0

        # Monotonically decay track confidence
        self.track_confidence = max(0.0, self.track_confidence - confidence_decay_rate)

        # Observed -> Stale -> Removed lifecycle
        if self.missed_frames > max_missed_frames or self.track_confidence < min_track_confidence:
            self.state = TrackState.REMOVED
        else:
            self.state = TrackState.STALE

        # Keep position history updated with coasted/predicted position
        self.add_position(self.kf.position, max_history_len)

        # Update bounding_region center to track predicted Kalman position
        b = self.bounding_region
        hx = (b[3] - b[0]) / 2.0
        hy = (b[4] - b[1]) / 2.0
        hz = (b[5] - b[2]) / 2.0
        px, py, pz = self.kf.position
        self.bounding_region = (
            float(px - hx),
            float(py - hy),
            float(pz - hz),
            float(px + hx),
            float(py + hy),
            float(pz + hz),
        )

    def to_tracked_object(self, timestamp: float, max_history_len: int) -> TrackedObject:
        """Construct public immutable TrackedObject contract instance."""
        # Calculate current bounding region centered at current Kalman filter position (self.kf.position).
        # This guarantees box position/size comes directly from current frame state,
        # translating accurately even when coasting without observations.
        b = self.bounding_region
        hx = (b[3] - b[0]) / 2.0
        hy = (b[4] - b[1]) / 2.0
        hz = (b[5] - b[2]) / 2.0
        px, py, pz = self.kf.position
        current_bbox = (
            float(px - hx),
            float(py - hy),
            float(pz - hz),
            float(px + hx),
            float(py + hy),
            float(pz + hz),
        )
        is_coasting = (self.state == TrackState.STALE) or (self.missed_frames > 0)
        return TrackedObject(
            object_id=self.track_id,
            track_id=self.track_id,
            semantic_class=self.semantic_class,
            position=self.kf.position,
            bounding_region=current_bbox,
            confidence=self.confidence,
            velocity=self.kf.velocity,
            direction=self.kf.direction,
            position_history=list(self.position_history),
            last_seen=self.last_seen,
            track_confidence=self.track_confidence,
            dynamic=self.is_dynamic,
            timestamp=float(timestamp),
            max_history_len=max_history_len,
            is_coasting=is_coasting,
        )


class MultiObjectTracker:
    """Multi-object tracker with Kalman kinematic filtering, data association, and stale decay.

    NOTE ON DATA ASSOCIATION:
    Nearest-Neighbor (NN) spatial distance gating is implemented for O(M * N) association.
    Global optimal assignment (Hungarian algorithm / Munkres) and JPDA (Joint Probabilistic
    Data Association) represent documented future upgrades for dense crowded scenes with
    overlapping bounding boxes.

    Attributes
    ----------
    gating_distance : float
        Maximum distance in meters allowed for detection-to-track matching.
    max_missed_frames : int
        Maximum consecutive frames a track may coast unmatched before removal.
    min_track_confidence : float
        Minimum track confirmation score before track removal.
    confidence_decay_rate : float
        Linear deduction from track_confidence per unmatched frame.
    static_speed_threshold : float
        Speed threshold in m/s to confirm dynamic state.
    min_moving_frames : int
        Consecutive frames exceeding static_speed_threshold to confirm dynamic=True.
    max_history_len : int
        Maximum number of historic positions retained per track.
    process_noise_std : float
        Kalman acceleration process noise standard deviation (m/s^2).
    measurement_noise_std : float
        Kalman position measurement noise standard deviation (m).
    """

    def __init__(
        self,
        gating_distance: float = 2.5,
        max_missed_frames: int = 3,
        min_track_confidence: float = 0.25,
        confidence_decay_rate: float = 0.20,
        static_speed_threshold: float = 0.30,
        min_moving_frames: int = 2,
        max_history_len: int = 50,
        process_noise_std: float = 1.0,
        measurement_noise_std: float = 0.15,
    ) -> None:
        self.gating_distance = float(gating_distance)
        self.max_missed_frames = int(max_missed_frames)
        self.min_track_confidence = float(min_track_confidence)
        self.confidence_decay_rate = float(confidence_decay_rate)
        self.static_speed_threshold = float(static_speed_threshold)
        self.min_moving_frames = int(min_moving_frames)
        self.max_history_len = int(max_history_len)
        self.process_noise_std = float(process_noise_std)
        self.measurement_noise_std = float(measurement_noise_std)

        self.tracks: List[_Track] = []
        self._track_id_counter: int = 0
        self.last_timestamp: Optional[float] = None

    def reset(self) -> None:
        """Reset all active tracks and timestamp state."""
        self.tracks.clear()
        self._track_id_counter = 0
        self.last_timestamp = None

    def _next_track_id(self) -> str:
        """Generate a persistent, stable track ID."""
        self._track_id_counter += 1
        return f"track_{self._track_id_counter}"

    @classmethod
    def load(
        cls, config: Optional[Union[Config, Dict[str, Any], Path, str]] = None
    ) -> MultiObjectTracker:
        """Instantiate MultiObjectTracker from system configuration."""
        if config is None:
            full_cfg = load_config()
        elif isinstance(config, (str, Path)):
            full_cfg = load_config(config)
        elif isinstance(config, (Config, dict)):
            full_cfg = config
        else:
            raise TypeError(f"Unsupported config type: {type(config).__name__}")

        track_cfg = full_cfg.get("tracking", {})
        if isinstance(track_cfg, Config):
            track_dict = track_cfg.to_dict()
        elif isinstance(track_cfg, dict):
            track_dict = track_cfg
        else:
            track_dict = {}

        return cls(
            gating_distance=float(track_dict.get("gating_distance", 2.5)),
            max_missed_frames=int(track_dict.get("max_missed_frames", 3)),
            min_track_confidence=float(track_dict.get("min_track_confidence", 0.25)),
            confidence_decay_rate=float(track_dict.get("confidence_decay_rate", 0.20)),
            static_speed_threshold=float(track_dict.get("static_speed_threshold", 0.30)),
            min_moving_frames=int(track_dict.get("min_moving_frames", 2)),
            max_history_len=int(track_dict.get("max_history_len", 50)),
            process_noise_std=float(track_dict.get("process_noise_std", 1.0)),
            measurement_noise_std=float(track_dict.get("measurement_noise_std", 0.15)),
        )

    def update(
        self, detections: Sequence[DetectedObject], timestamp: float
    ) -> List[TrackedObject]:
        """Update tracker with newly extracted detections for the current frame.

        Parameters
        ----------
        detections : Sequence[DetectedObject]
            Extracted object detections for the current frame.
        timestamp : float
            Current LiDAR scan timestamp in seconds.

        Returns
        -------
        List[TrackedObject]
            List of active (observed or coasting) tracks with persistent IDs and kinematic velocities.
        """
        # Step 1: Compute time step dt
        if self.last_timestamp is None:
            dt = 0.1
        else:
            dt = max(1e-4, timestamp - self.last_timestamp)
        self.last_timestamp = float(timestamp)

        # Step 2: Predict state for all existing tracks
        for track in self.tracks:
            track.predict(dt)

        # Step 3: Spatial association (Global Optimal Bipartite Assignment)
        M = len(self.tracks)
        N = len(detections)

        matched_track_indices: Set[int] = set()
        matched_det_indices: Set[int] = set()

        if M > 0 and N > 0:
            # Build cost matrix with Euclidean distance, class gating, and adaptive gating distance
            cost_matrix = np.full((M, N), 1e5, dtype=np.float64)
            dists = np.zeros((M, N), dtype=np.float64)
            gate_dists = np.zeros(M, dtype=np.float64)

            for i, track in enumerate(self.tracks):
                track_pos = np.array(track.kf.position)
                # Adaptive gating distance: expanding gate for coasted/uncertain tracks
                # to allow reliable re-acquisition after occlusion while maintaining
                # tight gating for observed tracks.
                gate_dist = self.gating_distance
                if track.missed_frames > 0:
                    speed = float(track.kf.speed)
                    gate_dist += 0.5 * track.missed_frames * min(speed, 5.0)
                gate_dists[i] = gate_dist

                for j, det in enumerate(detections):
                    det_pos = np.array(det.position)
                    d = float(np.linalg.norm(track_pos - det_pos))
                    dists[i, j] = d

                    if d <= gate_dist:
                        # Semantic class compatibility: penalize class mismatch
                        # to prevent ID swaps between different types (e.g. pedestrian and vehicle)
                        cls_penalty = 0.0
                        t_cls = str(track.semantic_class).upper()
                        d_cls = str(det.semantic_class).upper()
                        if t_cls != d_cls and t_cls != "UNKNOWN" and d_cls != "UNKNOWN":
                            cls_penalty = 20.0  # Significant penalty to prevent ID swap across classes

                        cost_matrix[i, j] = d + cls_penalty

            # Global optimal bipartite matching via Hungarian algorithm
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            for i, j in zip(row_ind, col_ind):
                i_idx = int(i)
                j_idx = int(j)
                if dists[i_idx, j_idx] <= gate_dists[i_idx] and cost_matrix[i_idx, j_idx] < 1e4:
                    matched_track_indices.add(i_idx)
                    matched_det_indices.add(j_idx)
                    self.tracks[i_idx].update_matched(
                        det=detections[j_idx],
                        timestamp=timestamp,
                        static_speed_threshold=self.static_speed_threshold,
                        min_moving_frames=self.min_moving_frames,
                        max_history_len=self.max_history_len,
                    )

        # Step 4: Update unmatched existing tracks (coasting & decay)
        for i, track in enumerate(self.tracks):
            if i not in matched_track_indices:
                track.update_unmatched(
                    confidence_decay_rate=self.confidence_decay_rate,
                    max_missed_frames=self.max_missed_frames,
                    min_track_confidence=self.min_track_confidence,
                    max_history_len=self.max_history_len,
                )

        # Step 5: Spawn new tracks for unmatched detections
        for j, det in enumerate(detections):
            if j not in matched_det_indices:
                new_track = _Track(
                    track_id=self._next_track_id(),
                    initial_det=det,
                    timestamp=timestamp,
                    process_noise_std=self.process_noise_std,
                    measurement_noise_std=self.measurement_noise_std,
                    max_history_len=self.max_history_len,
                )
                self.tracks.append(new_track)

        # Step 6: Purge removed tracks
        self.tracks = [t for t in self.tracks if t.state != TrackState.REMOVED]

        # Step 7: Return public immutable TrackedObject representations
        return [t.to_tracked_object(timestamp, self.max_history_len) for t in self.tracks]
