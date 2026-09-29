"""Synthetic point cloud scene generator for Fovea-LiDAR testing and benchmarking."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np

from fovea_lidar.contracts.point_cloud import FrameSource, PointCloudFrame


class SyntheticSceneGenerator:
    """Generates parameterized, deterministic synthetic point cloud scenarios.

    Produces sequences of PointCloudFrame objects alongside decoupled ground-truth
    annotations. Designed for pipeline validation without hardware/dataset dependencies.

    Attributes
    ----------
    seed : int
        Default random seed for deterministic generation.
    frame_id : str
        LiDAR coordinate frame identifier.
    dt : float
        Nominal timestep between consecutive frames in seconds (s).
    """

    def __init__(self, seed: int = 42, frame_id: str = "lidar_top", dt: float = 0.1) -> None:
        self.seed = seed
        self.frame_id = frame_id
        self.dt = dt

    # --------------------------------------------------------------------------
    # Primitive Geometry Synthesizers
    # --------------------------------------------------------------------------

    def _generate_ground_plane(
        self,
        rng: np.random.RandomState,
        num_points: int = 600,
        max_range: float = 50.0,
        min_range: float = 1.0,
        road_width: float = 12.0,
    ) -> np.ndarray:
        """Synthesize flat road surface points with minor sensor noise."""
        r = rng.uniform(min_range, max_range, size=num_points).astype(np.float32)
        y = rng.uniform(-road_width / 2.0, road_width / 2.0, size=num_points).astype(np.float32)
        # Forward distance
        x = np.sqrt(np.maximum(0.0, r**2 - y**2)).astype(np.float32)
        # Flat road around z=0 with slight gaussian noise
        z = rng.normal(loc=0.0, scale=0.02, size=num_points).astype(np.float32)
        intensity = rng.uniform(0.1, 0.3, size=num_points).astype(np.float32)
        return np.column_stack([x, y, z, intensity])

    def _generate_box_cluster(
        self,
        rng: np.random.RandomState,
        center: Sequence[float],
        size: Sequence[float],
        num_points: int,
        base_intensity: float = 0.8,
    ) -> np.ndarray:
        """Synthesize points uniformly distributed within an axis-aligned 3D volume."""
        cx, cy, cz = center
        sx, sy, sz = size
        x = rng.uniform(cx - sx / 2.0, cx + sx / 2.0, size=num_points).astype(np.float32)
        y = rng.uniform(cy - sy / 2.0, cy + sy / 2.0, size=num_points).astype(np.float32)
        z = rng.uniform(cz - sz / 2.0, cz + sz / 2.0, size=num_points).astype(np.float32)
        intensity = rng.uniform(base_intensity - 0.1, base_intensity + 0.1, size=num_points).astype(np.float32)
        return np.column_stack([x, y, z, intensity])

    def _generate_360_concentric_ground(
        self,
        rng: np.random.RandomState,
        num_rings: int = 28,
        min_range: float = 1.0,
        max_range: float = 48.0,
        points_per_ring: int = 150,
    ) -> np.ndarray:
        """Synthesize 360-degree concentric LiDAR ground scan rings with realistic 1/r density.

        Near the sensor (0m-10m), beam rings are tightly spaced, reproducing physical
        rotary LiDAR scan geometry where close-field point density is highest.
        """
        r_fractions = (np.linspace(0.0, 1.0, num_rings) ** 1.6).astype(np.float32)
        ring_radii = min_range + (max_range - min_range) * r_fractions

        all_points = []
        for r_k in ring_radii:
            azimuths = np.linspace(0.0, 2.0 * np.pi, points_per_ring, endpoint=False).astype(np.float32)
            az_noise = rng.normal(0.0, 0.005, size=points_per_ring).astype(np.float32)
            r_noise = rng.normal(0.0, 0.03, size=points_per_ring).astype(np.float32)
            r_actual = r_k + r_noise

            x = (r_actual * np.cos(azimuths + az_noise)).astype(np.float32)
            y = (r_actual * np.sin(azimuths + az_noise)).astype(np.float32)
            z = rng.normal(loc=0.0, scale=0.02, size=points_per_ring).astype(np.float32)
            intensity = (0.18 + 0.08 * np.sin(azimuths * 4.0)).astype(np.float32)
            all_points.append(np.column_stack([x, y, z, intensity]))

        return np.vstack(all_points).astype(np.float32)

    def _apply_ego_motion(
        self,
        pts: np.ndarray,
        ego_x: float,
        ego_y: float = 0.0,
        ego_heading: float = 0.0,
    ) -> np.ndarray:
        """Transform world-frame point cloud into sensor frame given ego pose.

        The ego vehicle starts at world origin (0, 0) and drives forward along
        +X.  At time t, ego is at world position (ego_x, ego_y).  All world-frame
        points must be shifted so that ego remains at sensor-frame origin.  This
        is the standard robotics convention: ego is always at (0,0), objects that
        are stationary in the world appear to drift toward -X.

        Parameters
        ----------
        pts : np.ndarray, shape (N, 4)  [x, y, z, intensity]
            Points in world frame.
        ego_x : float
            Ego world-frame X position (metres).
        ego_y : float
            Ego world-frame Y position (metres, default 0 for straight line).
        ego_heading : float
            Ego heading angle in radians (default 0 = facing +X).  Rotation
            about Z is applied after translation.

        Returns
        -------
        np.ndarray, shape (N, 4) — points in sensor frame.
        """
        out = pts.copy()
        out[:, 0] -= ego_x
        out[:, 1] -= ego_y
        if ego_heading != 0.0:
            cos_h = np.cos(-ego_heading)
            sin_h = np.sin(-ego_heading)
            x_rot = out[:, 0] * cos_h - out[:, 1] * sin_h
            y_rot = out[:, 0] * sin_h + out[:, 1] * cos_h
            out[:, 0] = x_rot.astype(np.float32)
            out[:, 1] = y_rot.astype(np.float32)
        return out.astype(np.float32)

    # --------------------------------------------------------------------------
    # Scenario Implementations
    # --------------------------------------------------------------------------

    def generate_empty_road(
        self,
        num_frames: int = 10,
        max_range: float = 50.0,
        num_ground_points: int = 600,
        seed: Optional[int] = None,
    ) -> Tuple[List[PointCloudFrame], Dict[str, Any]]:
        """Scenario: Flat road surface without obstacles."""
        rng = np.random.RandomState(self.seed if seed is None else seed)
        frames: List[PointCloudFrame] = []
        gt_frames: List[Dict[str, Any]] = []

        # Generate ground plane once for static scene to prevent artificial frame-to-frame point jitter
        ground_static = self._generate_ground_plane(rng, num_points=num_ground_points, max_range=max_range)

        for f_idx in range(num_frames):
            timestamp = f_idx * self.dt
            frame = PointCloudFrame(
                points=ground_static.copy(),
                timestamp=timestamp,
                frame_id=self.frame_id,
                source=FrameSource.MOCK,
                metadata={"scenario": "empty_road", "frame_idx": f_idx},
            )
            frames.append(frame)
            gt_frames.append({"frame_idx": f_idx, "timestamp": timestamp, "objects": []})

        ground_truth = {
            "scenario": "empty_road",
            "num_frames": num_frames,
            "frames": gt_frames,
        }
        return frames, ground_truth

    def generate_pedestrian_at_distance(
        self,
        num_frames: int = 100,
        initial_distance: float = 25.0,
        velocity: Tuple[float, float, float] = (0.0, 1.2, 0.0),
        ego_speed: float = 6.0,
        num_ground_points: int = 500,
        num_ped_points: int = 80,
        seed: Optional[int] = None,
    ) -> Tuple[List[PointCloudFrame], Dict[str, Any]]:
        """Scenario: Ground plane plus a moving pedestrian point cluster.

        Ego vehicle drives forward at ``ego_speed`` m/s along +X.  The pedestrian
        is given in world coordinates and transformed into sensor frame each frame.
        Over 10 s the ego covers 60 m, so the pedestrian that starts at world X=25
        will initially appear at +25 m sensor-frame, then slide through the fovea
        zone and disappear behind the ego as it passes.
        """
        rng = np.random.RandomState(self.seed if seed is None else seed)
        frames: List[PointCloudFrame] = []
        gt_frames: List[Dict[str, Any]] = []

        for f_idx in range(num_frames):
            timestamp = f_idx * self.dt
            ego_x = ego_speed * timestamp

            # Pedestrian world position (constant X, drifting Y)
            px_w = initial_distance + velocity[0] * timestamp
            py_w = velocity[1] * timestamp
            pz_w = 0.9 + velocity[2] * timestamp

            # World → sensor frame
            px = px_w - ego_x
            py = py_w

            ground_w = self._generate_ground_plane(rng, num_points=num_ground_points)
            ground = self._apply_ego_motion(ground_w, ego_x)

            ped_points = self._generate_box_cluster(
                rng,
                center=(px, py, pz_w),
                size=(0.5, 0.5, 1.8),
                num_points=num_ped_points,
                base_intensity=0.65,
            )
            pts = np.vstack([ground, ped_points])

            frame = PointCloudFrame(
                points=pts,
                timestamp=timestamp,
                frame_id=self.frame_id,
                source=FrameSource.MOCK,
                metadata={
                    "scenario": "pedestrian_at_distance",
                    "frame_idx": f_idx,
                    "ego_x": round(ego_x, 2),
                    "ego_speed_ms": ego_speed,
                },
            )
            frames.append(frame)
            gt_frames.append(
                {
                    "frame_idx": f_idx,
                    "timestamp": timestamp,
                    "ego_world_x": round(ego_x, 2),
                    "objects": [
                        {
                            "object_id": "ped_1",
                            "semantic_class": "PEDESTRIAN",
                            "position": [round(px, 3), round(py, 3), round(pz_w, 3)],
                            "world_position": [round(px_w, 3), round(py_w, 3), round(pz_w, 3)],
                            "size": [0.5, 0.5, 1.8],
                            "velocity": list(velocity),
                        }
                    ],
                }
            )

        ground_truth = {
            "scenario": "pedestrian_at_distance",
            "num_frames": num_frames,
            "ego_speed_ms": ego_speed,
            "frames": gt_frames,
        }
        return frames, ground_truth


    def generate_vehicle_crossing(
        self,
        num_frames: int = 100,
        initial_position: Tuple[float, float, float] = (30.0, -10.0, 0.75),
        velocity: Tuple[float, float, float] = (0.0, 3.5, 0.0),
        ego_speed: float = 5.0,
        num_ground_points: int = 500,
        num_veh_points: int = 250,
        seed: Optional[int] = None,
    ) -> Tuple[List[PointCloudFrame], Dict[str, Any]]:
        """Scenario: Vehicle moving laterally across the LiDAR field of view.

        Ego drives forward at ``ego_speed`` m/s.  The crossing vehicle starts at
        world X=30, Y=-10 and moves in the +Y direction.  As ego approaches, the
        vehicle appears to shrink in X range and sweep through the sensor's
        lateral field of view — producing a classic cut-in near-miss signature.
        """
        rng = np.random.RandomState(self.seed if seed is None else seed)
        frames: List[PointCloudFrame] = []
        gt_frames: List[Dict[str, Any]] = []

        for f_idx in range(num_frames):
            timestamp = f_idx * self.dt
            ego_x = ego_speed * timestamp

            # Crossing vehicle world position
            vx_w = initial_position[0] + velocity[0] * timestamp
            vy_w = initial_position[1] + velocity[1] * timestamp
            vz_w = initial_position[2] + velocity[2] * timestamp

            # Sensor frame
            vx = vx_w - ego_x
            vy = vy_w

            ground_w = self._generate_ground_plane(rng, num_points=num_ground_points)
            ground = self._apply_ego_motion(ground_w, ego_x)
            veh_points = self._generate_box_cluster(
                rng,
                center=(vx, vy, vz_w),
                size=(4.5, 2.0, 1.5),
                num_points=num_veh_points,
                base_intensity=0.85,
            )
            pts = np.vstack([ground, veh_points])

            frame = PointCloudFrame(
                points=pts,
                timestamp=timestamp,
                frame_id=self.frame_id,
                source=FrameSource.MOCK,
                metadata={
                    "scenario": "vehicle_crossing",
                    "frame_idx": f_idx,
                    "ego_x": round(ego_x, 2),
                    "ego_speed_ms": ego_speed,
                },
            )
            frames.append(frame)
            gt_frames.append(
                {
                    "frame_idx": f_idx,
                    "timestamp": timestamp,
                    "ego_world_x": round(ego_x, 2),
                    "objects": [
                        {
                            "object_id": "veh_crossing_1",
                            "semantic_class": "VEHICLE",
                            "position": [round(vx, 3), round(vy, 3), round(vz_w, 3)],
                            "world_position": [round(vx_w, 3), round(vy_w, 3), round(vz_w, 3)],
                            "size": [4.5, 2.0, 1.5],
                            "velocity": list(velocity),
                        }
                    ],
                }
            )

        ground_truth = {
            "scenario": "vehicle_crossing",
            "num_frames": num_frames,
            "ego_speed_ms": ego_speed,
            "frames": gt_frames,
        }

        return frames, ground_truth

    def generate_low_confidence_obstacle(
        self,
        num_frames: int = 10,
        position: Tuple[float, float, float] = (12.0, 1.0, 0.3),
        num_ground_points: int = 500,
        num_sparse_points: int = 12,
        seed: Optional[int] = None,
    ) -> Tuple[List[PointCloudFrame], Dict[str, Any]]:
        """Scenario: Sparse, ambiguous cluster intended to produce low classifier confidence."""
        rng = np.random.RandomState(self.seed if seed is None else seed)
        frames: List[PointCloudFrame] = []
        gt_frames: List[Dict[str, Any]] = []

        for f_idx in range(num_frames):
            timestamp = f_idx * self.dt
            ground = self._generate_ground_plane(rng, num_points=num_ground_points)
            # Sparse irregular points
            sparse_pts = self._generate_box_cluster(
                rng,
                center=position,
                size=(0.4, 0.4, 0.6),
                num_points=num_sparse_points,
                base_intensity=0.4,
            )
            pts = np.vstack([ground, sparse_pts])

            frame = PointCloudFrame(
                points=pts,
                timestamp=timestamp,
                frame_id=self.frame_id,
                source=FrameSource.MOCK,
                metadata={"scenario": "low_confidence_obstacle", "frame_idx": f_idx},
            )
            frames.append(frame)
            gt_frames.append(
                {
                    "frame_idx": f_idx,
                    "timestamp": timestamp,
                    "objects": [
                        {
                            "object_id": "ambiguous_1",
                            "semantic_class": "UNKNOWN_OBSTACLE",
                            "position": list(position),
                            "confidence_expected": "low",
                            "is_ambiguous": True,
                        }
                    ],
                }
            )

        ground_truth = {
            "scenario": "low_confidence_obstacle",
            "num_frames": num_frames,
            "frames": gt_frames,
        }
        return frames, ground_truth

    def generate_resolution_boundary_stress(
        self,
        num_frames: int = 10,
        boundary_distances: Sequence[float] = (10.0, 25.0, 50.0),
        num_ground_points: int = 300,
        seed: Optional[int] = None,
    ) -> Tuple[List[PointCloudFrame], Dict[str, Any]]:
        """Scenario: Points specifically clustered right across resolution zone boundaries."""
        rng = np.random.RandomState(self.seed if seed is None else seed)
        frames: List[PointCloudFrame] = []
        gt_frames: List[Dict[str, Any]] = []

        for f_idx in range(num_frames):
            timestamp = f_idx * self.dt
            ground = self._generate_ground_plane(rng, num_points=num_ground_points)

            # Insert boundary stress points (e.g., d-0.01, d, d+0.01)
            stress_points = []
            for d in boundary_distances:
                for delta in (-0.01, 0.0, 0.01):
                    for y_offset in (-2.0, 0.0, 2.0):
                        stress_points.append([d + delta, y_offset, 0.5, 0.9])

            pts = np.vstack([ground, np.array(stress_points, dtype=np.float32)])

            frame = PointCloudFrame(
                points=pts,
                timestamp=timestamp,
                frame_id=self.frame_id,
                source=FrameSource.MOCK,
                metadata={"scenario": "resolution_boundary_stress", "frame_idx": f_idx},
            )
            frames.append(frame)
            gt_frames.append(
                {
                    "frame_idx": f_idx,
                    "timestamp": timestamp,
                    "boundary_distances": list(boundary_distances),
                    "objects": [],
                }
            )

        ground_truth = {
            "scenario": "resolution_boundary_stress",
            "num_frames": num_frames,
            "frames": gt_frames,
        }
        return frames, ground_truth

    def generate_multi_object_dense(
        self,
        num_frames: int = 100,
        num_ground_points: int = 500,
        num_objects: Optional[int] = None,
        ego_speed: float = 6.0,
        seed: Optional[int] = None,
    ) -> Tuple[List[PointCloudFrame], Dict[str, Any]]:
        """Scenario: Multiple concurrent pedestrians and vehicles at varying distances.

        Ego drives at ``ego_speed`` m/s.  All world-frame object positions are
        transformed into sensor frame so ego appears stationary at origin while
        surrounding objects evolve according to their world-frame kinematics.
        """
        rng = np.random.RandomState(self.seed if seed is None else seed)
        frames: List[PointCloudFrame] = []
        gt_frames: List[Dict[str, Any]] = []

        # If num_objects is None or exactly 4, preserve canonical 4-object benchmark layout
        use_default_layout = (num_objects is None or num_objects == 4)

        for f_idx in range(num_frames):
            timestamp = f_idx * self.dt
            ego_x = ego_speed * timestamp
            ground_w = self._generate_ground_plane(rng, num_points=num_ground_points)
            ground = self._apply_ego_motion(ground_w, ego_x)

            if use_default_layout:
                # World-frame positions; sensor-frame = world - ego_x
                # Object 1: Near pedestrian (world X=8m stays ~constant relative while ego closes)
                p1_wx = 8.0 + (-0.3) * timestamp   # ped walking slightly toward traffic
                p1_wy = -1.0 + 0.5 * timestamp
                p1_pos = (p1_wx - ego_x, p1_wy, 0.9)
                p1_pts = self._generate_box_cluster(rng, p1_pos, (0.5, 0.5, 1.8), 70, 0.7)

                # Object 2: Mid-range crossing vehicle (world X=20m, moving Y)
                v1_wx = 20.0 + ego_speed * 0.4 * timestamp  # drives ~0.4x ego speed ahead
                v1_wy = -6.0 + 2.0 * timestamp
                v1_pos = (v1_wx - ego_x, v1_wy, 0.8)
                v1_pts = self._generate_box_cluster(rng, v1_pos, (4.2, 1.9, 1.6), 200, 0.85)

                # Object 3: Far-field truck stationary in world at X=60m
                v2_wx = 60.0
                v2_pos = (v2_wx - ego_x, 3.0, 1.5)
                v2_pts = self._generate_box_cluster(rng, v2_pos, (7.0, 2.5, 3.0), 300, 0.9)

                # Object 4: Second pedestrian (world X=15m, walking away)
                p2_wx = 15.0 + 0.5 * timestamp
                p2_wy = 4.0 - 0.8 * timestamp
                p2_pos = (p2_wx - ego_x, p2_wy, 0.9)
                p2_pts = self._generate_box_cluster(rng, p2_pos, (0.5, 0.5, 1.8), 60, 0.7)

                all_obj_pts = [p1_pts, v1_pts, v2_pts, p2_pts]
                gt_objects = [
                    {"object_id": "ped_near", "semantic_class": "PEDESTRIAN", "position": [round(x, 3) for x in p1_pos]},
                    {"object_id": "veh_mid", "semantic_class": "VEHICLE", "position": [round(x, 3) for x in v1_pos]},
                    {"object_id": "truck_far", "semantic_class": "TRUCK", "position": list(v2_pos)},
                    {"object_id": "ped_roadside", "semantic_class": "PEDESTRIAN", "position": [round(x, 3) for x in p2_pos]},
                ]
            else:
                all_obj_pts = []
                gt_objects = []
                # Parameterized multi-object scaling across range [5.0m, 45.0m] world frame
                assert num_objects is not None
                for obj_i in range(num_objects):
                    obj_type = "PEDESTRIAN" if obj_i % 2 == 0 else "VEHICLE"
                    dist_slot = 6.0 + (38.0 * obj_i / max(1, num_objects - 1))
                    y_slot = -4.0 + (8.0 * (obj_i % 3) / 2.0)
                    vx_w = -0.5 if obj_i % 2 == 0 else 1.2
                    cur_wx = dist_slot + vx_w * timestamp
                    cur_wy = y_slot + 0.3 * (1 if obj_i % 2 == 0 else -1) * timestamp
                    cur_z = 0.9 if obj_type == "PEDESTRIAN" else 0.8
                    size = (0.5, 0.5, 1.8) if obj_type == "PEDESTRIAN" else (4.0, 1.8, 1.5)
                    n_pts = 60 if obj_type == "PEDESTRIAN" else 150
                    intensity = 0.75 if obj_type == "PEDESTRIAN" else 0.85

                    pos = (float(cur_wx - ego_x), float(cur_wy), float(cur_z))
                    pts_i = self._generate_box_cluster(rng, pos, size, n_pts, intensity)
                    all_obj_pts.append(pts_i)
                    gt_objects.append(
                        {
                            "object_id": f"scaled_obj_{obj_i}",
                            "semantic_class": obj_type,
                            "position": [round(x, 3) for x in pos],
                        }
                    )

            pts = np.vstack([ground] + all_obj_pts) if all_obj_pts else ground

            frame = PointCloudFrame(
                points=pts,
                timestamp=timestamp,
                frame_id=self.frame_id,
                source=FrameSource.MOCK,
                metadata={
                    "scenario": "multi_object_dense",
                    "frame_idx": f_idx,
                    "num_objects": len(gt_objects),
                    "ego_x": round(ego_x, 2),
                    "ego_speed_ms": ego_speed,
                },
            )
            frames.append(frame)
            gt_frames.append(
                {
                    "frame_idx": f_idx,
                    "timestamp": timestamp,
                    "ego_world_x": round(ego_x, 2),
                    "objects": gt_objects,
                }
            )

        ground_truth = {
            "scenario": "multi_object_dense",
            "num_frames": num_frames,
            "ego_speed_ms": ego_speed,
            "num_objects": len(gt_frames[0]["objects"]) if gt_frames else 0,
            "frames": gt_frames,
        }
        return frames, ground_truth


    def generate_object_disappearing(
        self,
        num_frames: int = 10,
        disappear_after_frame: int = 5,
        object_class: str = "PEDESTRIAN",
        position: Tuple[float, float, float] = (12.0, 0.0, 0.9),
        num_ground_points: int = 500,
        seed: Optional[int] = None,
    ) -> Tuple[List[PointCloudFrame], Dict[str, Any]]:
        """Scenario: Object present for N frames, then disappears to test track decay."""
        rng = np.random.RandomState(self.seed if seed is None else seed)
        frames: List[PointCloudFrame] = []
        gt_frames: List[Dict[str, Any]] = []

        for f_idx in range(num_frames):
            timestamp = f_idx * self.dt
            ground = self._generate_ground_plane(rng, num_points=num_ground_points)

            active_objects: List[Dict[str, Any]] = []
            if f_idx < disappear_after_frame:
                obj_pts = self._generate_box_cluster(rng, position, (0.5, 0.5, 1.8), 80, 0.7)
                pts = np.vstack([ground, obj_pts])
                active_objects.append(
                    {
                        "object_id": "ephemeral_ped",
                        "semantic_class": object_class,
                        "position": list(position),
                    }
                )
            else:
                pts = ground

            frame = PointCloudFrame(
                points=pts,
                timestamp=timestamp,
                frame_id=self.frame_id,
                source=FrameSource.MOCK,
                metadata={"scenario": "object_disappearing", "frame_idx": f_idx},
            )
            frames.append(frame)
            gt_frames.append({"frame_idx": f_idx, "timestamp": timestamp, "objects": active_objects})

        ground_truth = {
            "scenario": "object_disappearing",
            "num_frames": num_frames,
            "disappear_after_frame": disappear_after_frame,
            "frames": gt_frames,
        }
        return frames, ground_truth

    def _generate_dense_road_ground(
        self,
        rng: np.random.RandomState,
        num_points: int = 1800,
        max_range: float = 55.0,
        min_range: float = 1.0,
        road_width: float = 12.0,
    ) -> np.ndarray:
        """Synthesize dense, realistic LiDAR ground scan returns in sensor frame."""
        u = rng.uniform(0.0, 1.0, size=num_points).astype(np.float32)
        x = (min_range + (max_range - min_range) * (u ** 1.15)).astype(np.float32)
        y = rng.uniform(-road_width / 2.0, road_width / 2.0, size=num_points).astype(np.float32)

        z = np.zeros(num_points, dtype=np.float32)
        intensity = np.zeros(num_points, dtype=np.float32)

        is_road = np.abs(y) <= 4.5
        is_curb = (np.abs(y) > 4.5) & (np.abs(y) <= 5.2)
        is_shoulder = np.abs(y) > 5.2

        z[is_road] = rng.normal(0.0, 0.015, size=int(np.sum(is_road))).astype(np.float32)
        road_ints = rng.uniform(0.20, 0.35, size=int(np.sum(is_road))).astype(np.float32)
        lane_mask = np.abs(y[is_road]) < 0.25
        road_ints[lane_mask] = rng.uniform(0.65, 0.85, size=int(np.sum(lane_mask))).astype(np.float32)
        intensity[is_road] = road_ints

        z[is_curb] = (0.10 + rng.normal(0.0, 0.02, size=int(np.sum(is_curb)))).astype(np.float32)
        intensity[is_curb] = rng.uniform(0.35, 0.50, size=int(np.sum(is_curb))).astype(np.float32)

        z[is_shoulder] = (0.12 + rng.normal(0.0, 0.03, size=int(np.sum(is_shoulder)))).astype(np.float32)
        intensity[is_shoulder] = rng.uniform(0.15, 0.25, size=int(np.sum(is_shoulder))).astype(np.float32)

        return np.column_stack([x, y, z, intensity]).astype(np.float32)

    def generate_flagship_demo(
        self,
        num_frames: int = 100,
        ped_start_pos: Tuple[float, float, float] = (55.0, 2.5, 0.9),
        ped_velocity: Tuple[float, float, float] = (-3.5, -2.0, 0.0),
        ped_enter_frame: int = 15,
        ped_exit_frame: int = 75,
        ego_speed: float = 8.0,
        num_ground_points: int = 1800,
        num_ped_points: int = 100,
        seed: Optional[int] = None,
    ) -> Tuple[List[PointCloudFrame], Dict[str, Any]]:
        """Flagship Jury Demo Scenario — 10-second ego-motion drive.

        Timeline (100 frames @ 10 Hz = 10 s):
        - Frames  0..14  (Empty Road):   Ego accelerates through clear road. Ground points
          stream backward as ego drives forward at 8 m/s. Fine 5cm fovea near-field active.
        - Frames 15..74  (Pedestrian):   Ped appears at world X=55m (~40m sensor-frame at entry).
          As ego closes, ped sensor range shrinks from 40m → fovea 10m zone, triggering adaptive
          5cm high-resolution tracking. Collision risk escalates through MEDIUM → HIGH.
        - Frames 75..99  (Relaxation):   Ped exits the road. Track decays, fovea relaxes to
          coarse baseline. Ego has now covered 80m of world distance.
        """
        rng = np.random.RandomState(self.seed if seed is None else seed)
        frames: List[PointCloudFrame] = []
        gt_frames: List[Dict[str, Any]] = []

        for f_idx in range(num_frames):
            timestamp = f_idx * self.dt
            ego_x = ego_speed * timestamp

            ground = self._generate_dense_road_ground(
                rng,
                num_points=max(num_ground_points, 1800),
                max_range=55.0,
                min_range=1.0,
                road_width=12.0,
            )

            active_objects: List[Dict[str, Any]] = []
            if ped_enter_frame <= f_idx < ped_exit_frame:
                t_rel = (f_idx - ped_enter_frame) * self.dt
                # World-frame ped position
                px_w = ped_start_pos[0] + ped_velocity[0] * t_rel
                py_w = ped_start_pos[1] + ped_velocity[1] * t_rel
                pz_w = ped_start_pos[2] + ped_velocity[2] * t_rel
                # Sensor frame
                px = px_w - ego_x
                py = py_w
                pz = pz_w

                ped_pts = self._generate_box_cluster(
                    rng,
                    center=(px, py, pz),
                    size=(0.5, 0.5, 1.8),
                    num_points=num_ped_points,
                    base_intensity=0.80,
                )
                pts = np.vstack([ground, ped_pts])
                active_objects.append(
                    {
                        "object_id": "flagship_pedestrian",
                        "semantic_class": "PEDESTRIAN",
                        "position": [float(px), float(py), float(pz)],
                        "world_position": [float(px_w), float(py_w), float(pz_w)],
                        "velocity": list(ped_velocity),
                    }
                )
            else:
                pts = ground

            frame = PointCloudFrame(
                points=pts,
                timestamp=timestamp,
                frame_id=self.frame_id,
                source=FrameSource.MOCK,
                metadata={
                    "scenario": "flagship_demo",
                    "frame_idx": f_idx,
                    "ego_x": round(ego_x, 2),
                    "ego_speed_ms": ego_speed,
                    "phase": (
                        "baseline_empty"
                        if f_idx < ped_enter_frame
                        else ("pedestrian_tracked" if f_idx < ped_exit_frame else "resolution_relaxation")
                    ),
                },
            )
            frames.append(frame)
            gt_frames.append(
                {
                    "frame_idx": f_idx,
                    "timestamp": timestamp,
                    "ego_world_x": round(ego_x, 2),
                    "objects": active_objects,
                }
            )

        ground_truth = {
            "scenario": "flagship_demo",
            "num_frames": num_frames,
            "ego_speed_ms": ego_speed,
            "ped_enter_frame": ped_enter_frame,
            "ped_exit_frame": ped_exit_frame,
            "frames": gt_frames,
        }
        return frames, ground_truth


    def generate_surround_360_city(
        self,
        num_frames: int = 100,
        num_ground_rings: int = 24,
        points_per_ring: int = 140,
        ego_speed: float = 7.0,
        seed: Optional[int] = None,
    ) -> Tuple[List[PointCloudFrame], Dict[str, Any]]:
        """True 360-degree Omnidirectional Urban LiDAR Scenario — 10-second ego drive.

        Ego vehicle drives at ``ego_speed`` m/s forward (+X world axis).  All 5 surrounding
        objects are given world-frame positions and transformed into sensor frame each step.

        World-frame object layout:
        1. CLOSE PEDESTRIAN crossing front-right: world (8m, 4m) — stays close initially
        2. LEAD VEHICLE:  world (22m, −0.6m) — drives at 1.8 m/s (slower than ego)
        3. REAR VEHICLE:  world (−5m, 0.4m)  — approaches at 9m/s (faster than ego)
        4. CYCLIST LEFT:  world (5m, 6.5m)   — overtaking at 9.8m/s (faster than ego)
        5. PARKED VEHICLE: world (15m, −6.8m) — stationary; ego drives past it
        """
        rng = np.random.RandomState(self.seed if seed is None else seed)
        frames: List[PointCloudFrame] = []
        gt_frames: List[Dict[str, Any]] = []

        # World-frame initial positions (ego starts at world origin 0,0)
        ped_world_x0, ped_world_y0 = 8.0, 4.0
        lead_world_x0, lead_world_y0 = 22.0, -0.6
        rear_world_x0, rear_world_y0 = -5.0, 0.4
        cyc_world_x0, cyc_world_y0 = 5.0, 6.5
        parked_world_x0, parked_world_y0 = 15.0, -6.8

        for f_idx in range(num_frames):
            timestamp = f_idx * self.dt
            ego_x = ego_speed * timestamp

            # 1. 360-degree concentric ground plane (always in sensor frame)
            ground = self._generate_360_concentric_ground(
                rng,
                num_rings=num_ground_rings,
                points_per_ring=points_per_ring,
                min_range=1.0,
                max_range=46.0,
            )

            # World-frame object positions at this timestep
            ped_wx = ped_world_x0 - 0.2 * timestamp   # ped crosses slowly inward
            ped_wy = ped_world_y0 - 0.7 * timestamp
            lead_wx = lead_world_x0 + 1.8 * timestamp  # lead car slower than ego
            lead_wy = lead_world_y0
            rear_wx = rear_world_x0 + 9.0 * timestamp  # rear car faster — overtakes
            rear_wy = rear_world_y0
            cyc_wx = cyc_world_x0 + 9.8 * timestamp   # cyclist faster — overtakes
            cyc_wy = cyc_world_y0 + 0.1 * timestamp
            parked_wx = parked_world_x0                # stationary in world
            parked_wy = parked_world_y0

            # Sensor-frame = world - ego_x (Y unchanged, straight-line drive)
            ped_x = ped_wx - ego_x;    ped_y = ped_wy
            lead_x = lead_wx - ego_x;  lead_y = lead_wy
            rear_x = rear_wx - ego_x;  rear_y = rear_wy
            cyc_x = cyc_wx - ego_x;    cyc_y = cyc_wy
            parked_x = parked_wx - ego_x; parked_y = parked_wy

            ped_pts = self._generate_box_cluster(rng, (ped_x, ped_y, 0.9), (0.5, 0.5, 1.8), 140, 0.85)
            lead_pts = self._generate_box_cluster(rng, (lead_x, lead_y, 0.75), (4.2, 1.8, 1.5), 150, 0.80)
            rear_pts = self._generate_box_cluster(rng, (rear_x, rear_y, 0.7), (4.0, 1.8, 1.4), 130, 0.75)
            cyc_pts = self._generate_box_cluster(rng, (cyc_x, cyc_y, 0.85), (1.6, 0.6, 1.6), 90, 0.78)
            parked_pts = self._generate_box_cluster(rng, (parked_x, parked_y, 0.8), (4.5, 2.0, 1.6), 120, 0.70)

            pts = np.vstack([ground, ped_pts, lead_pts, rear_pts, cyc_pts, parked_pts])

            frame = PointCloudFrame(
                points=pts,
                timestamp=timestamp,
                frame_id=self.frame_id,
                source=FrameSource.MOCK,
                metadata={
                    "scenario": "surround_360_city",
                    "frame_idx": f_idx,
                    "coverage": "360_omni",
                    "ego_x": round(ego_x, 2),
                    "ego_speed_ms": ego_speed,
                },
            )
            frames.append(frame)

            gt_objects = [
                {
                    "object_id": "ped_close_crossing",
                    "semantic_class": "PEDESTRIAN",
                    "position": [round(ped_x, 3), round(ped_y, 3), 0.9],
                    "world_position": [round(ped_wx, 3), round(ped_wy, 3), 0.9],
                    "size": [0.5, 0.5, 1.8],
                    "velocity": [-0.2, -0.7, 0.0],
                },
                {
                    "object_id": "veh_lead",
                    "semantic_class": "VEHICLE",
                    "position": [round(lead_x, 3), round(lead_y, 3), 0.75],
                    "world_position": [round(lead_wx, 3), round(lead_wy, 3), 0.75],
                    "size": [4.2, 1.8, 1.5],
                    "velocity": [1.8 - ego_speed, 0.0, 0.0],
                },
                {
                    "object_id": "veh_rear_approaching",
                    "semantic_class": "VEHICLE",
                    "position": [round(rear_x, 3), round(rear_y, 3), 0.7],
                    "world_position": [round(rear_wx, 3), round(rear_wy, 3), 0.7],
                    "size": [4.0, 1.8, 1.4],
                    "velocity": [9.0 - ego_speed, 0.0, 0.0],
                },
                {
                    "object_id": "cyclist_left",
                    "semantic_class": "CYCLIST",
                    "position": [round(cyc_x, 3), round(cyc_y, 3), 0.85],
                    "world_position": [round(cyc_wx, 3), round(cyc_wy, 3), 0.85],
                    "size": [1.6, 0.6, 1.6],
                    "velocity": [9.8 - ego_speed, 0.1, 0.0],
                },
                {
                    "object_id": "veh_parked_right",
                    "semantic_class": "VEHICLE",
                    "position": [round(parked_x, 3), round(parked_y, 3), 0.8],
                    "world_position": [round(parked_wx, 3), round(parked_wy, 3), 0.8],
                    "size": [4.5, 2.0, 1.6],
                    "velocity": [-ego_speed, 0.0, 0.0],
                },
            ]
            gt_frames.append({"frame_idx": f_idx, "timestamp": timestamp, "ego_world_x": round(ego_x, 2), "objects": gt_objects})

        ground_truth = {
            "scenario": "surround_360_city",
            "num_frames": num_frames,
            "ego_speed_ms": ego_speed,
            "coverage": "360_degrees",
            "frames": gt_frames,
        }
        return frames, ground_truth



    # --------------------------------------------------------------------------
    # Universal Generator Dispatcher
    # --------------------------------------------------------------------------

    def generate(
        self,
        scenario_name: str,
        **kwargs: Any,
    ) -> Tuple[List[PointCloudFrame], Dict[str, Any]]:
        """Generate a named scenario by key."""
        scenarios: Dict[str, Callable[..., Tuple[List[PointCloudFrame], Dict[str, Any]]]] = {
            "empty_road": self.generate_empty_road,
            "pedestrian_at_distance": self.generate_pedestrian_at_distance,
            "vehicle_crossing": self.generate_vehicle_crossing,
            "low_confidence_obstacle": self.generate_low_confidence_obstacle,
            "resolution_boundary_stress": self.generate_resolution_boundary_stress,
            "multi_object_dense": self.generate_multi_object_dense,
            "object_disappearing": self.generate_object_disappearing,
            "flagship_demo": self.generate_flagship_demo,
            "surround_360_city": self.generate_surround_360_city,
        }

        canonical = scenario_name.strip().lower()
        if canonical not in scenarios:
            available = sorted(scenarios.keys())
            raise ValueError(f"Unknown scenario '{scenario_name}'. Available: {available}")

        return scenarios[canonical](**kwargs)

    def generate_interactive_frame(
        self,
        scenario_name: str = "flagship_demo",
        ego_x: float = 0.0,
        ego_y: float = 0.0,
        ego_heading: float = 0.0,
        ego_speed: float = 0.0,
        timestamp: float = 0.0,
        frame_idx: int = 0,
        seed: Optional[int] = None,
    ) -> PointCloudFrame:
        """Synthesize a single LiDAR point cloud frame at an arbitrary ego vehicle pose.

        Supports dynamic real-time driving controls where the ego vehicle position
        (ego_x, ego_y), heading, and speed are interactively controlled by the user.

        Key Design Rules for Realistic Demonstration:
        1. Fixed World Roadway: The road corridor is permanently anchored at |Y_world| <= 4.5m.
           Outside (|Y_world| > 4.5m) is elevated curb and rough off-road terrain.
           If the user steers off the road, the road stays where it belongs, and the vehicle
           enters rough off-road terrain.
        2. Stationary / Fixed Obstacles: All obstacles (pedestrians, vehicles, hazard debris)
           are stationary at fixed world coordinates (X, Y). The ego vehicle drives freely
           among them.
        """
        rng = np.random.RandomState(int((timestamp * 1000) % 1000000) if seed is None else seed)
        canonical = scenario_name.strip().lower()

        # ----------------------------------------------------------------------
        # 1. World Road & Terrain Generation (Anchored in World Space)
        # ----------------------------------------------------------------------
        # Sample ground points in polar coordinates around current ego vehicle pose
        num_ground_pts = 1800
        max_lidar_range = 50.0
        r_samples = rng.uniform(1.2, max_lidar_range, size=num_ground_pts).astype(np.float32)
        phi_samples = rng.uniform(-np.pi, np.pi, size=num_ground_pts).astype(np.float32)

        # Candidate points in world frame around ego position
        # Heading angle theta rotates the sensor cone
        dx_w = r_samples * np.cos(phi_samples + ego_heading)
        dy_w = r_samples * np.sin(phi_samples + ego_heading)
        xw = ego_x + dx_w
        yw = ego_y + dy_w

        # Determine surface height, roughness, and intensity based on true world Y
        # Road corridor: |Y_world| <= 4.5m
        # Curb step: 4.5m < |Y_world| <= 5.2m
        # Off-road / Grass / Gravel: |Y_world| > 5.2m
        zw = np.zeros(num_ground_pts, dtype=np.float32)
        intensities = np.zeros(num_ground_pts, dtype=np.float32)

        is_road = np.abs(yw) <= 4.5
        is_curb = (np.abs(yw) > 4.5) & (np.abs(yw) <= 5.2)
        is_offroad = np.abs(yw) > 5.2

        # Road points: flat z=0 with micro sensor noise
        zw[is_road] = rng.normal(0.0, 0.015, size=np.sum(is_road)).astype(np.float32)
        intensities[is_road] = (0.24 + 0.05 * np.sin(xw[is_road] * 2.0)).astype(np.float32)

        # Curb points: elevated step z=0.12m
        zw[is_curb] = (0.12 + rng.normal(0.0, 0.02, size=np.sum(is_curb))).astype(np.float32)
        intensities[is_curb] = (0.35 + rng.uniform(-0.05, 0.05, size=np.sum(is_curb))).astype(np.float32)

        # Off-road terrain points: rough, uneven surface z=0.15m + bumps
        zw[is_offroad] = (0.15 + rng.normal(0.0, 0.06, size=np.sum(is_offroad))).astype(np.float32)
        intensities[is_offroad] = (0.12 + rng.uniform(-0.04, 0.04, size=np.sum(is_offroad))).astype(np.float32)

        ground_world = np.column_stack([xw, yw, zw, intensities]).astype(np.float32)

        # Transform world ground points into ego sensor frame
        ground_sensor = self._apply_ego_motion(ground_world, ego_x, ego_y, ego_heading)

        # ----------------------------------------------------------------------
        # 2. Fixed Stationary Obstacles in World Space
        # ----------------------------------------------------------------------
        # All obstacles sit at constant, fixed world coordinates (X, Y, Z)
        stationary_obstacles = [
            # Obstacle 1: Pedestrian at X=32m, Y=1.8m (in left half of right lane)
            {"pos": (32.0, 1.8, 0.9), "size": (0.6, 0.6, 1.8), "pts": 100, "intensity": 0.85},
            # Obstacle 2: Stopped Vehicle at X=65m, Y=-2.0m (in right lane)
            {"pos": (65.0, -2.0, 0.75), "size": (4.5, 2.0, 1.5), "pts": 180, "intensity": 0.80},
            # Obstacle 3: Road Hazard Debris at X=100m, Y=1.5m (in left lane)
            {"pos": (100.0, 1.5, 0.6), "size": (1.8, 1.8, 1.2), "pts": 110, "intensity": 0.75},
            # Obstacle 4: Parked Truck on Right Shoulder at X=140m, Y=-4.6m
            {"pos": (140.0, -4.6, 1.4), "size": (6.5, 2.4, 2.6), "pts": 220, "intensity": 0.82},
            # Obstacle 5: Stationary Cyclist at X=180m, Y=-2.2m
            {"pos": (180.0, -2.2, 0.85), "size": (1.6, 0.7, 1.6), "pts": 95, "intensity": 0.78},
            # Obstacle 6: Pedestrian at X=225m, Y=0.0m (center line)
            {"pos": (225.0, 0.0, 0.9), "size": (0.6, 0.6, 1.8), "pts": 105, "intensity": 0.85},
        ]

        all_obj_sensor_pts: List[np.ndarray] = []
        for obs in stationary_obstacles:
            pos_w = obs["pos"]
            # Only synthesize points if obstacle is within reasonable LiDAR range of vehicle
            dist_to_ego = float(np.hypot(pos_w[0] - ego_x, pos_w[1] - ego_y))
            if dist_to_ego <= max_lidar_range + 5.0:
                cluster_w = self._generate_box_cluster(
                    rng,
                    center=pos_w,
                    size=obs["size"],
                    num_points=obs["pts"],
                    base_intensity=obs["intensity"],
                )
                cluster_sensor = self._apply_ego_motion(cluster_w, ego_x, ego_y, ego_heading)
                all_obj_sensor_pts.append(cluster_sensor)

        if all_obj_sensor_pts:
            pts = np.vstack([ground_sensor] + all_obj_sensor_pts).astype(np.float32)
        else:
            pts = ground_sensor

        return PointCloudFrame(
            points=pts,
            timestamp=timestamp,
            frame_id=self.frame_id,
            source=FrameSource.MOCK,
            metadata={
                "scenario": canonical,
                "frame_idx": frame_idx,
                "ego_x": round(float(ego_x), 2),
                "ego_y": round(float(ego_y), 2),
                "ego_heading": round(float(ego_heading), 3),
                "ego_speed_ms": round(float(ego_speed), 1),
            },
        )


