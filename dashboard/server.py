"""FastAPI real-time dashboard server for Fovea-LiDAR perception pipeline streaming."""

from __future__ import annotations

import asyncio
import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import numpy as np

from fovea_lidar.contracts.mapping import TraversabilityMap
from fovea_lidar.contracts.pipeline_result import PipelineResult
from fovea_lidar.contracts.point_cloud import PointCloudFrame
from fovea_lidar.io.replay import ReplayController
from fovea_lidar.io.synthetic import SyntheticSceneGenerator
from fovea_lidar.pipeline import FoveaPipeline

logger = logging.getLogger("fovea_dashboard")
logging.basicConfig(level=logging.INFO)

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
SAMPLES_DIR = BASE_DIR / "data" / "samples" / "synthetic"
FRONTEND_DIR = Path(__file__).resolve().parent / "frontend"

app = FastAPI(
    title="Fovea-LiDAR Real-Time Perception Dashboard",
    description="Live WebSocket streaming of multi-resolution adaptive perception, terrain, and tracking.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def add_no_cache_headers(request, call_next):
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

# Mount static frontend files
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


def list_available_scenarios() -> List[str]:
    """Discover available scenario sequences from data/samples/synthetic or fallback."""
    scenarios: List[str] = []
    if SAMPLES_DIR.exists() and SAMPLES_DIR.is_dir():
        for d in sorted(SAMPLES_DIR.iterdir()):
            if d.is_dir() and any(
                f.suffix.lower() in (".pcd", ".ply") for f in d.iterdir()
            ):
                scenarios.append(d.name)

        if "flagship_demo" in scenarios:
            scenarios.remove("flagship_demo")
            scenarios.insert(0, "flagship_demo")

    if not scenarios:
        # Fallback list from SyntheticSceneGenerator
        scenarios = [
            "flagship_demo",
            "multi_object_dense",
            "pedestrian_at_distance",
            "vehicle_crossing",
            "empty_road",
            "object_disappearing",
            "low_confidence_obstacle",
            "resolution_boundary_stress",
        ]
    return scenarios


def load_scenario_replay(scenario_name: str, loop: bool = True) -> ReplayController:
    """Load ReplayController for the given scenario name.

    Prefers pre-generated dataset files from data/samples/synthetic for real datasets;
    for synthetic scenarios, uses full 100-frame sequences with ego motion.
    """
    scenario_path = SAMPLES_DIR / scenario_name
    if scenario_path.exists() and scenario_path.is_dir():
        try:
            rc = ReplayController(scenario_path, speed=1.0, loop=loop)
            # Use pre-recorded files for real_world_highway or if >= 50 frames
            if rc.total_frames >= 50 or scenario_name == "real_world_highway":
                return rc
        except Exception as e:
            logger.warning(
                "Failed to load scenario files from %s (%s); falling back to generator.",
                scenario_path,
                e,
            )

    gen = SyntheticSceneGenerator(seed=42)
    frames, _ = gen.generate(scenario_name, num_frames=100)
    return ReplayController(frames, speed=1.0, loop=loop)



def serialize_frame_message(
    result: PipelineResult,
    scenario: str,
    total_frames: int,
    max_transport_points: int = 12000,
    max_transport_cells: int = 10000,
    session: Optional[Any] = None,
) -> Dict[str, Any]:
    """Serialize a PipelineResult instance into a schema-valid dashboard frame message.

    NOTE ON TRANSPORT DOWNSAMPLING (Rendering-Only Reduction):
    LiDAR point clouds frequently contain thousands of points per frame.
    To ensure low network bandwidth, 60fps browser rendering, and smooth WebSocket
    transmission, raw and semantic points are uniformly downsampled to at most
    max_transport_points samples.
    This transport downsampling is strictly rendering-only; the underlying
    FoveaPipeline operates on 100% of the authentic sensor returns.
    """
    # 1. Downsample raw points for transport
    raw_pts_arr = result.raw_frame.points
    n_raw = raw_pts_arr.shape[0]
    if n_raw > max_transport_points:
        step = max(1, n_raw // max_transport_points)
        raw_sampled = raw_pts_arr[::step][:max_transport_points]
    else:
        raw_sampled = raw_pts_arr

    has_intensity = raw_pts_arr.shape[1] >= 4
    raw_points_list: List[List[float]] = [
        [round(float(p[0]), 3), round(float(p[1]), 3), round(float(p[2]), 3)]
        for p in raw_sampled
    ]
    if has_intensity:
        raw_max = float(np.max(raw_sampled[:, 3])) if raw_sampled.shape[0] > 0 else 1.0
        norm_scale = (1.0 / raw_max) if raw_max > 1.0 else 1.0
        raw_intensities_list: List[float] = [
            round(float(np.clip(float(p[3]) * norm_scale, 0.0, 1.0)), 3)
            for p in raw_sampled
        ]
    else:
        raw_intensities_list: List[float] = [
            round(max(0.08, min(1.0, 1.0 - ((float(p[0])**2 + float(p[1])**2)**0.5) / 55.0)), 3)
            for p in raw_sampled
        ]

    # 2. Downsample semantic points for transport
    sem_pts = result.semantic_points
    n_sem = len(sem_pts)
    if n_sem > max_transport_points:
        step = max(1, n_sem // max_transport_points)
        sem_sampled = sem_pts[::step][:max_transport_points]
    else:
        sem_sampled = sem_pts

    semantic_points_list: List[Dict[str, Any]] = [
        {
            "x": round(float(sp.coordinates[0]), 3),
            "y": round(float(sp.coordinates[1]), 3),
            "z": round(float(sp.coordinates[2]), 3),
            "class": str(sp.semantic_class),
            "confidence": round(float(sp.confidence), 2),
        }
        for sp in sem_sampled
    ]

    # Build track risk mapping from risk_regions
    track_risk_map: Dict[str, str] = {}
    for rr in getattr(result, "risk_regions", []):
        if getattr(rr, "associated_object_id", None) is not None:
            obj_id = str(rr.associated_object_id)
            lvl_val = rr.risk_level.value if hasattr(rr.risk_level, "value") else str(rr.risk_level)
            track_risk_map[obj_id] = lvl_val

    # 3. Tracked objects
    tracked_list: List[Dict[str, Any]] = []
    for trk in result.tracked_objects:
        pos = [round(float(coord), 3) for coord in trk.position]
        vel = (
            [round(float(v), 3) for v in trk.velocity]
            if trk.velocity is not None
            else [0.0, 0.0, 0.0]
        )
        bbox = [round(float(b), 3) for b in trk.bounding_region]
        risk_lvl = "LOW"
        if hasattr(trk, "risk_level") and trk.risk_level is not None:
            risk_lvl = trk.risk_level.value if hasattr(trk.risk_level, "value") else str(trk.risk_level)
        elif str(trk.track_id) in track_risk_map:
            risk_lvl = track_risk_map[str(trk.track_id)]
        elif hasattr(trk, "object_id") and str(trk.object_id) in track_risk_map:
            risk_lvl = track_risk_map[str(trk.object_id)]

        is_coasting = bool(
            getattr(trk, "is_coasting", False)
            or (hasattr(trk, "last_seen") and trk.last_seen < getattr(result, "timestamp", 0.0) - 1e-4)
        )
        speed_ms = round(float(np.linalg.norm(vel)), 2)

        tracked_list.append(
            {
                "track_id": str(trk.track_id),
                "semantic_class": str(trk.semantic_class),
                "position": pos,
                "velocity": vel,
                "speed_ms": speed_ms,
                "bounding_region": bbox,
                "confidence": round(float(trk.confidence), 2),
                "dynamic": bool(trk.dynamic),
                "risk_level": risk_lvl,
                "is_coasting": is_coasting,
            }
        )

    # 4. Active fovea regions
    fovea_list = list(result.active_fovea_regions)

    # 5. Grid Cells & Traversability View
    tmap: Optional[TraversabilityMap] = result.traversability_map
    cells_list: List[Dict[str, Any]] = []
    map_cells = getattr(result.adaptive_map, "cells", {})

    for idx, (key, cell) in enumerate(map_cells.items()):
        if idx >= max_transport_cells:
            break
        lbl = tmap.get_label(key) if tmap else "SAFE"
        lbl_str = lbl.value if hasattr(lbl, "value") else str(lbl)
        sc = tmap.get_score(key) if tmap else float(cell.traversability_score)
        is_fov = bool(isinstance(key, tuple) and str(key[0]).startswith("fovea"))

        cells_list.append(
            {
                "key": list(key) if isinstance(key, (tuple, list)) else str(key),
                "bounds": [round(float(b), 3) for b in cell.spatial_bounds],
                "resolution": round(float(cell.resolution), 3),
                "elevation": round(float(cell.elevation), 3),
                "traversability_label": lbl_str,
                "traversability_score": round(float(sc), 2),
                "semantic_class": str(cell.semantic_class or "UNKNOWN"),
                "occupancy": round(float(getattr(cell, "occupancy", 0.0)), 3),
                "is_fovea": is_fov,
            }
        )

    # 6. Performance Telemetry Metrics
    metrics_snapshot = dict(result.metrics)
    fps = round(float(metrics_snapshot.get("fps", 0.0)), 1)
    total_ms = round(float(metrics_snapshot.get("total_ms", 0.0)), 2)

    meta = result.raw_frame.metadata or {}
    ego_x = getattr(session, "ego_x", None) if session else None
    if ego_x is None:
        ego_x = float(meta.get("ego_x", 0.0))
    ego_y = getattr(session, "ego_y", None) if session else None
    if ego_y is None:
        ego_y = float(meta.get("ego_y", 0.0))
    ego_heading = getattr(session, "ego_heading", None) if session else None
    if ego_heading is None:
        ego_heading = float(meta.get("ego_heading", 0.0))
    ego_speed = getattr(session, "ego_speed", None) if session else None
    if ego_speed is None:
        ego_speed = float(meta.get("ego_speed_ms", 0.0))
    target_speed = getattr(session, "target_speed", ego_speed) if session else ego_speed
    steering = getattr(session, "steering", 0.0) if session else 0.0
    gear = getattr(session, "gear", "D") if session else "D"
    drive_mode = getattr(session, "drive_mode", "manual") if session else "manual"

    return {
        "type": "frame",
        "frame_index": result.frame_index,
        "timestamp": round(float(result.timestamp), 3),
        "scenario": scenario,
        "total_frames": total_frames,
        "ego_x": round(float(ego_x), 2),
        "ego_y": round(float(ego_y), 2),
        "ego_heading": round(float(ego_heading), 3),
        "ego_speed_ms": round(float(ego_speed), 1),
        "target_speed_ms": round(float(target_speed), 1),
        "steering": round(float(steering), 3),
        "gear": str(gear),
        "drive_mode": str(drive_mode),
        "raw_points": raw_points_list,
        "raw_intensities": raw_intensities_list,
        "semantic_points": semantic_points_list,
        "tracked_objects": tracked_list,
        "active_fovea_regions": fovea_list,
        "cells": cells_list,
        "metrics": {
            "fps": fps,
            "total_ms": total_ms,
            "preprocessing_ms": metrics_snapshot.get("preprocessing_ms", 0.0),
            "perception_ms": metrics_snapshot.get("perception_ms", 0.0),
            "terrain_ms": metrics_snapshot.get("terrain_ms", 0.0),
            "extraction_ms": metrics_snapshot.get("extraction_ms", 0.0),
            "tracking_ms": metrics_snapshot.get("tracking_ms", 0.0),
            "risk_ms": metrics_snapshot.get("risk_ms", 0.0),
            "resolution_ms": metrics_snapshot.get("resolution_ms", 0.0),
            "mapping_ms": metrics_snapshot.get("mapping_ms", 0.0),
            "traversability_ms": metrics_snapshot.get("traversability_ms", 0.0),
        },
        "counts": {
            "raw_points": int(n_raw),
            "cleaned_points": int(result.cleaned_frame.points.shape[0]),
            "semantic_points": int(n_sem),
            "detected_objects": len(result.detected_objects),
            "tracked_objects": len(result.tracked_objects),
            "active_foveas": len(fovea_list),
            "total_cells": int(result.adaptive_map.total_cells),
        },
    }




class ConnectionSession:
    """Per-connection session encapsulating dedicated pipeline, replay, and interactive drive engine."""

    def __init__(self, scenario: str) -> None:
        self.scenario = scenario
        demo_cfg = BASE_DIR / "configs" / "demo.yaml"
        self.pipeline = FoveaPipeline(config=demo_cfg if demo_cfg.exists() else None)
        self.generator = SyntheticSceneGenerator(seed=42)
        self.replay = load_scenario_replay(scenario, loop=True)

        # Dynamic ego vehicle motion state
        self.ego_x: float = 0.0
        self.ego_y: float = 0.0
        self.ego_heading: float = 0.0
        self.ego_speed: float = 0.0
        self.target_speed: float = 0.0
        self.steering: float = 0.0
        self.gear: str = "P"
        self.drive_mode: str = "inspect"  # "inspect" (primary stationary LiDAR view) or "manual" (interactive drive demo)

        self.speed: float = 1.0
        self.paused: bool = False
        self.frame_index: int = 0
        self.sim_time: float = 0.0
        self.active_layers: Set[str] = {
            "raw",
            "semantics",
            "cells",
            "foveas",
            "tracks",
        }

    def reset_car(self) -> None:
        """Reset vehicle pose to starting line."""
        self.ego_x = 0.0
        self.ego_y = 0.0
        self.ego_heading = 0.0
        self.ego_speed = 8.0 if self.drive_mode == "manual" else 0.0
        self.target_speed = self.ego_speed
        self.steering = 0.0
        self.gear = "D" if self.drive_mode == "manual" else "P"
        self.frame_index = 0
        self.sim_time = 0.0
        self.pipeline.reset()
        if self.replay:
            self.replay.seek(0)

    def change_scenario(self, new_scenario: str) -> None:
        """Switch scenario, cleanly resetting pipeline and replay state."""
        self.scenario = new_scenario
        self.pipeline.reset()
        self.replay = load_scenario_replay(new_scenario, loop=True)
        self.replay.set_speed(self.speed)
        self.reset_car()

    def seek_and_reconstruct(self, target_frame: int) -> None:
        """Seek to target_frame and accurately reconstruct pipeline state."""
        clamped = max(0, min(int(target_frame), self.replay.total_frames - 1))
        self.pipeline.reset()
        for idx in range(clamped):
            if idx < len(self.replay.frames):
                frame = self.replay.frames[idx]
                self.pipeline.process_frame(frame)
        self.replay.seek(clamped)


# REST Endpoints
@app.get("/")
async def get_index() -> FileResponse:
    """Serve dashboard frontend web page."""
    index_file = FRONTEND_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return JSONResponse(
        {"message": "Fovea-LiDAR Dashboard API running. Frontend not found."},
        status_code=404,
    )


@app.get("/api/scenarios")
async def get_scenarios() -> List[str]:
    """List available sample scenarios."""
    return list_available_scenarios()


# WebSocket Streaming Endpoint
@app.websocket("/ws/stream")
async def websocket_stream(
    websocket: WebSocket,
    scenario: Optional[str] = Query("flagship_demo"),
) -> None:
    """WebSocket stream emitting real-time frame telemetry and accepting client controls."""
    await websocket.accept()
    active_scenario = scenario or "flagship_demo"
    session = ConnectionSession(active_scenario)

    logger.info("New WebSocket client connected for scenario: %s", active_scenario)

    async def control_consumer() -> None:
        """Process control messages received from the client."""
        while True:
            try:
                raw_data = await websocket.receive_text()
                data = json.loads(raw_data)
                msg_type = data.get("type", "")

                if msg_type == "drive":
                    # Interactive driving controls from client (W/S/A/D, speed slider, D-pad)
                    session.drive_mode = "manual"
                    if "speed" in data:
                        session.target_speed = max(-6.0, min(35.0, float(data["speed"])))
                    if "steer" in data:
                        session.steering = max(-0.85, min(0.85, float(data["steer"])))
                    if "gear" in data:
                        session.gear = str(data["gear"])
                elif msg_type == "set_speed":
                    session.drive_mode = "manual"
                    session.target_speed = max(0.0, min(35.0, float(data.get("speed", 0.0))))
                elif msg_type == "drive_mode":
                    mode = str(data.get("mode", "inspect")).lower()
                    session.drive_mode = mode
                    if mode in ("inspect", "cruise"):
                        session.ego_x = 0.0
                        session.ego_y = 0.0
                        session.ego_heading = 0.0
                        session.ego_speed = 0.0
                        session.target_speed = 0.0
                        session.steering = 0.0
                        session.gear = "P"
                        session.pipeline.reset()
                        if session.replay:
                            session.replay.seek(0)
                    elif mode in ("manual", "drive"):
                        session.drive_mode = "manual"
                        session.ego_speed = 8.0
                        session.target_speed = 8.0
                        session.gear = "D"
                elif msg_type == "reset_car":
                    session.reset_car()
                elif msg_type == "pause":
                    session.paused = True
                    session.replay.pause()
                elif msg_type == "resume":
                    session.paused = False
                    session.replay.resume()
                elif msg_type == "seek":
                    frame_num = int(data.get("frame", 0))
                    session.seek_and_reconstruct(frame_num)
                elif msg_type == "speed":
                    new_speed = float(data.get("value", 1.0))
                    session.speed = new_speed
                    session.replay.set_speed(new_speed)
                elif msg_type == "select_layer":
                    session.active_layers = set(data.get("layers", []))
                elif msg_type == "change_scenario":
                    new_scen = str(data.get("scenario", "")).strip()
                    if new_scen:
                        session.change_scenario(new_scen)
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.warning("Error processing client control message: %s", e)
                await asyncio.sleep(0.05)

    async def stream_producer() -> None:
        """Continuously process and emit frames over WebSocket with authentic pipeline execution."""
        dt = 0.10
        while True:
            try:
                if session.paused:
                    await asyncio.sleep(0.05)
                    continue

                frame: Optional[PointCloudFrame] = None

                if session.drive_mode == "manual" and session.scenario != "real_world_highway":
                    # --- Real-Time Interactive Driving Physics ---
                    rate = 0.35 if session.target_speed >= session.ego_speed else 0.55
                    session.ego_speed += (session.target_speed - session.ego_speed) * rate
                    if abs(session.ego_speed) < 0.04 and abs(session.target_speed) < 0.04:
                        session.ego_speed = 0.0

                    # Realistic vehicle kinematics: turn rate scales with vehicle velocity
                    eff_speed = session.ego_speed if abs(session.ego_speed) > 0.4 else (0.4 if abs(session.steering) > 0.05 else 0.0)
                    speed_sign = -1.0 if session.ego_speed < -0.05 else 1.0
                    turn_rate = session.steering * (min(abs(eff_speed), 10.0) / 4.5) * speed_sign
                    session.ego_heading += turn_rate * dt
                    session.ego_x += session.ego_speed * math.cos(session.ego_heading) * dt
                    session.ego_y += session.ego_speed * math.sin(session.ego_heading) * dt

                    session.sim_time += dt
                    session.frame_index += 1

                    # Dynamically synthesize LiDAR frame for current ego pose
                    frame = session.generator.generate_interactive_frame(
                        scenario_name=session.scenario,
                        ego_x=session.ego_x,
                        ego_y=session.ego_y,
                        ego_heading=session.ego_heading,
                        ego_speed=session.ego_speed,
                        timestamp=session.sim_time,
                        frame_idx=session.frame_index,
                    )
                else:
                    # --- Scenario Replay / Autonomous Cruise Mode ---
                    frame = session.replay.step()
                    if frame is None:
                        if session.replay.loop:
                            session.replay.seek(0)
                            session.pipeline.reset()
                            frame = session.replay.step()
                        else:
                            session.paused = True
                            await asyncio.sleep(0.1)
                            continue

                    # In LiDAR inspection mode: vehicle is stationary at sensor coordinate origin (0, 0, 0)
                    if session.drive_mode != "manual":
                        session.ego_x = 0.0
                        session.ego_y = 0.0
                        session.ego_heading = 0.0
                        session.ego_speed = 0.0
                        session.target_speed = 0.0
                        session.steering = 0.0
                        session.gear = "P"
                    elif frame is not None and frame.metadata:
                        session.ego_x = float(frame.metadata.get("ego_x", session.ego_x))
                        session.ego_y = float(frame.metadata.get("ego_y", session.ego_y))
                        session.ego_heading = float(frame.metadata.get("ego_heading", session.ego_heading))
                        session.ego_speed = float(frame.metadata.get("ego_speed_ms", session.ego_speed))
                        session.target_speed = session.ego_speed

                if frame is not None:
                    # Run authentic FoveaPipeline on the frame
                    result = session.pipeline.process_frame(frame)
                    msg = serialize_frame_message(
                        result=result,
                        scenario=session.scenario,
                        total_frames=session.replay.total_frames or 100,
                        session=session,
                    )
                    await websocket.send_text(json.dumps(msg))

                # Pacing delay scaled by playback speed (target ~100ms base frame dt)
                delay = max(0.02, dt / max(0.1, session.speed))
                await asyncio.sleep(delay)

            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error("Error in stream_producer: %s", e)
                try:
                    await websocket.send_text(
                        json.dumps({"type": "error", "message": str(e)})
                    )
                except Exception:
                    pass
                await asyncio.sleep(0.5)

    consumer_task = asyncio.create_task(control_consumer())
    producer_task = asyncio.create_task(stream_producer())

    try:
        done, pending = await asyncio.wait(
            [consumer_task, producer_task],
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in pending:
            task.cancel()
    except Exception as e:
        logger.warning("WebSocket session error: %s", e)
    finally:
        session.pipeline.reset()
        logger.info("WebSocket client disconnected.")

