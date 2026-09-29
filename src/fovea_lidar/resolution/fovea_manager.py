"""Fovea overlay lifecycle manager coordinating ResolutionController and AdaptiveGridEngine."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Union
import numpy as np

from fovea_lidar.contracts.mapping import AdaptiveMap
from fovea_lidar.contracts.tracking import TrackedObject
from fovea_lidar.mapping.fovea_overlay import FoveaOverlay
from fovea_lidar.resolution.controller import ResolutionController
from fovea_lidar.utils.config import Config, load_config


class FoveaManager:
    """Manages creation, frame-to-frame tracking, and removal of local FoveaOverlays.

    Wires ResolutionController output into the grid engine by activating refined
    foveation overlays whenever the computed resolution is meaningfully finer than
    the base distance-band resolution, and removing them cleanly when no longer justified.

    Attributes
    ----------
    controller : ResolutionController
        Resolution decision engine.
    fovea_radius : float
        Radius in meters defining the spatial footprint of each fovea overlay.
    fovea_activation_threshold : float
        Minimum resolution improvement in meters (base_res - desired_res) needed to
        activate an overlay.
    active_overlays : Dict[str, FoveaOverlay]
        Dictionary of currently registered overlays keyed by region_id.
    """

    def __init__(
        self,
        controller: Optional[ResolutionController] = None,
        fovea_radius: float = 2.0,
        fovea_activation_threshold: float = 0.02,
    ) -> None:
        self.controller = controller or ResolutionController()
        self.fovea_radius = float(fovea_radius)
        self.fovea_activation_threshold = float(fovea_activation_threshold)
        self.active_overlays: Dict[str, FoveaOverlay] = {}

    def reset(self) -> None:
        """Clear all registered fovea overlays."""
        self.active_overlays.clear()

    @classmethod
    def load(
        cls, config: Optional[Union[Config, Dict[str, Any], Path, str]] = None
    ) -> FoveaManager:
        """Instantiate FoveaManager from system configuration."""
        if config is None:
            full_cfg = load_config()
        elif isinstance(config, (str, Path)):
            full_cfg = load_config(config)
        elif isinstance(config, (Config, dict)):
            full_cfg = config
        else:
            raise TypeError(f"Unsupported config type: {type(config).__name__}")

        fovea_cfg = full_cfg.get("semantic_foveation", {})
        if isinstance(fovea_cfg, Config):
            fovea_dict = fovea_cfg.to_dict()
        elif isinstance(fovea_cfg, dict):
            fovea_dict = fovea_cfg
        else:
            fovea_dict = {}

        controller = ResolutionController.load(full_cfg)
        return cls(
            controller=controller,
            fovea_radius=float(fovea_dict.get("fovea_radius", 2.0)),
            fovea_activation_threshold=float(
                fovea_dict.get("fovea_activation_threshold", 0.02)
            ),
        )

    def update(
        self,
        tracked_objects: Sequence[TrackedObject],
        adaptive_map: AdaptiveMap,
        terrain: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Update active fovea overlays based on current tracking and map state.

        Parameters
        ----------
        tracked_objects : Sequence[TrackedObject]
            Active tracks from MultiObjectTracker.
        adaptive_map : AdaptiveMap
            Target adaptive map instance to populate with fovea overlays.
        terrain : Optional[Dict[str, Any]]
            Optional terrain classification or hazard map.
        """
        context = {
            "tracked_objects": tracked_objects,
            "adaptive_map": adaptive_map,
            "terrain": terrain,
        }
        desired_resolutions = self.controller.compute(context)

        current_active_ids: Set[str] = set()

        for track in tracked_objects:
            region_id = f"fovea_{track.track_id}"
            desired_res = desired_resolutions.get(region_id)
            if desired_res is None:
                continue

            # Compare against base grid band resolution at track distance
            dist = float(np.hypot(track.position[0], track.position[1]))
            band_idx = adaptive_map.grid_engine.band_for_distance(dist)
            if band_idx is None:
                continue
            base_res = adaptive_map.grid_engine.resolution_for_band(band_idx)

            # Activate fovea only if resolution is meaningfully finer than base band
            if desired_res <= (base_res - self.fovea_activation_threshold):
                current_active_ids.add(region_id)
                center = (float(track.position[0]), float(track.position[1]))

                # Cleanly update overlay (remove previous displaced cells to avoid orphaned state)
                if region_id in self.active_overlays:
                    adaptive_map.grid_engine.remove_fovea_overlay(region_id)

                overlay = FoveaOverlay(
                    overlay_id=region_id,
                    center=center,
                    radius=self.fovea_radius,
                    resolution=desired_res,
                    associated_track_id=track.track_id,
                )
                self.active_overlays[region_id] = overlay
                adaptive_map.grid_engine.add_fovea_overlay(overlay)

        # Remove overlays for regions that no longer qualify
        stale_ids = [oid for oid in self.active_overlays if oid not in current_active_ids]
        for oid in stale_ids:
            self.active_overlays.pop(oid, None)
            adaptive_map.grid_engine.remove_fovea_overlay(oid)

    def get_active_fovea_regions(self) -> List[Dict[str, Any]]:
        """Return list of active foveation regions with bounds and resolution for dashboard."""
        regions: List[Dict[str, Any]] = []
        for overlay in self.active_overlays.values():
            regions.append(
                {
                    "region_id": overlay.overlay_id,
                    "center": overlay.center,
                    "radius": overlay.radius,
                    "resolution": overlay.resolution,
                    "region_bounds": overlay.bounds,
                    "associated_track_id": overlay.associated_track_id,
                }
            )
        return regions
