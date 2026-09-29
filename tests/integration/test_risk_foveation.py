"""Integration test for risk-driven adaptive foveation and resolution refinement ablation."""

import numpy as np
import pytest

from fovea_lidar.contracts.risk import RiskLevel
from fovea_lidar.contracts.tracking import TrackedObject
from fovea_lidar.io.synthetic import SyntheticSceneGenerator
from fovea_lidar.mapping.adaptive_map import AdaptiveMap
from fovea_lidar.objects.extraction import ObjectExtractor
from fovea_lidar.perception.mock_model import MockModel
from fovea_lidar.preprocessing.pipeline import Preprocessor
from fovea_lidar.resolution.controller import ResolutionController
from fovea_lidar.resolution.fovea_manager import FoveaManager
from fovea_lidar.risk.engine import RiskEngine
from fovea_lidar.tracking.tracker import MultiObjectTracker
from fovea_lidar.utils.config import load_config


def test_risk_acceptance_criteria_high_vs_low_risk_objects():
    """Verify Acceptance Criteria 1:

    Fast pedestrian approaching ego produces a CRITICAL or HIGH risk region;
    a receding vehicle at 30m produces LOW risk.
    """
    engine = RiskEngine.load()

    # Fast pedestrian approaching ego at 6m distance, moving 2.5 m/s toward (0, 0)
    fast_ped = TrackedObject(
        object_id="ped_fast",
        track_id="ped_fast",
        semantic_class="PEDESTRIAN",
        position=(6.0, 0.0, 0.8),
        bounding_region=(5.5, -0.4, 0.0, 6.5, 0.4, 1.7),
        confidence=0.95,
        velocity=(-2.5, 0.0, 0.0),  # heading directly into ego
        dynamic=True,
        track_confidence=1.0,
    )
    risk_ped, level_ped = engine.compute_object_risk(fast_ped)

    # Receding vehicle at 30m distance, moving away from ego at 1.0 m/s
    receding_veh = TrackedObject(
        object_id="veh_receding",
        track_id="veh_receding",
        semantic_class="VEHICLE",
        position=(30.0, 0.0, 1.0),
        bounding_region=(28.0, -1.0, 0.0, 32.0, 1.0, 2.0),
        confidence=0.85,
        velocity=(1.0, 0.0, 0.0),  # heading away from ego
        dynamic=True,
        track_confidence=1.0,
    )
    risk_veh, level_veh = engine.compute_object_risk(receding_veh)

    print("\n" + "=" * 70)
    print("Risk Assessment Acceptance Criteria Verification:")
    print(f"  Fast Approaching Pedestrian: Risk = {risk_ped:.4f}, Level = {level_ped.value}")
    print(f"  Receding Vehicle at 30m:     Risk = {risk_veh:.4f}, Level = {level_veh.value}")
    print("=" * 70)

    # Acceptance Criteria 1 checks
    assert level_ped in (RiskLevel.CRITICAL, RiskLevel.HIGH), (
        f"Expected CRITICAL or HIGH risk for approaching pedestrian, got {level_ped.value}"
    )
    assert level_veh == RiskLevel.LOW, (
        f"Expected LOW risk for receding vehicle at 30m, got {level_veh.value} (score={risk_veh:.4f})"
    )
    assert risk_veh < 0.25


def test_risk_refinement_produces_measurably_finer_resolution():
    """Verify Acceptance Criteria 2 & 3:

    Disabling risk_refinement.enabled produces measurably coarser resolution around
    the high-risk object than when enabled.
    """
    config = load_config()

    # Controller with risk refinement enabled
    controller_risk_on = ResolutionController.load(config)
    assert controller_risk_on.risk_refinement_enabled is True

    # Controller with risk refinement ablated (disabled)
    cfg_disabled = config.to_dict()
    cfg_disabled["risk_refinement"]["enabled"] = False
    controller_risk_off = ResolutionController.load(cfg_disabled)
    assert controller_risk_off.risk_refinement_enabled is False

    # High-risk approaching obstacle at 20m distance
    # Distance baseline at 20m: R_dist = 0.05 + 0.007 * 20 = 0.190m
    high_risk_obstacle = TrackedObject(
        object_id="hazard_obj",
        track_id="hazard_obj",
        semantic_class="UNKNOWN_OBSTACLE",
        position=(20.0, 0.0, 0.8),
        bounding_region=(19.0, -1.0, 0.0, 21.0, 1.0, 1.8),
        confidence=0.90,
        velocity=(-4.0, 0.0, 0.0),  # 4 m/s directly approaching sensor origin
        dynamic=True,
    )

    context = {"tracked_objects": [high_risk_obstacle], "adaptive_map": None}

    res_on = controller_risk_on.compute(context)["fovea_hazard_obj"]
    res_off = controller_risk_off.compute(context)["fovea_hazard_obj"]

    resolution_diff = res_off - res_on
    improvement_pct = (resolution_diff / res_off) * 100.0

    print("\n" + "=" * 70)
    print("Risk Refinement Ablation Evaluation on High-Risk Obstacle:")
    print(f"  Resolution (Risk Refinement ENABLED):  {res_on:.4f} m")
    print(f"  Resolution (Risk Refinement DISABLED): {res_off:.4f} m")
    print(f"  Absolute Resolution Delta (Coarser - Finer): {resolution_diff:.4f} m")
    print(f"  Relative Grid Refinement Gain:              {improvement_pct:.2f} %")
    print("=" * 70)

    # Assert that enabled produces strictly and measurably finer resolution
    assert res_on < res_off
    assert resolution_diff >= 0.02, (
        f"Expected at least 0.02m (2cm) finer resolution with risk enabled, got {resolution_diff:.4f}m"
    )


def test_independent_ablation_risk_and_uncertainty():
    """Verify that risk and uncertainty refinement signals can be ablated completely independently."""
    base_cfg = load_config().to_dict()

    # 1. Both enabled
    ctrl_both = ResolutionController.load(base_cfg)

    # 2. Risk off, uncertainty on
    cfg_risk_off = load_config().to_dict()
    cfg_risk_off["risk_refinement"]["enabled"] = False
    ctrl_risk_off = ResolutionController.load(cfg_risk_off)

    # 3. Risk on, uncertainty off
    cfg_unc_off = load_config().to_dict()
    cfg_unc_off["uncertainty_refinement"]["enabled"] = False
    ctrl_unc_off = ResolutionController.load(cfg_unc_off)

    # 4. Both off
    cfg_both_off = load_config().to_dict()
    cfg_both_off["risk_refinement"]["enabled"] = False
    cfg_both_off["uncertainty_refinement"]["enabled"] = False
    ctrl_both_off = ResolutionController.load(cfg_both_off)

    # Evaluate on track with high risk and high uncertainty (low detection confidence)
    track = TrackedObject(
        object_id="ambiguous_speeding_obj",
        track_id="ambiguous_speeding_obj",
        semantic_class="PEDESTRIAN",
        position=(18.0, 0.0, 0.8),
        bounding_region=(17.5, -0.5, 0.0, 18.5, 0.5, 1.8),
        confidence=0.50,  # high uncertainty: 1 - 0.50 = 0.50
        velocity=(-3.0, 0.0, 0.0),  # approaching ego -> high risk
        dynamic=True,
    )
    context = {"tracked_objects": [track], "adaptive_map": None}

    r_both = ctrl_both.compute(context)["fovea_ambiguous_speeding_obj"]
    r_risk_off = ctrl_risk_off.compute(context)["fovea_ambiguous_speeding_obj"]
    r_unc_off = ctrl_unc_off.compute(context)["fovea_ambiguous_speeding_obj"]
    r_both_off = ctrl_both_off.compute(context)["fovea_ambiguous_speeding_obj"]

    print("\n" + "=" * 70)
    print("Multi-Signal Ablation Matrix for Ambiguous Approaching Obstacle:")
    print(f"  Both Active:       {r_both:.4f} m")
    print(f"  Risk Ablated:      {r_risk_off:.4f} m")
    print(f"  Uncertainty Ablated: {r_unc_off:.4f} m")
    print(f"  Both Ablated:      {r_both_off:.4f} m")
    print("=" * 70)

    # Turning risk off must increase cell size (make it coarser)
    assert r_both < r_risk_off
    # Turning uncertainty off must increase cell size compared to both on
    assert r_both < r_unc_off
    # Both off must be the coarsest
    assert r_both_off >= r_risk_off
    assert r_both_off >= r_unc_off
    assert r_both_off > r_both


def test_replay_pipeline_with_risk_refinement_overlay():
    """Verify complete replay through Preprocessor, MockModel, ObjectExtractor,

    MultiObjectTracker, RiskEngine, and FoveaManager with risk refinement.
    """
    gen = SyntheticSceneGenerator(seed=123)
    frames, _ = gen.generate("pedestrian_at_distance", num_frames=5)

    preprocessor = Preprocessor()
    model = MockModel()
    extractor = ObjectExtractor.load()
    tracker = MultiObjectTracker.load()
    fovea_manager = FoveaManager.load()
    adaptive_map = AdaptiveMap()

    for f_idx, raw_frame in enumerate(frames):
        cleaned_frame, _ = preprocessor.process(raw_frame)
        sem_points = model.predict(cleaned_frame)
        detections = extractor.extract(sem_points)
        tracked_objects = tracker.update(detections, timestamp=cleaned_frame.timestamp)

        fovea_manager.update(
            tracked_objects=tracked_objects,
            adaptive_map=adaptive_map,
        )
        adaptive_map.update(cleaned_frame, semantic_points=sem_points)

        active_foveas = fovea_manager.get_active_fovea_regions()
        assert len(active_foveas) >= 1
        # Target resolution must be refined to fine scale <= 0.08m
        assert active_foveas[0]["resolution"] <= 0.08
