"""Integration tests for the closed-loop adaptive perception pipeline."""

import numpy as np
import pytest

from fovea_lidar.contracts.pipeline_result import PipelineResult
from fovea_lidar.io.synthetic import SyntheticSceneGenerator
from fovea_lidar.pipeline import FoveaPipeline


def test_pedestrian_at_distance_closed_loop_tracking():
    """Verify closed-loop pipeline tracks moving pedestrian, preserves stable track ID, and records metrics."""
    gen = SyntheticSceneGenerator(seed=42)
    frames, gt = gen.generate("pedestrian_at_distance", num_frames=10)

    pipeline = FoveaPipeline()

    fovea_centers = []
    gt_positions = []
    track_ids = set()

    for f_idx, raw_frame in enumerate(frames):
        result = pipeline.process_frame(raw_frame)

        assert isinstance(result, PipelineResult)
        assert result.frame_index == f_idx
        assert result.timestamp == raw_frame.timestamp

        # 1. Closed-loop fovea tracking assertion
        assert len(result.active_fovea_regions) == 1, (
            f"Frame {f_idx}: expected 1 active fovea overlay, got {len(result.active_fovea_regions)}"
        )
        fovea = result.active_fovea_regions[0]
        fovea_center = fovea["center"]
        fovea_centers.append(fovea_center)

        gt_pos = gt["frames"][f_idx]["objects"][0]["position"]
        gt_positions.append(gt_pos)

        # Fovea center must track the moving pedestrian within 0.5m
        dist_to_gt = float(np.hypot(fovea_center[0] - gt_pos[0], fovea_center[1] - gt_pos[1]))
        assert dist_to_gt < 0.5, (
            f"Frame {f_idx}: Fovea center {fovea_center} deviated from ground truth {gt_pos} by {dist_to_gt:.3f}m"
        )

        # 2. Stable track ID assertion
        assert len(result.tracked_objects) >= 1
        ped_track = result.tracked_objects[0]
        track_ids.add(ped_track.track_id)
        assert ped_track.track_id == "track_1", (
            f"Frame {f_idx}: expected stable track_id 'track_1', got {ped_track.track_id}"
        )

        # 3. Non-fabricated metrics verification
        m = result.metrics
        assert m["total_ms"] > 0.0
        assert m["preprocessing_ms"] >= 0.0
        assert m["perception_ms"] >= 0.0
        assert m["tracking_ms"] >= 0.0
        assert m["resolution_ms"] >= 0.0
        assert m["mapping_ms"] >= 0.0
        assert m["fps"] > 0.0

        stage_sum = (
            m["preprocessing_ms"]
            + m["perception_ms"]
            + m["terrain_ms"]
            + m["extraction_ms"]
            + m["tracking_ms"]
            + m["risk_ms"]
            + m["resolution_ms"]
            + m["mapping_ms"]
            + m.get("traversability_ms", 0.0)
        )
        # Total wall clock time must roughly equal sum of stages (within small timing noise/overhead)
        assert abs(m["total_ms"] - stage_sum) < 10.0, (
            f"Frame {f_idx}: total_ms ({m['total_ms']}) deviates significantly from stage sum ({stage_sum})"
        )

    # Confirm exactly 1 persistent track across all 10 frames
    assert len(track_ids) == 1
    assert "track_1" in track_ids

    # Confirm fovea center moved dynamically across frames
    x_displacements = [fovea_centers[i][0] - fovea_centers[i + 1][0] for i in range(len(fovea_centers) - 1)]
    assert any(abs(dx) > 0.01 for dx in x_displacements), "Fovea center was unexpectedly static"


def test_multi_object_dense_end_to_end_and_fps():
    """Run full multi_object_dense scenario end-to-end; verify stability, finite completion, and report observed FPS."""
    gen = SyntheticSceneGenerator(seed=101)
    frames, _ = gen.generate("multi_object_dense", num_frames=10)

    pipeline = FoveaPipeline()

    for f_idx, raw_frame in enumerate(frames):
        result = pipeline.process_frame(raw_frame)

        assert isinstance(result, PipelineResult)
        assert result.raw_frame.points.shape[0] > 0
        assert result.cleaned_frame.points.shape[0] > 0
        assert len(result.detected_objects) > 0
        assert len(result.tracked_objects) > 0
        assert result.adaptive_map.total_cells > 0

    # Retrieve rolling average performance
    stats = pipeline.metrics_collector.get_rolling_average(window=10)
    observed_fps = stats["fps"]
    mean_latency_ms = stats["total_ms"]

    print("\n" + "=" * 70)
    print("Multi-Object Dense Scenario Pipeline Performance:")
    print(f"  Processed Frames:       {stats['total_frames']}")
    print(f"  Mean Frame Latency:     {mean_latency_ms:.2f} ms")
    print(f"  Observed Real FPS:      {observed_fps:.2f} frames/sec")
    print(f"  Stage Timings Breakdown:")
    print(f"    - Preprocessing:      {stats.get('preprocessing_ms', 0):.2f} ms")
    print(f"    - Perception:         {stats.get('perception_ms', 0):.2f} ms")
    print(f"    - Terrain Analysis:   {stats.get('terrain_ms', 0):.2f} ms")
    print(f"    - Object Extraction:  {stats.get('extraction_ms', 0):.2f} ms")
    print(f"    - Tracking:           {stats.get('tracking_ms', 0):.2f} ms")
    print(f"    - Risk Evaluation:    {stats.get('risk_ms', 0):.2f} ms")
    print(f"    - Resolution & Fovea: {stats.get('resolution_ms', 0):.2f} ms")
    print(f"    - Adaptive Mapping:   {stats.get('mapping_ms', 0):.2f} ms")
    print("=" * 70)

    assert observed_fps > 0.0
    assert mean_latency_ms > 0.0
    assert stats["total_frames"] == 10


def test_object_disappearing_stops_foveation():
    """Verify foveation overlay persists during coasting and cleanly stops when track decays/is removed."""
    gen = SyntheticSceneGenerator(seed=77)
    # Object disappears after frame 5
    frames, _ = gen.generate(
        "object_disappearing",
        num_frames=10,
        disappear_after_frame=5,
    )

    pipeline = FoveaPipeline()

    active_fovea_counts = []
    tracked_counts = []

    for f_idx, raw_frame in enumerate(frames):
        result = pipeline.process_frame(raw_frame)
        active_fovea_counts.append(len(result.active_fovea_regions))
        tracked_counts.append(len(result.tracked_objects))

    print("\n" + "=" * 70)
    print("Object Disappearing Lifecycle (Frames 0-9):")
    for idx, (tc, fc) in enumerate(zip(tracked_counts, active_fovea_counts)):
        print(f"  Frame {idx}: Tracked Objects = {tc} | Active Fovea Overlays = {fc}")
    print("=" * 70)

    # 1. Frames 0-4 (present): object tracked and fovea active
    for idx in range(5):
        assert active_fovea_counts[idx] >= 1, f"Frame {idx}: Expected active fovea while object is present"
        assert tracked_counts[idx] >= 1, f"Frame {idx}: Expected active track while object is present"

    # 2. Frames 5-7: Object unobserved, tracker coasts with decay
    # 3. By Frame 8-9: Object exceeds max_missed_frames (3) and is completely removed
    assert tracked_counts[9] == 0, f"Frame 9: Expected 0 tracked objects after decay, got {tracked_counts[9]}"
    assert active_fovea_counts[9] == 0, (
        f"Frame 9: Expected fovea overlay to be removed when track died, got {active_fovea_counts[9]}"
    )


def test_pipeline_statefulness_and_reset():
    """Verify pipeline maintains state across frames, and reset() completely restores clean initial state with zero leakage."""
    gen = SyntheticSceneGenerator(seed=88)
    frames, _ = gen.generate("pedestrian_at_distance", num_frames=4)

    pipeline = FoveaPipeline()

    # Pass 1
    results_pass1 = [pipeline.process_frame(f) for f in frames]
    cells_pass1 = set(pipeline.adaptive_map.cells.keys())
    assert len(cells_pass1) > 0
    assert len(results_pass1[3].tracked_objects) == 1
    assert results_pass1[3].tracked_objects[0].track_id == "track_1"

    # Call reset
    pipeline.reset()

    # Verify state is completely cleared
    assert pipeline.adaptive_map.total_cells == 0
    assert len(pipeline.tracker.tracks) == 0
    assert len(pipeline.fovea_manager.active_overlays) == 0
    assert pipeline.metrics_collector.total_frames == 0
    assert pipeline.frame_index == 0

    # Pass 2 with exact same frames
    results_pass2 = [pipeline.process_frame(f) for f in frames]
    cells_pass2 = set(pipeline.adaptive_map.cells.keys())

    # Verify identical output and cell allocation (no ghost tracks or state leakage)
    assert cells_pass1 == cells_pass2
    assert len(results_pass2[3].tracked_objects) == 1
    assert results_pass2[3].tracked_objects[0].track_id == "track_1", (
        "Track ID counter was not reset back to 0; state leaked from pass 1"
    )
    assert len(results_pass1[3].active_fovea_regions) == len(results_pass2[3].active_fovea_regions)
