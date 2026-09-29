"""Unit tests for ConstantVelocityKalmanFilter."""

import numpy as np
import pytest

from fovea_lidar.tracking.kalman import ConstantVelocityKalmanFilter


def test_kalman_initialization():
    """Verify initial state, covariance, and property getters."""
    kf = ConstantVelocityKalmanFilter(
        initial_position=(10.0, -5.0, 1.5),
        initial_velocity=(0.0, 0.0, 0.0),
        process_noise_std=1.0,
        measurement_noise_std=0.15,
    )
    assert kf.position == (10.0, -5.0, 1.5)
    assert kf.velocity == (0.0, 0.0, 0.0)
    assert kf.speed == pytest.approx(0.0)
    assert kf.direction == (1.0, 0.0, 0.0)
    assert kf.state.shape == (6,)
    assert kf.covariance.shape == (6, 6)


def test_kalman_predict_constant_velocity():
    """Verify state prediction adheres to kinematic constant-velocity equation."""
    kf = ConstantVelocityKalmanFilter(
        initial_position=(0.0, 0.0, 0.0),
        initial_velocity=(2.0, -4.0, 1.0),
    )
    pred_state = kf.predict(dt=0.5)
    assert pred_state.shape == (6,)
    np.testing.assert_allclose(kf.position, (1.0, -2.0, 0.5), atol=1e-5)
    np.testing.assert_allclose(kf.velocity, (2.0, -4.0, 1.0), atol=1e-5)


def test_kalman_zero_or_negative_dt_handling():
    """Verify filter gracefully ignores dt <= 0 without error or modification."""
    kf = ConstantVelocityKalmanFilter(initial_position=(1.0, 2.0, 3.0))
    state_before = kf.state.copy()
    kf.predict(dt=0.0)
    np.testing.assert_array_equal(kf.state, state_before)
    kf.predict(dt=-0.5)
    np.testing.assert_array_equal(kf.state, state_before)


def test_kalman_velocity_convergence_on_synthetic_trajectory():
    """Verify Kalman filter velocity estimate converges to within 0.2 m/s of ground truth."""
    rng = np.random.RandomState(42)
    true_velocity = np.array([3.0, -2.0, 0.0])  # m/s
    initial_pos = np.array([5.0, 10.0, 1.0])    # m
    dt = 0.1                                    # 10 Hz (100 ms)

    kf = ConstantVelocityKalmanFilter(
        initial_position=(initial_pos[0], initial_pos[1], initial_pos[2]),
        initial_velocity=(0.0, 0.0, 0.0),  # zero initial velocity belief
        process_noise_std=1.0,
        measurement_noise_std=0.10,
    )

    current_pos = initial_pos.copy()
    for step in range(35):
        current_pos = current_pos + true_velocity * dt
        noise = rng.normal(0, 0.05, size=3)
        measured_pos = current_pos + noise

        kf.predict(dt=dt)
        kf.update(measured_pos)

    # Verify velocity estimate has converged within documented tolerance (0.20 m/s)
    est_vel = np.array(kf.velocity)
    vel_error = np.linalg.norm(est_vel - true_velocity)
    assert vel_error < 0.20, f"Kalman velocity error {vel_error:.3f} exceeds 0.20 m/s tolerance"

    # Verify position tracking accuracy
    pos_error = np.linalg.norm(np.array(kf.position) - current_pos)
    assert pos_error < 0.15, f"Kalman position error {pos_error:.3f} exceeds 0.15 m tolerance"


def test_kalman_direction_unit_vector():
    """Verify direction property produces a valid 3D unit vector when moving."""
    kf = ConstantVelocityKalmanFilter(
        initial_position=(0.0, 0.0, 0.0),
        initial_velocity=(3.0, 4.0, 0.0),  # speed = 5.0 m/s
    )
    assert kf.speed == pytest.approx(5.0)
    direction = kf.direction
    assert np.linalg.norm(direction) == pytest.approx(1.0, abs=1e-5)
    assert direction == pytest.approx((0.6, 0.8, 0.0), abs=1e-5)
