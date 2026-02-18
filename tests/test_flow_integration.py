import numpy as np
import pytest
from scipy.integrate import cumulative_trapezoid

from signal_toolkit.flow_integration import integrate_flow, integrate_flow_to_volume


def test_integrate_flow_scalar_constant() -> None:
    time = np.array([0.0, 1.0, 2.0, 3.0])
    flow = np.array([2.0, 2.0, 2.0, 2.0])
    result = integrate_flow(time, flow, cumulative=False)
    assert result == pytest.approx(6.0)


def test_integrate_flow_cumulative_matches_scipy() -> None:
    time = np.linspace(0.0, 10.0, 200)
    flow = 3.0 + 1.5 * np.sin(2 * np.pi * time / 5.0)
    expected = cumulative_trapezoid(flow, time, initial=0.0)
    result = integrate_flow(time, flow, cumulative=True)
    assert np.array_equal(result, expected)


def test_integrate_flow_validation() -> None:
    time = np.array([0.0, 1.0, 2.0])
    flow = np.array([1.0, 2.0, 3.0])

    with pytest.raises(AssertionError, match='matching shapes'):
        integrate_flow(time, flow[:-1], cumulative=False)

    with pytest.raises(AssertionError, match='At least 2 data points'):
        integrate_flow(time[:1], flow[:1], cumulative=False)

    bad_time = np.array([0.0, 2.0, 1.0])
    with pytest.raises(AssertionError, match='monotonically increasing'):
        integrate_flow(bad_time, flow, cumulative=False)


def test_integrate_flow_to_volume_constant() -> None:
    time = np.linspace(0.0, 60.0, 200)
    flow = np.full_like(time, 2.0)
    volume, error = integrate_flow_to_volume(time, flow)

    assert volume == pytest.approx(120.0, abs=1e-9)
    assert error < 1e-6


def test_integrate_flow_to_volume_sinusoidal() -> None:
    duration = 10.0
    period = 5.0
    omega = 2.0 * np.pi / period
    mean = 5.0
    amp = 2.0

    time = np.linspace(0.0, duration, 500)
    flow = mean + amp * np.sin(omega * time)
    volume, error = integrate_flow_to_volume(time, flow)

    assert volume == pytest.approx(mean * duration, rel=1e-3)
    assert error > 0.0


def test_integrate_flow_to_volume_theta_preserves_full_window_volume() -> None:
    time = np.linspace(0.0, 30.0, 300)
    flow = 3.0 + 1.5 * np.sin(2 * np.pi * time / 10.0)

    unshifted, _ = integrate_flow_to_volume(time, flow, theta=0.0)
    shifted, _ = integrate_flow_to_volume(time, flow, theta=1.0)

    assert shifted == pytest.approx(unshifted, rel=1e-2)


def test_integrate_flow_to_volume_template_scaling() -> None:
    measured_period = 50.0
    time = np.linspace(0.0, measured_period, 300)
    flow = np.full_like(time, 2.5)

    raw, _ = integrate_flow_to_volume(time, flow)
    normalized, _ = integrate_flow_to_volume(time, flow, template_period=55.0)

    assert normalized == pytest.approx(raw * (55.0 / 50.0), abs=1e-9)


def test_integrate_flow_to_volume_validation() -> None:
    time = np.linspace(0.0, 60.0, 100)
    flow = np.full_like(time, 2.0)

    with pytest.raises(AssertionError, match='matching shapes'):
        integrate_flow_to_volume(time, flow[:-1])

    with pytest.raises(AssertionError, match='At least 3 data points'):
        integrate_flow_to_volume(time[:2], flow[:2])

    bad_time = time.copy()
    bad_time[50] = bad_time[49]
    with pytest.raises(AssertionError, match='strictly increasing'):
        integrate_flow_to_volume(bad_time, flow)

    bad_flow = flow.copy()
    bad_flow[10] = -0.1
    with pytest.raises(AssertionError, match='non-negative'):
        integrate_flow_to_volume(time, bad_flow)

    with pytest.raises(AssertionError, match='10% of measurement'):
        integrate_flow_to_volume(time, flow, theta=10.0)

    with pytest.raises(AssertionError, match='template_period must be positive'):
        integrate_flow_to_volume(time, flow, template_period=-1.0)
