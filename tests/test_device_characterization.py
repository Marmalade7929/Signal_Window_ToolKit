"""Tests for device characterization core logic."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

import signal_toolkit.device_characterization as dc
from signal_toolkit.device_characterization import (
    CHARACTERIZATION_SCHEMA_VERSION,
    CharacterizationResult,
    MeasurementRun,
    build_characterization,
    prepare_measurement_run,
)


@dataclass
class DatasetStub:
    frame: pd.DataFrame
    time_points: np.ndarray


def _make_synthetic_run(run_id: str, phase_shift: float = 0.0) -> MeasurementRun:
    time = np.linspace(0.0, 8.0, 801)
    base_flow = 100.0
    amplitude = 12.0
    flow = base_flow + amplitude * np.sin(2 * np.pi * (time + phase_shift) / 2.0)
    return MeasurementRun(
        device_id="pump-1",
        run_id=run_id,
        time=time,
        flow=flow,
    )


def test_build_characterization_detects_cycle_period() -> None:
    runs = [
        _make_synthetic_run("run-a", phase_shift=0.0),
        _make_synthetic_run("run-b", phase_shift=0.1),
    ]
    result = build_characterization(runs, trigger_period=2.0)

    assert isinstance(result, CharacterizationResult)
    assert abs(result.cycle_period_s - 2.0) < 0.05
    assert result.cycle_count >= 3


def test_characterization_integration_matches_expected_volume() -> None:
    runs = [_make_synthetic_run("run-main")]
    result = build_characterization(runs, trigger_period=2.0)

    expected_volume = 100.0 * result.cycle_period_s / 3600.0
    predicted_volume = result.integrate_volume(result.cycle_period_s)

    assert predicted_volume == pytest.approx(expected_volume, rel=1e-3)


def test_characterization_integrate_volume_from_time_supports_wrap() -> None:
    time = np.linspace(0.0, 8.0, 801)
    flow = np.full_like(time, 100.0)
    runs = [
        MeasurementRun(
            device_id="pump-const",
            run_id="run-time",
            time=time,
            flow=flow,
        )
    ]
    result = build_characterization(runs, trigger_period=2.0)

    duration_volume = result.integrate_volume(2.5)
    absolute_volume = result.integrate_volume_from_time(1.0, 3.5)

    assert absolute_volume > 0
    assert duration_volume == pytest.approx(absolute_volume, rel=1e-3)


def test_characterization_round_trip_serialization() -> None:
    runs = [_make_synthetic_run("run-serial")]
    result = build_characterization(runs, trigger_period=2.0)

    payload = result.to_dict()
    restored = CharacterizationResult.from_dict(payload)

    assert restored.device_id == result.device_id
    assert np.allclose(restored.time_axis, result.time_axis)
    assert np.allclose(restored.mean_flow_mlh, result.mean_flow_mlh)
    assert restored.cycle_period_s == pytest.approx(result.cycle_period_s)


def test_flow_validation_flags_out_of_range() -> None:
    runs = [_make_synthetic_run("run-out")]
    result = build_characterization(
        runs,
        trigger_period=2.0,
        expected_flow_range=(150.0, 160.0),
    )

    validation = result.metadata.get("flow_validation")
    assert isinstance(validation, dict)
    assert validation["within_bounds"] is False


def test_characterization_from_dict_rejects_future_schema_version() -> None:
    runs = [_make_synthetic_run("run-schema")]
    result = build_characterization(runs, trigger_period=2.0)
    payload = result.to_dict()
    payload["schema_version"] = CHARACTERIZATION_SCHEMA_VERSION + 1

    with pytest.raises(ValueError, match="Unsupported characterization version"):
        CharacterizationResult.from_dict(payload)


def test_characterization_from_dict_rejects_missing_required_keys() -> None:
    runs = [_make_synthetic_run("run-missing")]
    result = build_characterization(runs, trigger_period=2.0)
    payload = result.to_dict()
    payload.pop("cycle_flow_integral")

    with pytest.raises(ValueError, match="missing required keys"):
        CharacterizationResult.from_dict(payload)


def test_extract_trigger_points_filters_and_sorts_metadata_values() -> None:
    time = np.linspace(0.0, 8.0, 801)
    flow = 100.0 + 12.0 * np.sin(2 * np.pi * time / 2.0)
    metadata = {
        "trigger_times_s": [np.nan, -1.0, 4.0, 0.0, 2.0, 20.0],
    }
    run = MeasurementRun(
        device_id="pump-trigger",
        run_id="run-trigger",
        time=time,
        flow=flow,
        metadata=metadata,
    )

    trigger_points = dc._extract_trigger_points([run])
    assert trigger_points is not None
    assert np.array_equal(trigger_points, np.array([0.0, 2.0, 4.0], dtype=np.float64))

    result = build_characterization([run], trigger_period=2.0)
    assert result.metadata.get("trigger_source") == "metadata"
    assert result.metadata.get("trigger_point_count") == 3


def test_prepare_measurement_run_interpolates_non_finite_flow() -> None:
    time = np.linspace(0.0, 10.0, 101)
    flow = np.full_like(time, 60.0)
    flow[50] = np.nan
    frame = pd.DataFrame({"flow_rate": flow})
    dataset = DatasetStub(frame=frame, time_points=time)

    run = prepare_measurement_run(dataset, device_id="d1", run_id="r1")
    assert np.all(np.isfinite(run.flow))


def test_prepare_measurement_run_rejects_missing_flow_column() -> None:
    time = np.linspace(0.0, 10.0, 101)
    frame = pd.DataFrame({"not_flow_rate": np.ones_like(time)})
    dataset = DatasetStub(frame=frame, time_points=time)

    with pytest.raises(ValueError, match="missing 'flow_rate' column"):
        prepare_measurement_run(dataset, device_id="d1")


def test_slice_cycles_rejects_non_positive_trigger_period() -> None:
    axis = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    with pytest.raises(ValueError, match="Trigger period must be positive"):
        dc._slice_cycles_by_period(axis, trigger_period=0.0)
    with pytest.raises(ValueError, match="Trigger period must be positive"):
        dc._slice_cycles_by_period(axis, trigger_period=-1.0)


def test_build_cycle_template_rejects_empty_cycles() -> None:
    axis = np.linspace(0.0, 5.0, 51, dtype=np.float64)
    flow = np.full(axis.shape, 60.0, dtype=np.float64)
    with pytest.raises(ValueError, match="At least one cycle is required"):
        dc._build_cycle_template(axis, flow, [])
