"""Tests for pure orchestration helpers and quality policy."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import pytest

from signal_toolkit.app_logic import (
    build_prediction_export_frame,
    compute_drift_pct,
    evaluate_cycle_quality,
    normalize_uploaded_files,
    scope_filters_changed,
    should_use_fault_period,
)


def test_quality_policy_stable_boundaries() -> None:
    result = evaluate_cycle_quality(
        mean_period_s=110.0,
        model_period_s=100.0,
        cv_pct=5.0,
        valid_cycles=5,
    )
    assert result.quality == "Stable"


def test_quality_policy_watch_cv_boundary() -> None:
    result = evaluate_cycle_quality(
        mean_period_s=100.0,
        model_period_s=100.0,
        cv_pct=12.0,
        valid_cycles=5,
    )
    assert result.quality == "Watch"


def test_quality_policy_watch_drift_boundary() -> None:
    result = evaluate_cycle_quality(
        mean_period_s=120.0,
        model_period_s=100.0,
        cv_pct=1.0,
        valid_cycles=5,
    )
    assert result.quality == "Watch"


def test_quality_policy_unreliable_cv() -> None:
    result = evaluate_cycle_quality(
        mean_period_s=100.0,
        model_period_s=100.0,
        cv_pct=12.1,
        valid_cycles=5,
    )
    assert result.quality == "Unreliable"


def test_quality_policy_unreliable_drift() -> None:
    result = evaluate_cycle_quality(
        mean_period_s=121.0,
        model_period_s=100.0,
        cv_pct=1.0,
        valid_cycles=5,
    )
    assert result.quality == "Unreliable"


def test_quality_policy_unreliable_with_few_cycles() -> None:
    result = evaluate_cycle_quality(
        mean_period_s=100.0,
        model_period_s=100.0,
        cv_pct=1.0,
        valid_cycles=2,
    )
    assert result.quality == "Unreliable"


def test_quality_policy_defaults_to_watch_when_cv_and_drift_non_finite() -> None:
    result = evaluate_cycle_quality(
        mean_period_s=100.0,
        model_period_s=float("nan"),
        cv_pct=float("nan"),
        valid_cycles=5,
    )
    assert result.quality == "Watch"
    assert math.isnan(result.drift_pct)
    assert math.isnan(result.normalization_ratio)


def test_compute_drift_pct_returns_nan_for_invalid_inputs() -> None:
    assert math.isnan(compute_drift_pct(100.0, 0.0))
    assert math.isnan(compute_drift_pct(100.0, -1.0))
    assert math.isnan(compute_drift_pct(float("nan"), 100.0))
    assert math.isnan(compute_drift_pct(100.0, float("nan")))


def test_normalize_uploaded_files_handles_common_shapes() -> None:
    file_obj = object()
    assert normalize_uploaded_files(None) == []
    assert normalize_uploaded_files([file_obj]) == [file_obj]
    assert normalize_uploaded_files((file_obj,)) == [file_obj]
    assert normalize_uploaded_files(file_obj) == [file_obj]


def test_normalize_uploaded_files_handles_generators() -> None:
    first = object()
    second = object()

    def _gen():
        yield first
        yield second

    assert normalize_uploaded_files(_gen()) == [first, second]


def test_scope_filters_changed_detects_delta() -> None:
    current = {"fault_high": 2.5, "fault_low": 1.5}
    previous = {"fault_high": 2.5, "fault_low": 1.5}
    assert scope_filters_changed(previous, current, has_timing=True) is False
    assert scope_filters_changed({"fault_high": 2.7, "fault_low": 1.5}, current, has_timing=True) is True
    assert scope_filters_changed(None, current, has_timing=True) is True
    assert scope_filters_changed(previous, current, has_timing=False) is False


def test_should_use_fault_period_only_for_stable_watch() -> None:
    assert should_use_fault_period(True, "Stable") is True
    assert should_use_fault_period(True, "Watch") is True
    assert should_use_fault_period(True, "Unreliable") is False
    assert should_use_fault_period(False, "Stable") is False


@dataclass(frozen=True)
class PredictionSample:
    predicted_volume_ml: float
    known_volume_ml: Optional[float]
    error_pct: Optional[float]
    transit_start_s: float
    transit_end_s: float
    transit_duration_s: float
    cycles_elapsed: float
    measured_cycle_period_s: float
    measured_cycle_std_s: float
    cycle_period_min_s: Optional[float]
    cycle_period_max_s: Optional[float]
    cycle_period_cv_pct: float
    cycle_valid_count: int
    cycle_discarded_count: int
    model_cycle_period_s: float
    normalization_ratio: float
    time_offset_s: Optional[float]
    time_offset_provenance: Optional[str]
    search_range_s: Optional[float]


def test_build_prediction_export_frame_contains_provenance() -> None:
    prediction = PredictionSample(
        predicted_volume_ml=1.23,
        known_volume_ml=1.2,
        error_pct=2.5,
        transit_start_s=0.1,
        transit_end_s=1.1,
        transit_duration_s=1.0,
        cycles_elapsed=0.5,
        measured_cycle_period_s=2.0,
        measured_cycle_std_s=0.2,
        cycle_period_min_s=1.8,
        cycle_period_max_s=2.2,
        cycle_period_cv_pct=4.0,
        cycle_valid_count=10,
        cycle_discarded_count=1,
        model_cycle_period_s=2.0,
        normalization_ratio=1.0,
        time_offset_s=0.25,
        time_offset_provenance="calibrated",
        search_range_s=30.0,
    )
    frame = build_prediction_export_frame(prediction)
    assert frame.shape == (1, len(frame.columns))
    assert frame.loc[0, "time_offset_provenance"] == "calibrated"
    assert frame.loc[0, "search_range_s"] == pytest.approx(30.0)
    assert frame.loc[0, "predicted_volume_ml"] == pytest.approx(1.23)
