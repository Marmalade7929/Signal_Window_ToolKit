"""Pure helper logic for orchestration and cycle-quality policy."""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Optional, Protocol

import numpy as np
import pandas as pd

CV_STABLE_MAX_PCT = 5.0
CV_WATCH_MAX_PCT = 12.0
DRIFT_STABLE_MAX_PCT = 10.0
DRIFT_WATCH_MAX_PCT = 20.0
MIN_VALID_CYCLES = 3


@dataclass(frozen=True)
class CycleQuality:
    quality: str
    color: str
    message: str
    recommendation: str
    normalization_ratio: float
    cv_pct: float
    drift_pct: float
    valid_cycles: int


class PredictionExportInput(Protocol):
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


def compute_drift_pct(mean_period_s: float, model_period_s: float) -> float:
    """Compute absolute cycle-period drift relative to the model period."""
    if (
        not np.isfinite(mean_period_s)
        or not np.isfinite(model_period_s)
        or model_period_s <= 0
    ):
        return float("nan")
    return float(abs(mean_period_s - model_period_s) / model_period_s * 100.0)


def _metric_severity(value: float, *, stable_max: float, watch_max: float) -> Optional[int]:
    if not np.isfinite(value):
        return None
    if value <= stable_max:
        return 0
    if value <= watch_max:
        return 1
    return 2


def evaluate_cycle_quality(
    mean_period_s: float,
    model_period_s: float,
    cv_pct: float,
    valid_cycles: int,
) -> CycleQuality:
    """Apply CV/drift policy for cycle timing confidence."""
    ratio = float("nan")
    if model_period_s > 0 and np.isfinite(model_period_s) and np.isfinite(mean_period_s):
        ratio = float(mean_period_s) / float(model_period_s)

    cv_value = float(cv_pct) if np.isfinite(cv_pct) else float("nan")
    drift_value = compute_drift_pct(mean_period_s, model_period_s)
    cycles = int(valid_cycles or 0)

    if cycles < MIN_VALID_CYCLES or not np.isfinite(mean_period_s):
        return CycleQuality(
            quality="Unreliable",
            color="#e76f51",
            message="Fewer than 3 usable cycles.",
            recommendation="Capture a longer trace before relying on calibration.",
            normalization_ratio=ratio,
            cv_pct=cv_value,
            drift_pct=drift_value,
            valid_cycles=cycles,
        )

    cv_severity = _metric_severity(
        cv_value,
        stable_max=CV_STABLE_MAX_PCT,
        watch_max=CV_WATCH_MAX_PCT,
    )
    drift_severity = _metric_severity(
        drift_value,
        stable_max=DRIFT_STABLE_MAX_PCT,
        watch_max=DRIFT_WATCH_MAX_PCT,
    )

    severities = [severity for severity in (cv_severity, drift_severity) if severity is not None]
    worst = max(severities) if severities else 1

    cv_text = f"{cv_value:.1f}%" if np.isfinite(cv_value) else "n/a"
    drift_text = f"{drift_value:.1f}%" if np.isfinite(drift_value) else "n/a"

    if worst >= 2:
        return CycleQuality(
            quality="Unreliable",
            color="#e76f51",
            message=f"Cycle timing is unreliable (CV {cv_text}, drift {drift_text}).",
            recommendation="Inspect jitter/drift and re-characterize before trusting calibrated offsets.",
            normalization_ratio=ratio,
            cv_pct=cv_value,
            drift_pct=drift_value,
            valid_cycles=cycles,
        )
    if worst == 1:
        return CycleQuality(
            quality="Watch",
            color="#f4a261",
            message=f"Cycle timing needs review (CV {cv_text}, drift {drift_text}).",
            recommendation="Review jitter/drift and apply calibration for best accuracy.",
            normalization_ratio=ratio,
            cv_pct=cv_value,
            drift_pct=drift_value,
            valid_cycles=cycles,
        )
    return CycleQuality(
        quality="Stable",
        color="#2a9d8f",
        message=f"Consistent cycle measurement (CV {cv_text}, drift {drift_text}).",
        recommendation="",
        normalization_ratio=ratio,
        cv_pct=cv_value,
        drift_pct=drift_value,
        valid_cycles=cycles,
    )


def normalize_uploaded_files(uploaded_files: Any) -> list[Any]:
    """Normalize uploader output into a list."""
    if uploaded_files is None:
        return []
    if isinstance(uploaded_files, list):
        return uploaded_files
    if isinstance(uploaded_files, tuple):
        return list(uploaded_files)
    if isinstance(uploaded_files, Iterable) and not isinstance(
        uploaded_files,
        (str, bytes, bytearray, Mapping),
    ):
        return list(uploaded_files)
    return [uploaded_files]


def scope_filters_changed(
    previous_filters: Optional[Mapping[str, object]],
    current_filters: Mapping[str, object],
    *,
    has_timing: bool,
) -> bool:
    """Return True when scope filter values changed after timing is available."""
    if not has_timing:
        return False
    if not previous_filters:
        return True
    return any(current_filters[key] != previous_filters.get(key) for key in current_filters)


def should_use_fault_period(has_signal: bool, quality: str) -> bool:
    """Only Stable/Watch timing should drive cycle-period normalization."""
    return bool(has_signal and quality in {"Stable", "Watch"})


def build_prediction_export_frame(prediction: PredictionExportInput) -> pd.DataFrame:
    """Create a one-row export payload for prediction metadata."""
    return pd.DataFrame(
        {
            "predicted_volume_ml": [prediction.predicted_volume_ml],
            "known_volume_ml": [prediction.known_volume_ml],
            "error_pct": [prediction.error_pct],
            "transit_start_s": [prediction.transit_start_s],
            "transit_end_s": [prediction.transit_end_s],
            "transit_duration_s": [prediction.transit_duration_s],
            "cycles_elapsed": [prediction.cycles_elapsed],
            "measured_cycle_period_s": [prediction.measured_cycle_period_s],
            "measured_cycle_std_s": [prediction.measured_cycle_std_s],
            "cycle_period_min_s": [prediction.cycle_period_min_s],
            "cycle_period_max_s": [prediction.cycle_period_max_s],
            "cycle_period_cv_pct": [prediction.cycle_period_cv_pct],
            "cycle_valid_count": [prediction.cycle_valid_count],
            "cycle_discarded_count": [prediction.cycle_discarded_count],
            "model_cycle_period_s": [prediction.model_cycle_period_s],
            "normalization_ratio": [prediction.normalization_ratio],
            "time_offset_s": [prediction.time_offset_s],
            "time_offset_provenance": [prediction.time_offset_provenance],
            "search_range_s": [prediction.search_range_s],
        }
    )


__all__ = [
    "CycleQuality",
    "build_prediction_export_frame",
    "compute_drift_pct",
    "evaluate_cycle_quality",
    "normalize_uploaded_files",
    "scope_filters_changed",
    "should_use_fault_period",
]
