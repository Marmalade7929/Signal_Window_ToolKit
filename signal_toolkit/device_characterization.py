"""Device characterization pipeline for periodic flow modeling."""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from .flow_integration import integrate_flow

logger = logging.getLogger(__name__)

CHARACTERIZATION_SCHEMA_VERSION = 1


class DatasetLike(Protocol):
    """Duck-typed dataset interface used by prepare_measurement_run."""

    frame: pd.DataFrame
    time_points: Sequence[float]


@dataclass
class MeasurementRun:
    """Uniform representation of a single clean pump run."""

    device_id: str
    run_id: str
    time: np.ndarray
    flow: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.time = np.asarray(self.time, dtype=np.float64)
        self.flow = np.asarray(self.flow, dtype=np.float64)
        if self.time.size != self.flow.size:
            raise ValueError("Time and flow arrays must have equal length")
        if self.time.size < 2:
            raise ValueError("Measurement run requires at least two samples")
        order = np.argsort(self.time)
        self.time = self.time[order]
        self.flow = self.flow[order]
        self.time = self.time - float(self.time[0])

    @property
    def duration(self) -> float:
        return float(self.time[-1]) if self.time.size else 0.0

    @property
    def median_spacing(self) -> float:
        if self.time.size < 2:
            return 0.0
        diffs = np.diff(self.time)
        diffs = diffs[diffs > 0]
        return float(np.median(diffs)) if diffs.size else 0.0


@dataclass
class RunSummary:
    """Lightweight descriptor surfaced in UI/persistence."""

    run_id: str
    duration_s: float
    mean_flow_mlh: float
    sample_count: int


@dataclass
class CharacterizationResult:
    """Aggregated pump model derived from multiple measurement runs."""

    device_id: str
    run_ids: List[str]
    time_axis: np.ndarray
    mean_flow_mlh: np.ndarray
    std_flow_mlh: np.ndarray
    sample_counts: np.ndarray
    cycle_period_s: float
    cycle_period_std_s: float
    cycle_count: int
    cycle_phase: np.ndarray
    cycle_flow_mlh: np.ndarray
    cycle_flow_integral: np.ndarray
    run_summaries: List[RunSummary]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def integrate_volume(
        self,
        duration_s: float,
        *,
        measured_cycle_period: Optional[float] = None,
    ) -> float:
        """Integrate characterized flow over a duration (seconds)."""
        if duration_s <= 0:
            return 0.0
        base_period = float(measured_cycle_period or self.cycle_period_s)
        if base_period <= 0:
            raise ValueError("Cycle period must be positive for integration")

        cycles = duration_s / base_period
        full_cycles = int(np.floor(cycles))
        fractional_phase = cycles - full_cycles

        per_cycle_integral = self.cycle_flow_integral[-1]
        volume_ml = (per_cycle_integral * base_period / 3600.0) * full_cycles

        if fractional_phase > 0:
            partial_integral = float(
                np.interp(
                    fractional_phase,
                    self.cycle_phase,
                    self.cycle_flow_integral,
                )
            )
            volume_ml += partial_integral * base_period / 3600.0

        return float(volume_ml)

    def integrate_volume_from_time(
        self,
        start_time_s: float,
        end_time_s: float,
        *,
        measured_cycle_period: Optional[float] = None,
    ) -> float:
        """Integrate the model between absolute times."""
        if end_time_s <= start_time_s:
            return 0.0

        base_period = float(measured_cycle_period or self.cycle_period_s)
        if base_period <= 0:
            raise ValueError("Cycle period must be positive for integration")

        start_cycle_index = int(np.floor(start_time_s / base_period))
        end_cycle_index = int(np.floor(end_time_s / base_period))
        boundaries_crossed = end_cycle_index - start_cycle_index

        start_phase = (start_time_s % base_period) / base_period
        end_phase = (end_time_s % base_period) / base_period

        start_integral = float(np.interp(start_phase, self.cycle_phase, self.cycle_flow_integral))
        end_integral = float(np.interp(end_phase, self.cycle_phase, self.cycle_flow_integral))
        per_cycle_integral = self.cycle_flow_integral[-1]

        if boundaries_crossed == 0:
            net_integral = end_integral - start_integral
        else:
            first_segment = per_cycle_integral - start_integral
            last_segment = end_integral
            middle_cycles = boundaries_crossed - 1
            net_integral = first_segment + (middle_cycles * per_cycle_integral) + last_segment

        volume_ml = net_integral * base_period / 3600.0
        return float(volume_ml)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize characterization into JSON-friendly primitives."""
        return {
            "schema_version": CHARACTERIZATION_SCHEMA_VERSION,
            "device_id": self.device_id,
            "run_ids": list(self.run_ids),
            "time_axis": self.time_axis.tolist(),
            "mean_flow_mlh": self.mean_flow_mlh.tolist(),
            "std_flow_mlh": self.std_flow_mlh.tolist(),
            "sample_counts": self.sample_counts.tolist(),
            "cycle_period_s": float(self.cycle_period_s),
            "cycle_period_std_s": float(self.cycle_period_std_s),
            "cycle_count": int(self.cycle_count),
            "cycle_phase": self.cycle_phase.tolist(),
            "cycle_flow_mlh": self.cycle_flow_mlh.tolist(),
            "cycle_flow_integral": self.cycle_flow_integral.tolist(),
            "run_summaries": [asdict(summary) for summary in self.run_summaries],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "CharacterizationResult":
        """Rehydrate characterization from serialized form."""
        if not isinstance(payload, dict):
            raise ValueError("Characterization payload must be a dictionary")

        metadata = dict(payload.get("metadata") or {})
        schema_version = int(payload.get("schema_version", 0))
        if schema_version > CHARACTERIZATION_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported characterization version {schema_version}; update toolkit to load this file."
            )

        required_keys = [
            "device_id",
            "time_axis",
            "mean_flow_mlh",
            "std_flow_mlh",
            "sample_counts",
            "cycle_period_s",
            "cycle_phase",
            "cycle_flow_mlh",
            "cycle_flow_integral",
        ]
        missing_keys = [key for key in required_keys if key not in payload]
        if missing_keys:
            raise ValueError(
                f"Invalid characterization file: missing required keys: {', '.join(missing_keys)}"
            )

        run_summaries_payload = payload.get("run_summaries", [])
        run_summaries = [RunSummary(**summary) for summary in run_summaries_payload]

        return cls(
            device_id=str(payload["device_id"]),
            run_ids=list(payload.get("run_ids", [])),
            time_axis=np.asarray(payload["time_axis"], dtype=np.float64),
            mean_flow_mlh=np.asarray(payload["mean_flow_mlh"], dtype=np.float64),
            std_flow_mlh=np.asarray(payload["std_flow_mlh"], dtype=np.float64),
            sample_counts=np.asarray(payload["sample_counts"], dtype=np.int64),
            cycle_period_s=float(payload["cycle_period_s"]),
            cycle_period_std_s=float(payload.get("cycle_period_std_s", 0.0)),
            cycle_count=int(payload.get("cycle_count", 0)),
            cycle_phase=np.asarray(payload["cycle_phase"], dtype=np.float64),
            cycle_flow_mlh=np.asarray(payload["cycle_flow_mlh"], dtype=np.float64),
            cycle_flow_integral=np.asarray(payload["cycle_flow_integral"], dtype=np.float64),
            run_summaries=run_summaries,
            metadata=metadata,
        )


def save_characterization(result: CharacterizationResult, destination: Union[str, Path]) -> None:
    """Persist characterization result to a JSON file."""
    path = Path(destination)
    path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")


def load_characterization(source: Union[str, Path]) -> CharacterizationResult:
    """Load characterization result from disk."""
    path = Path(source)
    payload = json.loads(path.read_text(encoding="utf-8"))
    return CharacterizationResult.from_dict(payload)


def prepare_measurement_run(
    dataset: DatasetLike,
    device_id: str,
    *,
    run_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> MeasurementRun:
    """Create a MeasurementRun from a dataset-like object."""
    if "flow_rate" not in dataset.frame.columns:
        raise ValueError("Dataset is missing 'flow_rate' column")

    time = np.asarray(dataset.time_points, dtype=np.float64)
    flow = dataset.frame["flow_rate"].to_numpy(dtype=np.float64)

    finite = np.isfinite(flow)
    if not np.all(finite):
        if np.count_nonzero(finite) < 2:
            raise ValueError("Insufficient finite flow samples for interpolation")
        flow = np.interp(time, time[finite], flow[finite])

    resolved_run_id = run_id or f"run-{abs(hash((device_id, float(time[0]), float(time[-1]))))}"
    return MeasurementRun(
        device_id=device_id,
        run_id=resolved_run_id,
        time=time,
        flow=flow,
        metadata=dict(metadata or {}),
    )


def _resolved_resolution(runs: Sequence[MeasurementRun], *, default: float = 0.1) -> float:
    spacings = [run.median_spacing for run in runs if run.median_spacing > 0]
    if not spacings:
        return float(default)
    return float(max(min(np.median(spacings), 0.5), 0.01))


def _resample_run(time_axis: np.ndarray, run: MeasurementRun) -> np.ndarray:
    resampled = np.interp(time_axis, run.time, run.flow, left=np.nan, right=np.nan)
    resampled[time_axis > run.time[-1]] = np.nan
    return resampled


def _extract_trigger_points(runs: Sequence[MeasurementRun]) -> Optional[np.ndarray]:
    """Collect trigger instants from run metadata when available."""
    candidate_keys = (
        "trigger_points_s",
        "trigger_times_s",
        "trigger_times",
        "hardware_trigger_points_s",
        "hardware_trigger_times_s",
        "hardware_triggers_s",
        "hardware_triggers",
    )
    for run in runs:
        for key in candidate_keys:
            if key not in run.metadata:
                continue
            values = np.asarray(run.metadata[key], dtype=np.float64)
            finite_mask = np.isfinite(values)
            if np.count_nonzero(finite_mask) == 0:
                continue
            filtered = values[finite_mask]
            filtered = filtered[(filtered >= 0.0) & (filtered <= run.duration + 1e-9)]
            if filtered.size == 0:
                continue
            return np.sort(filtered.astype(np.float64))
    return None


def _slice_cycles_by_period(
    time_axis: np.ndarray,
    trigger_period: float,
    *,
    trigger_points: Optional[np.ndarray] = None,
) -> List[Dict[str, float]]:
    """Construct fixed-duration cycle windows aligned to trigger instants."""
    if time_axis.size < 2:
        return []
    if trigger_period <= 0:
        raise ValueError("Trigger period must be positive")

    start_time = float(time_axis[0])
    end_time = float(time_axis[-1])
    total_span = end_time - start_time
    if total_span < trigger_period:
        return []

    epsilon = np.finfo(np.float64).eps * max(1.0, abs(end_time))

    if trigger_points is not None and trigger_points.size:
        points = np.asarray(trigger_points, dtype=np.float64)
        points = points[np.isfinite(points)]
        points = points[(points >= start_time - epsilon) & (points < end_time)]
        points = np.unique(np.sort(points))
    else:
        count = int(np.floor(total_span / trigger_period))
        points = start_time + np.arange(count, dtype=np.float64) * trigger_period

    cycles: List[Dict[str, float]] = []
    if points.size == 0:
        return cycles

    for trigger_time in points:
        t_start = float(trigger_time)
        t_end = t_start + trigger_period
        if t_end > end_time + epsilon:
            break
        mask = (time_axis >= t_start - epsilon) & (time_axis <= t_end + epsilon)
        if np.count_nonzero(mask) < 2:
            continue
        cycles.append({"t_start": t_start, "t_end": t_end})

    return cycles


def _build_cycle_template(
    time_axis: np.ndarray,
    mean_flow_mlh: np.ndarray,
    cycles: Sequence[Dict[str, float]],
    *,
    samples_per_cycle: int = 200,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not cycles:
        raise ValueError("At least one cycle is required to build the template")

    phase_axis = np.linspace(0.0, 1.0, samples_per_cycle)
    valid_mask = np.isfinite(mean_flow_mlh)
    valid_time = time_axis[valid_mask]
    valid_flow = mean_flow_mlh[valid_mask]
    if valid_time.size < 2:
        raise ValueError("No valid samples available for cycle template")

    cycle_samples: List[np.ndarray] = []
    for cycle in cycles:
        start = cycle["t_start"]
        end = cycle["t_end"]
        if end <= start:
            continue
        sample_times = start + phase_axis * (end - start)
        sample_flow = np.interp(sample_times, valid_time, valid_flow)
        if np.count_nonzero(np.isfinite(sample_flow)) < samples_per_cycle // 2:
            continue
        if not np.all(np.isfinite(sample_flow)):
            continue
        cycle_samples.append(sample_flow)

    if not cycle_samples:
        raise ValueError("Unable to construct cycle template from data")

    stacked = np.vstack(cycle_samples)
    cycle_mean = np.nanmean(stacked, axis=0)
    integral = integrate_flow(phase_axis, cycle_mean, cumulative=True)
    return phase_axis, cycle_mean, integral


def build_characterization(
    runs: Sequence[MeasurementRun],
    *,
    trigger_period: float,
    resolution: Optional[float] = None,
    expected_period_range: Optional[Tuple[Optional[float], Optional[float]]] = None,
    expected_flow_range: Optional[Tuple[Optional[float], Optional[float]]] = None,
) -> CharacterizationResult:
    """Aggregate measurement runs into a trigger-period characterization model."""
    if not runs:
        raise ValueError("At least one measurement run is required")

    device_ids = {run.device_id for run in runs}
    if len(device_ids) > 1:
        raise ValueError("Characterization requires runs from a single device")
    device_id = runs[0].device_id

    trigger_period = float(trigger_period)
    if not np.isfinite(trigger_period) or trigger_period <= 0:
        raise ValueError("trigger_period must be a positive finite value")

    trigger_points = _extract_trigger_points(runs)

    trigger_within_expected: Optional[bool] = None
    if expected_period_range is not None:
        lower, upper = expected_period_range
        trigger_within_expected = True
        if lower is not None and trigger_period < lower:
            trigger_within_expected = False
        if upper is not None and trigger_period > upper:
            trigger_within_expected = False

    target_resolution = resolution or _resolved_resolution(runs)
    max_duration = float(max(run.duration for run in runs))

    max_time_points = 100_000
    expected_size = int(np.ceil(max_duration / target_resolution)) + 1
    if expected_size > max_time_points:
        adjusted_resolution = max_duration / (max_time_points - 1)
        logger.warning(
            "Resolution adjusted from %.4fs to %.4fs to cap points at %d",
            target_resolution,
            adjusted_resolution,
            max_time_points,
        )
        target_resolution = adjusted_resolution

    time_axis = np.arange(0.0, max_duration + target_resolution, target_resolution)

    resampled = np.vstack([_resample_run(time_axis, run) for run in runs])
    sample_counts = np.sum(~np.isnan(resampled), axis=0)

    mean_flow = np.full(time_axis.shape, np.nan, dtype=np.float64)
    std_flow = np.full(time_axis.shape, np.nan, dtype=np.float64)
    valid_column_indices = np.where(sample_counts > 0)[0]
    if valid_column_indices.size:
        valid_resampled = resampled[:, valid_column_indices]
        mean_flow[valid_column_indices] = np.nanmean(valid_resampled, axis=0)

        std_values = np.zeros(valid_column_indices.size, dtype=np.float64)
        multi_sample_mask = sample_counts[valid_column_indices] > 1
        if np.any(multi_sample_mask):
            std_values[multi_sample_mask] = np.nanstd(
                valid_resampled[:, multi_sample_mask],
                axis=0,
            )
        std_flow[valid_column_indices] = std_values

    valid_mask = sample_counts > 0
    if np.count_nonzero(valid_mask) < 10:
        raise ValueError("Insufficient overlap between runs for characterization")

    valid_time_axis = time_axis[valid_mask]
    cycles = _slice_cycles_by_period(
        valid_time_axis,
        trigger_period,
        trigger_points=trigger_points,
    )
    if not cycles:
        raise ValueError("Cycle slicing failed; verify trigger period and run duration provide complete cycles")

    durations = np.array([c["t_end"] - c["t_start"] for c in cycles], dtype=np.float64)
    expected_cycles = int(np.floor((valid_time_axis[-1] - valid_time_axis[0]) / trigger_period))
    skipped_cycles = max(expected_cycles - len(cycles), 0)
    cycle_period = float(trigger_period)
    cycle_period_std = float(np.std(durations)) if durations.size > 1 else 0.0

    phase_axis, cycle_flow, cycle_integral = _build_cycle_template(
        time_axis,
        mean_flow,
        cycles,
    )

    per_cycle_volume_ml = 0.0
    implied_flow_mlh = float("nan")
    if cycle_integral.size and cycle_period > 0:
        per_cycle_volume_ml = float(cycle_integral[-1] * cycle_period / 3600.0)
        implied_flow_mlh = float(per_cycle_volume_ml * 3600.0 / cycle_period)

    flow_validation: Optional[Dict[str, Any]] = None
    if expected_flow_range is not None:
        flow_min, flow_max = expected_flow_range
        flow_min = float(flow_min) if flow_min is not None and flow_min > 0 else None
        flow_max = float(flow_max) if flow_max is not None and flow_max > 0 else None
        if flow_min is not None or flow_max is not None:
            within_bounds = True
            messages: List[str] = []
            reference = None
            if flow_min is not None and (not np.isfinite(implied_flow_mlh) or implied_flow_mlh < flow_min):
                within_bounds = False
                messages.append(f"Below expected minimum ({flow_min:.2f} mL/h)")
                reference = flow_min
            if flow_max is not None and (not np.isfinite(implied_flow_mlh) or implied_flow_mlh > flow_max):
                within_bounds = False
                messages.append(f"Above expected maximum ({flow_max:.2f} mL/h)")
                if reference is None:
                    reference = flow_max
            deviation_pct = None
            if reference is not None and reference > 0 and np.isfinite(implied_flow_mlh):
                deviation_pct = float(abs(implied_flow_mlh - reference) / reference * 100.0)
            flow_validation = {
                "expected_min_mlh": flow_min,
                "expected_max_mlh": flow_max,
                "implied_flow_mlh": implied_flow_mlh,
                "within_bounds": within_bounds,
                "deviation_pct": deviation_pct,
                "messages": messages,
            }

    run_summaries = [
        RunSummary(
            run_id=run.run_id,
            duration_s=run.duration,
            mean_flow_mlh=float(np.mean(run.flow)),
            sample_count=int(run.time.size),
        )
        for run in runs
    ]

    metadata = dict(runs[0].metadata)
    metadata.update(
        {
            "resolution_s": target_resolution,
            "trigger_period_s": trigger_period,
            "trigger_source": "metadata" if trigger_points is not None else "uniform_period",
            "per_cycle_volume_ml": per_cycle_volume_ml,
            "implied_flow_mlh": implied_flow_mlh,
        }
    )
    if trigger_points is not None:
        metadata["trigger_point_count"] = int(trigger_points.size)
    if skipped_cycles > 0:
        metadata["cycles_skipped_due_to_partial_data"] = int(skipped_cycles)
    if trigger_within_expected is not None:
        metadata["trigger_period_within_expected"] = trigger_within_expected
    if flow_validation is not None:
        metadata["flow_validation"] = flow_validation
    if expected_period_range is not None:
        metadata["expected_period_range"] = tuple(
            float(value) if value is not None else None for value in expected_period_range
        )
    if expected_flow_range is not None:
        metadata["expected_flow_range"] = tuple(
            float(value) if value is not None else None for value in expected_flow_range
        )

    return CharacterizationResult(
        device_id=device_id,
        run_ids=[run.run_id for run in runs],
        time_axis=time_axis,
        mean_flow_mlh=mean_flow,
        std_flow_mlh=std_flow,
        sample_counts=sample_counts,
        cycle_period_s=cycle_period,
        cycle_period_std_s=cycle_period_std,
        cycle_count=len(cycles),
        cycle_phase=phase_axis,
        cycle_flow_mlh=cycle_flow,
        cycle_flow_integral=cycle_integral,
        run_summaries=run_summaries,
        metadata=metadata,
    )


__all__ = [
    "CHARACTERIZATION_SCHEMA_VERSION",
    "MeasurementRun",
    "RunSummary",
    "CharacterizationResult",
    "prepare_measurement_run",
    "build_characterization",
    "save_characterization",
    "load_characterization",
]
