"""Oscilloscope timing helpers for timing-window analysis."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Dict, Any

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


@dataclass
class DetectionWindow:
    """Continuous interval where the AIL signal is below threshold."""

    start_s: float
    end_s: float

    @property
    def duration_s(self) -> float:
        return float(self.end_s - self.start_s)


@dataclass
class DiscardedCycleDetail:
    """Context about an individual cycle that was removed during filtering."""

    timestamp_s: float
    period_s: float
    reason: str
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class FaultTiming:
    """Hall-effect fault sensor timing diagnostics with variability metrics."""

    transition_times: List[float]
    cycle_periods: np.ndarray
    raw_cycle_periods: np.ndarray
    mean_period_s: float
    std_period_s: float
    min_period_s: float
    max_period_s: float
    coefficient_of_variation_pct: float
    valid_cycle_count: int
    discarded_cycle_count: int
    discarded_cycle_details: List[DiscardedCycleDetail]
    detection_quality_metrics: Dict[str, int]
    debug: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ScopeTiming:
    """Structured timing information extracted from a two-channel trace."""

    time: np.ndarray
    ail_voltage: np.ndarray
    fault_voltage: np.ndarray
    fault: FaultTiming
    ail_windows: List[DetectionWindow]


class ScopeProcessingError(RuntimeError):
    pass


def _ensure_sorted(time: np.ndarray, *signals: np.ndarray) -> tuple[np.ndarray, ...]:
    order = np.argsort(time)
    sorted_time = time[order]
    sorted_signals = tuple(sig[order] for sig in signals)
    return (sorted_time, *sorted_signals)


SMOOTHING_WINDOW_SECONDS = 10.0


def _estimate_sampling_interval(time: np.ndarray) -> Optional[float]:
    """Return the median positive sampling interval, or None if unavailable."""
    if time.size < 2:
        return None
    diffs = np.diff(time)
    finite = diffs[np.isfinite(diffs) & (diffs > 0)]
    if finite.size == 0:
        return None
    return float(np.median(finite))


def _smooth_signal(
    time: np.ndarray,
    values: np.ndarray,
    *,
    window_seconds: float = SMOOTHING_WINDOW_SECONDS,
) -> tuple[np.ndarray, Dict[str, Any]]:
    """Apply a moving-average smoother sized to approximately ``window_seconds``."""
    if values.size < 3:
        return values.astype(np.float64, copy=True), {'window_samples': 1, 'window_seconds': 0.0}

    total_duration = float(time[-1] - time[0]) if time.size >= 2 else 0.0
    if total_duration <= 0:
        return values.astype(np.float64, copy=True), {'window_samples': 1, 'window_seconds': 0.0}

    sampling_interval = _estimate_sampling_interval(time)
    if sampling_interval is None or sampling_interval <= 0:
        return values.astype(np.float64, copy=True), {'window_samples': 1, 'window_seconds': 0.0}

    quarter_duration = 0.25 * total_duration if total_duration > 0 else window_seconds
    effective_window_seconds = min(window_seconds, quarter_duration) if quarter_duration > 0 else window_seconds
    effective_window_seconds = max(effective_window_seconds, 3.0 * sampling_interval)

    window_samples = int(max(3, round(effective_window_seconds / sampling_interval)))
    max_reasonable = max(3, values.size // 5)
    window_samples = min(window_samples, max_reasonable)
    if window_samples % 2 == 0:
        window_samples += 1
    if window_samples >= values.size:
        window_samples = values.size - 1 if values.size % 2 == 0 else values.size
    if window_samples < 3:
        return values.astype(np.float64, copy=True), {'window_samples': 1, 'window_seconds': 0.0}

    kernel = np.ones(window_samples, dtype=np.float64) / float(window_samples)
    smoothed = np.convolve(values, kernel, mode='same')
    return smoothed.astype(np.float64, copy=False), {
        'window_samples': int(window_samples),
        'window_seconds': float(window_samples * sampling_interval),
    }


def _build_cycle_debug_plot(
    time: np.ndarray,
    raw_voltage: np.ndarray,
    smoothed_voltage: np.ndarray,
    transitions: List[float],
    high_threshold: float,
    low_threshold: float,
) -> Optional[Any]:
    """Return a matplotlib figure visualising detection diagnostics, if available."""
    try:
        import matplotlib.pyplot as plt
    except Exception:  # pragma: no cover - plotting fallback
        logger.debug('matplotlib unavailable; skipping cycle debug plot.')
        return None

    try:
        fig, ax = plt.subplots(figsize=(10, 4))
    except Exception:  # pragma: no cover - headless backends without Tk
        logger.debug('matplotlib backend unavailable; skipping cycle debug plot.', exc_info=True)
        return None
    ax.plot(time, raw_voltage, color='#9ca3af', linewidth=1.0, alpha=0.6, label='Raw fault')
    ax.plot(time, smoothed_voltage, color='#2563eb', linewidth=1.6, label='Smoothed (MA)')
    for boundary in transitions:
        ax.axvline(boundary, color='#ef4444', linestyle='--', linewidth=1.0, alpha=0.6)
    ax.axhline(high_threshold, color='#22c55e', linestyle=':', linewidth=1.0, alpha=0.8, label='High threshold')
    ax.axhline(low_threshold, color='#f97316', linestyle='-.', linewidth=1.0, alpha=0.8, label='Low threshold')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Fault voltage (V)')
    ax.set_title('Fault cycle detection debug view')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    return fig


def _detect_falling_transitions(
    time: np.ndarray,
    series: np.ndarray,
    high_threshold: float,
    low_threshold: float,
) -> List[float]:
    """Identify high-to-low crossings for a given signal."""
    if time.size < 2:
        return []
    state = 'high' if float(series[0]) >= high_threshold else 'low'
    transitions: List[float] = []
    for idx in range(1, time.size):
        prev = float(series[idx - 1])
        sample = float(series[idx])
        if state == 'high' and sample <= low_threshold and prev > low_threshold:
            crossing = _interpolate_crossing(
                float(time[idx - 1]),
                prev,
                float(time[idx]),
                sample,
                low_threshold,
            )
            transitions.append(crossing)
            state = 'low'
        elif state == 'low' and sample >= high_threshold and prev < high_threshold:
            state = 'high'
    return transitions


def parse_scope_csv(
    source: str | bytes | Iterable[str],
    *,
    time_column: str = 'time',
    ail_column: str = 'AIL_voltage',
    fault_column: str = 'fault_voltage',
) -> ScopeTiming:
    """Parse a CSV dump from the oscilloscope."""
    try:
        # Read all lines first to properly handle different formats
        if hasattr(source, 'read'):
            # File-like object - check if seekable before seeking
            # Type narrowing: source is a file-like object with read() method
            if hasattr(source, 'seekable') and callable(getattr(source, 'seekable')):
                if getattr(source, 'seekable')():  # type: ignore[misc]
                    getattr(source, 'seek')(0)  # type: ignore[misc]
            elif hasattr(source, 'seek'):
                # Has seek method but no seekable() check - try it
                try:
                    getattr(source, 'seek')(0)  # type: ignore[misc]
                except (AttributeError, OSError, IOError):
                    pass  # Already at position or non-seekable
            # If no seek method at all, just read from current position
            raw_content = getattr(source, 'read')()  # type: ignore[misc]
            if isinstance(raw_content, bytes):
                raw_content = raw_content.decode('utf-8')
            lines = raw_content.splitlines()
        elif isinstance(source, (str, bytes)):
            if isinstance(source, bytes):
                source = source.decode('utf-8')
            lines = source.splitlines()
        else:
            # Iterable of strings
            lines = list(source)
        
        if not lines:
            raise ScopeProcessingError('CSV file is empty')
        
        # Find the first row with numeric data
        data_start_idx = None
        for idx, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            # Try splitting by common delimiters
            tokens = None
            for delimiter in [',', '\t', ';', ' ']:
                parts = [p.strip() for p in line.split(delimiter) if p.strip()]
                if len(parts) >= 2:
                    tokens = parts
                    break
            
            if tokens and len(tokens) >= 2:
                try:
                    # Try to convert first two tokens to float
                    float(tokens[0])
                    float(tokens[1])
                    data_start_idx = idx
                    break
                except (ValueError, TypeError):
                    continue
        
        if data_start_idx is None:
            raise ScopeProcessingError('No numeric data found in CSV file')
        
        # Create a StringIO from the valid data lines
        import io
        valid_lines = '\n'.join(lines[data_start_idx:])
        data_stream = io.StringIO(valid_lines)
        
        # Now read with pandas, trying different delimiters
        frame = None
        for delimiter in [',', '\t', ';', r'\s+']:
            try:
                data_stream.seek(0)
                frame = pd.read_csv(data_stream, header=None, sep=delimiter, skip_blank_lines=True, engine='python')
                if not frame.empty and len(frame.columns) >= 2:
                    break
            except Exception:  # noqa: BLE001
                continue
        
        if frame is None or frame.empty:
            raise ScopeProcessingError('Failed to parse CSV data with any common delimiter')
        
        if len(frame.columns) < 2:
            raise ScopeProcessingError(f'CSV must have at least 2 columns, found {len(frame.columns)}')
        
        if len(frame) < 2:
            raise ScopeProcessingError(f'Insufficient data rows (found {len(frame)})')
            
    except ScopeProcessingError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise ScopeProcessingError(f'Failed to read scope data: {exc}') from exc

    # Use positional columns since we've stripped headers
    if len(frame.columns) < 2:
        raise ScopeProcessingError(f"Scope data must have at least 2 columns, found {len(frame.columns)}")
    
    # Try to use first column as time values
    try:
        first_col = frame.iloc[:, 0].to_numpy(dtype=np.float64)
    except (ValueError, TypeError) as exc:
        raise ScopeProcessingError(f'First column contains non-numeric data: {exc}') from exc
    
    if len(first_col) < 2:
        raise ScopeProcessingError("Scope data must have at least 2 rows")
    
    # Calculate differences to check if monotonic
    diffs = np.diff(first_col)
    is_monotonic = np.all(diffs >= 0) or np.all(diffs <= 0)

    # Check if all time differences are zero (repeated timestamps) - edge case
    # Use a tight tolerance for floating-point comparison
    has_zero_diffs = np.allclose(diffs, 0.0, atol=1e-12)

    if is_monotonic and np.all(np.isfinite(first_col)) and not has_zero_diffs:
        # Use first column as time, make it start at 0
        time = first_col - first_col[0]
    else:
        # Generate synthetic time based on inferred sampling
        # Use differences between values in first column if they look like intervals
        median_diff = float(np.median(np.abs(diffs[diffs != 0]))) if np.any(diffs != 0) else 1.0

        if median_diff > 0 and median_diff < 1.0:
            # Looks like intervals in seconds
            interval = median_diff
        else:
            # Default to unit interval when no valid time info is available
            interval = 1.0

        if has_zero_diffs:
            logger.warning(
                'Detected zero time differences in input data (all timestamps identical). '
                'Generating synthetic evenly-spaced time with interval=%.4f seconds.',
                interval,
            )

        time = np.arange(len(frame), dtype=np.float64) * interval
    
    # Second column is the voltage signal (AIL)
    try:
        ail = frame.iloc[:, 1].to_numpy(dtype=np.float64)
    except (ValueError, TypeError) as exc:
        raise ScopeProcessingError(f'Second column contains non-numeric data: {exc}') from exc
    
    # Third column is fault if it exists, otherwise duplicate AIL
    if len(frame.columns) >= 3:
        try:
            fault = frame.iloc[:, 2].to_numpy(dtype=np.float64)
        except (ValueError, TypeError):
            # If third column isn't numeric, use zeros
            fault = np.zeros_like(ail)
    else:
        # No fault signal, create a dummy one
        fault = np.zeros_like(ail)

    sorted_time, sorted_ail, sorted_fault = _ensure_sorted(time, ail, fault)

    fault_timing = extract_fault_cycles(sorted_time, sorted_fault)
    ail_windows = detect_ail_windows(sorted_time, sorted_ail)

    return ScopeTiming(
        time=sorted_time,
        ail_voltage=sorted_ail,
        fault_voltage=sorted_fault,
        fault=fault_timing,
        ail_windows=ail_windows,
    )




def extract_fault_cycles(
    time: np.ndarray,
    voltage: np.ndarray,
    *,
    high_threshold: float = 2.5,
    low_threshold: float = 1.5,
    min_cycle_s: float = 30.0,
    mad_threshold_factor: float = 3.5,
    enable_outlier_filtering: bool = True,
) -> FaultTiming:
    """Detect high-to-low transitions corresponding to pump cycles.

    The returned ``FaultTiming`` includes variability metrics that help
    validate temporal calibration quality. Periods that fall below
    ``min_cycle_s`` or appear as robust outliers are discarded so that
    noisy transitions do not skew the mean cycle estimate. A moving-average
    smoother is applied up-front (targeting roughly ten seconds or one-third
    of ``min_cycle_s``) to suppress high-frequency noise before evaluating the
    dual-threshold state machine.

    Parameters
    ----------
    time:
        Time axis for the trace.
    voltage:
        Fault signal samples.
    high_threshold:
        Threshold for detecting a falling edge start (transition to low).
    low_threshold:
        Threshold for confirming the low state.
    min_cycle_s:
        Minimum acceptable cycle period in seconds.
    mad_threshold_factor:
        Scaling factor applied to the MAD-derived robust limit.
    enable_outlier_filtering:
        When False, MAD-based outlier rejection is bypassed (useful for diagnostics).
    """
    if time.size < 2:
        raise ScopeProcessingError('Fault trace must include at least two samples')
    if voltage.size != time.size:
        raise ScopeProcessingError('Fault trace length must match time axis length')

    smoothing_target_seconds = min(SMOOTHING_WINDOW_SECONDS, max(0.25, min_cycle_s / 3.0))
    smoothed_voltage, smoothing_meta = _smooth_signal(
        time,
        voltage,
        window_seconds=smoothing_target_seconds,
    )
    smoothing_meta['requested_window_seconds'] = float(smoothing_target_seconds)
    smoothing_meta['initial_window_seconds'] = float(smoothing_meta.get('window_seconds', 0.0))
    smoothing_meta['initial_window_samples'] = int(smoothing_meta.get('window_samples', 0))
    smoothing_meta['fallback_used'] = False

    transitions = _detect_falling_transitions(time, smoothed_voltage, high_threshold, low_threshold)

    if len(transitions) < 2 and smoothing_meta.get('window_seconds', 0.0) > 0:
        fallback_target = max(0.05, smoothing_meta['window_seconds'] / 4.0)
        if fallback_target < smoothing_meta['window_seconds'] - 1e-6:
            fallback_smoothed, fallback_meta = _smooth_signal(
                time,
                voltage,
                window_seconds=fallback_target,
            )
            fallback_transitions = _detect_falling_transitions(
                time,
                fallback_smoothed,
                high_threshold,
                low_threshold,
            )
            if len(fallback_transitions) >= len(transitions):
                transitions = fallback_transitions
                smoothed_voltage = fallback_smoothed
                smoothing_meta.update(fallback_meta)
                smoothing_meta['fallback_used'] = True
                smoothing_meta['fallback_requested_window_seconds'] = float(fallback_target)

    if len(transitions) < 2:
        periods = np.array([], dtype=np.float64)
        raw_periods = np.array([], dtype=np.float64)
        discarded_details: List[DiscardedCycleDetail] = []
        detection_metrics = {
            'total_transitions_detected': len(transitions),
            'transitions_passing_min_cycle_s': 0,
            'transitions_failing_mad_filter': 0,
        }
    else:
        cycle_records: List[Dict[str, Any]] = []
        for idx in range(len(transitions) - 1):
            start_ts = float(transitions[idx])
            end_ts = float(transitions[idx + 1])
            period = float(end_ts - start_ts)
            cycle_records.append(
                {
                    'index': idx,
                    'start_timestamp': start_ts,
                    'end_timestamp': end_ts,
                    'period_s': period,
                    'discard_reason': None,
                    'metadata': {},
                },
            )

        raw_periods = np.asarray([record['period_s'] for record in cycle_records], dtype=np.float64)
        finite_mask = np.isfinite(raw_periods)
        finite_periods = raw_periods[finite_mask]

        if finite_periods.size:
            finite_mean = float(np.mean(finite_periods))
            finite_std = float(np.std(finite_periods)) if finite_periods.size > 1 else 0.0
            finite_min = float(np.min(finite_periods))
            finite_max = float(np.max(finite_periods))
            logger.debug(
                (
                    'Detected %d raw fault cycles (finite values): mean=%.4fs std=%.4fs '
                    'min=%.4fs max=%.4fs'
                ),
                finite_periods.size,
                finite_mean,
                finite_std,
                finite_min,
                finite_max,
            )

        discarded_details = []

        for record, is_finite in zip(cycle_records, finite_mask):
            if not is_finite:
                record['discard_reason'] = 'non_finite_period'
                detail_metadata = {'cycle_index': record['index']}
                discarded_details.append(
                    DiscardedCycleDetail(
                        timestamp_s=record['start_timestamp'],
                        period_s=record['period_s'],
                        reason='non_finite_period',
                        metadata=detail_metadata,
                    ),
                )

        # Filter by minimum cycle threshold
        min_filtered_records = []
        for record in cycle_records:
            if record['discard_reason']:
                continue
            if record['period_s'] < min_cycle_s:
                record['discard_reason'] = 'below_min_cycle'
                detail_metadata = {'cycle_index': record['index']}
                discarded_details.append(
                    DiscardedCycleDetail(
                        timestamp_s=record['start_timestamp'],
                        period_s=record['period_s'],
                        reason='below_min_cycle',
                        metadata=detail_metadata,
                    ),
                )
                logger.debug(
                    'Discarding cycle starting at %.4fs (period %.4fs) for being below min_cycle_s %.4fs',
                    record['start_timestamp'],
                    record['period_s'],
                    min_cycle_s,
                )
                continue
            min_filtered_records.append(record)

        detection_metrics = {
            'total_transitions_detected': len(transitions),
            'transitions_passing_min_cycle_s': len(min_filtered_records),
            'transitions_failing_mad_filter': 0,
        }

        if enable_outlier_filtering and len(min_filtered_records) >= 3:
            candidate_periods = np.asarray([record['period_s'] for record in min_filtered_records], dtype=np.float64)
            median = float(np.median(candidate_periods))
            deviations = np.abs(candidate_periods - median)
            mad = float(np.median(deviations))
            if mad > 0:
                robust_limit = mad_threshold_factor * 1.4826 * mad
                keep_records = []
                for record, deviation in zip(min_filtered_records, deviations):
                    if deviation <= robust_limit:
                        keep_records.append(record)
                    else:
                        record['discard_reason'] = 'mad_outlier'
                        metadata = {'deviation_s': float(deviation), 'robust_limit_s': float(robust_limit)}
                        metadata['cycle_index'] = record['index']
                        record['metadata'].update(metadata)
                        discarded_details.append(
                            DiscardedCycleDetail(
                                timestamp_s=record['start_timestamp'],
                                period_s=record['period_s'],
                                reason='mad_outlier',
                                metadata=metadata,
                            ),
                        )
                        detection_metrics['transitions_failing_mad_filter'] += 1
                        logger.debug(
                            (
                                'Discarding cycle starting at %.4fs (period %.4fs) as MAD outlier; '
                                'deviation %.4fs exceeds limit %.4fs (median %.4fs)'
                            ),
                            record['start_timestamp'],
                            record['period_s'],
                            deviation,
                            robust_limit,
                            median,
                        )
                min_filtered_records = keep_records
            else:
                logger.debug(
                    (
                        'MAD filtering skipped: zero dispersion detected among %d candidate periods '
                        '(median %.4fs)'
                    ),
                    len(min_filtered_records),
                    median,
                )
        elif not enable_outlier_filtering:
            logger.debug('MAD filtering disabled; using %d cycles that passed min_cycle_s check', len(min_filtered_records))

        periods = np.asarray([record['period_s'] for record in min_filtered_records], dtype=np.float64)

    valid_count = int(periods.size)
    total_cycles = len(transitions) - 1 if len(transitions) >= 1 else 0
    discarded_count = int(max(0, total_cycles - valid_count))

    mean_period = float(np.mean(periods)) if periods.size else float('nan')
    std_period = float(np.std(periods)) if periods.size > 1 else 0.0
    min_period = float(np.min(periods)) if periods.size else float('nan')
    max_period = float(np.max(periods)) if periods.size else float('nan')
    if periods.size and mean_period > 0:
        coefficient_of_variation = float(std_period / mean_period * 100.0)
    else:
        coefficient_of_variation = float('nan')

    detection_metrics['smoothing_window_samples'] = int(smoothing_meta.get('window_samples', 0))
    detection_metrics['smoothing_window_seconds'] = float(smoothing_meta.get('window_seconds', 0.0))

    transition_list = [float(ts) for ts in transitions]
    debug_data: Dict[str, Any] = {
        'time': np.asarray(time, dtype=np.float64),
        'raw_voltage': np.asarray(voltage, dtype=np.float64),
        'smoothed_voltage': np.asarray(smoothed_voltage, dtype=np.float64),
        'transitions': transition_list,
        'smoothing': smoothing_meta,
    }
    debug_plot = _build_cycle_debug_plot(
        np.asarray(time, dtype=np.float64),
        np.asarray(voltage, dtype=np.float64),
        np.asarray(smoothed_voltage, dtype=np.float64),
        transition_list,
        high_threshold,
        low_threshold,
    )
    if debug_plot is not None:
        debug_data['plot'] = debug_plot

    return FaultTiming(
        transition_times=transition_list,
        cycle_periods=periods,
        raw_cycle_periods=raw_periods,
        mean_period_s=mean_period,
        std_period_s=std_period,
        min_period_s=min_period,
        max_period_s=max_period,
        coefficient_of_variation_pct=coefficient_of_variation,
        valid_cycle_count=valid_count,
        discarded_cycle_count=discarded_count,
        discarded_cycle_details=discarded_details,
        detection_quality_metrics=detection_metrics,
        debug=debug_data,
    )


def _interpolate_crossing(
    t0: float,
    v0: float,
    t1: float,
    v1: float,
    threshold: float,
) -> float:
    if v1 == v0:
        return t1
    ratio = (threshold - v0) / (v1 - v0)
    ratio = float(np.clip(ratio, 0.0, 1.0))
    return float(t0 + ratio * (t1 - t0))


def detect_ail_windows(
    time: np.ndarray,
    voltage: np.ndarray,
    *,
    threshold: float = 1.0,
    min_duration_s: float = 0.01,
) -> List[DetectionWindow]:
    """Identify continuous windows where the AIL signal falls below the threshold."""
    if time.size == 0:
        return []

    below = voltage <= threshold
    windows: List[DetectionWindow] = []
    start_time: Optional[float] = float(time[0]) if below[0] else None

    for idx in range(1, time.size):
        prev_below = bool(below[idx - 1])
        curr_below = bool(below[idx])
        if not prev_below and curr_below:
            start_time = _interpolate_crossing(
                float(time[idx - 1]),
                float(voltage[idx - 1]),
                float(time[idx]),
                float(voltage[idx]),
                threshold,
            )
        elif prev_below and not curr_below and start_time is not None:
            end_time = _interpolate_crossing(
                float(time[idx - 1]),
                float(voltage[idx - 1]),
                float(time[idx]),
                float(voltage[idx]),
                threshold,
            )
            if end_time - start_time >= min_duration_s:
                windows.append(DetectionWindow(start_s=start_time, end_s=end_time))
            start_time = None

    if bool(below[-1]) and start_time is not None:
        end_time = float(time[-1])
        if end_time - start_time >= min_duration_s:
            windows.append(DetectionWindow(start_s=start_time, end_s=end_time))

    return windows


__all__ = [
    'DetectionWindow',
    'FaultTiming',
    'ScopeTiming',
    'ScopeProcessingError',
    'parse_scope_csv',
    'extract_fault_cycles',
    'detect_ail_windows',
]
