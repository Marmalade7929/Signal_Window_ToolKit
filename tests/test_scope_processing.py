"""Tests for oscilloscope scope processing helpers."""
import io

import numpy as np
import pytest

import signal_toolkit.scope_processing as sp
from signal_toolkit.scope_processing import (
    ScopeProcessingError,
    detect_ail_windows,
    extract_fault_cycles,
    parse_scope_csv,
)


def test_extract_fault_cycles_square_wave() -> None:
    time = np.linspace(0.0, 6.0, 6001)
    fault_voltage = np.where(np.sin(2 * np.pi * time) > 0, 3.3, 0.0)

    timing = extract_fault_cycles(
        time,
        fault_voltage,
        high_threshold=2.5,
        low_threshold=1.5,
        min_cycle_s=0.5,
    )

    assert timing.cycle_periods.size >= 4
    assert timing.mean_period_s == pytest.approx(1.0, rel=1e-2)
    assert timing.raw_cycle_periods.size >= timing.cycle_periods.size
    assert 'total_transitions_detected' in timing.detection_quality_metrics
    assert 'smoothed_voltage' in timing.debug
    assert timing.debug['smoothed_voltage'].shape == fault_voltage.shape
    assert timing.debug['smoothing']['window_samples'] >= 3


def test_extract_fault_cycles_respects_outlier_filter_toggle() -> None:
    time = np.linspace(0.0, 8.0, 80001)  # 0.0001s resolution
    fault_voltage = np.full_like(time, 3.3)

    transition_times = [1.0, 2.1, 3.0, 4.05, 6.6]
    low_width = 0.05
    for transition in transition_times:
        mask = (time >= transition) & (time < transition + low_width)
        fault_voltage[mask] = 0.0

    filtered = extract_fault_cycles(
        time,
        fault_voltage,
        high_threshold=2.5,
        low_threshold=1.5,
        min_cycle_s=0.5,
        mad_threshold_factor=2.0,
        enable_outlier_filtering=True,
    )
    unfiltered = extract_fault_cycles(
        time,
        fault_voltage,
        high_threshold=2.5,
        low_threshold=1.5,
        min_cycle_s=0.5,
        mad_threshold_factor=2.0,
        enable_outlier_filtering=False,
    )

    assert unfiltered.valid_cycle_count == 4
    assert filtered.valid_cycle_count == 3
    reasons = {detail.reason for detail in filtered.discarded_cycle_details}
    assert 'mad_outlier' in reasons
    assert filtered.detection_quality_metrics['transitions_failing_mad_filter'] == 1


def test_detect_ail_windows_identifies_pulses() -> None:
    time = np.linspace(0.0, 5.0, 5001)
    voltage = np.full_like(time, 2.5)
    voltage[(time >= 1.0) & (time <= 1.4)] = 0.2
    voltage[(time >= 3.0) & (time <= 3.2)] = 0.3

    windows = detect_ail_windows(time, voltage, threshold=1.0, min_duration_s=0.05)

    assert len(windows) == 2
    assert windows[0].start_s == pytest.approx(1.0, abs=0.01)
    assert windows[0].end_s == pytest.approx(1.4, abs=0.01)
    assert windows[1].duration_s == pytest.approx(0.2, abs=0.02)


# ============================================================================
# PHASE 1: Zero Time Difference and Edge Case Tests
# ============================================================================


def test_parse_scope_csv_with_zero_time_differences() -> None:
    """Test that parse_scope_csv handles data with all identical timestamps.

    This edge case occurs when oscilloscope data has repeated timestamps,
    which would cause division by zero in sampling interval estimation.
    The parser should generate synthetic evenly-spaced time values.
    """
    # All timestamps are identical (zero time differences)
    csv_data = """0.0,2.5,3.3
0.0,2.4,3.2
0.0,2.6,3.3
0.0,2.5,3.3
0.0,2.4,3.2"""

    result = parse_scope_csv(csv_data)

    # Should successfully parse and generate synthetic time
    assert result.time.size == 5
    # Time should be evenly spaced, starting at 0
    assert result.time[0] == pytest.approx(0.0)
    # Time differences should be non-zero (synthetic)
    time_diffs = np.diff(result.time)
    assert np.all(time_diffs > 0)
    # Should be evenly spaced
    assert np.allclose(time_diffs, time_diffs[0])
    # Voltage data should be preserved
    assert result.ail_voltage.size == 5
    assert result.fault_voltage.size == 5


def test_parse_scope_csv_with_negative_time_differences() -> None:
    """Test that parse_scope_csv handles non-monotonic time (decreasing timestamps).

    When time values decrease, the parser should generate synthetic time
    rather than using the potentially corrupted time column.
    """
    csv_data = """5.0,2.5,3.3
4.0,2.4,3.2
3.0,2.6,3.3
2.0,2.5,3.3
1.0,2.4,3.2"""

    result = parse_scope_csv(csv_data)

    # Should generate synthetic monotonic time
    assert result.time.size == 5
    time_diffs = np.diff(result.time)
    # All differences should be positive (monotonic increasing)
    assert np.all(time_diffs > 0)
    # Should be evenly spaced
    assert np.allclose(time_diffs, time_diffs[0])


def test_parse_scope_csv_with_mixed_zero_and_nonzero_diffs() -> None:
    """Test handling of partially repeated timestamps.

    Some data points may have identical timestamps while others differ.
    This tests the median-based sampling interval estimation.
    """
    csv_data = """0.0,2.5,3.3
0.0,2.4,3.2
0.001,2.6,3.3
0.002,2.5,3.3
0.002,2.4,3.2
0.003,2.3,3.1"""

    result = parse_scope_csv(csv_data)

    # Should successfully parse
    assert result.time.size == 6
    # Time should be monotonic
    assert np.all(np.diff(result.time) >= 0)


def test_parse_scope_csv_empty_file() -> None:
    """Test that empty CSV raises appropriate error."""
    csv_data = ""

    with pytest.raises(ScopeProcessingError, match="CSV file is empty"):
        parse_scope_csv(csv_data)


def test_parse_scope_csv_no_numeric_data() -> None:
    """Test that CSV with only headers/text raises error."""
    csv_data = """time,voltage,fault
header,text,only"""

    with pytest.raises(ScopeProcessingError, match="No numeric data found"):
        parse_scope_csv(csv_data)


def test_parse_scope_csv_insufficient_rows() -> None:
    """Test that CSV with only one data row raises error."""
    csv_data = "0.0,2.5,3.3"

    with pytest.raises(ScopeProcessingError, match="Insufficient data rows"):
        parse_scope_csv(csv_data)


def test_parse_scope_csv_insufficient_columns() -> None:
    """Test that CSV with only one column raises error."""
    csv_data = """0.0
0.001
0.002"""

    # Parser requires at least 2 columns, detected during delimiter detection
    with pytest.raises(ScopeProcessingError, match="No numeric data found"):
        parse_scope_csv(csv_data)


def test_parse_scope_csv_handles_different_delimiters() -> None:
    """Test that parser correctly handles various CSV delimiters."""
    test_cases = [
        ("0.0,2.5,3.3\n0.001,2.4,3.2\n0.002,2.6,3.3", "comma"),
        ("0.0\t2.5\t3.3\n0.001\t2.4\t3.2\n0.002\t2.6\t3.3", "tab"),
        ("0.0;2.5;3.3\n0.001;2.4;3.2\n0.002;2.6;3.3", "semicolon"),
        ("0.0 2.5 3.3\n0.001 2.4 3.2\n0.002 2.6 3.3", "space"),
    ]

    for csv_data, delimiter_name in test_cases:
        result = parse_scope_csv(csv_data)
        assert result.time.size == 3, f"Failed for {delimiter_name} delimiter"
        assert result.ail_voltage.size == 3
        assert result.fault_voltage.size == 3


def test_parse_scope_csv_with_headers() -> None:
    """Test that parser skips non-numeric header rows."""
    csv_data = """Oscilloscope Data Export
Time (s),Channel 1 (V),Channel 2 (V)
0.0,2.5,3.3
0.001,2.4,3.2
0.002,2.6,3.3"""

    result = parse_scope_csv(csv_data)

    assert result.time.size == 3
    assert result.time[0] == pytest.approx(0.0)
    assert result.ail_voltage[0] == pytest.approx(2.5)


def test_parse_scope_csv_two_columns_only() -> None:
    """Test parsing CSV with only two columns (time and one voltage)."""
    csv_data = """0.0,2.5
0.001,2.4
0.002,2.6
0.003,2.5"""

    result = parse_scope_csv(csv_data)

    assert result.time.size == 4
    assert result.ail_voltage.size == 4
    # Fault voltage should be zeros when third column missing
    assert result.fault_voltage.size == 4
    assert np.all(result.fault_voltage == 0.0)


def test_parse_scope_csv_with_file_like_object() -> None:
    """Test parsing from file-like objects (StringIO, BytesIO)."""
    csv_data = """0.0,2.5,3.3
0.001,2.4,3.2
0.002,2.6,3.3"""

    # Test with StringIO
    string_io = io.StringIO(csv_data)
    result = parse_scope_csv(string_io)
    assert result.time.size == 3

    # Test with BytesIO
    bytes_io = io.BytesIO(csv_data.encode('utf-8'))
    result = parse_scope_csv(bytes_io)
    assert result.time.size == 3


def test_extract_fault_cycles_with_zero_time_span() -> None:
    """Test that extract_fault_cycles handles zero time span gracefully.

    When all time values are identical, total_duration is zero,
    which should be handled by the smoothing function.
    """
    time = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    voltage = np.array([3.3, 0.0, 3.3, 0.0, 3.3])

    # Should not crash, but may return no cycles due to insufficient data
    result = extract_fault_cycles(time, voltage)

    # With zero time span, no meaningful cycles can be detected
    assert result.cycle_periods.size == 0


def test_extract_fault_cycles_insufficient_samples() -> None:
    """Test that extract_fault_cycles rejects data with too few samples."""
    time = np.array([0.0])
    voltage = np.array([3.3])

    with pytest.raises(ScopeProcessingError, match="at least two samples"):
        extract_fault_cycles(time, voltage)


def test_extract_fault_cycles_mismatched_lengths() -> None:
    """Test that mismatched time and voltage arrays raise error."""
    time = np.array([0.0, 1.0, 2.0])
    voltage = np.array([3.3, 0.0])  # One element short

    with pytest.raises(ScopeProcessingError, match="length must match"):
        extract_fault_cycles(time, voltage)


def test_extract_fault_cycles_smoothing_fallback() -> None:
    """Test that smoothing fallback activates when initial smoothing over-smooths.

    When the smoothing window is too large and misses transitions,
    a narrower window should be tried automatically.
    """
    # Create signal with very short pulses that might be smoothed away
    time = np.linspace(0.0, 10.0, 10001)
    voltage = np.full_like(time, 3.3)

    # Add very narrow pulses
    pulse_times = [2.0, 5.0, 8.0]
    pulse_width = 0.01  # 10ms pulses

    for pulse_t in pulse_times:
        mask = (time >= pulse_t) & (time < pulse_t + pulse_width)
        voltage[mask] = 0.0

    result = extract_fault_cycles(
        time,
        voltage,
        high_threshold=2.5,
        low_threshold=1.5,
        min_cycle_s=1.0,
    )

    # Should detect some cycles, potentially using fallback smoothing
    assert 'fallback_used' in result.debug['smoothing']


def test_detect_ail_windows_with_zero_time_span() -> None:
    """Test AIL window detection with identical timestamps."""
    time = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    voltage = np.array([0.2, 0.3, 0.2, 0.3, 0.2])

    # Should not crash even with zero time differences
    windows = detect_ail_windows(time, voltage, threshold=1.0, min_duration_s=0.01)

    # With zero time span, duration calculations may be zero
    # Function should handle this gracefully
    assert isinstance(windows, list)


def test_parse_scope_csv_handles_seek_oserror_and_falls_back_to_read() -> None:
    class SeekRaisesSource:
        def __init__(self, text: str) -> None:
            self._text = text

        def seek(self, _: int) -> None:
            raise OSError('not seekable')

        def read(self) -> str:
            return self._text

    source = SeekRaisesSource("0.0,2.5,3.3\n0.1,2.4,3.2\n0.2,2.3,3.1\n")
    result = parse_scope_csv(source)
    assert result.time.size == 3
    assert result.ail_voltage[0] == pytest.approx(2.5)


def test_parse_scope_csv_wraps_unexpected_read_errors() -> None:
    class ReadFailsSource:
        def read(self) -> str:
            raise RuntimeError('boom')

    with pytest.raises(ScopeProcessingError, match='Failed to read scope data'):
        parse_scope_csv(ReadFailsSource())


def test_extract_fault_cycles_discards_non_finite_periods_with_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    time = np.linspace(0.0, 10.0, 101)
    voltage = np.full(time.shape, 3.3)

    monkeypatch.setattr(sp, '_detect_falling_transitions', lambda *args, **kwargs: [0.0, float('nan'), 5.0])
    monkeypatch.setattr(
        sp,
        '_smooth_signal',
        lambda *args, **kwargs: (
            np.asarray(args[1], dtype=np.float64),
            {'window_samples': 3, 'window_seconds': 1.0},
        ),
    )
    monkeypatch.setattr(sp, '_build_cycle_debug_plot', lambda *args, **kwargs: None)

    result = extract_fault_cycles(time, voltage, min_cycle_s=0.5)

    assert result.valid_cycle_count == 0
    assert result.raw_cycle_periods.size == 2
    assert not np.any(np.isfinite(result.raw_cycle_periods))
    reasons = [detail.reason for detail in result.discarded_cycle_details]
    assert reasons and all(reason == 'non_finite_period' for reason in reasons)
    assert result.detection_quality_metrics['transitions_passing_min_cycle_s'] == 0


def test_detect_ail_windows_returns_empty_for_empty_input() -> None:
    windows = detect_ail_windows(np.array([], dtype=np.float64), np.array([], dtype=np.float64))
    assert windows == []


def test_interpolate_crossing_returns_t1_when_voltage_is_constant() -> None:
    crossing = sp._interpolate_crossing(1.0, 2.0, 2.0, 2.0, threshold=1.5)
    assert crossing == pytest.approx(2.0)
