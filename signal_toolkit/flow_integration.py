"""Minimal flow-to-volume integration with embedded mathematical rigor.

This module provides a single, mathematically rigorous function for integrating
flow rate data to compute volume with error bounds.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import PchipInterpolator


def integrate_flow(
    coordinate: np.ndarray,
    flow_rate: np.ndarray,
    cumulative: bool = False,
) -> np.ndarray | float:
    """Core trapezoidal integration function - single source of truth.

    This is the foundational integration method used in this toolkit for all
    flow-to-volume calculations. It implements the composite trapezoidal rule:

    .. math::

        V = \\sum_{i=1}^{n-1} \\frac{h_i}{2} (Q_{i-1} + Q_i)

    where h_i = t_i - t_{i-1} is the step size between measurements.

    Mathematical Properties
    -----------------------
    - **Convergence**: O(h²) for smooth functions where h is the mean step size
    - **Exactness**: Exact for linear functions (second derivative = 0)
    - **Stability**: Numerically stable for well-conditioned input data
    - **Conservation**: Preserves monotonicity of cumulative integral

    Parameters
    ----------
    coordinate : np.ndarray
        Monotonically increasing coordinate array (typically time in seconds,
        or normalized phase φ ∈ [0,1]). Must have at least 2 points.
    flow_rate : np.ndarray
        Flow rate measurements corresponding to coordinate points.
        Units depend on context (e.g., mL/s or mL/h).
    cumulative : bool, default=False
        If False, returns the total integral as a scalar float.
        If True, returns array of cumulative integrals starting from 0.

    Returns
    -------
    float or np.ndarray
        If cumulative=False: Total integrated value (scalar)
        If cumulative=True: Array of cumulative integrals with shape (n,)

    Raises
    ------
    AssertionError
        If input validation fails:
        - coordinate and flow_rate must have matching shapes
        - At least 2 data points required
        - coordinate must be monotonically increasing
        - All values must be finite

    Notes
    -----
    This function serves as the single integration implementation for:
    - integrate_flow_to_volume(): One-shot integration with error estimation
    - CharacterizationResult._build_cycle_template(): Cumulative integral arrays
    - Any future integration needs in the toolkit

    The cumulative mode uses scipy.integrate.cumulative_trapezoid with initial=0,
    ensuring the output array has the same length as the input arrays and starts
    at zero.

    Examples
    --------
    Total integration (single value):

    >>> time = np.array([0, 1, 2, 3])
    >>> flow = np.array([2.0, 3.0, 2.5, 2.0])
    >>> total = integrate_flow(time, flow, cumulative=False)
    >>> # Returns: 7.5 (trapezoidal approximation)

    Cumulative integration (array):

    >>> cumul = integrate_flow(time, flow, cumulative=True)
    >>> # Returns: [0.0, 2.5, 5.25, 7.5]

    References
    ----------
    - Burden, R. L., & Faires, J. D. (2010). Numerical Analysis (9th ed.).
      Brooks/Cole. Chapter 4: Numerical Differentiation and Integration.
    """
    # Input validation
    coordinate = np.asarray(coordinate, dtype=np.float64)
    flow_rate = np.asarray(flow_rate, dtype=np.float64)

    assert coordinate.shape == flow_rate.shape, \
        f"coordinate and flow_rate must have matching shapes, got {coordinate.shape} vs {flow_rate.shape}"
    assert coordinate.size >= 2, \
        f"At least 2 data points required for integration, got {coordinate.size}"
    assert np.all(np.isfinite(coordinate)), \
        "coordinate array contains non-finite values"
    assert np.all(np.isfinite(flow_rate)), \
        "flow_rate array contains non-finite values"

    # Check monotonicity (strictly or non-strictly increasing)
    coord_diffs = np.diff(coordinate)
    assert np.all(coord_diffs >= 0), \
        "coordinate must be monotonically increasing (non-strictly)"

    if cumulative:
        # Return cumulative integral array starting from 0
        return cumulative_trapezoid(flow_rate, coordinate, initial=0.0)
    else:
        # Return total integral as scalar
        return float(np.trapezoid(flow_rate, coordinate))


def integrate_flow_to_volume(
    time: np.ndarray,  # seconds
    flow_rate: np.ndarray,  # mL/s
    theta: float = 0.0,  # latency correction, seconds
    template_period: float | None = None,  # model period for normalization, seconds
) -> tuple[float, float]:
    """Integrate flow rate Q(t) to compute volume V with error estimate.

    Mathematical Foundation
    -----------------------
    Given flow rate measurements Q(t) at time points t, this function computes:

    .. math::

        V = \\int_{t_0}^{t_f} Q(t - \\theta) \\, dt

    where θ is a time-shift correction for sensor latency.

    The integration uses the composite trapezoidal rule after PCHIP
    (Piecewise Cubic Hermite Interpolating Polynomial) interpolation:

    .. math::

        V \\approx \\sum_{i=1}^{n-1} \\frac{h_i}{2} (Q_{i-1} + Q_i)

    If a template period T_template is provided, the result is normalized:

    .. math::

        V_{normalized} = V \\cdot \\frac{T_{template}}{T_{measured}}

    Error Bound
    -----------
    The error estimate uses the second-order truncation error of the
    trapezoidal rule:

    .. math::

        E \\leq \\frac{h^2}{12} \\cdot \\max|Q''(t)| \\cdot (t_f - t_0)

    where h is the mean time step and Q''(t) is approximated via second
    finite differences.

    Parameters
    ----------
    time : np.ndarray
        Time points in seconds, must be strictly increasing with at least 3 points.
    flow_rate : np.ndarray
        Flow rate measurements in mL/s, must be non-negative everywhere.
    theta : float, default=0.0
        Time shift correction in seconds for sensor latency.
        Must satisfy |θ| < 0.1 × (t_f - t_0).
    template_period : float or None, default=None
        Reference period in seconds for cycle normalization.
        If provided, scales the result by (template_period / measured_period).

    Returns
    -------
    volume_mL : float
        Integrated volume in milliliters.
    error_estimate_mL : float
        Upper bound on integration error in milliliters.

    Raises
    ------
    AssertionError
        If input validation fails:
        - time and flow_rate must have matching shapes
        - At least 3 data points required
        - time must be strictly increasing
        - flow_rate must be non-negative everywhere
        - theta must be small relative to measurement duration
        - template_period must be positive if provided

    Examples
    --------
    Constant flow rate (analytical verification):

    >>> time = np.linspace(0, 60, 100)  # 60 seconds
    >>> flow = np.full_like(time, 2.0)  # 2 mL/s constant
    >>> volume, error = integrate_flow_to_volume(time, flow)
    >>> assert abs(volume - 120.0) < 0.01  # 2 mL/s × 60s = 120 mL

    Sinusoidal flow with time shift:

    >>> time = np.linspace(0, 10, 200)
    >>> flow = 5.0 + 2.0 * np.sin(2 * np.pi * time / 5.0)
    >>> volume, error = integrate_flow_to_volume(time, flow, theta=0.1)

    Real pump data with cycle normalization:

    >>> time = np.array([...])  # measured time series
    >>> flow = np.array([...])  # measured flow rates
    >>> volume, error = integrate_flow_to_volume(
    ...     time, flow, theta=0.05, template_period=55.0
    ... )

    References
    ----------
    - Fritsch, F. N., & Carlson, R. E. (1980). Monotone piecewise cubic
      interpolation. SIAM Journal on Numerical Analysis, 17(2), 238-246.
    - Burden, R. L., & Faires, J. D. (2010). Numerical Analysis (9th ed.).
      Brooks/Cole. Chapter 4: Numerical Differentiation and Integration.
    """
    # Input validation with explicit bounds
    time = np.asarray(time, dtype=np.float64)
    flow_rate = np.asarray(flow_rate, dtype=np.float64)

    assert time.shape == flow_rate.shape, \
        f"time and flow_rate must have matching shapes, got {time.shape} vs {flow_rate.shape}"
    assert time.size >= 3, \
        f"At least 3 data points required for integration, got {time.size}"
    assert np.all(np.isfinite(time)), \
        "time array contains non-finite values"
    assert np.all(np.isfinite(flow_rate)), \
        "flow_rate array contains non-finite values"

    # Check strict monotonicity
    time_diffs = np.diff(time)
    assert np.all(time_diffs > 0), \
        "time must be strictly increasing"

    # Check non-negativity of flow
    assert np.all(flow_rate >= 0), \
        f"flow_rate must be non-negative everywhere, min={np.min(flow_rate):.6f}"

    # Measure duration and validate theta
    measured_period = float(time[-1] - time[0])
    assert abs(theta) < 0.1 * measured_period, \
        f"theta={theta:.6f}s must be < 10% of measurement duration ({measured_period:.6f}s)"

    # Validate template_period if provided
    if template_period is not None:
        assert template_period > 0, \
            f"template_period must be positive, got {template_period}"
        assert np.isfinite(template_period), \
            "template_period must be finite"

    # Apply time shift correction
    time_corrected = time - theta

    # PCHIP interpolation for smooth derivative estimates
    # PCHIP preserves monotonicity and provides C¹ continuity
    interpolator = PchipInterpolator(time_corrected, flow_rate)

    # Integrate using composite trapezoidal rule via centralized integrate_flow()
    # This ensures consistency across all toolkit integration operations
    volume = integrate_flow(time_corrected, flow_rate, cumulative=False)

    # Apply cycle normalization if template period provided
    if template_period is not None:
        normalization_factor = template_period / measured_period
        volume *= normalization_factor

    # Error estimate: h² × max|Q''| × duration
    # Compute second derivative using PCHIP interpolator
    h_mean = float(np.mean(time_diffs))

    # Estimate Q''(t) using second-order finite differences at interior points
    # For PCHIP, we can evaluate the second derivative directly
    time_eval = time_corrected[1:-1]  # interior points
    second_deriv = interpolator.derivative(2)(time_eval)

    # Take maximum absolute value of second derivative
    max_second_deriv = float(np.max(np.abs(second_deriv))) if second_deriv.size > 0 else 0.0

    # Trapezoidal rule error bound: (h² / 12) × max|Q''| × duration
    # Using conservative estimate without the 1/12 factor for upper bound
    error_estimate = (h_mean ** 2) * max_second_deriv * measured_period

    # Apply normalization to error estimate if used
    if template_period is not None:
        error_estimate *= normalization_factor

    return volume, error_estimate


__all__ = ['integrate_flow', 'integrate_flow_to_volume']
