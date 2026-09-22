"""Physical Discrepancy, Moments, and Response Curve Analytics.

This module implements Stage 4 diagnostics for measuring the physical discrepancy
between private beliefs x and expressed public stances y, tracking response curves,
and identifying reversal / threshold-crossing endpoints under external perturbations.

Simulation Stage Mapping:
    Stage 4: Measurement, Analytics & Neural Diagnostics
    (Observes output from Stage 0-3 simulation trajectories).

Key Capabilities:
    - Physical discrepancy d_2 = (4ND)^{-1} sum (x - y)^2 in [0, 1].
    - First and second moment calculations per belief component (m_x, m_y, M_x^(2), M_y^(2)).
    - Response curve extraction across trajectory time series.
    - Rigorous reversal endpoint detection with initial eligibility, amplitude crossing,
      and temporal persistence criteria.
    - General threshold-crossing endpoint detector with holding duration verification.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Union
import numpy as np


def calculate_population_discrepancy(x: np.ndarray, y: np.ndarray) -> float:
    """Calculate normalized population discrepancy d_2 in [0.0, 1.0].

    [Analytics]
    Computes:
        d_2 = (1 / (4 * N * D)) * sum_{i=1}^N sum_{alpha=1}^D (x_{i, alpha} - y_{i, alpha})^2

    Mathematical Properties:
        - Exactly 0.0 when private and public states are identical.
        - Exactly 1.0 when private and public states are completely opposite binary states (+1 vs -1).
        - Exactly 0.0 when continuous states are centered at zero (x = y = 0.0).
        - Expected value 1/6 for independent Uniform[-1, 1] continuous states.

    Args:
        x: Private belief array of shape (N, D) or (N,).
        y: Expressed public stance array of shape (N, D) or (N,).

    Returns:
        Discrepancy scalar d_2 in [0.0, 1.0].
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if x.shape != y.shape:
        raise ValueError(f"Shape mismatch: x {x.shape} vs y {y.shape}")

    if x.size == 0:
        return 0.0

    diff = x - y
    d2 = float(np.mean(diff * diff) / 4.0)
    return float(np.clip(d2, 0.0, 1.0))


def calculate_second_moment(x: np.ndarray) -> np.ndarray:
    """Compute second moment M_alpha^(2) = (1/N) sum_i x_{i, alpha}^2 for each component.

    [Analytics]
    Distinguishes weak continuous opinions near zero (M^(2) ~ 0) from sharply
    polarized bimodal populations split at +/- 1 (M^(2) ~ 1) when the mean is zero.

    Args:
        x: Belief array of shape (N, D) or (N,).

    Returns:
        1D numpy array of length D containing second moments for each dimension.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, np.newaxis]
    elif x.ndim != 2:
        raise ValueError(f"Expected 1D or 2D array, got shape {x.shape}")

    N, _ = x.shape
    if N == 0:
        return np.array([], dtype=np.float64)

    return np.mean(x * x, axis=0)


def calculate_moments(x: np.ndarray) -> Dict[str, np.ndarray]:
    """Compute component means and second moments for belief states.

    [Analytics]
    Args:
        x: Belief array of shape (N, D) or (N,).

    Returns:
        Dictionary containing:
            "mean": 1D array of component means m_alpha = (1/N) sum_i x_{i, alpha}.
            "second_moment": 1D array of second moments M_alpha^(2) = (1/N) sum_i x_{i, alpha}^2.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, np.newaxis]
    elif x.ndim != 2:
        raise ValueError(f"Expected 1D or 2D array, got shape {x.shape}")

    N, _ = x.shape
    if N == 0:
        return {
            "mean": np.array([], dtype=np.float64),
            "second_moment": np.array([], dtype=np.float64),
        }

    return {
        "mean": np.mean(x, axis=0),
        "second_moment": np.mean(x * x, axis=0),
    }


def track_response_curve(
    times: np.ndarray,
    x_series: np.ndarray,
    y_series: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Track response curves of discrepancy, means, and second moments over time.

    [Analytics]
    Args:
        times: 1D array of T simulation timestamps.
        x_series: Array of private states over time, shape (T, N, D).
        y_series: Array of public states over time, shape (T, N, D).

    Returns:
        Dictionary containing:
            "times": 1D array of timestamps (T,).
            "discrepancy": 1D array of instantaneous d_2(t) values (T,).
            "mean_x": 2D array of private component means (T, D).
            "mean_y": 2D array of public component means (T, D).
            "second_moment_x": 2D array of private second moments (T, D).
            "second_moment_y": 2D array of public second moments (T, D).
    """
    times = np.asarray(times, dtype=np.float64)
    x_series = np.asarray(x_series, dtype=np.float64)
    y_series = np.asarray(y_series, dtype=np.float64)

    if len(times) != len(x_series) or len(x_series) != len(y_series):
        raise ValueError(
            f"Length mismatch: times ({len(times)}), x_series ({len(x_series)}), y_series ({len(y_series)})"
        )

    T = len(times)
    d2_vals = np.zeros(T, dtype=np.float64)

    if T == 0:
        return {
            "times": times,
            "discrepancy": d2_vals,
            "mean_x": np.zeros((0, 0)),
            "mean_y": np.zeros((0, 0)),
            "second_moment_x": np.zeros((0, 0)),
            "second_moment_y": np.zeros((0, 0)),
        }

    D = x_series.shape[2] if x_series.ndim == 3 else 1
    mean_x = np.zeros((T, D), dtype=np.float64)
    mean_y = np.zeros((T, D), dtype=np.float64)
    m2_x = np.zeros((T, D), dtype=np.float64)
    m2_y = np.zeros((T, D), dtype=np.float64)

    for t in range(T):
        xt = x_series[t]
        yt = y_series[t]
        d2_vals[t] = calculate_population_discrepancy(xt, yt)
        mx_dict = calculate_moments(xt)
        my_dict = calculate_moments(yt)
        mean_x[t] = mx_dict["mean"]
        mean_y[t] = my_dict["mean"]
        m2_x[t] = mx_dict["second_moment"]
        m2_y[t] = my_dict["second_moment"]

    return {
        "times": times,
        "discrepancy": d2_vals,
        "mean_x": mean_x,
        "mean_y": mean_y,
        "second_moment_x": m2_x,
        "second_moment_y": m2_y,
    }


def detect_reversal_endpoint(
    times: np.ndarray,
    values: np.ndarray,
    initial_threshold: float = 0.5,
    reversal_threshold: float = 0.5,
    persistence_duration: float = 0.0,
) -> Dict[str, Any]:
    """Detect sustained opinion reversal endpoint from a magnetization trajectory.

    [Analytics]
    Implements the methodological rule from the measurement specification:
    1. Initial Eligibility: |m(0)| >= initial_threshold. If not, the run is classified
       as ineligible for reversal rather than counted as a failure.
    2. Crossing: Crossing to the opposite sign beyond reversal_threshold:
       -sgn(m(0)) * m(t) >= reversal_threshold.
    3. Persistence: Must remain beyond reversal_threshold for at least persistence_duration.

    Args:
        times: 1D array of non-decreasing simulation timestamps.
        values: 1D array of single-component mean magnetizations m(t).
        initial_threshold: Minimum initial magnitude required for eligibility.
        reversal_threshold: Minimum opposite-sign magnitude required for reversal.
        persistence_duration: Duration for which opposite state must persist uninterrupted.

    Returns:
        Dictionary containing:
            "eligible": bool, whether the trajectory met the initial alignment criterion.
            "reversed": bool, whether eligible and a persistent reversal occurred.
            "reversal_time": float or None, timestamp of first qualifying reversal.
            "recovery_time": float or None, timestamp if trajectory later recovered original sign.
            "persistence_achieved": bool, whether the required holding duration was fulfilled.
    """
    times = np.asarray(times, dtype=np.float64).ravel()
    values = np.asarray(values, dtype=np.float64).ravel()

    if len(times) != len(values):
        raise ValueError(f"Length mismatch: times ({len(times)}) vs values ({len(values)})")

    if len(times) == 0:
        return {
            "eligible": False,
            "reversed": False,
            "reversal_time": None,
            "recovery_time": None,
            "persistence_achieved": False,
        }

    # 1. Initial eligibility
    m0 = float(values[0])
    if abs(m0) < initial_threshold:
        return {
            "eligible": False,
            "reversed": False,
            "reversal_time": None,
            "recovery_time": None,
            "persistence_achieved": False,
        }

    orig_sign = 1.0 if m0 > 0 else -1.0
    reversed_condition = (-orig_sign * values) >= reversal_threshold

    reversal_time = None
    persistence_achieved = False
    recovery_time = None

    # Scan for first persistent crossing
    T = len(times)
    i = 0
    while i < T:
        if reversed_condition[i]:
            start_t = times[i]
            # Check how long this episode lasts
            j = i
            while j < T and reversed_condition[j]:
                j += 1
            held_duration = times[j - 1] - start_t
            if held_duration >= persistence_duration:
                reversal_time = start_t
                persistence_achieved = True
                # Check for subsequent recovery back to original sign
                for k in range(j, T):
                    if (orig_sign * values[k]) >= 0.0:
                        recovery_time = times[k]
                        break
                break
            i = j
        else:
            i += 1

    return {
        "eligible": True,
        "reversed": persistence_achieved,
        "reversal_time": reversal_time,
        "recovery_time": recovery_time,
        "persistence_achieved": persistence_achieved,
    }


def detect_crossing_endpoint(
    times: np.ndarray,
    values: np.ndarray,
    threshold: float,
    direction: str = "above",
    persistence_duration: float = 0.0,
) -> Dict[str, Any]:
    """Detect threshold-crossing endpoint with optional persistence duration check.

    [Analytics]
    Args:
        times: 1D array of simulation timestamps.
        values: 1D array of observable values (e.g. participation fraction n_mean(t)).
        threshold: Threshold value to test against.
        direction: "above" (values >= threshold) or "below" (values <= threshold).
        persistence_duration: Uninterrupted duration required to satisfy holding criterion.

    Returns:
        Dictionary containing:
            "crossed": bool, whether threshold was crossed.
            "crossing_time": float or None, first timestamp of crossing.
            "holding_satisfied": bool, whether the condition held for persistence_duration.
            "duration_held": float, maximum continuous duration held.
    """
    times = np.asarray(times, dtype=np.float64).ravel()
    values = np.asarray(values, dtype=np.float64).ravel()

    if direction not in ("above", "below"):
        raise ValueError(f"direction must be 'above' or 'below', got {direction!r}")

    if len(times) != len(values):
        raise ValueError(f"Length mismatch: times ({len(times)}) vs values ({len(values)})")

    if len(times) == 0:
        return {
            "crossed": False,
            "crossing_time": None,
            "holding_satisfied": False,
            "duration_held": 0.0,
        }

    condition = (values >= threshold) if direction == "above" else (values <= threshold)
    if not np.any(condition):
        return {
            "crossed": False,
            "crossing_time": None,
            "holding_satisfied": False,
            "duration_held": 0.0,
        }

    T = len(times)
    first_crossing = None
    max_held = 0.0
    holding_satisfied = False

    i = 0
    while i < T:
        if condition[i]:
            if first_crossing is None:
                first_crossing = float(times[i])
            start_t = times[i]
            j = i
            while j < T and condition[j]:
                j += 1
            held = float(times[j - 1] - start_t)
            if held > max_held:
                max_held = held
            if held >= persistence_duration:
                holding_satisfied = True
            i = j
        else:
            i += 1

    return {
        "crossed": True,
        "crossing_time": first_crossing,
        "holding_satisfied": holding_satisfied,
        "duration_held": max_held,
    }
