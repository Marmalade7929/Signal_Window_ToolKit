"""Minimal public Streamlit shell for Signal Window Toolkit."""
from __future__ import annotations

import io
from collections.abc import Sequence

import numpy as np
import pandas as pd

from signal_toolkit import integrate_flow_to_volume, parse_scope_csv

FLOW_TIME_KEYS = {"time", "time_s", "seconds", "t"}
FLOW_RATE_KEYS = {"flow", "flow_rate", "flow_mls", "flow_ml_s", "q"}


def decode_uploaded_content(payload: bytes | str) -> str:
    """Decode uploaded payload into UTF-8 text."""
    if isinstance(payload, bytes):
        return payload.decode("utf-8", errors="replace")
    return str(payload)


def _is_numeric_like(value: object) -> bool:
    try:
        float(str(value))
        return True
    except (TypeError, ValueError):
        return False


def _looks_like_no_header(columns: Sequence[object]) -> bool:
    if not columns:
        return True
    return all(_is_numeric_like(column) for column in columns)


def parse_flow_frame(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Extract validated time/flow arrays from a parsed CSV frame."""
    if frame.shape[1] < 2:
        raise ValueError("Flow CSV must contain at least two columns.")

    normalized = {str(col).strip().lower(): col for col in frame.columns}
    time_col = next((normalized[key] for key in FLOW_TIME_KEYS if key in normalized), frame.columns[0])
    flow_col = next((normalized[key] for key in FLOW_RATE_KEYS if key in normalized), frame.columns[1])

    time_series = pd.to_numeric(frame[time_col], errors="coerce")
    flow_series = pd.to_numeric(frame[flow_col], errors="coerce")

    if time_series.isna().any() or flow_series.isna().any():
        raise ValueError("Time and flow columns must be numeric.")

    time_s = time_series.to_numpy(dtype=np.float64)
    flow_mls = flow_series.to_numpy(dtype=np.float64)

    if time_s.size < 3:
        raise ValueError("At least three samples are required.")
    if not np.all(np.diff(time_s) > 0):
        raise ValueError("Time samples must be strictly increasing.")
    if np.any(flow_mls < 0):
        raise ValueError("Flow rate must be non-negative.")

    return time_s, flow_mls


def parse_flow_csv_text(csv_text: str) -> tuple[np.ndarray, np.ndarray]:
    """Parse flow CSV text into validated time and flow arrays."""
    stream = io.StringIO(csv_text)
    try:
        frame = pd.read_csv(stream, sep=None, engine="python")
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Unable to parse flow CSV: {exc}") from exc

    if frame.shape[1] < 2 or _looks_like_no_header(list(frame.columns)):
        stream.seek(0)
        frame = pd.read_csv(stream, sep=None, engine="python", header=None)
        frame.columns = [f"col_{idx}" for idx in range(frame.shape[1])]

    return parse_flow_frame(frame)


def run_app() -> None:
    """Render the public Streamlit app shell."""
    import streamlit as st

    st.set_page_config(page_title="Signal Window Toolkit", layout="wide")
    st.title("Signal Window Toolkit")
    st.caption("Public app shell for flow integration and scope timing exploration.")

    flow_tab, scope_tab, about_tab = st.tabs(["Flow Integration", "Scope Timing", "About"])

    with flow_tab:
        st.subheader("Flow Integration")
        st.write("Upload a CSV with time and flow columns (expected units: seconds and mL/s).")
        flow_file = st.file_uploader(
            "Flow CSV",
            type=["csv", "txt"],
            key="flow_csv",
            help="Supported formats: header columns (time, flow_rate) or two numeric columns.",
        )

        if flow_file is not None:
            try:
                text = decode_uploaded_content(flow_file.getvalue())
                time_s, flow_mls = parse_flow_csv_text(text)
                volume_ml, error_ml = integrate_flow_to_volume(time_s, flow_mls)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Flow processing failed: {exc}")
            else:
                col_a, col_b, col_c = st.columns(3)
                col_a.metric("Integrated Volume (mL)", f"{volume_ml:.4f}")
                col_b.metric("Error Estimate (mL)", f"{error_ml:.6f}")
                col_c.metric("Samples", f"{time_s.size}")

                chart_df = pd.DataFrame({"time_s": time_s, "flow_mls": flow_mls})
                st.line_chart(chart_df.set_index("time_s"))
                st.dataframe(chart_df.head(25), use_container_width=True)

    with scope_tab:
        st.subheader("Scope Timing")
        st.write("Upload oscilloscope CSV to detect timing windows and cycle metrics.")
        scope_file = st.file_uploader(
            "Scope CSV",
            type=["csv", "txt"],
            key="scope_csv",
        )

        if scope_file is not None:
            try:
                text = decode_uploaded_content(scope_file.getvalue())
                scope = parse_scope_csv(text)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Scope processing failed: {exc}")
            else:
                mean_period = scope.fault.mean_period_s
                mean_period_label = f"{mean_period:.3f}" if np.isfinite(mean_period) else "n/a"

                col_a, col_b, col_c = st.columns(3)
                col_a.metric("Samples", f"{scope.time.size}")
                col_b.metric("Valid Fault Cycles", f"{scope.fault.valid_cycle_count}")
                col_c.metric("Mean Period (s)", mean_period_label)

                window_rows = [
                    {
                        "start_s": float(window.start_s),
                        "end_s": float(window.end_s),
                        "duration_s": float(window.duration_s),
                    }
                    for window in scope.ail_windows
                ]
                st.metric("Detected Windows", f"{len(window_rows)}")
                if window_rows:
                    st.dataframe(pd.DataFrame(window_rows), use_container_width=True, hide_index=True)
                else:
                    st.info("No timing windows detected with current signal thresholds.")

                trace_df = pd.DataFrame(
                    {
                        "time_s": scope.time,
                        "AIL_voltage": scope.ail_voltage,
                        "fault_voltage": scope.fault_voltage,
                    }
                )
                st.line_chart(trace_df.set_index("time_s"))

    with about_tab:
        st.subheader("Links")
        st.markdown("- GitHub: https://github.com/Marmalade7929/Signal_Window_ToolKit")
        st.markdown("- GitHub Pages: https://marmalade7929.github.io/Signal_Window_ToolKit/")
        st.markdown("- Live Streamlit: https://signal-window-toolkit.streamlit.app/")


if __name__ == "__main__":
    run_app()
