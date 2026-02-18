"""Tests for minimal Streamlit app parsing helpers."""
from __future__ import annotations

import pytest

from app import decode_uploaded_content, parse_flow_csv_text


def test_decode_uploaded_content_bytes() -> None:
    assert decode_uploaded_content(b"abc") == "abc"


def test_parse_flow_csv_text_with_header_columns() -> None:
    csv_text = "time,flow_rate\n0.0,1.0\n1.0,1.5\n2.0,2.0\n"
    time_s, flow_mls = parse_flow_csv_text(csv_text)
    assert time_s.tolist() == [0.0, 1.0, 2.0]
    assert flow_mls.tolist() == [1.0, 1.5, 2.0]


def test_parse_flow_csv_text_without_header() -> None:
    csv_text = "0.0,1.0\n1.0,1.5\n2.0,2.0\n"
    time_s, flow_mls = parse_flow_csv_text(csv_text)
    assert time_s.tolist() == [0.0, 1.0, 2.0]
    assert flow_mls.tolist() == [1.0, 1.5, 2.0]


def test_parse_flow_csv_text_rejects_non_monotonic_time() -> None:
    csv_text = "time,flow_rate\n0.0,1.0\n2.0,1.5\n1.0,2.0\n"
    with pytest.raises(ValueError, match="strictly increasing"):
        parse_flow_csv_text(csv_text)


def test_parse_flow_csv_text_rejects_negative_flow() -> None:
    csv_text = "time,flow_rate\n0.0,1.0\n1.0,-0.1\n2.0,2.0\n"
    with pytest.raises(ValueError, match="non-negative"):
        parse_flow_csv_text(csv_text)
