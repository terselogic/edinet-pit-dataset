"""Tests for Point-in-Time integrity and zero look-ahead bias."""

from datetime import datetime
from pathlib import Path

from edinet_pit.pit_schema import (
    build_pit_dataframe,
    export_dataset,
    query_as_of,
    validate_pit_integrity,
)
import pytest


def test_query_as_of_prevents_lookahead_bias() -> None:
    """Ensure amendments filed later do not leak into earlier as_of queries."""
    df = build_pit_dataframe(
        [
            {
                "doc_id": "S100ORIG",
                "edinet_code": "E00001",
                "sec_code": "10010",
                "filer_name": "Sample Corp",
                "doc_type_code": "120",
                "is_amendment": False,
                "period_end": "2024-03-31",
                "submit_datetime": "2024-06-25 09:00:00",
                "metric_name": "net_sales",
                "metric_value": 100_000.0,
                "risk_change_score": 0.12,
            },
            {
                "doc_id": "S100AMEND",
                "edinet_code": "E00001",
                "sec_code": "10010",
                "filer_name": "Sample Corp",
                "doc_type_code": "130",
                "is_amendment": True,
                "period_end": "2024-03-31",
                "submit_datetime": "2024-09-10 15:30:00",
                "metric_name": "net_sales",
                "metric_value": 95_000.0,
                "risk_change_score": 0.45,
            },
        ]
    )

    validate_pit_integrity(df)

    before_amend = query_as_of(df, datetime(2024, 7, 1, 0, 0, 0))
    assert before_amend.height == 1
    assert before_amend["doc_id"][0] == "S100ORIG"
    assert before_amend["metric_value"][0] == 100_000.0

    after_amend = query_as_of(df, datetime(2024, 9, 11, 0, 0, 0))
    assert after_amend.height == 1
    assert after_amend["doc_id"][0] == "S100AMEND"
    assert after_amend["metric_value"][0] == 95_000.0


def test_validate_rejects_submit_before_period_end() -> None:
    """Ensure rows where submit_datetime precedes period_end are rejected."""
    df = build_pit_dataframe(
        [
            {
                "doc_id": "S100BAD",
                "edinet_code": "E00002",
                "sec_code": "10020",
                "filer_name": "Bad Date Corp",
                "doc_type_code": "120",
                "is_amendment": False,
                "period_end": "2025-03-31",
                "submit_datetime": "2025-03-01 09:00:00",
                "metric_name": "net_sales",
                "metric_value": 50_000.0,
                "risk_change_score": None,
            }
        ]
    )

    with pytest.raises(ValueError, match="precedes period_end"):
        validate_pit_integrity(df)


def test_export_dataset_writes_parquet_and_duckdb(tmp_path: Path) -> None:
    """Verify Parquet and DuckDB artifacts are created cleanly."""
    df = build_pit_dataframe(
        [
            {
                "doc_id": "S100TEST",
                "edinet_code": "E00003",
                "sec_code": "10030",
                "filer_name": "Export Corp",
                "doc_type_code": "120",
                "is_amendment": False,
                "period_end": "2025-03-31",
                "submit_datetime": "2025-06-28 10:00:00",
                "metric_name": "operating_income",
                "metric_value": 12_345.0,
                "risk_change_score": 0.08,
            }
        ]
    )
    outputs = export_dataset(df, tmp_path, name="sample")
    assert outputs["parquet"].exists()
    assert outputs["duckdb"].exists()
