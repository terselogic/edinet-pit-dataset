"""Point-in-Time (PiT) schema builder, validator, and exporter."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Iterable

import duckdb
import polars as pl

PIT_COLUMNS: dict[str, pl.DataType] = {
    "doc_id": pl.String,
    "edinet_code": pl.String,
    "sec_code": pl.String,
    "filer_name": pl.String,
    "doc_type_code": pl.String,
    "is_amendment": pl.Boolean,
    "period_end": pl.Date,
    "submit_datetime": pl.Datetime("us"),
    "metric_name": pl.String,
    "metric_value": pl.Float64,
    "risk_change_score": pl.Float64,
}


def build_pit_dataframe(records: Iterable[dict[str, object]]) -> pl.DataFrame:
    """Create a typed and deterministically sorted Point-in-Time DataFrame."""
    rows = list(records)
    if not rows:
        return pl.DataFrame(schema=PIT_COLUMNS)

    df = pl.DataFrame(rows)
    period_expr = (
        pl.col("period_end").str.to_date("%Y-%m-%d")
        if df.schema.get("period_end") == pl.String
        else pl.col("period_end").cast(pl.Date)
    )
    submit_expr = (
        pl.col("submit_datetime").str.to_datetime("%Y-%m-%d %H:%M:%S").cast(pl.Datetime("us"))
        if df.schema.get("submit_datetime") == pl.String
        else pl.col("submit_datetime").cast(pl.Datetime("us"))
    )
    df = df.with_columns(
        period_expr.alias("period_end"),
        submit_expr.alias("submit_datetime"),
        pl.col("is_amendment").cast(pl.Boolean),
        pl.col("metric_value").cast(pl.Float64),
        pl.col("risk_change_score").cast(pl.Float64),
    )
    return df.select(list(PIT_COLUMNS.keys())).sort(
        ["edinet_code", "period_end", "metric_name", "submit_datetime"]
    )


def validate_pit_integrity(df: pl.DataFrame) -> None:
    """Validate Point-in-Time invariants; raise ValueError on violations."""
    if df.is_empty():
        return

    null_keys = df.filter(
        pl.col("doc_id").is_null()
        | pl.col("edinet_code").is_null()
        | pl.col("submit_datetime").is_null()
        | pl.col("metric_name").is_null()
    )
    if not null_keys.is_empty():
        raise ValueError(f"Found {null_keys.height} rows with null required keys.")

    # 決算期末日よりも前に提出日時が来ている異常値（先読み/メタデータ破損）を検知
    invalid_dates = df.filter(
        pl.col("period_end").is_not_null()
        & (pl.col("submit_datetime").dt.date() < pl.col("period_end"))
    )
    if not invalid_dates.is_empty():
        raise ValueError(
            f"Found {invalid_dates.height} rows where submit_datetime precedes period_end."
        )


def query_as_of(df: pl.DataFrame, as_of: datetime) -> pl.DataFrame:
    """Return the latest known value per (edinet_code, period_end, metric_name) at as_of."""
    eligible = df.filter(pl.col("submit_datetime") <= pl.lit(as_of))
    return (
        eligible.sort("submit_datetime")
        .group_by(["edinet_code", "period_end", "metric_name"], maintain_order=True)
        .last()
    )


def attach_drift_scores(df: pl.DataFrame, drift_df: pl.DataFrame) -> pl.DataFrame:
    """Attach Colab-computed business_risks drift scores onto PiT metrics."""
    if df.is_empty() or drift_df.is_empty():
        return df

    risk_scores = (
        drift_df.filter(pl.col("section") == "business_risks")
        .select(["doc_id", pl.col("drift_score").alias("_new_risk_score")])
    )
    return (
        df.join(risk_scores, on="doc_id", how="left")
        .with_columns(
            pl.coalesce(["_new_risk_score", "risk_change_score"]).alias("risk_change_score")
        )
        .drop("_new_risk_score")
    )


def export_dataset(df: pl.DataFrame, output_dir: Path, name: str = "edinet_pit") -> dict[str, Path]:
    """Validate and export the dataset to Parquet and DuckDB."""
    validate_pit_integrity(df)
    output_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = output_dir / f"{name}.parquet"
    duckdb_path = output_dir / f"{name}.duckdb"

    df.write_parquet(parquet_path, compression="zstd")

    con = duckdb.connect(str(duckdb_path))
    try:
        con.execute(
            "CREATE OR REPLACE TABLE pit_metrics AS SELECT * FROM read_parquet(?)",
            [str(parquet_path)],
        )
    finally:
        con.close()

    return {"parquet": parquet_path, "duckdb": duckdb_path}
