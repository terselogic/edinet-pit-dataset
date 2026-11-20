"""Build a multi-year sample Point-in-Time EDINET dataset (Parquet + DuckDB + texts)."""

from __future__ import annotations

from datetime import date, timedelta
import os
from pathlib import Path

import polars as pl

from edinet_pit.client import EdinetClient, EdinetDocument
from edinet_pit.parser import parse_filing_zip
from edinet_pit.pit_schema import (
    attach_drift_scores,
    build_pit_dataframe,
    export_dataset,
    write_license_attributions,
)


def _load_dotenv(env_path: Path) -> None:
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, val = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), val.strip())


def _select_listed_filings(
    docs: list[EdinetDocument],
    limit_annual: int = 8,
    limit_amend: int = 2,
) -> list[EdinetDocument]:
    annual: list[EdinetDocument] = []
    amend: list[EdinetDocument] = []
    for d in docs:
        if not d.sec_code or not d.csv_flag:
            continue
        if d.doc_type_code == "120" and len(annual) < limit_annual:
            annual.append(d)
        elif d.doc_type_code == "130" and len(amend) < limit_amend:
            amend.append(d)
        if len(annual) >= limit_annual and len(amend) >= limit_amend:
            break
    return annual + amend


def _find_prior_year_filings(
    client: EdinetClient,
    edinet_codes: set[str],
    start: date,
    end: date,
) -> list[EdinetDocument]:
    matched: list[EdinetDocument] = []
    seen: set[str] = set()
    cur = start
    while cur <= end and len(seen) < len(edinet_codes):
        for d in client.list_documents(cur, doc_type_codes={"120"}):
            if d.edinet_code in edinet_codes and d.edinet_code not in seen and d.csv_flag:
                matched.append(d)
                seen.add(d.edinet_code)
        cur += timedelta(days=1)
    return matched


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    _load_dotenv(root / ".env")

    raw_dir = root / "data" / "raw"
    out_dir = root / "data" / "out"

    target_day = date(2026, 6, 25)
    metric_rows: list[dict[str, object]] = []
    text_rows: list[dict[str, object]] = []

    with EdinetClient() as client:
        docs_2026 = client.list_documents(target_day, doc_type_codes={"120", "130"})
        selected_2026 = _select_listed_filings(docs_2026, limit_annual=8, limit_amend=2)
        target_codes = {
            d.edinet_code for d in selected_2026 if d.doc_type_code == "120" and d.edinet_code
        }
        prior_2025 = _find_prior_year_filings(
            client, target_codes, date(2025, 6, 20), date(2025, 6, 30)
        )
        all_docs = prior_2025 + selected_2026
        print(f"Selected {len(all_docs)} filings (2025={len(prior_2025)}, 2026={len(selected_2026)}).")

        for doc in all_docs:
            zip_path = raw_dir / f"{doc.doc_id}_type5.zip"
            if not zip_path.exists():
                zip_path = client.download_document(doc.doc_id, raw_dir, file_type=5)
            parsed = parse_filing_zip(zip_path, doc)
            metric_rows.extend(parsed.metrics)
            resolved_period_end = (
                parsed.metrics[0].period_end if parsed.metrics else doc.period_end
            )
            for section, content in parsed.texts.items():
                text_rows.append(
                    {
                        "doc_id": doc.doc_id,
                        "edinet_code": doc.edinet_code,
                        "sec_code": doc.sec_code,
                        "filer_name": doc.filer_name,
                        "doc_type_code": doc.doc_type_code,
                        "is_amendment": doc.is_amendment,
                        "period_end": resolved_period_end,
                        "submit_datetime": doc.submit_datetime,
                        "section": section,
                        "text_length": len(content),
                        "text": content,
                    }
                )

    df = build_pit_dataframe(metric_rows)
    drift_path = out_dir / "edinet_pit_sample_drift.parquet"
    if drift_path.exists():
        df = attach_drift_scores(df, pl.read_parquet(drift_path))
    outputs = export_dataset(df, out_dir, name="edinet_pit_sample")
    write_license_attributions(out_dir)

    sample_dir = root / "sample"
    export_dataset(df, sample_dir, name="edinet_pit_sample")
    write_license_attributions(sample_dir)

    if text_rows:
        texts_df = pl.DataFrame(text_rows).sort(["edinet_code", "section", "submit_datetime"])
        texts_path = out_dir / "edinet_pit_sample_texts.parquet"
        texts_df.write_parquet(texts_path, compression="zstd")
        print(f"Texts parquet: {texts_path} ({texts_df.height} rows)")

    print(f"Metrics rows: {df.height}, companies: {df['edinet_code'].n_unique()}")
    print("Outputs:", {k: str(v) for k, v in outputs.items()})


if __name__ == "__main__":
    main()
