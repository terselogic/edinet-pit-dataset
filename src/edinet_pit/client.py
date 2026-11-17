"""EDINET API v2 client for fetching document lists and downloading filings."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import os
from pathlib import Path
from typing import Iterable

import httpx

BASE_URL = "https://api.edinet-fsa.go.jp/api/v2"

# 主要な書類種別コード（有報、訂正有報、四半期/半期、大量保有・変更）
DEFAULT_DOC_TYPE_CODES: frozenset[str] = frozenset(
    {"120", "130", "140", "150", "160", "170", "350", "360"}
)


@dataclass(frozen=True)
class EdinetDocument:
    """Normalized metadata for a single EDINET filing."""

    doc_id: str
    edinet_code: str | None
    sec_code: str | None
    filer_name: str
    doc_type_code: str
    period_start: str | None
    period_end: str | None
    submit_datetime: str
    doc_description: str
    xbrl_flag: bool
    csv_flag: bool

    @property
    def is_amendment(self) -> bool:
        """Return True if the filing is an amendment report."""
        return self.doc_type_code in {"130", "150", "170", "360"}


class EdinetClient:
    """Minimal synchronous client for EDINET API v2."""

    def __init__(
        self,
        api_key: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.api_key = api_key or os.environ.get("EDINET_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "EDINET_API_KEY is required. Set it in env or pass api_key."
            )
        self._client = httpx.Client(
            base_url=BASE_URL,
            timeout=timeout,
            params={"Subscription-Key": self.api_key},
        )

    def __enter__(self) -> EdinetClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._client.close()

    def list_documents(
        self,
        target_date: date,
        doc_type_codes: Iterable[str] = DEFAULT_DOC_TYPE_CODES,
    ) -> list[EdinetDocument]:
        """Fetch document metadata for a specific submission date."""
        resp = self._client.get(
            "/documents.json",
            params={"date": target_date.isoformat(), "type": 2},
        )
        resp.raise_for_status()
        payload = resp.json()

        allowed = set(doc_type_codes)
        results: list[EdinetDocument] = []
        for row in payload.get("results", []):
            code = str(row.get("docTypeCode") or "")
            if allowed and code not in allowed:
                continue
            if str(row.get("withdrawalStatus") or "0") != "0":
                continue
            results.append(
                EdinetDocument(
                    doc_id=str(row["docID"]),
                    edinet_code=row.get("edinetCode"),
                    sec_code=row.get("secCode"),
                    filer_name=str(row.get("filerName") or ""),
                    doc_type_code=code,
                    period_start=row.get("periodStart"),
                    period_end=row.get("periodEnd"),
                    submit_datetime=str(row.get("submitDateTime") or ""),
                    doc_description=str(row.get("docDescription") or ""),
                    xbrl_flag=str(row.get("xbrlFlag") or "0") == "1",
                    csv_flag=str(row.get("csvFlag") or "0") == "1",
                )
            )
        return results

    def download_document(
        self,
        doc_id: str,
        output_dir: Path,
        file_type: int = 5,
    ) -> Path:
        """Download a filing archive (type=1: XBRL, type=5: CSV) to output_dir."""
        output_dir.mkdir(parents=True, exist_ok=True)
        dest = output_dir / f"{doc_id}_type{file_type}.zip"
        resp = self._client.get(f"/documents/{doc_id}", params={"type": file_type})
        resp.raise_for_status()
        dest.write_bytes(resp.content)
        return dest
