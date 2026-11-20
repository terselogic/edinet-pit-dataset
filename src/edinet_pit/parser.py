"""Parser for EDINET type=5 (XBRL-to-CSV) ZIP archives."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import io
from pathlib import Path
import re
import zipfile

from edinet_pit.client import EdinetDocument

# 主要財務指標の要素ID -> 正規化メトリクス名マッピング（日本基準・IFRS等を統一）
METRIC_ELEMENT_MAP: dict[str, str] = {
    "jpcrp_cor:NetSalesSummaryOfBusinessResults": "net_sales",
    "jpcrp_cor:RevenueIFRSSummaryOfBusinessResults": "net_sales",
    "jpcrp_cor:OperatingRevenue1SummaryOfBusinessResults": "net_sales",
    "jppfs_cor:OperatingIncome": "operating_income",
    "jpcrp_cor:OperatingIncomeSummaryOfBusinessResults": "operating_income",
    "jpcrp_cor:ProfitLossBeforeTaxIFRSSummaryOfBusinessResults": "ordinary_income",
    "jpcrp_cor:OrdinaryIncomeLossSummaryOfBusinessResults": "ordinary_income",
    "jpcrp_cor:ProfitLossAttributableToOwnersOfParentSummaryOfBusinessResults": "net_income",
    "jpcrp_cor:ProfitLossAttributableToOwnersOfParentIFRSSummaryOfBusinessResults": "net_income",
    "jpcrp_cor:NetAssetsSummaryOfBusinessResults": "net_assets",
    "jpcrp_cor:EquityAttributableToOwnersOfParentIFRSSummaryOfBusinessResults": "net_assets",
    "jpcrp_cor:TotalAssetsSummaryOfBusinessResults": "total_assets",
    "jpcrp_cor:TotalAssetsIFRSSummaryOfBusinessResults": "total_assets",
    "jpcrp_cor:NetCashProvidedByUsedInOperatingActivitiesSummaryOfBusinessResults": "operating_cf",
    "jpcrp_cor:CashFlowsFromUsedInOperatingActivitiesIFRSSummaryOfBusinessResults": "operating_cf",
    "jpcrp_cor:CashFlowsFromOperatingActivitiesSummaryOfBusinessResults": "operating_cf",
    "jppfs_cor:NetCashProvidedByUsedInOperatingActivities": "operating_cf",
}

DEI_PERIOD_END_ELEMENTS: tuple[str, ...] = (
    "jpdei_cor:CurrentPeriodEndDateDEI",
    "jpdei_cor:CurrentFiscalYearEndDateDEI",
)

ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
CSV_FILENAME_PERIOD_RE = re.compile(r"_[A-Z]\d{5}-\d{3}_(\d{4}-\d{2}-\d{2})_")

TEXT_BLOCK_ELEMENTS: dict[str, str] = {
    "jpcrp_cor:BusinessRisksTextBlock": "business_risks",
    "jpcrp_cor:ManagementAnalysisOfFinancialPositionOperatingResultsAndCashFlowsTextBlock": "mda_analysis",
}

PREFERRED_CONTEXTS = ("CurrentYearDuration", "CurrentYearInstant")
NON_CONSOLIDATED_CONTEXTS = (
    "CurrentYearDuration_NonConsolidatedMember",
    "CurrentYearInstant_NonConsolidatedMember",
)


@dataclass(frozen=True)
class ParsedFiling:
    """Extracted metrics and text blocks from a single EDINET filing."""

    metrics: list[dict[str, object]]
    texts: dict[str, str]


def _parse_float(raw_value: str) -> float | None:
    cleaned = raw_value.strip().replace(",", "")
    if not cleaned or cleaned in {"-", "－"}:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def _pick_main_csv_name(names: list[str]) -> str | None:
    for name in names:
        if "jpcrp" in name and name.endswith(".csv"):
            return name
    return None


def parse_filing_zip(zip_path: Path, doc: EdinetDocument) -> ParsedFiling:
    """Extract normalized current-period metrics and risk/MD&A text blocks."""
    with zipfile.ZipFile(zip_path) as zf:
        csv_name = _pick_main_csv_name(zf.namelist())
        if not csv_name:
            return ParsedFiling(metrics=[], texts={})
        raw_bytes = zf.read(csv_name)

    try:
        text = raw_bytes.decode("utf-16")
    except UnicodeDecodeError:
        text = raw_bytes.decode("utf-16", errors="replace")
    reader = csv.reader(io.StringIO(text), delimiter="\t")
    header = next(reader, None)
    if not header:
        return ParsedFiling(metrics=[], texts={})

    # 連結優先、なければ個別を採用するため一時保持
    consolidated: dict[str, float] = {}
    non_consolidated: dict[str, float] = {}
    texts: dict[str, str] = {}
    dei_period_end: dict[str, str] = {}

    for row in reader:
        if len(row) < 9:
            continue
        elem_id, context_id, val = row[0], row[2], row[8]

        if elem_id in DEI_PERIOD_END_ELEMENTS and val.strip():
            cand = val.strip()
            if ISO_DATE_RE.match(cand) and elem_id not in dei_period_end:
                dei_period_end[elem_id] = cand
            continue

        if elem_id in TEXT_BLOCK_ELEMENTS and context_id in {
            "FilingDateInstant",
            "CurrentYearDuration",
            "CurrentYearDuration_NonConsolidatedMember",
        }:
            section = TEXT_BLOCK_ELEMENTS[elem_id]
            if section not in texts and val.strip():
                texts[section] = val.strip()
            continue

        metric_name = METRIC_ELEMENT_MAP.get(elem_id)
        if not metric_name:
            continue

        num = _parse_float(val)
        if num is None:
            continue

        if context_id in PREFERRED_CONTEXTS and metric_name not in consolidated:
            consolidated[metric_name] = num
        elif context_id in NON_CONSOLIDATED_CONTEXTS and metric_name not in non_consolidated:
            non_consolidated[metric_name] = num

    # 訂正報告書(130等)ではAPIメタデータのperiod_endがNoneになるためDEI/ファイル名から補完
    resolved_period_end = doc.period_end
    if not resolved_period_end:
        for key in DEI_PERIOD_END_ELEMENTS:
            if key in dei_period_end:
                resolved_period_end = dei_period_end[key]
                break
    if not resolved_period_end:
        fname_match = CSV_FILENAME_PERIOD_RE.search(csv_name)
        if fname_match:
            resolved_period_end = fname_match.group(1)

    merged = {**non_consolidated, **consolidated}
    # CSVダウンロード時は submit_datetime が "YYYY-MM-DD HH:MM" なので秒を補完
    submit_ts = (
        f"{doc.submit_datetime}:00"
        if len(doc.submit_datetime) == 16
        else doc.submit_datetime
    )

    metrics = [
        {
            "doc_id": doc.doc_id,
            "edinet_code": doc.edinet_code,
            "sec_code": doc.sec_code,
            "filer_name": doc.filer_name,
            "doc_type_code": doc.doc_type_code,
            "is_amendment": doc.is_amendment,
            "period_end": resolved_period_end,
            "submit_datetime": submit_ts,
            "metric_name": m_name,
            "metric_value": m_val,
            "risk_change_score": None,
        }
        for m_name, m_val in sorted(merged.items())
    ]
    return ParsedFiling(metrics=metrics, texts=texts)
