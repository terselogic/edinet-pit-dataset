# EDINET Point-in-Time Dataset (`edinet-pit-dataset`)

日本株クオンツ・システムトレーダー向けの、**サバイバーシップバイアス除去済み・Point-in-Time（時点整合性）保証・有報テキスト変化スコア付き** 静的データセット（Parquet / DuckDB）および構築パイプラインです。

> **無料サンプル（10社・2年分）**: [`sample/edinet_pit_sample.parquet`](sample/edinet_pit_sample.parquet) / [`sample/edinet_pit_sample.duckdb`](sample/edinet_pit_sample.duckdb)

---

## 既存データとの違い

1. **厳密な Point-in-Time（先読みバイアス排除）**
   * 訂正報告書（`130` 等）が後から提出された場合でも、過去の `as_of` 時点では訂正前の数値のみが参照される履歴構造（`submit_datetime` × `period_end`）。
2. **上場廃止銘柄の保持（生存者バイアス排除）**
   * EDINET提出書類ベースで過去の廃止銘柄も保持する設計。
3. **GPUバッチ処理による定性テキストの数値化（`risk_change_score`）**
   * 有価証券報告書の「事業等のリスク」を多言語Embedding（`intfloat/multilingual-e5-large`）で前年比較し、記述変化度（コサイン距離）を事前計算して付与。

---

## スキーマ定義（`edinet_pit_sample.parquet` / `pit_metrics` テーブル）

| カラム名 | 型 | 説明 |
| :--- | :--- | :--- |
| `doc_id` | `String` | EDINET 書類管理番号（例：`S100YDF3`） |
| `edinet_code` | `String` | EDINET コード（例：`E01441`） |
| `sec_code` | `String` | 証券コード5桁（例：`59730`） |
| `filer_name` | `String` | 提出者名 |
| `doc_type_code` | `String` | 書類種別コード（`120`: 有報, `130`: 訂正有報 等） |
| `is_amendment` | `Boolean` | 訂正報告書フラグ |
| `period_end` | `Date` | 会計期末日（`YYYY-MM-DD`） |
| `submit_datetime` | `Datetime` | 提出日時（先読み防止の基準時刻） |
| `metric_name` | `String` | 正規化指標名（`net_sales`, `operating_income`, `ordinary_income`, `net_income`, `net_assets`, `total_assets`, `operating_cf`） |
| `metric_value` | `Float64` | 指標値（円） |
| `risk_change_score` | `Float64` | 有報「事業等のリスク」の前年比変化スコア（コサイン距離） |

---

## クイックスタート（無料サンプルを3行で試す）

### Polars で特定時点（`as_of`）のデータを取得

```python
from datetime import datetime
import polars as pl

df = pl.read_parquet("sample/edinet_pit_sample.parquet")
as_of = datetime(2026, 6, 26, 0, 0, 0)

pit_snapshot = (
    df.filter(pl.col("submit_datetime") <= as_of)
    .sort("submit_datetime")
    .group_by(["edinet_code", "period_end", "metric_name"])
    .last()
)
print(pit_snapshot.head())
```

### DuckDB でそのまま SQL 照会

```sql
SELECT sec_code, filer_name, period_end, metric_name, metric_value, risk_change_score
FROM read_parquet('sample/edinet_pit_sample.parquet')
WHERE submit_datetime <= TIMESTAMP '2026-06-26 00:00:00'
QUALIFY ROW_NUMBER() OVER (
    PARTITION BY edinet_code, period_end, metric_name
    ORDER BY submit_datetime DESC
) = 1;
```

---

## 148社コーホート検証結果（2025年 vs 2026年有報）

上場企業148社の「事業等のリスク」前年比変化スコア（`risk_change_score`）を計測したところ、以下の傾向が確認されました。

* **中央値（50%）**: `0.001543`（下位25%は `0.000253` 以下で前年踏襲）
* **最大値**: `0.023372`（中央値の約15倍）
* 上位には経営環境・事業構造やリスク開示を大きく書き換えた銘柄（例：ANAホールディングス `0.023372`、大東建託 `0.018248` 等）が明確に分離されました。

---

## 全上場企業・フル版（先行案内）

全上場企業（上場廃止含む）× 過去5〜10年分のフルデータセット（初期買い切り 20,000円想定 / 月次更新サブスク 3,500円想定）の先行リリース通知・追加項目リクエストを受け付けています。

* **先行案内・リクエスト登録**: `https://terselogic.com/edinet-pit/`（※準備中・公開用URLへ差し替え）

---

## ローカル開発・テスト

```bash
uv sync --extra dev
uv run pytest -v
```
