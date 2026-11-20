"""Google Colab (GPU) batch script for computing YoY filing text drift.

Reads `edinet_pit_sample_texts.parquet`, computes chunk-averaged cosine
drift using `intfloat/multilingual-e5-large` on GPU, and writes
`edinet_pit_sample_drift.parquet`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl


def _chunk_text(text: str, chunk_chars: int = 400, max_chunks: int = 12) -> list[str]:
    cleaned = " ".join(text.split())
    if not cleaned:
        return [""]
    chunks = [
        cleaned[i : i + chunk_chars]
        for i in range(0, len(cleaned), chunk_chars)
    ][:max_chunks]
    return chunks


def _encode_doc(model: object, text: str) -> np.ndarray:
    chunks = _chunk_text(text)
    embs = model.encode(
        [f"passage: {c}" for c in chunks],
        batch_size=16,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    mean_emb = np.mean(embs, axis=0)
    norm = np.linalg.norm(mean_emb)
    return mean_emb / norm if norm > 0 else mean_emb


def main(
    input_path: Path = Path("edinet_pit_sample_texts.parquet"),
    output_path: Path = Path("edinet_pit_sample_drift.parquet"),
    model_name: str = "intfloat/multilingual-e5-large",
) -> None:
    import torch
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {model_name} on {device}...")
    model = SentenceTransformer(model_name, device=device)

    df = pl.read_parquet(input_path)
    if "is_amendment" in df.columns:
        df = df.filter(~pl.col("is_amendment"))
    elif "doc_type_code" in df.columns:
        df = df.filter(pl.col("doc_type_code") == "120")
    df = df.sort(["edinet_code", "section", "submit_datetime"])
    rows: list[dict[str, object]] = []

    for (edinet_code, section), group in df.group_by(
        ["edinet_code", "section"], maintain_order=True
    ):
        records = group.to_dicts()
        if len(records) < 2:
            continue
        prev_emb = _encode_doc(model, str(records[0]["text"]))
        for rec in records[1:]:
            curr_emb = _encode_doc(model, str(rec["text"]))
            drift = float(1.0 - np.dot(prev_emb, curr_emb))
            rows.append(
                {
                    "doc_id": rec["doc_id"],
                    "edinet_code": edinet_code,
                    "section": section,
                    "drift_score": round(drift, 6),
                }
            )
            prev_emb = curr_emb

    out_df = pl.DataFrame(rows)
    out_df.write_parquet(output_path, compression="zstd")
    print(f"Saved {out_df.height} drift scores to {output_path}")
    print(out_df)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="edinet_pit_sample_texts.parquet")
    parser.add_argument("--output", default="edinet_pit_sample_drift.parquet")
    args = parser.parse_args()
    main(input_path=Path(args.input), output_path=Path(args.output))
