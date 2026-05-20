from __future__ import annotations

import csv
import hashlib
import json
import urllib.request
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "mirage"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "mirage"
TABLES_DIR = ARTIFACTS_DIR / "tables"
PARQUET_URL = "https://huggingface.co/datasets/nlpai-lab/mirage/resolve/main/data/train-00000-of-00001.parquet"


def ensure_dirs() -> None:
    for path in [DATA_DIR, ARTIFACTS_DIR, TABLES_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def download_if_needed(path: Path) -> None:
    if path.exists():
        return
    urllib.request.urlretrieve(PARQUET_URL, path)


def passage_id(mapped_id: str, doc_chunk: str) -> str:
    digest = hashlib.sha1(f"{mapped_id}\n{doc_chunk}".encode("utf-8")).hexdigest()[:16]
    return f"mirage-{digest}"


def iter_pool(doc_pool: dict[str, list]) -> list[dict[str, object]]:
    return [
        {
            "mapped_id": mapped_id,
            "doc_name": doc_name,
            "doc_chunk": doc_chunk,
            "support": int(support),
        }
        for mapped_id, doc_name, doc_chunk, support in zip(
            doc_pool["mapped_id"],
            doc_pool["doc_name"],
            doc_pool["doc_chunk"],
            doc_pool["support"],
            strict=True,
        )
    ]


def main() -> None:
    ensure_dirs()
    parquet_path = DATA_DIR / "train.parquet"
    download_if_needed(parquet_path)

    rows = pq.read_table(parquet_path).to_pylist()

    documents: dict[str, dict[str, str]] = {}
    queries: list[dict[str, object]] = []
    qrels_rows: list[dict[str, object]] = []
    query_audit: list[dict[str, object]] = []

    for row in rows:
        query_id = row["query_id"]
        query_text = row["query"].strip()
        queries.append(
            {
                "query_id": query_id,
                "source": row["source"],
                "query": query_text,
                "num_doc_labels": int(row["num_doc_labels"]),
                "doc_name": row["doc_name"],
            }
        )

        positives = 0
        for item in iter_pool(row["doc_pool"]):
            doc_id = passage_id(item["mapped_id"], item["doc_chunk"])
            documents.setdefault(
                doc_id,
                {
                    "doc_id": doc_id,
                    "mapped_id": item["mapped_id"],
                    "doc_name": item["doc_name"],
                    "text": item["doc_chunk"].strip(),
                },
            )
            if item["support"] > 0:
                positives += 1
                qrels_rows.append(
                    {
                        "query_id": query_id,
                        "iteration": "0",
                        "doc_id": doc_id,
                        "relevance": item["support"],
                    }
                )

        oracle = row["oracle"]
        oracle_doc_id = passage_id(oracle["mapped_id"], oracle["doc_chunk"])
        documents.setdefault(
            oracle_doc_id,
            {
                "doc_id": oracle_doc_id,
                "mapped_id": oracle["mapped_id"],
                "doc_name": oracle["doc_name"],
                "text": oracle["doc_chunk"].strip(),
            },
        )
        if not any(
            qrel["query_id"] == query_id and qrel["doc_id"] == oracle_doc_id for qrel in qrels_rows
        ):
            qrels_rows.append(
                {
                    "query_id": query_id,
                    "iteration": "0",
                    "doc_id": oracle_doc_id,
                    "relevance": int(oracle["support"]),
                }
            )
            positives += int(oracle["support"] > 0)

        query_audit.append(
            {
                "query_id": query_id,
                "source": row["source"],
                "query_length_words": len(query_text.split()),
                "relevant_passages": positives,
                "pool_size": len(row["doc_pool"]["mapped_id"]),
            }
        )

    documents_csv = ARTIFACTS_DIR / "documents.csv"
    queries_csv = ARTIFACTS_DIR / "queries.csv"
    qrels_path = ARTIFACTS_DIR / "qrels.trec"

    pd.DataFrame(sorted(documents.values(), key=lambda item: item["doc_id"])).to_csv(documents_csv, index=False)
    pd.DataFrame(queries).to_csv(queries_csv, index=False)
    pd.DataFrame(query_audit).to_csv(TABLES_DIR / "mirage_query_audit.csv", index=False)
    pd.DataFrame(qrels_rows).to_csv(TABLES_DIR / "mirage_qrels_table.csv", index=False)

    with qrels_path.open("w", encoding="utf-8") as f:
        for row in qrels_rows:
            f.write(f"{row['query_id']} {row['iteration']} {row['doc_id']} {row['relevance']}\n")

    summary = pd.DataFrame(
        [
            {
                "queries": len(queries),
                "documents": len(documents),
                "qrels": len(qrels_rows),
                "mean_query_length_words": round(pd.DataFrame(query_audit)["query_length_words"].mean(), 2),
                "mean_relevant_passages": round(pd.DataFrame(query_audit)["relevant_passages"].mean(), 2),
                "mean_pool_size": round(pd.DataFrame(query_audit)["pool_size"].mean(), 2),
            }
        ]
    )
    summary.to_csv(TABLES_DIR / "mirage_collection_summary.csv", index=False)

    metadata = {
        "parquet_path": str(parquet_path),
        "documents_csv": str(documents_csv),
        "queries_csv": str(queries_csv),
        "qrels_path": str(qrels_path),
    }
    with (ARTIFACTS_DIR / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
