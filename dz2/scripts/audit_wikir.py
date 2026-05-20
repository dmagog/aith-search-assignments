from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WIKIR_ROOT = PROJECT_ROOT.parent / "dz1" / "wikIR1k" / "wikIR1k"
OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "tables"


def count_documents(documents_path: Path) -> dict[str, object]:
    doc_count = 0
    empty_docs = 0
    token_count = 0
    min_doc_len: int | None = None
    max_doc_len = 0

    with documents_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            doc_count += 1
            text = row["text_right"].strip()
            length = len(text.split()) if text else 0
            token_count += length
            if length == 0:
                empty_docs += 1
            min_doc_len = length if min_doc_len is None else min(min_doc_len, length)
            max_doc_len = max(max_doc_len, length)

    avg_doc_len = token_count / doc_count if doc_count else 0.0
    return {
        "documents": doc_count,
        "empty_documents": empty_docs,
        "tokens": token_count,
        "avg_document_length": avg_doc_len,
        "min_document_length": min_doc_len or 0,
        "max_document_length": max_doc_len,
    }


def read_queries(queries_path: Path) -> tuple[list[dict[str, object]], set[str]]:
    rows: list[dict[str, object]] = []
    ids: set[str] = set()
    with queries_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            query_id = row["id_left"].strip()
            text = row["text_left"].strip()
            length = len(text.split()) if text else 0
            rows.append(
                {
                    "query_id": query_id,
                    "text": text,
                    "length_words": length,
                }
            )
            ids.add(query_id)
    return rows, ids


def read_qrels(qrels_path: Path) -> tuple[list[dict[str, object]], Counter[int], Counter[str], set[str]]:
    rows: list[dict[str, object]] = []
    relevance_counter: Counter[int] = Counter()
    per_query_counts: Counter[str] = Counter()
    doc_ids: set[str] = set()
    with qrels_path.open("r", encoding="utf-8", newline="") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            query_id, iteration, doc_id, relevance = line.split()
            rel_value = int(relevance)
            rows.append(
                {
                    "query_id": query_id,
                    "iteration": iteration,
                    "doc_id": doc_id,
                    "relevance": rel_value,
                }
            )
            relevance_counter[rel_value] += 1
            per_query_counts[query_id] += 1
            doc_ids.add(doc_id)
    return rows, relevance_counter, per_query_counts, doc_ids


def summarize_query_lengths(rows: list[dict[str, object]]) -> dict[str, float | int]:
    lengths = [int(row["length_words"]) for row in rows]
    if not lengths:
        return {
            "queries": 0,
            "avg_query_length": 0.0,
            "min_query_length": 0,
            "max_query_length": 0,
        }
    lengths_sorted = sorted(lengths)
    n = len(lengths_sorted)
    median = (
        lengths_sorted[n // 2]
        if n % 2 == 1
        else (lengths_sorted[n // 2 - 1] + lengths_sorted[n // 2]) / 2
    )
    return {
        "queries": n,
        "avg_query_length": sum(lengths_sorted) / n,
        "median_query_length": median,
        "min_query_length": lengths_sorted[0],
        "max_query_length": lengths_sorted[-1],
    }


def summarize_per_query_qrels(per_query_counts: Counter[str]) -> dict[str, float | int]:
    counts = sorted(per_query_counts.values())
    if not counts:
        return {
            "queries_with_qrels": 0,
            "avg_relevant_docs_per_query": 0.0,
            "min_relevant_docs_per_query": 0,
            "max_relevant_docs_per_query": 0,
        }
    n = len(counts)
    median = counts[n // 2] if n % 2 == 1 else (counts[n // 2 - 1] + counts[n // 2]) / 2
    return {
        "queries_with_qrels": n,
        "avg_relevant_docs_per_query": sum(counts) / n,
        "median_relevant_docs_per_query": median,
        "min_relevant_docs_per_query": counts[0],
        "max_relevant_docs_per_query": counts[-1],
    }


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    documents_path = WIKIR_ROOT / "documents.csv"
    splits = {
        "train": WIKIR_ROOT / "train",
        "validation": WIKIR_ROOT / "validation",
        "test": WIKIR_ROOT / "test",
    }

    document_summary = count_documents(documents_path)
    document_ids: set[str] = set()
    with documents_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            document_ids.add(row["id_right"].strip())

    overall_summary: dict[str, object] = {
        "dataset_root": str(WIKIR_ROOT),
        "documents": document_summary,
        "splits": {},
    }

    split_rows: list[dict[str, object]] = []

    for split_name, split_dir in splits.items():
        queries_path = split_dir / "queries.csv"
        qrels_path = split_dir / "qrels"

        query_rows, query_ids = read_queries(queries_path)
        qrel_rows, relevance_counter, per_query_counts, qrel_doc_ids = read_qrels(qrels_path)

        queries_without_qrels = sorted(query_ids - set(per_query_counts.keys()))
        qrels_without_queries = sorted(set(per_query_counts.keys()) - query_ids)
        qrels_missing_documents = sorted(qrel_doc_ids - document_ids)
        empty_queries = sum(1 for row in query_rows if int(row["length_words"]) == 0)

        query_summary = summarize_query_lengths(query_rows)
        qrel_summary = summarize_per_query_qrels(per_query_counts)

        split_summary = {
            "queries": query_summary,
            "qrels": {
                "rows": len(qrel_rows),
                "relevance_distribution": dict(relevance_counter),
                **qrel_summary,
            },
            "sanity_checks": {
                "empty_queries": empty_queries,
                "queries_without_qrels": len(queries_without_qrels),
                "qrels_without_queries": len(qrels_without_queries),
                "qrels_missing_documents": len(qrels_missing_documents),
            },
        }
        overall_summary["splits"][split_name] = split_summary

        split_rows.append(
            {
                "split": split_name,
                "queries": query_summary["queries"],
                "avg_query_length": query_summary["avg_query_length"],
                "median_query_length": query_summary["median_query_length"],
                "min_query_length": query_summary["min_query_length"],
                "max_query_length": query_summary["max_query_length"],
                "qrels_rows": len(qrel_rows),
                "avg_relevant_docs_per_query": qrel_summary["avg_relevant_docs_per_query"],
                "median_relevant_docs_per_query": qrel_summary["median_relevant_docs_per_query"],
                "min_relevant_docs_per_query": qrel_summary["min_relevant_docs_per_query"],
                "max_relevant_docs_per_query": qrel_summary["max_relevant_docs_per_query"],
                "empty_queries": empty_queries,
                "queries_without_qrels": len(queries_without_qrels),
                "qrels_without_queries": len(qrels_without_queries),
                "qrels_missing_documents": len(qrels_missing_documents),
            }
        )

        write_csv(
            OUTPUT_DIR / f"{split_name}_queries_audit.csv",
            ["query_id", "text", "length_words"],
            query_rows,
        )

        write_csv(
            OUTPUT_DIR / f"{split_name}_qrels_per_query.csv",
            ["query_id", "relevant_docs"],
            [
                {"query_id": query_id, "relevant_docs": count}
                for query_id, count in sorted(per_query_counts.items(), key=lambda x: int(x[0]))
            ],
        )

    write_csv(
        OUTPUT_DIR / "wikir_split_summary.csv",
        [
            "split",
            "queries",
            "avg_query_length",
            "median_query_length",
            "min_query_length",
            "max_query_length",
            "qrels_rows",
            "avg_relevant_docs_per_query",
            "median_relevant_docs_per_query",
            "min_relevant_docs_per_query",
            "max_relevant_docs_per_query",
            "empty_queries",
            "queries_without_qrels",
            "qrels_without_queries",
            "qrels_missing_documents",
        ],
        split_rows,
    )

    with (OUTPUT_DIR / "wikir_audit_summary.json").open("w", encoding="utf-8") as f:
        json.dump(overall_summary, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
