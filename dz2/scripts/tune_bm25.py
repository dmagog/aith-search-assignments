from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import ir_measures
from ir_measures import AP, P, ScoredDoc, nDCG
from rank_bm25 import BM25Okapi

from run_retrieval import (
    PROJECT_ROOT,
    WIKIR_ROOT,
    ensure_dirs,
    get_processor,
    load_documents,
    load_queries,
    processed_documents_path,
    processed_queries_path,
    read_processed_csv,
    stable_topk,
    write_processed_csv,
)


MEASURES = [P@1, P@10, P@20, AP, nDCG@20]
MEASURE_BY_NAME = {
    "P@1": P@1,
    "P@10": P@10,
    "P@20": P@20,
    "AP": AP,
    "nDCG@20": nDCG@20,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["original", "stemmed", "lemmatized"], required=True)
    parser.add_argument("--split", choices=["validation"], default="validation")
    parser.add_argument("--top-k", type=int, default=1000)
    parser.add_argument("--optimize", choices=sorted(MEASURE_BY_NAME), default="AP")
    parser.add_argument("--k1-values", default="0.6,0.9,1.2,1.5,1.8,2.0")
    parser.add_argument("--b-values", default="0.2,0.4,0.6,0.75,0.9")
    return parser.parse_args()


def parse_float_list(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def load_or_prepare_documents(variant: str) -> tuple[list[str], list[list[str]], bool]:
    cache_path = processed_documents_path(variant)
    if cache_path.exists():
        rows = read_processed_csv(cache_path, "doc_id")
        return [row_id for row_id, _ in rows], [tokens for _, tokens in rows], True

    docs = load_documents()
    processor = get_processor(variant)
    doc_ids = [doc.doc_id for doc in docs]
    doc_tokens = processor([doc.text for doc in docs])
    write_processed_csv(cache_path, "doc_id", list(zip(doc_ids, doc_tokens, strict=True)))
    return doc_ids, doc_tokens, False


def load_or_prepare_queries(variant: str, split: str) -> tuple[list[str], list[list[str]], bool]:
    cache_path = processed_queries_path(variant, split)
    if cache_path.exists():
        rows = read_processed_csv(cache_path, "query_id")
        return [row_id for row_id, _ in rows], [tokens for _, tokens in rows], True

    queries = load_queries(split)
    processor = get_processor(variant)
    query_ids = [query.query_id for query in queries]
    query_tokens = processor([query.text for query in queries])
    write_processed_csv(cache_path, "query_id", list(zip(query_ids, query_tokens, strict=True)))
    return query_ids, query_tokens, False


def main() -> None:
    args = parse_args()
    ensure_dirs()
    tuning_dir = PROJECT_ROOT / "artifacts" / "bm25_tuning"
    tuning_dir.mkdir(parents=True, exist_ok=True)

    optimize_measure = MEASURE_BY_NAME[args.optimize]
    k1_values = parse_float_list(args.k1_values)
    b_values = parse_float_list(args.b_values)

    prep_start = time.perf_counter()
    doc_ids, doc_tokens, used_cached_documents = load_or_prepare_documents(args.variant)
    query_ids, query_tokens, used_cached_queries = load_or_prepare_queries(args.variant, args.split)
    preprocess_seconds = time.perf_counter() - prep_start

    qrels = list(ir_measures.read_trec_qrels(str(WIKIR_ROOT / args.split / "qrels")))

    rows: list[dict[str, float | str | bool]] = []
    best_row: dict[str, float | str | bool] | None = None
    best_run: list[ScoredDoc] | None = None

    for k1 in k1_values:
        for b in b_values:
            build_start = time.perf_counter()
            bm25 = BM25Okapi(doc_tokens, k1=k1, b=b)
            build_seconds = time.perf_counter() - build_start

            run: list[ScoredDoc] = []
            total_query_seconds = 0.0
            for query_id, tokens in zip(query_ids, query_tokens, strict=True):
                query_start = time.perf_counter()
                scores = bm25.get_scores(tokens)
                top_docs = stable_topk(doc_ids, scores, args.top_k)
                total_query_seconds += time.perf_counter() - query_start
                for doc_id, score in top_docs:
                    run.append(ScoredDoc(query_id=query_id, doc_id=doc_id, score=float(score)))

            aggregate = ir_measures.calc_aggregate(MEASURES, qrels, run)
            row = {
                "variant": args.variant,
                "split": args.split,
                "k1": k1,
                "b": b,
                "P@1": float(aggregate[P@1]),
                "P@10": float(aggregate[P@10]),
                "P@20": float(aggregate[P@20]),
                "AP": float(aggregate[AP]),
                "nDCG@20": float(aggregate[nDCG@20]),
                "build_seconds": build_seconds,
                "total_query_seconds": total_query_seconds,
                "avg_query_seconds": total_query_seconds / len(query_ids) if query_ids else 0.0,
                "used_cached_documents": used_cached_documents,
                "used_cached_queries": used_cached_queries,
                "preprocess_seconds": preprocess_seconds,
            }
            rows.append(row)

            if best_row is None or float(row[args.optimize]) > float(best_row[args.optimize]):
                best_row = row
                best_run = run

    assert best_row is not None
    assert best_run is not None

    base_name = f"bm25_tuning_{args.variant}_{args.split}"
    summary_csv = tuning_dir / f"{base_name}.csv"
    summary_json = tuning_dir / f"{base_name}_best.json"
    best_run_path = tuning_dir / f"{base_name}_best_run.trec"

    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "variant",
                "split",
                "k1",
                "b",
                "P@1",
                "P@10",
                "P@20",
                "AP",
                "nDCG@20",
                "build_seconds",
                "total_query_seconds",
                "avg_query_seconds",
                "used_cached_documents",
                "used_cached_queries",
                "preprocess_seconds",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    with best_run_path.open("w", encoding="utf-8", newline="") as f:
        current_query_id: str | None = None
        rank = -1
        for scored_doc in best_run:
            if scored_doc.query_id != current_query_id:
                current_query_id = scored_doc.query_id
                rank = 0
            else:
                rank += 1
            f.write(
                f"{scored_doc.query_id} Q0 {scored_doc.doc_id} {rank} "
                f"{scored_doc.score:.12f} best_bm25_{args.variant}_{args.split}\n"
            )

    result = {
        "variant": args.variant,
        "split": args.split,
        "optimize": args.optimize,
        "best": best_row,
        "summary_csv": str(summary_csv),
        "best_run_path": str(best_run_path),
        "grid_size": len(k1_values) * len(b_values),
        "k1_values": k1_values,
        "b_values": b_values,
    }
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
