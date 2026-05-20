from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import spacy
from nltk.stem import PorterStemmer
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WIKIR_ROOT = PROJECT_ROOT.parent / "dz1" / "wikIR1k" / "wikIR1k"
PROCESSED_DIR = PROJECT_ROOT / "artifacts" / "processed"


@dataclass
class Query:
    query_id: str
    text: str


@dataclass
class Document:
    doc_id: str
    text: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["tfidf", "bm25"], required=True)
    parser.add_argument("--variant", choices=["original", "stemmed", "lemmatized"], required=True)
    parser.add_argument("--split", choices=["train", "validation", "test"], required=True)
    parser.add_argument("--top-k", type=int, default=1000)
    parser.add_argument("--k1", type=float, default=1.5)
    parser.add_argument("--b", type=float, default=0.75)
    parser.add_argument("--max-docs", type=int, default=None)
    parser.add_argument("--max-queries", type=int, default=None)
    return parser.parse_args()


def load_documents(limit: int | None = None) -> list[Document]:
    path = WIKIR_ROOT / "documents.csv"
    rows: list[Document] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if limit is not None and i >= limit:
                break
            rows.append(Document(doc_id=row["id_right"].strip(), text=row["text_right"].strip()))
    return rows


def load_queries(split: str, limit: int | None = None) -> list[Query]:
    path = WIKIR_ROOT / split / "queries.csv"
    rows: list[Query] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if limit is not None and i >= limit:
                break
            rows.append(Query(query_id=row["id_left"].strip(), text=row["text_left"].strip()))
    return rows


def get_processor(variant: str) -> Callable[[list[str]], list[list[str]]]:
    if variant == "original":
        return lambda texts: [text.split() for text in texts]

    if variant == "stemmed":
        stemmer = PorterStemmer()

        def stem_texts(texts: list[str]) -> list[list[str]]:
            return [[stemmer.stem(token) for token in text.split()] for text in texts]

        return stem_texts

    if variant == "lemmatized":
        try:
            nlp = spacy.blank("en")
            nlp.add_pipe("lemmatizer", config={"mode": "lookup"})
            nlp.initialize()
        except ValueError as exc:
            raise RuntimeError(
                "spaCy lookup lemmatizer is unavailable. Install spacy-lookups-data in the active environment."
            ) from exc

        def lemmatize_texts(texts: list[str]) -> list[list[str]]:
            results: list[list[str]] = []
            for doc in nlp.pipe(texts, batch_size=128):
                results.append([token.lemma_ for token in doc])
            return results

        return lemmatize_texts

    raise ValueError(f"Unsupported variant: {variant}")


def ensure_dirs() -> None:
    for rel in [
        "artifacts/runs",
        "artifacts/tables",
        "artifacts/timings",
        "artifacts/processed",
    ]:
        (PROJECT_ROOT / rel).mkdir(parents=True, exist_ok=True)


def processed_documents_path(variant: str) -> Path:
    return PROCESSED_DIR / f"documents_{variant}.csv"


def processed_queries_path(variant: str, split: str) -> Path:
    return PROCESSED_DIR / f"queries_{split}_{variant}.csv"


def write_processed_csv(path: Path, id_field: str, rows: list[tuple[str, list[str]]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[id_field, "processed_text"])
        writer.writeheader()
        for row_id, tokens in rows:
            writer.writerow({id_field: row_id, "processed_text": " ".join(tokens)})


def read_processed_csv(path: Path, id_field: str) -> list[tuple[str, list[str]]]:
    rows: list[tuple[str, list[str]]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append((row[id_field].strip(), row["processed_text"].split()))
    return rows


def run_name(
    method: str,
    variant: str,
    split: str,
    k1: float,
    b: float,
    max_docs: int | None,
    max_queries: int | None,
) -> str:
    if method == "bm25":
        base = f"{method}_{variant}_{split}_k1_{k1:g}_b_{b:g}"
    else:
        base = f"{method}_{variant}_{split}"

    suffix_parts: list[str] = []
    if max_docs is not None:
        suffix_parts.append(f"docs_{max_docs}")
    if max_queries is not None:
        suffix_parts.append(f"queries_{max_queries}")
    if suffix_parts:
        return f"{base}_{'_'.join(suffix_parts)}"
    return base


def stable_topk(doc_ids: list[str], scores: np.ndarray, top_k: int) -> list[tuple[str, float]]:
    limit = min(top_k, len(doc_ids))
    candidate_idx = np.argpartition(scores, -limit)[-limit:]
    candidates = [(doc_ids[idx], float(scores[idx])) for idx in candidate_idx]
    candidates.sort(key=lambda item: (-item[1], int(item[0]) if item[0].isdigit() else item[0]))
    return candidates[:limit]


def write_run(path: Path, run_id: str, rows: Iterable[tuple[str, str, int, float]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        for query_id, doc_id, rank, score in rows:
            f.write(f"{query_id} Q0 {doc_id} {rank} {score:.12f} {run_id}\n")


def write_query_timings(path: Path, timings: list[dict[str, float | str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["query_id", "seconds"])
        writer.writeheader()
        writer.writerows(timings)


def build_tfidf_retriever(
    doc_tokens: list[list[str]],
) -> tuple[Callable[[list[str]], np.ndarray], float]:
    doc_texts = [" ".join(tokens) for tokens in doc_tokens]
    vectorizer = TfidfVectorizer(
        tokenizer=str.split,
        preprocessor=None,
        lowercase=False,
        token_pattern=None,
        use_idf=True,
        norm="l2",
    )

    start = time.perf_counter()
    doc_matrix = vectorizer.fit_transform(doc_texts)
    build_seconds = time.perf_counter() - start

    def retrieve(query_tokens: list[str]) -> np.ndarray:
        query_text = " ".join(query_tokens)
        query_vector = vectorizer.transform([query_text])
        return (query_vector @ doc_matrix.T).toarray().ravel()

    return retrieve, build_seconds


def build_bm25_retriever(
    doc_tokens: list[list[str]],
    k1: float,
    b: float,
) -> tuple[Callable[[list[str]], np.ndarray], float]:
    start = time.perf_counter()
    bm25 = BM25Okapi(doc_tokens, k1=k1, b=b)
    build_seconds = time.perf_counter() - start

    def retrieve(query_tokens: list[str]) -> np.ndarray:
        return bm25.get_scores(query_tokens)

    return retrieve, build_seconds


def main() -> None:
    args = parse_args()
    ensure_dirs()

    docs = load_documents(limit=args.max_docs)
    queries = load_queries(args.split, limit=args.max_queries)
    processor = get_processor(args.variant)

    can_cache = args.max_docs is None and args.max_queries is None
    docs_cache_path = processed_documents_path(args.variant)
    queries_cache_path = processed_queries_path(args.variant, args.split)

    preprocess_start = time.perf_counter()
    used_cached_documents = False
    used_cached_queries = False

    if can_cache and docs_cache_path.exists():
        cached_docs = read_processed_csv(docs_cache_path, "doc_id")
        doc_ids = [doc_id for doc_id, _ in cached_docs]
        doc_tokens = [tokens for _, tokens in cached_docs]
        used_cached_documents = True
    else:
        doc_ids = [doc.doc_id for doc in docs]
        doc_tokens = processor([doc.text for doc in docs])
        if can_cache:
            write_processed_csv(
                docs_cache_path,
                "doc_id",
                list(zip(doc_ids, doc_tokens, strict=True)),
            )

    if can_cache and queries_cache_path.exists():
        cached_queries = read_processed_csv(queries_cache_path, "query_id")
        query_ids = [query_id for query_id, _ in cached_queries]
        query_text_map = {query.query_id: query.text for query in queries}
        queries = [Query(query_id=query_id, text=query_text_map[query_id]) for query_id in query_ids]
        query_tokens = [tokens for _, tokens in cached_queries]
        used_cached_queries = True
    else:
        query_tokens = processor([query.text for query in queries])
        if can_cache:
            write_processed_csv(
                queries_cache_path,
                "query_id",
                [(query.query_id, tokens) for query, tokens in zip(queries, query_tokens, strict=True)],
            )

    preprocess_seconds = time.perf_counter() - preprocess_start

    if args.method == "tfidf":
        retrieve_scores, build_seconds = build_tfidf_retriever(doc_tokens)
    else:
        retrieve_scores, build_seconds = build_bm25_retriever(doc_tokens, args.k1, args.b)

    run_id = run_name(
        args.method,
        args.variant,
        args.split,
        args.k1,
        args.b,
        args.max_docs,
        args.max_queries,
    )
    run_path = PROJECT_ROOT / "artifacts" / "runs" / f"{run_id}.trec"
    timings_csv = PROJECT_ROOT / "artifacts" / "timings" / f"{run_id}_query_times.csv"
    timings_json = PROJECT_ROOT / "artifacts" / "timings" / f"{run_id}_timings.json"

    rows: list[tuple[str, str, int, float]] = []
    query_timings: list[dict[str, float | str]] = []
    total_query_seconds = 0.0

    for query, query_token_list in zip(queries, query_tokens, strict=True):
        start = time.perf_counter()
        scores = retrieve_scores(query_token_list)
        top_docs = stable_topk(doc_ids, scores, args.top_k)
        elapsed = time.perf_counter() - start
        total_query_seconds += elapsed
        query_timings.append({"query_id": query.query_id, "seconds": elapsed})

        for rank, (doc_id, score) in enumerate(top_docs):
            rows.append((query.query_id, doc_id, rank, score))

    write_run(run_path, run_id, rows)
    write_query_timings(timings_csv, query_timings)

    timing_summary = {
        "run_id": run_id,
        "method": args.method,
        "variant": args.variant,
        "split": args.split,
        "documents": len(docs),
        "queries": len(queries),
        "top_k": args.top_k,
        "build_seconds": build_seconds,
        "preprocess_seconds": preprocess_seconds,
        "total_query_seconds": total_query_seconds,
        "avg_query_seconds": total_query_seconds / len(queries) if queries else 0.0,
        "k1": args.k1 if args.method == "bm25" else None,
        "b": args.b if args.method == "bm25" else None,
        "used_cached_documents": used_cached_documents,
        "used_cached_queries": used_cached_queries,
        "run_path": str(run_path),
    }
    with timings_json.open("w", encoding="utf-8") as f:
        json.dump(timing_summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(timing_summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
