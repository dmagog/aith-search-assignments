from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import spacy
from nltk.stem import PorterStemmer
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MIRAGE_DIR = PROJECT_ROOT / "artifacts" / "mirage"
PROCESSED_DIR = MIRAGE_DIR / "processed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["tfidf", "bm25"], required=True)
    parser.add_argument("--variant", choices=["original", "stemmed", "lemmatized"], required=True)
    parser.add_argument("--top-k", type=int, default=1000)
    parser.add_argument("--k1", type=float, default=1.5)
    parser.add_argument("--b", type=float, default=0.75)
    return parser.parse_args()


def ensure_dirs() -> None:
    for path in [MIRAGE_DIR / "runs", MIRAGE_DIR / "timings", PROCESSED_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def get_processor(variant: str):
    if variant == "original":
        return lambda texts: [text.split() for text in texts]
    if variant == "stemmed":
        stemmer = PorterStemmer()
        return lambda texts: [[stemmer.stem(token) for token in text.split()] for text in texts]
    if variant == "lemmatized":
        nlp = spacy.blank("en")
        nlp.add_pipe("lemmatizer", config={"mode": "lookup"})
        nlp.initialize()
        return lambda texts: [[token.lemma_ for token in doc] for doc in nlp.pipe(texts, batch_size=128)]
    raise ValueError(variant)


def stable_topk(doc_ids: list[str], scores: np.ndarray, top_k: int) -> list[tuple[str, float]]:
    limit = min(top_k, len(doc_ids))
    idx = np.argpartition(scores, -limit)[-limit:]
    rows = [(doc_ids[i], float(scores[i])) for i in idx]
    rows.sort(key=lambda item: (-item[1], item[0]))
    return rows[:limit]


def build_tfidf(doc_tokens: list[list[str]]):
    vectorizer = TfidfVectorizer(
        tokenizer=str.split,
        preprocessor=None,
        lowercase=False,
        token_pattern=None,
        use_idf=True,
        norm="l2",
    )
    matrix = vectorizer.fit_transform([" ".join(tokens) for tokens in doc_tokens])
    return vectorizer, matrix


def build_fast_bm25(doc_tokens: list[list[str]], k1: float, b: float):
    vectorizer = CountVectorizer(
        tokenizer=str.split,
        preprocessor=None,
        lowercase=False,
        token_pattern=None,
    )
    counts = vectorizer.fit_transform([" ".join(tokens) for tokens in doc_tokens]).tocsr()
    doc_lengths = np.asarray(counts.sum(axis=1)).ravel()
    avgdl = float(doc_lengths.mean())
    num_docs = counts.shape[0]
    df = np.asarray((counts > 0).sum(axis=0)).ravel()
    idf = np.log((num_docs - df + 0.5) / (df + 0.5) + 1.0)

    weights = counts.copy().astype(np.float64)
    rows = np.repeat(np.arange(num_docs), np.diff(counts.indptr))
    tf = weights.data
    denom = tf + k1 * (1 - b + b * doc_lengths[rows] / avgdl)
    weights.data = idf[counts.indices] * (tf * (k1 + 1) / denom)
    return vectorizer, weights.tocsr()


def main() -> None:
    args = parse_args()
    ensure_dirs()

    docs = pd.read_csv(MIRAGE_DIR / "documents.csv")
    queries = pd.read_csv(MIRAGE_DIR / "queries.csv")
    processor = get_processor(args.variant)

    preprocess_start = time.perf_counter()
    doc_tokens = processor(docs["text"].tolist())
    query_tokens = processor(queries["query"].tolist())
    preprocess_seconds = time.perf_counter() - preprocess_start

    if args.method == "tfidf":
        build_start = time.perf_counter()
        vectorizer, matrix = build_tfidf(doc_tokens)
        build_seconds = time.perf_counter() - build_start

    else:
        build_start = time.perf_counter()
        bm25_vectorizer, bm25_matrix = build_fast_bm25(doc_tokens, k1=args.k1, b=args.b)
        build_seconds = time.perf_counter() - build_start

        def score_query_batch(texts: list[str]) -> sparse.csr_matrix:
            query_counts = bm25_vectorizer.transform(texts)
            return (query_counts @ bm25_matrix.T).tocsr()

    run_id = f"mirage_{args.method}_{args.variant}"
    if args.method == "bm25":
        run_id += f"_k1_{args.k1:g}_b_{args.b:g}"

    run_rows: list[tuple[str, str, int, float]] = []
    timings: list[dict[str, object]] = []
    doc_ids = docs["doc_id"].tolist()

    if args.method == "tfidf":
        batch_size = 128
        query_texts = [" ".join(tokens) for tokens in query_tokens]
        for start_idx in range(0, len(query_texts), batch_size):
            end_idx = min(start_idx + batch_size, len(query_texts))
            batch_texts = query_texts[start_idx:end_idx]
            batch_ids = queries["query_id"].iloc[start_idx:end_idx].tolist()
            batch_start = time.perf_counter()
            batch_scores = (vectorizer.transform(batch_texts) @ matrix.T).toarray()
            batch_elapsed = time.perf_counter() - batch_start
            per_query_elapsed = batch_elapsed / len(batch_ids)
            for query_id, scores in zip(batch_ids, batch_scores, strict=True):
                timings.append({"query_id": query_id, "seconds": per_query_elapsed})
                for rank, (doc_id, score) in enumerate(stable_topk(doc_ids, scores, args.top_k), start=1):
                    run_rows.append((query_id, doc_id, rank, score))
    else:
        batch_size = 128
        query_texts = [" ".join(tokens) for tokens in query_tokens]
        for start_idx in range(0, len(query_texts), batch_size):
            end_idx = min(start_idx + batch_size, len(query_texts))
            batch_texts = query_texts[start_idx:end_idx]
            batch_ids = queries["query_id"].iloc[start_idx:end_idx].tolist()
            batch_start = time.perf_counter()
            batch_scores = score_query_batch(batch_texts)
            batch_elapsed = time.perf_counter() - batch_start
            per_query_elapsed = batch_elapsed / len(batch_ids)
            for row_idx, query_id in enumerate(batch_ids):
                dense_scores = batch_scores.getrow(row_idx).toarray().ravel()
                timings.append({"query_id": query_id, "seconds": per_query_elapsed})
                for rank, (doc_id, score) in enumerate(stable_topk(doc_ids, dense_scores, args.top_k), start=1):
                    run_rows.append((query_id, doc_id, rank, score))

    run_path = MIRAGE_DIR / "runs" / f"{run_id}.trec"
    with run_path.open("w", encoding="utf-8") as f:
        for query_id, doc_id, rank, score in run_rows:
            f.write(f"{query_id} Q0 {doc_id} {rank} {score:.12f} {run_id}\n")

    pd.DataFrame(timings).to_csv(MIRAGE_DIR / "timings" / f"{run_id}_query_times.csv", index=False)
    summary = {
        "run_id": run_id,
        "method": args.method,
        "variant": args.variant,
        "k1": args.k1,
        "b": args.b,
        "documents": len(docs),
        "queries": len(queries),
        "preprocess_seconds": preprocess_seconds,
        "build_seconds": build_seconds,
        "avg_query_seconds": sum(row["seconds"] for row in timings) / len(timings),
        "run_path": str(run_path),
    }
    with (MIRAGE_DIR / "timings" / f"{run_id}_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
