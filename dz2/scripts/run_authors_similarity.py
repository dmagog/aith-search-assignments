from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "authors_search"
TOKEN_RE = re.compile(r"[A-Za-zА-Яа-яЁё]+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=3)
    return parser.parse_args()


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


def stable_topk(candidate_ids: list[str], scores: np.ndarray, top_k: int) -> list[tuple[str, float]]:
    limit = min(top_k, len(candidate_ids))
    candidate_idx = np.argpartition(scores, -limit)[-limit:]
    rows = [(candidate_ids[i], float(scores[i])) for i in candidate_idx]
    rows.sort(key=lambda item: (-item[1], item[0]))
    return rows[:limit]


def build_tfidf(doc_tokens: list[list[str]]) -> tuple[TfidfVectorizer, object]:
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


def build_fast_bm25(doc_tokens: list[list[str]], k1: float = 1.5, b: float = 0.75):
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
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    paragraphs = pd.read_csv(ARTIFACTS_DIR / "paragraphs.csv")
    paragraphs["tokens"] = paragraphs["text"].map(tokenize)
    row_by_id = paragraphs.set_index("paragraph_id").to_dict("index")

    records: list[dict[str, object]] = []
    experiment_pairs = [
        ("tfidf", "global"),
        ("tfidf", "local"),
        ("bm25", "global"),
    ]
    setups = [
        ("within_tolstoy", "tolstoy", "tolstoy"),
        ("within_dostoevsky", "dostoevsky", "dostoevsky"),
        ("tolstoy_to_dostoevsky", "tolstoy", "dostoevsky"),
        ("dostoevsky_to_tolstoy", "dostoevsky", "tolstoy"),
    ]

    global_tokens = paragraphs["tokens"].tolist()
    global_ids = paragraphs["paragraph_id"].tolist()
    global_tfidf_vectorizer, global_tfidf_matrix = build_tfidf(global_tokens)
    global_bm25_vectorizer, global_bm25_matrix = build_fast_bm25(global_tokens)
    id_to_index = {pid: idx for idx, pid in enumerate(global_ids)}

    for setup_name, query_author, doc_author in setups:
        query_df = paragraphs[paragraphs["author"] == query_author].copy()
        doc_df = paragraphs[paragraphs["author"] == doc_author].copy()
        candidate_ids = doc_df["paragraph_id"].tolist()
        candidate_positions = [id_to_index[candidate_id] for candidate_id in candidate_ids]
        query_records = list(query_df.itertuples(index=False))
        query_texts = [" ".join(row.tokens) for row in query_records]

        local_tokens = doc_df["tokens"].tolist()
        local_ids = candidate_ids
        local_tfidf_vectorizer, local_tfidf_matrix = build_tfidf(local_tokens)
        local_bm25_vectorizer, local_bm25_matrix = build_fast_bm25(local_tokens)

        for method, idf_scheme in experiment_pairs:
            if method == "tfidf" and idf_scheme == "local":
                query_matrix = local_tfidf_vectorizer.transform(query_texts)
                score_source = local_tfidf_matrix
                use_global_positions = False
            elif method == "tfidf" and idf_scheme == "global":
                query_matrix = global_tfidf_vectorizer.transform(query_texts)
                score_source = global_tfidf_matrix
                use_global_positions = True
            elif method == "bm25" and idf_scheme == "local":
                query_matrix = local_bm25_vectorizer.transform(query_texts)
                score_source = local_bm25_matrix
                use_global_positions = False
            else:
                query_matrix = global_bm25_vectorizer.transform(query_texts)
                score_source = global_bm25_matrix
                use_global_positions = True

            batch_size = 128
            for start_idx in range(0, len(query_records), batch_size):
                end_idx = min(start_idx + batch_size, len(query_records))
                batch_records = query_records[start_idx:end_idx]
                batch_scores = (query_matrix[start_idx:end_idx] @ score_source.T).toarray()

                for row_offset, row in enumerate(batch_records):
                    all_scores = batch_scores[row_offset]
                    scores = (
                        np.array([all_scores[pos] for pos in candidate_positions])
                        if use_global_positions
                        else all_scores.copy()
                    )

                    if query_author == doc_author:
                        for i, candidate_id in enumerate(candidate_ids):
                            if candidate_id == row.paragraph_id:
                                scores[i] = -np.inf

                    top_matches = stable_topk(candidate_ids, scores, args.top_k)
                    for rank, (candidate_id, score) in enumerate(top_matches, start=1):
                        match_row = row_by_id[candidate_id]
                        records.append(
                            {
                                "setup": setup_name,
                                "method": method,
                                "idf_scheme": idf_scheme,
                                "query_id": row.paragraph_id,
                                "query_author": row.author,
                                "query_source_file": row.source_file,
                                "query_section": row.section,
                                "query_text": row.text,
                                "doc_id": candidate_id,
                                "doc_author": match_row["author"],
                                "doc_source_file": match_row["source_file"],
                                "doc_section": match_row["section"],
                                "doc_text": match_row["text"],
                                "rank": rank,
                                "score": score,
                            }
                        )

    out_csv = ARTIFACTS_DIR / "similarity_matches.csv"
    pd.DataFrame.from_records(records).to_csv(out_csv, index=False)

    metadata = {
        "top_k": args.top_k,
        "paragraphs_csv": str(ARTIFACTS_DIR / "paragraphs.csv"),
        "matches_csv": str(out_csv),
        "setups": [setup[0] for setup in setups],
        "methods": ["tfidf", "bm25"],
        "idf_schemes": ["local", "global"],
    }
    with (ARTIFACTS_DIR / "similarity_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(out_csv)


if __name__ == "__main__":
    main()
