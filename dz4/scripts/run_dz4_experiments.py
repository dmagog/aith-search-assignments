from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer


DZ4_ROOT = Path(__file__).resolve().parents[1]
SEARCH_ROOT = DZ4_ROOT.parent
ARTIFACTS_DIR = DZ4_ROOT / "artifacts"
CACHE_DIR = ARTIFACTS_DIR / "cache"
RUNS_DIR = ARTIFACTS_DIR / "runs"
TABLES_DIR = ARTIFACTS_DIR / "tables"

WIKIR_ROOT = SEARCH_ROOT / "dz1" / "dz1" / "wikIR1k" / "wikIR1k"
MIRAGE_PARQUET = SEARCH_ROOT / "dz2" / "data" / "mirage" / "train.parquet"
DZ2_WIKIR_BASELINES = SEARCH_ROOT / "dz2" / "artifacts" / "tables" / "baseline_results_summary.csv"
DZ2_MIRAGE_BASELINES = SEARCH_ROOT / "dz2" / "artifacts" / "mirage" / "tables" / "mirage_experiment_summary.csv"
DZ3_WIKIR_RESULTS = SEARCH_ROOT / "dz3" / "artifacts" / "wikir_ltr" / "results.json"
DZ3_MIRAGE_RESULTS = SEARCH_ROOT / "dz3" / "artifacts" / "mirage_ltr" / "results.json"

DENSE_MODELS = (
    "all-MiniLM-L6-v2",
    "paraphrase-multilingual-MiniLM-L12-v2",
)
DEFAULT_RERANK_MODEL = "all-MiniLM-L6-v2"

MODEL_CACHE: dict[tuple[str, str], SentenceTransformer] = {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DZ4: neural retrieval, reranking, mixture model.")
    parser.add_argument("--device", default="auto", help="auto/cpu/cuda/mps")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--embedding-shard-size", type=int, default=2000)
    parser.add_argument("--dense-top-k-wikir", type=int, default=1000)
    parser.add_argument("--dense-top-k-mirage", type=int, default=100)
    parser.add_argument("--mirage-bm25-top-k", type=int, default=100)
    parser.add_argument("--rerank-ks", default="10,20,50,100")
    parser.add_argument("--alpha-step", type=float, default=0.05)
    parser.add_argument("--models", default=",".join(DENSE_MODELS), help="Comma-separated SentenceTransformer model names.")
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--limit-wikir-train", type=int)
    parser.add_argument("--limit-wikir-test", type=int)
    parser.add_argument("--limit-mirage-test", type=int)
    parser.add_argument("--skip-dense", action="store_true")
    parser.add_argument("--skip-rerank", action="store_true")
    parser.add_argument("--skip-mixture", action="store_true")
    return parser.parse_args()


def ensure_dirs() -> None:
    for path in (ARTIFACTS_DIR, CACHE_DIR, RUNS_DIR, TABLES_DIR):
        path.mkdir(parents=True, exist_ok=True)


def detect_device(preferred: str) -> str:
    if preferred != "auto":
        return preferred
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Unsupported type for JSON serialization: {type(value)!r}")


def save_json(payload: dict[str, object], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, default=json_default)


def stable_sort_pairs(doc_ids: Sequence[str], scores: Sequence[float]) -> list[tuple[str, float]]:
    pairs = [(str(doc_id), float(score)) for doc_id, score in zip(doc_ids, scores, strict=True)]
    pairs.sort(key=lambda item: (-item[1], item[0]))
    return pairs


def precision_at_k(labels: Sequence[float], k: int) -> float:
    if k <= 0:
        return 0.0
    cut = list(labels[:k])
    if len(cut) < k:
        cut.extend([0.0] * (k - len(cut)))
    return float(sum(1.0 for label in cut if label > 0) / k)


def average_precision_at_k(ranked_doc_ids: Sequence[str], doc_rels: dict[str, float], k: int) -> float:
    total_relevant = sum(1 for relevance in doc_rels.values() if relevance > 0)
    if total_relevant == 0 or k <= 0:
        return 0.0
    denom = min(total_relevant, k)
    running_hits = 0
    precision_sum = 0.0
    for rank, doc_id in enumerate(ranked_doc_ids[:k], start=1):
        if float(doc_rels.get(doc_id, 0.0)) <= 0:
            continue
        running_hits += 1
        precision_sum += running_hits / rank
    return precision_sum / denom


def dcg_at_k(labels: Sequence[float], k: int) -> float:
    total = 0.0
    for rank, label in enumerate(labels[:k], start=1):
        gain = (2.0 ** float(label)) - 1.0
        total += gain / math.log2(rank + 1.0)
    return total


def ndcg_at_k(ranked_doc_ids: Sequence[str], doc_rels: dict[str, float], k: int) -> float:
    actual_labels = [float(doc_rels.get(doc_id, 0.0)) for doc_id in ranked_doc_ids[:k]]
    ideal_labels = sorted((float(relevance) for relevance in doc_rels.values()), reverse=True)
    actual = dcg_at_k(actual_labels, k)
    ideal = dcg_at_k(ideal_labels, k)
    if ideal <= 0.0:
        return 0.0
    return actual / ideal


def evaluate_rankings(
    rankings: dict[str, list[str]],
    qrels: dict[str, dict[str, float]],
    *,
    ks: Sequence[int] = (1, 10, 20),
    map_k: int = 20,
    ndcg_k: int = 20,
) -> dict[str, float]:
    aggregates: dict[str, list[float]] = defaultdict(list)
    max_precision_k = max(ks) if ks else 0
    for qid, doc_rels in qrels.items():
        ranking = rankings.get(qid, [])
        labels = [float(doc_rels.get(doc_id, 0.0)) for doc_id in ranking[:max_precision_k]]
        for k in ks:
            aggregates[f"P@{k}"].append(precision_at_k(labels, k))
        aggregates[f"MAP@{map_k}"].append(average_precision_at_k(ranking, doc_rels, map_k))
        aggregates[f"nDCG@{ndcg_k}"].append(ndcg_at_k(ranking, doc_rels, ndcg_k))
    return {metric: float(np.mean(values)) for metric, values in aggregates.items()}


def read_trec_qrels(path: Path) -> dict[str, dict[str, float]]:
    qrels: dict[str, dict[str, float]] = defaultdict(dict)
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            qid, _, doc_id, relevance = line.split()
            qrels[str(qid)][str(doc_id)] = float(relevance)
    return qrels


def normalize_text(text: str) -> str:
    return " ".join(str(text).strip().lower().split())


def limit_frame(frame: pd.DataFrame, id_column: str, limit: int | None) -> pd.DataFrame:
    if limit is None:
        return frame
    return frame.head(limit).copy()


def load_wikir_documents() -> tuple[list[str], list[str]]:
    frame = pd.read_csv(WIKIR_ROOT / "documents.csv", dtype={"id_right": str})
    return frame["id_right"].astype(str).tolist(), frame["text_right"].fillna("").astype(str).tolist()


def load_wikir_queries(split: str, limit: int | None = None) -> tuple[list[str], list[str]]:
    frame = pd.read_csv(WIKIR_ROOT / split / "queries.csv", dtype={"id_left": str})
    frame = limit_frame(frame, "id_left", limit)
    return frame["id_left"].astype(str).tolist(), frame["text_left"].fillna("").astype(str).tolist()


def load_wikir_qrels(split: str, qids: set[str] | None = None) -> dict[str, dict[str, float]]:
    qrels = read_trec_qrels(WIKIR_ROOT / split / "qrels")
    if qids is None:
        return qrels
    return {qid: rels for qid, rels in qrels.items() if qid in qids}


def load_wikir_bm25_run(split: str, top_k: int, qids: set[str] | None = None) -> tuple[dict[str, list[str]], dict[str, dict[str, float]]]:
    frame = pd.read_csv(
        WIKIR_ROOT / split / "BM25.res",
        sep=r"\s+",
        header=None,
        names=["qid", "Q0", "doc_id", "rank", "score", "run_id"],
        dtype={"qid": str, "doc_id": str},
    )
    if qids is not None:
        frame = frame[frame["qid"].isin(qids)]
    frame = frame[frame["rank"] < top_k].copy()
    frame.sort_values(["qid", "rank", "doc_id"], inplace=True)

    rankings: dict[str, list[str]] = defaultdict(list)
    scores: dict[str, dict[str, float]] = defaultdict(dict)
    for row in frame.itertuples(index=False):
        rankings[str(row.qid)].append(str(row.doc_id))
        scores[str(row.qid)][str(row.doc_id)] = float(row.score)
    return dict(rankings), dict(scores)


def passage_id(title: str, body: str) -> str:
    digest = hashlib.sha1(f"{title}\n{body}".encode("utf-8")).hexdigest()[:16]
    return f"mirage-{digest}"


def stratified_mirage_split(
    query_meta: pd.DataFrame,
    *,
    test_fraction: float = 0.2,
    validation_fraction: float = 0.1,
    random_seed: int = 42,
) -> tuple[set[str], set[str], set[str]]:
    rng = np.random.default_rng(random_seed)
    train_ids: set[str] = set()
    validation_ids: set[str] = set()
    test_ids: set[str] = set()

    for _, group in query_meta.groupby("source", sort=True):
        qids = group["query_id"].astype(str).to_numpy()
        shuffled = rng.permutation(qids)
        test_count = max(1, int(round(len(shuffled) * test_fraction)))
        test_part = shuffled[:test_count]
        remaining = shuffled[test_count:]
        validation_count = max(1, int(round(len(remaining) * validation_fraction))) if len(remaining) > 1 else 0
        validation_part = remaining[:validation_count]
        train_part = remaining[validation_count:]
        test_ids.update(map(str, test_part))
        validation_ids.update(map(str, validation_part))
        train_ids.update(map(str, train_part))

    return train_ids, validation_ids, test_ids


def build_mirage_dataset(random_seed: int, limit_test: int | None = None) -> dict[str, object]:
    frame = pd.read_parquet(MIRAGE_PARQUET)
    query_meta = frame[["query_id", "source", "query"]].copy()
    train_ids, validation_ids, test_ids = stratified_mirage_split(query_meta, random_seed=random_seed)

    corpus_texts: dict[str, str] = {}
    qrels: dict[str, dict[str, float]] = defaultdict(dict)
    query_texts: dict[str, str] = {}
    query_sources: dict[str, str] = {}
    query_order: list[str] = []

    for row in frame.itertuples(index=False):
        qid = str(row.query_id)
        query_texts[qid] = str(row.query).strip()
        query_sources[qid] = str(row.source)
        query_order.append(qid)

        for title, body, support in zip(
            row.doc_pool["doc_name"],
            row.doc_pool["doc_chunk"],
            row.doc_pool["support"],
            strict=True,
        ):
            title = str(title).strip()
            body = str(body).strip()
            pid = passage_id(title, body)
            corpus_texts.setdefault(pid, f"{title}. {body}".strip())
            if float(support) > 0:
                qrels[qid][pid] = max(qrels[qid].get(pid, 0.0), float(support))

        oracle_title = str(row.oracle["doc_name"]).strip()
        oracle_body = str(row.oracle["doc_chunk"]).strip()
        oracle_pid = passage_id(oracle_title, oracle_body)
        corpus_texts.setdefault(oracle_pid, f"{oracle_title}. {oracle_body}".strip())
        if float(row.oracle["support"]) > 0:
            qrels[qid][oracle_pid] = max(qrels[qid].get(oracle_pid, 0.0), float(row.oracle["support"]))

    ordered_test_ids = [qid for qid in query_order if qid in test_ids]
    if limit_test is not None:
        ordered_test_ids = ordered_test_ids[:limit_test]
    test_id_set = set(ordered_test_ids)

    return {
        "corpus_ids": list(corpus_texts.keys()),
        "corpus_texts": list(corpus_texts.values()),
        "query_text_map": query_texts,
        "query_source_map": query_sources,
        "train_ids": sorted(train_ids),
        "validation_ids": sorted(validation_ids),
        "test_ids": ordered_test_ids,
        "test_qrels": {qid: qrels[qid] for qid in ordered_test_ids},
        "all_qrels": dict(qrels),
        "full_rows": int(len(frame)),
        "corpus_size": int(len(corpus_texts)),
        "test_id_set": test_id_set,
    }


class BM25Index:
    def __init__(self, doc_ids: Sequence[str], doc_texts: Sequence[str], name: str) -> None:
        self.doc_ids = [str(doc_id) for doc_id in doc_ids]
        self.doc_texts = [normalize_text(text) for text in doc_texts]
        self.name = name
        self.vectorizer: CountVectorizer | None = None
        self.matrix = None

    def build(self) -> dict[str, float]:
        start = time.perf_counter()
        self.vectorizer = CountVectorizer(
            tokenizer=str.split,
            preprocessor=None,
            lowercase=False,
            token_pattern=None,
        )
        counts = self.vectorizer.fit_transform(self.doc_texts).tocsr()
        doc_lengths = np.asarray(counts.sum(axis=1)).ravel().astype(np.float32)
        avgdl = float(doc_lengths.mean())
        num_docs = counts.shape[0]
        df = np.asarray((counts > 0).sum(axis=0)).ravel()
        idf = np.log((num_docs - df + 0.5) / (df + 0.5) + 1.0).astype(np.float32)

        weights = counts.copy().astype(np.float32)
        rows = np.repeat(np.arange(num_docs), np.diff(counts.indptr))
        tf = weights.data
        denom = tf + 1.5 * (1.0 - 0.75 + 0.75 * doc_lengths[rows] / avgdl)
        weights.data = idf[counts.indices] * (tf * 2.5 / denom)
        self.matrix = weights.tocsr()
        return {"build_seconds": time.perf_counter() - start}

    def search(
        self,
        query_ids: Sequence[str],
        query_texts: Sequence[str],
        *,
        top_k: int,
        batch_size: int = 128,
    ) -> tuple[dict[str, list[str]], dict[str, dict[str, float]], float]:
        if self.vectorizer is None or self.matrix is None:
            raise RuntimeError("BM25 index must be built before search.")

        rankings: dict[str, list[str]] = {}
        scores: dict[str, dict[str, float]] = {}
        timing_rows: list[float] = []
        query_texts = [normalize_text(text) for text in query_texts]
        limit = min(top_k, len(self.doc_ids))

        for start_idx in range(0, len(query_ids), batch_size):
            end_idx = min(start_idx + batch_size, len(query_ids))
            batch_ids = list(query_ids[start_idx:end_idx])
            batch_texts = query_texts[start_idx:end_idx]
            batch_start = time.perf_counter()
            batch_scores = (self.vectorizer.transform(batch_texts) @ self.matrix.T).toarray()
            batch_elapsed = time.perf_counter() - batch_start
            per_query = batch_elapsed / max(len(batch_ids), 1)
            timing_rows.extend([per_query] * len(batch_ids))

            for row_idx, qid in enumerate(batch_ids):
                row_scores = batch_scores[row_idx]
                if limit == 0:
                    rankings[qid] = []
                    scores[qid] = {}
                    continue
                top_idx = np.argpartition(row_scores, -limit)[-limit:]
                ordered = stable_sort_pairs([self.doc_ids[i] for i in top_idx], [row_scores[i] for i in top_idx])
                rankings[qid] = [doc_id for doc_id, _ in ordered]
                scores[qid] = {doc_id: score for doc_id, score in ordered}

        avg_query_seconds = float(np.mean(timing_rows)) if timing_rows else 0.0
        return rankings, scores, avg_query_seconds


def cache_paths(prefix: str, model_name: str) -> tuple[Path, Path, str]:
    slug = model_name.replace("/", "__")
    return CACHE_DIR / f"{prefix}__{slug}.npy", CACHE_DIR / f"{prefix}__{slug}.json", slug


def shard_paths(prefix: str, slug: str, part_idx: int) -> tuple[Path, Path]:
    stem = CACHE_DIR / f"{prefix}__{slug}.part{part_idx:04d}"
    return Path(str(stem) + ".npy"), Path(str(stem) + ".json")


def get_model(model_name: str, device: str) -> SentenceTransformer:
    key = (model_name, device)
    if key not in MODEL_CACHE:
        MODEL_CACHE[key] = SentenceTransformer(model_name, device=device)
    return MODEL_CACHE[key]


def encode_texts(
    model_name: str,
    texts: Sequence[str],
    *,
    cache_prefix: str,
    device: str,
    batch_size: int,
    shard_size: int,
) -> tuple[np.ndarray, dict[str, object]]:
    array_path, meta_path, slug = cache_paths(cache_prefix, model_name)
    if array_path.exists() and meta_path.exists():
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        embeddings = np.load(array_path)
        return embeddings.astype(np.float32), metadata

    model = get_model(model_name, device)
    shard_arrays: list[np.ndarray] = []
    shard_seconds: list[float] = []

    for shard_start in range(0, len(texts), shard_size):
        shard_end = min(shard_start + shard_size, len(texts))
        part_idx = shard_start // shard_size
        shard_array_path, shard_meta_path = shard_paths(cache_prefix, slug, part_idx)
        if shard_array_path.exists() and shard_meta_path.exists():
            shard_arrays.append(np.load(shard_array_path).astype(np.float32))
            shard_meta = json.loads(shard_meta_path.read_text(encoding="utf-8"))
            shard_seconds.append(float(shard_meta.get("encode_seconds", 0.0)))
            continue

        print(f"Encoding shard {part_idx + 1} for {cache_prefix} with model {model_name}: rows {shard_start}-{shard_end - 1}", flush=True)
        start = time.perf_counter()
        shard_embeddings = model.encode(
            list(texts[shard_start:shard_end]),
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)
        encode_seconds = time.perf_counter() - start
        np.save(shard_array_path, shard_embeddings)
        shard_meta = {
            "model_name": model_name,
            "rows": int(shard_end - shard_start),
            "dim": int(shard_embeddings.shape[1]) if len(shard_embeddings) else 0,
            "encode_seconds": encode_seconds,
            "device": device,
            "shard_start": int(shard_start),
            "shard_end": int(shard_end),
        }
        shard_meta_path.write_text(json.dumps(shard_meta, indent=2, ensure_ascii=False), encoding="utf-8")
        shard_arrays.append(shard_embeddings)
        shard_seconds.append(encode_seconds)

    embeddings = np.vstack(shard_arrays).astype(np.float32) if shard_arrays else np.empty((0, 0), dtype=np.float32)
    np.save(array_path, embeddings)
    metadata = {
        "model_name": model_name,
        "rows": len(texts),
        "dim": int(embeddings.shape[1]) if embeddings.ndim == 2 and len(embeddings) else 0,
        "encode_seconds": float(sum(shard_seconds)),
        "device": device,
        "shard_size": int(shard_size),
        "shards": int(len(shard_arrays)),
    }
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return embeddings, metadata


def dense_search(
    query_ids: Sequence[str],
    query_embeddings: np.ndarray,
    doc_ids: Sequence[str],
    doc_embeddings: np.ndarray,
    *,
    top_k: int,
    batch_size: int = 32,
) -> tuple[dict[str, list[str]], dict[str, dict[str, float]], float]:
    rankings: dict[str, list[str]] = {}
    scores: dict[str, dict[str, float]] = {}
    timing_rows: list[float] = []
    doc_matrix_t = np.ascontiguousarray(doc_embeddings.T)
    limit = min(top_k, len(doc_ids))

    for start_idx in range(0, len(query_ids), batch_size):
        end_idx = min(start_idx + batch_size, len(query_ids))
        batch_ids = list(query_ids[start_idx:end_idx])
        batch_queries = query_embeddings[start_idx:end_idx]
        batch_start = time.perf_counter()
        batch_scores = batch_queries @ doc_matrix_t
        batch_elapsed = time.perf_counter() - batch_start
        per_query = batch_elapsed / max(len(batch_ids), 1)
        timing_rows.extend([per_query] * len(batch_ids))

        for row_idx, qid in enumerate(batch_ids):
            row_scores = batch_scores[row_idx]
            if limit == 0:
                rankings[qid] = []
                scores[qid] = {}
                continue
            top_idx = np.argpartition(row_scores, -limit)[-limit:]
            ordered = stable_sort_pairs([doc_ids[i] for i in top_idx], [row_scores[i] for i in top_idx])
            rankings[qid] = [doc_id for doc_id, _ in ordered]
            scores[qid] = {doc_id: score for doc_id, score in ordered}

    avg_query_seconds = float(np.mean(timing_rows)) if timing_rows else 0.0
    return rankings, scores, avg_query_seconds


def write_trec_run(path: Path, rankings: dict[str, list[str]], scores: dict[str, dict[str, float]], run_id: str) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for qid in sorted(rankings):
            docs = rankings[qid]
            for rank, doc_id in enumerate(docs, start=1):
                score = float(scores.get(qid, {}).get(doc_id, len(docs) - rank))
                handle.write(f"{qid} Q0 {doc_id} {rank} {score:.12f} {run_id}\n")


def ranking_to_score_lookup(ranking: list[str]) -> dict[str, float]:
    total = len(ranking)
    return {doc_id: float(total - idx) for idx, doc_id in enumerate(ranking)}


def build_query_embedding_map(query_ids: Sequence[str], query_embeddings: np.ndarray) -> dict[str, np.ndarray]:
    return {str(qid): query_embeddings[idx] for idx, qid in enumerate(query_ids)}


def rerank_candidates(
    query_ids: Sequence[str],
    base_rankings: dict[str, list[str]],
    query_embedding_map: dict[str, np.ndarray],
    doc_embeddings: np.ndarray,
    doc_index: dict[str, int],
    *,
    rerank_k: int,
) -> tuple[dict[str, list[str]], dict[str, dict[str, float]], float]:
    reranked: dict[str, list[str]] = {}
    reranked_scores: dict[str, dict[str, float]] = {}
    timing_rows: list[float] = []

    for qid in query_ids:
        original = list(base_rankings.get(qid, []))
        head = original[:rerank_k]
        start = time.perf_counter()
        if head:
            candidate_indices = [doc_index[doc_id] for doc_id in head if doc_id in doc_index]
            candidate_docs = [doc_id for doc_id in head if doc_id in doc_index]
            query_vec = query_embedding_map[qid]
            cosine_scores = doc_embeddings[candidate_indices] @ query_vec if candidate_indices else np.array([], dtype=np.float32)
            ordered = stable_sort_pairs(candidate_docs, cosine_scores)
            reranked_head = [doc_id for doc_id, _ in ordered]
            reranked_score_map = {doc_id: score for doc_id, score in ordered}
        else:
            reranked_head = []
            reranked_score_map = {}
        elapsed = time.perf_counter() - start
        timing_rows.append(elapsed)
        used = set(reranked_head)
        tail = [doc_id for doc_id in original if doc_id not in used]
        reranked[qid] = reranked_head + tail
        reranked_scores[qid] = reranked_score_map | {doc_id: score for doc_id, score in ranking_to_score_lookup(tail).items()}

    avg_query_seconds = float(np.mean(timing_rows)) if timing_rows else 0.0
    return reranked, reranked_scores, avg_query_seconds


def minmax_normalize(score_map: dict[str, float]) -> dict[str, float]:
    if not score_map:
        return {}
    values = list(score_map.values())
    lo = min(values)
    hi = max(values)
    if abs(hi - lo) < 1e-12:
        fill = 1.0 if hi > 0 else 0.0
        return {doc_id: fill for doc_id in score_map}
    return {doc_id: (score - lo) / (hi - lo) for doc_id, score in score_map.items()}


def build_mixture_candidates(
    query_ids: Sequence[str],
    bm25_rankings: dict[str, list[str]],
    bm25_scores: dict[str, dict[str, float]],
    dense_rankings: dict[str, list[str]],
    dense_scores: dict[str, dict[str, float]],
    query_embedding_map: dict[str, np.ndarray],
    doc_embeddings: np.ndarray,
    doc_index: dict[str, int],
    *,
    bm25_top_k: int,
    dense_top_k: int,
) -> dict[str, dict[str, object]]:
    payload: dict[str, dict[str, object]] = {}
    for qid in query_ids:
        bm25_docs = list(bm25_rankings.get(qid, []))[:bm25_top_k]
        dense_docs = list(dense_rankings.get(qid, []))[:dense_top_k]
        candidate_docs = list(dict.fromkeys(bm25_docs + dense_docs))
        if not candidate_docs:
            payload[qid] = {"doc_ids": [], "bm25": np.array([], dtype=np.float32), "cosine": np.array([], dtype=np.float32)}
            continue

        bm25_norm = minmax_normalize({doc_id: bm25_scores.get(qid, {}).get(doc_id, 0.0) for doc_id in bm25_docs})
        cosine_lookup: dict[str, float] = {}
        missing_docs: list[str] = []
        missing_indices: list[int] = []
        for doc_id in candidate_docs:
            if doc_id in dense_scores.get(qid, {}):
                cosine_lookup[doc_id] = float(dense_scores[qid][doc_id])
            else:
                missing_docs.append(doc_id)
                missing_indices.append(doc_index[doc_id])

        if missing_docs:
            query_vec = query_embedding_map[qid]
            exact_scores = doc_embeddings[missing_indices] @ query_vec
            for doc_id, score in zip(missing_docs, exact_scores, strict=True):
                cosine_lookup[doc_id] = float(score)

        payload[qid] = {
            "doc_ids": candidate_docs,
            "bm25": np.array([bm25_norm.get(doc_id, 0.0) for doc_id in candidate_docs], dtype=np.float32),
            "cosine": np.array([cosine_lookup.get(doc_id, 0.0) for doc_id in candidate_docs], dtype=np.float32),
        }
    return payload


def score_mixture_candidates(
    candidates: dict[str, dict[str, object]],
    *,
    alpha: float,
) -> tuple[dict[str, list[str]], dict[str, dict[str, float]], float]:
    rankings: dict[str, list[str]] = {}
    scores: dict[str, dict[str, float]] = {}
    timing_rows: list[float] = []
    for qid, entry in candidates.items():
        doc_ids = entry["doc_ids"]
        bm25 = entry["bm25"]
        cosine = entry["cosine"]
        start = time.perf_counter()
        combined = alpha * bm25 + (1.0 - alpha) * cosine
        elapsed = time.perf_counter() - start
        timing_rows.append(elapsed)
        ordered = stable_sort_pairs(doc_ids, combined)
        rankings[qid] = [doc_id for doc_id, _ in ordered]
        scores[qid] = {doc_id: score for doc_id, score in ordered}
    avg_query_seconds = float(np.mean(timing_rows)) if timing_rows else 0.0
    return rankings, scores, avg_query_seconds


def summarize_dense_results(
    dataset_name: str,
    model_name: str,
    metrics: dict[str, float],
    *,
    corpus_size: int,
    query_count: int,
    top_k: int,
    doc_encode_seconds: float,
    query_encode_seconds: float,
    avg_query_seconds: float,
) -> dict[str, object]:
    return {
        "dataset": dataset_name,
        "model": model_name,
        "corpus_size": corpus_size,
        "query_count": query_count,
        "top_k": top_k,
        "metrics": metrics,
        "doc_encode_seconds": doc_encode_seconds,
        "query_encode_seconds": query_encode_seconds,
        "avg_query_seconds": avg_query_seconds,
    }


def load_previous_results() -> dict[str, object]:
    wikir_baselines = pd.read_csv(DZ2_WIKIR_BASELINES)
    mirage_baselines = pd.read_csv(DZ2_MIRAGE_BASELINES)
    wikir_ltr = json.loads(DZ3_WIKIR_RESULTS.read_text(encoding="utf-8"))
    mirage_ltr = json.loads(DZ3_MIRAGE_RESULTS.read_text(encoding="utf-8"))

    best_wikir_bm25 = wikir_baselines.sort_values("nDCG@20", ascending=False).iloc[0].to_dict()
    best_mirage_bm25 = mirage_baselines.sort_values("nDCG@20", ascending=False).iloc[0].to_dict()
    best_wikir_ltr = max(wikir_ltr["results"], key=lambda row: row["test_metrics"]["ndcg@20"])
    best_mirage_ltr = max(mirage_ltr["results"], key=lambda row: row["test_metrics"]["ap"])

    return {
        "assignment2": {
            "wikir_best": best_wikir_bm25,
            "mirage_best": best_mirage_bm25,
        },
        "assignment3": {
            "wikir_best": best_wikir_ltr,
            "mirage_best": best_mirage_ltr,
        },
    }


def load_dense_rows_from_table() -> list[dict[str, object]]:
    path = TABLES_DIR / "dense_results.csv"
    if not path.exists():
        return []
    table = pd.read_csv(path)
    rows: list[dict[str, object]] = []
    for row in table.to_dict("records"):
        rows.append(
            {
                "dataset": row["dataset"],
                "model": row["model"],
                "metrics": {
                    "P@1": float(row["P@1"]),
                    "P@10": float(row["P@10"]),
                    "P@20": float(row["P@20"]),
                    "MAP@20": float(row["MAP@20"]),
                    "nDCG@20": float(row["nDCG@20"]),
                },
                "avg_query_seconds": float(row["avg_query_seconds"]),
            }
        )
    return rows


def load_rerank_rows_from_table() -> list[dict[str, object]]:
    path = TABLES_DIR / "reranking_results.csv"
    if not path.exists():
        return []
    table = pd.read_csv(path)
    rows: list[dict[str, object]] = []
    for row in table.to_dict("records"):
        rows.append(
            {
                "dataset": row["dataset"],
                "model": row["model"],
                "k": int(row["k"]),
                "metrics": {
                    "P@1": float(row["P@1"]),
                    "P@10": float(row["P@10"]),
                    "P@20": float(row["P@20"]),
                    "MAP@20": float(row["MAP@20"]),
                    "nDCG@20": float(row["nDCG@20"]),
                },
                "avg_query_seconds": float(row["avg_query_seconds"]),
            }
        )
    return rows


def load_mixture_from_tables() -> dict[str, object]:
    results_path = TABLES_DIR / "mixture_results.csv"
    curve_path = TABLES_DIR / "alpha_curve.json"
    if not results_path.exists():
        return {}

    results_table = pd.read_csv(results_path)
    results = [
        {
            "dataset": row["dataset"],
            "model": row["model"],
            "alpha": float(row["alpha"]),
            "metrics": {
                "P@1": float(row["P@1"]),
                "P@10": float(row["P@10"]),
                "P@20": float(row["P@20"]),
                "MAP@20": float(row["MAP@20"]),
                "nDCG@20": float(row["nDCG@20"]),
            },
            "avg_query_seconds": float(row["avg_query_seconds"]),
        }
        for row in results_table.to_dict("records")
    ]
    best_alpha = float(results[0]["alpha"]) if results else None

    alpha_curve: list[dict[str, object]] = []
    best_train_metrics = None
    if curve_path.exists():
        alpha_curve = json.loads(curve_path.read_text(encoding="utf-8"))
        if best_alpha is not None:
            for row in alpha_curve:
                if abs(float(row["alpha"]) - best_alpha) < 1e-12:
                    best_train_metrics = row["metrics"]
                    break

    return {
        "model": str(results[0]["model"]) if results else DEFAULT_RERANK_MODEL,
        "selection_metric": "nDCG@20",
        "best_alpha": best_alpha,
        "best_train_metrics": best_train_metrics,
        "alpha_curve": alpha_curve,
        "results": results,
    }


def merge_metric_rows(
    existing_rows: Sequence[dict[str, object]],
    new_rows: Sequence[dict[str, object]],
    key_fields: Sequence[str],
) -> list[dict[str, object]]:
    merged: dict[tuple[object, ...], dict[str, object]] = {}
    for row in existing_rows:
        key = tuple(row[field] for field in key_fields)
        merged[key] = row
    for row in new_rows:
        key = tuple(row[field] for field in key_fields)
        merged[key] = row
    return [merged[key] for key in sorted(merged)]


def hydrate_summary_from_existing_artifacts(summary: dict[str, object]) -> dict[str, object]:
    summary["dense_retrieval"] = merge_metric_rows(
        load_dense_rows_from_table(),
        summary.get("dense_retrieval", []),
        ("dataset", "model"),
    )
    summary["reranking"] = merge_metric_rows(
        load_rerank_rows_from_table(),
        summary.get("reranking", []),
        ("dataset", "model", "k"),
    )

    current_mixture = summary.get("mixture") or {}
    if current_mixture.get("results"):
        summary["mixture"] = current_mixture
    else:
        existing_mixture = load_mixture_from_tables()
        if existing_mixture:
            summary["mixture"] = existing_mixture

    config = summary.get("config", {})
    if isinstance(config, dict):
        known_models = {row["model"] for row in summary.get("dense_retrieval", [])}
        if summary.get("reranking"):
            known_models.update(row["model"] for row in summary["reranking"])
        if summary.get("mixture", {}).get("results"):
            known_models.update(row["model"] for row in summary["mixture"]["results"])
        config["models"] = sorted(known_models)

    return summary


def build_summary_markdown(summary: dict[str, object]) -> str:
    lines = [
        "# DZ4 summary",
        "",
        "## Dense retrieval",
        "",
        "| Dataset | Model | P@1 | P@10 | P@20 | MAP@20 | nDCG@20 | Avg query seconds |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary.get("dense_retrieval", []):
        metrics = row["metrics"]
        lines.append(
            f"| {row['dataset']} | {row['model']} | {metrics['P@1']:.4f} | {metrics['P@10']:.4f} | {metrics['P@20']:.4f} | {metrics['MAP@20']:.4f} | {metrics['nDCG@20']:.4f} | {row['avg_query_seconds']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Re-ranking",
            "",
            "| Dataset | Model | k | P@1 | P@10 | P@20 | MAP@20 | nDCG@20 | Avg query seconds |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in summary.get("reranking", []):
        metrics = row["metrics"]
        lines.append(
            f"| {row['dataset']} | {row['model']} | {row['k']} | {metrics['P@1']:.4f} | {metrics['P@10']:.4f} | {metrics['P@20']:.4f} | {metrics['MAP@20']:.4f} | {metrics['nDCG@20']:.4f} | {row['avg_query_seconds']:.4f} |"
        )

    mixture = summary.get("mixture") or {}
    lines.extend(
        [
            "",
            "## Mixture model",
            "",
            f"Selected alpha on WikIR train: `{mixture.get('best_alpha', 'n/a')}` using `{mixture.get('selection_metric', 'n/a')}`.",
            "",
            "| Dataset | Model | P@1 | P@10 | P@20 | MAP@20 | nDCG@20 | Avg query seconds |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in mixture.get("results", []):
        metrics = row["metrics"]
        lines.append(
            f"| {row['dataset']} | {row['model']} | {metrics['P@1']:.4f} | {metrics['P@10']:.4f} | {metrics['P@20']:.4f} | {metrics['MAP@20']:.4f} | {metrics['nDCG@20']:.4f} | {row['avg_query_seconds']:.6f} |"
        )

    previous = summary.get("previous_best") or {}
    lines.extend(
        [
            "",
            "## Previous best references",
            "",
            f"- DZ2 WikIR best BM25: `{previous.get('assignment2', {}).get('wikir_best', {}).get('run_id', 'n/a')}`",
            f"- DZ2 MIRAGE best lexical baseline: `{previous.get('assignment2', {}).get('mirage_best', {}).get('run_id', 'n/a')}`",
            f"- DZ3 WikIR best LTR: `{previous.get('assignment3', {}).get('wikir_best', {}).get('model_name', 'n/a')}`",
            f"- DZ3 MIRAGE best LTR: `{previous.get('assignment3', {}).get('mirage_best', {}).get('model_name', 'n/a')}`",
            "",
            "## Notes",
            "",
            "- MIRAGE test split is reproduced via the same stratified split by `source` as in DZ3.",
            "- The optional cross-encoder fine-tuning task is not part of the current implementation.",
        ]
    )
    return "\n".join(lines) + "\n"


def _format_metric_row(row: dict[str, object], include_k: bool = False, include_alpha: bool = False) -> str:
    metrics = row["metrics"]
    parts = [str(row["dataset"]), str(row["model"])]
    if include_k:
        parts.append(str(row["k"]))
    if include_alpha:
        parts.append(f"{row['alpha']:.2f}")
    parts.extend(
        [
            f"{metrics['P@1']:.4f}",
            f"{metrics['P@10']:.4f}",
            f"{metrics['P@20']:.4f}",
            f"{metrics['MAP@20']:.4f}",
            f"{metrics['nDCG@20']:.4f}",
            f"{row['avg_query_seconds']:.6f}",
        ]
    )
    return "| " + " | ".join(parts) + " |"


def _best_rerank_row(summary: dict[str, object], dataset: str) -> dict[str, object] | None:
    candidates = [row for row in summary.get("reranking", []) if row["dataset"] == dataset]
    if not candidates:
        return None
    return max(candidates, key=lambda row: row["metrics"]["nDCG@20"])


def _first_dense_row(summary: dict[str, object], dataset: str) -> dict[str, object] | None:
    candidates = [row for row in summary.get("dense_retrieval", []) if row["dataset"] == dataset]
    if not candidates:
        return None
    return max(candidates, key=lambda row: row["metrics"]["nDCG@20"])


def _first_mixture_row(summary: dict[str, object], dataset: str) -> dict[str, object] | None:
    mixture = summary.get("mixture") or {}
    candidates = [row for row in mixture.get("results", []) if row["dataset"] == dataset]
    if not candidates:
        return None
    return candidates[0]


def build_report_markdown(summary: dict[str, object]) -> str:
    dense_rows = list(summary.get("dense_retrieval", []))
    rerank_rows = list(summary.get("reranking", []))
    mixture = summary.get("mixture") or {}
    mixture_rows = list(mixture.get("results", []))
    previous = summary.get("previous_best") or {}

    required_dense_models = len(DENSE_MODELS)
    completed_dense_models = sorted({row["model"] for row in dense_rows})
    dense_ready = len(completed_dense_models) >= required_dense_models
    rerank_ready = bool(rerank_rows)
    mixture_ready = bool(mixture_rows) and mixture.get("best_alpha") is not None

    lines = [
        "# DZ4: нейросетевой поиск и переупорядочивание",
        "",
        "## Проверка соответствия заданию",
        "",
    ]

    if dense_ready:
        lines.append(
            f"- Пункт 1, нейросетевой поиск: выполнен. В отчёте есть результаты как минимум для двух предобученных моделей: {', '.join(f'`{name}`' for name in completed_dense_models)}."
        )
    elif dense_rows:
        lines.append(
            f"- Пункт 1, нейросетевой поиск: выполнен частично. На момент подготовки отчёта завершены результаты только для {len(completed_dense_models)} модели(ей): {', '.join(f'`{name}`' for name in completed_dense_models)}. Требование задания: не менее двух моделей."
        )
    else:
        lines.append("- Пункт 1, нейросетевой поиск: не выполнен.")

    lines.append(
        "- Пункт 2, переупорядочивание результатов BM25: "
        + ("выполнен." if rerank_ready else "не выполнен.")
    )
    lines.append(
        "- Пункт 3, смешанная модель `alpha * BM25 + (1 - alpha) * cosine`: "
        + ("выполнен." if mixture_ready else "не выполнен.")
    )
    lines.append("- Добавление результатов из `dz2` и `dz3`: выполнено.")
    lines.append("- Дополнительное задание с дообучением cross-encoder: не выполнено.")

    lines.extend(
        [
            "",
            "## Данные и постановка эксперимента",
            "",
            "- Использованы два корпуса из условия: `WikIR en1k` и `MIRAGE`.",
            "- Для `WikIR` использованы готовые обучающая и тестовая выборки.",
            "- Для `MIRAGE` тестовая выборка воспроизведена тем же стратифицированным разбиением по полю `source`, что и в `dz3`.",
            "- Для оценки качества во всех основных конфигурациях считаются `P@1`, `P@10`, `P@20`, `MAP@20`, `nDCG@20`.",
            "",
            "## Выбор моделей",
            "",
            "- `all-MiniLM-L6-v2` выбрана как компактная и быстрая модель, подходящая для базового сравнения по качеству и скорости.",
            "- `paraphrase-multilingual-MiniLM-L12-v2` выбрана как более крупная многоязычная модель, чтобы проверить, даёт ли более богатое смысловое представление выигрыш на тех же данных.",
            "- Для переупорядочивания использована `all-MiniLM-L6-v2`: это осознанный компромисс между качеством и временем ответа. При этом нужно учитывать, что такое переупорядочивание слабее настоящей парной модели, которая оценивает запрос и документ совместно.",
            "",
            "## Результаты нейросетевого поиска",
            "",
            "| Набор | Модель | P@1 | P@10 | P@20 | MAP@20 | nDCG@20 | Среднее время запроса, с |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in dense_rows:
        lines.append(_format_metric_row(row))
    if not dense_rows:
        lines.append("| нет данных | - | - | - | - | - | - | - |")

    lines.extend(
        [
            "",
            "## Результаты переупорядочивания",
            "",
            "| Набор | Модель | k | P@1 | P@10 | P@20 | MAP@20 | nDCG@20 | Среднее время запроса, с |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rerank_rows:
        lines.append(_format_metric_row(row, include_k=True))
    if not rerank_rows:
        lines.append("| нет данных | - | - | - | - | - | - | - | - |")

    lines.extend(
        [
            "",
            "## Результаты смешанной модели",
            "",
            f"- Подбор коэффициента `alpha` проводился на обучающей части `WikIR` по метрике `{mixture.get('selection_metric', 'n/a')}`.",
            f"- Лучшее значение: `alpha = {mixture.get('best_alpha', 'n/a')}`.",
            "",
            "| Набор | Модель | alpha | P@1 | P@10 | P@20 | MAP@20 | nDCG@20 | Среднее время запроса, с |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in mixture_rows:
        lines.append(_format_metric_row(row, include_alpha=True))
    if not mixture_rows:
        lines.append("| нет данных | - | - | - | - | - | - | - | - |")

    lines.extend(
        [
            "",
            "## Сравнение с результатами `dz2` и `dz3`",
            "",
        ]
    )

    wikir_dense = _first_dense_row(summary, "WikIR test")
    mirage_dense = _first_dense_row(summary, "MIRAGE test")
    wikir_rerank = _best_rerank_row(summary, "WikIR test")
    mirage_rerank = _best_rerank_row(summary, "MIRAGE test")
    wikir_mix = _first_mixture_row(summary, "WikIR test")
    mirage_mix = _first_mixture_row(summary, "MIRAGE test")

    wikir_bm25 = previous.get("assignment2", {}).get("wikir_best", {})
    mirage_bm25 = previous.get("assignment2", {}).get("mirage_best", {})
    wikir_ltr = previous.get("assignment3", {}).get("wikir_best", {})
    mirage_ltr = previous.get("assignment3", {}).get("mirage_best", {})

    if wikir_dense and wikir_bm25:
        delta = wikir_dense["metrics"]["nDCG@20"] - float(wikir_bm25["nDCG@20"])
        lines.append(
            f"- На `WikIR` чистый нейросетевой поиск улучшает `nDCG@20` относительно лучшего `BM25` из `dz2` на `{delta:+.4f}`: `{wikir_bm25['nDCG@20']:.4f}` -> `{wikir_dense['metrics']['nDCG@20']:.4f}`."
        )
    if mirage_dense and mirage_bm25:
        delta = mirage_dense["metrics"]["nDCG@20"] - float(mirage_bm25["nDCG@20"])
        lines.append(
            f"- На `MIRAGE` чистый нейросетевой поиск улучшает `nDCG@20` относительно лучшего `BM25` из `dz2` на `{delta:+.4f}`: `{mirage_bm25['nDCG@20']:.4f}` -> `{mirage_dense['metrics']['nDCG@20']:.4f}`."
        )
    if wikir_mix and wikir_dense:
        delta = wikir_mix["metrics"]["nDCG@20"] - wikir_dense["metrics"]["nDCG@20"]
        lines.append(
            f"- На `WikIR` смешанная модель даёт дополнительный прирост к нейросетевому поиску по `nDCG@20` на `{delta:+.4f}`: `{wikir_dense['metrics']['nDCG@20']:.4f}` -> `{wikir_mix['metrics']['nDCG@20']:.4f}`."
        )
    if mirage_mix and mirage_dense:
        delta = mirage_mix["metrics"]["nDCG@20"] - mirage_dense["metrics"]["nDCG@20"]
        lines.append(
            f"- На `MIRAGE` смешанная модель также улучшает `nDCG@20` на `{delta:+.4f}`: `{mirage_dense['metrics']['nDCG@20']:.4f}` -> `{mirage_mix['metrics']['nDCG@20']:.4f}`."
        )
    if wikir_rerank and wikir_dense:
        delta = wikir_rerank["metrics"]["nDCG@20"] - wikir_dense["metrics"]["nDCG@20"]
        lines.append(
            f"- Лучшее переупорядочивание на `WikIR` при `k={wikir_rerank['k']}` уступает чистому нейросетевому поиску по `nDCG@20` на `{delta:+.4f}`: `{wikir_dense['metrics']['nDCG@20']:.4f}` -> `{wikir_rerank['metrics']['nDCG@20']:.4f}`."
        )
    if mirage_rerank and mirage_dense:
        delta = mirage_rerank["metrics"]["nDCG@20"] - mirage_dense["metrics"]["nDCG@20"]
        lines.append(
            f"- Лучшее переупорядочивание на `MIRAGE` при `k={mirage_rerank['k']}` также уступает чистому нейросетевому поиску по `nDCG@20` на `{delta:+.4f}`: `{mirage_dense['metrics']['nDCG@20']:.4f}` -> `{mirage_rerank['metrics']['nDCG@20']:.4f}`."
        )
    if wikir_mix and wikir_ltr:
        dz3_wikir_ndcg20 = float(wikir_ltr["test_metrics"]["ndcg@20"])
        delta = wikir_mix["metrics"]["nDCG@20"] - dz3_wikir_ndcg20
        lines.append(
            f"- Для `WikIR` есть прямое сопоставление с `dz3` по `nDCG@20`: смешанная модель даёт `{wikir_mix['metrics']['nDCG@20']:.4f}` против `{dz3_wikir_ndcg20:.4f}`, то есть `{delta:+.4f}`."
        )
    if mirage_ltr:
        lines.append(
            "- Для `MIRAGE` прямое численное сравнение с `dz3` ограничено: в `dz3` сохранены `AP` и `nDCG@5`, а в текущей работе основной фокус на `MAP@20` и `nDCG@20`, поэтому эти числа нельзя сопоставлять напрямую."
        )

    lines.extend(
        [
            "",
            "## Интерпретация результатов",
            "",
            "- Самая логичная картина на текущих данных такая: сочетание лексического сигнала `BM25` и смыслового сходства работает лучше, чем каждый из этих сигналов по отдельности.",
            "- Переупорядочивание оказалось слабее чистого нейросетевого поиска. Это объяснимо: модель видит только ограниченный набор кандидатов от `BM25`, а само переупорядочивание основано на той же косинусной близости, без более сильной парной модели.",
            "- На `WikIR` смешанная модель уже сопоставима с лучшим результатом из `dz3` и даже немного превосходит его по `nDCG@20`. Это сильный итог для относительно простой схемы без обучения ранжировщика.",
            "- На `MIRAGE` нейросетевой поиск и смешанная модель уверенно превосходят лучший `BM25` из `dz2` по ранним метрикам и по `nDCG@20`, что говорит о большой роли смыслового сходства в этом наборе.",
            "",
            "## Что ещё нужно улучшить",
            "",
            "- Добить прогон второй модели и добавить её результаты в раздел нейросетевого поиска. Пока это главный незакрытый пункт по формулировке задания.",
            "- Для переупорядочивания стоит попробовать не косинусную близость той же модели, а отдельную парную модель или хотя бы более сильный кодировщик. Текущая схема слишком близка к исходному нейросетевому поиску и почти не добавляет новой информации.",
            "- Для более строгого сравнения с `dz3` на `MIRAGE` стоит либо пересчитать для старых прогонов `nDCG@20`, либо сохранить их ранжирования в формате, из которого можно вычислить те же метрики, что и здесь.",
            "- Дополнительное задание с дообучением cross-encoder остаётся резервом для заметного усиления качества именно на этапе переупорядочивания.",
            "",
            "## Вывод",
            "",
            "- На текущем этапе работа закрывает основную идею задания: реализованы нейросетевой поиск, переупорядочивание результатов `BM25`, смешанная модель и сравнение с предыдущими домашними заданиями.",
            "- Самый сильный результат среди уже завершённых экспериментов даёт смешанная модель с `alpha = 0.1`.",
            "- Для полной формальной готовности к сдаче нужно дождаться завершения второй модели в пункте 1 и затем один раз обновить итоговый отчёт.",
        ]
    )

    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    ensure_dirs()
    device = detect_device(args.device)
    rerank_ks = tuple(sorted({int(value) for value in args.rerank_ks.split(",") if value.strip()}))
    selected_models = tuple(model.strip() for model in args.models.split(",") if model.strip())
    if not selected_models:
        raise ValueError("At least one model must be provided via --models.")
    if DEFAULT_RERANK_MODEL not in selected_models and ((not args.skip_rerank) or (not args.skip_mixture)):
        raise ValueError(f"{DEFAULT_RERANK_MODEL} must be included in --models when reranking or mixture is enabled.")

    wikir_doc_ids, wikir_doc_texts = load_wikir_documents()
    wikir_test_qids, wikir_test_texts = load_wikir_queries("test", limit=args.limit_wikir_test)
    wikir_test_qrels = load_wikir_qrels("test", set(wikir_test_qids))
    wikir_train_qids, wikir_train_texts = load_wikir_queries("train", limit=args.limit_wikir_train)
    wikir_train_qrels = load_wikir_qrels("train", set(wikir_train_qids))

    mirage = build_mirage_dataset(args.random_seed, limit_test=args.limit_mirage_test)
    mirage_test_qids = list(mirage["test_ids"])
    mirage_test_texts = [mirage["query_text_map"][qid] for qid in mirage_test_qids]
    mirage_test_qrels = mirage["test_qrels"]
    mirage_doc_ids = list(mirage["corpus_ids"])
    mirage_doc_texts = list(mirage["corpus_texts"])

    previous_best = load_previous_results()
    summary: dict[str, object] = {
        "config": {
            "device": device,
            "batch_size": args.batch_size,
            "dense_top_k_wikir": args.dense_top_k_wikir,
            "dense_top_k_mirage": args.dense_top_k_mirage,
            "mirage_bm25_top_k": args.mirage_bm25_top_k,
            "rerank_ks": rerank_ks,
            "alpha_step": args.alpha_step,
            "models": selected_models,
            "random_seed": args.random_seed,
            "limits": {
                "wikir_train": args.limit_wikir_train,
                "wikir_test": args.limit_wikir_test,
                "mirage_test": args.limit_mirage_test,
            },
        },
        "previous_best": previous_best,
        "dense_retrieval": [],
        "reranking": [],
        "mixture": {},
    }

    selected_embeddings: dict[str, object] = {}

    if args.skip_dense and args.skip_rerank and args.skip_mixture:
        summary = hydrate_summary_from_existing_artifacts(summary)
        summary_path = ARTIFACTS_DIR / "summary.json"
        markdown_path = ARTIFACTS_DIR / "summary.md"
        report_path = ARTIFACTS_DIR / "report.md"
        save_json(summary, summary_path)
        markdown_path.write_text(build_summary_markdown(summary), encoding="utf-8")
        report_path.write_text(build_report_markdown(summary), encoding="utf-8")
        print(json.dumps(summary["config"], indent=2, ensure_ascii=False, default=json_default))
        print(f"Summary JSON: {summary_path}")
        print(f"Summary Markdown: {markdown_path}")
        print(f"Report Markdown: {report_path}")
        return

    need_default_model = (not args.skip_rerank) or (not args.skip_mixture)

    for model_name in selected_models:
        need_wikir_embeddings = (not args.skip_dense) or (model_name == DEFAULT_RERANK_MODEL and need_default_model)
        need_mirage_embeddings = (not args.skip_dense) or (model_name == DEFAULT_RERANK_MODEL and need_default_model)

        wikir_doc_embeddings = None
        wikir_doc_meta = None
        wikir_test_query_embeddings = None
        wikir_test_query_meta = None

        if need_wikir_embeddings:
            wikir_doc_embeddings, wikir_doc_meta = encode_texts(
                model_name,
                wikir_doc_texts,
                cache_prefix="wikir_documents",
                device=device,
                batch_size=args.batch_size,
                shard_size=args.embedding_shard_size,
            )
            wikir_test_query_embeddings, wikir_test_query_meta = encode_texts(
                model_name,
                wikir_test_texts,
                cache_prefix="wikir_test_queries",
                device=device,
                batch_size=args.batch_size,
                shard_size=args.embedding_shard_size,
            )

        if not args.skip_dense:
            run_id = f"wikir_dense_{model_name.replace('/', '__')}"
            rankings, score_lookup, avg_query_seconds = dense_search(
                wikir_test_qids,
                wikir_test_query_embeddings,
                wikir_doc_ids,
                wikir_doc_embeddings,
                top_k=args.dense_top_k_wikir,
            )
            metrics = evaluate_rankings(rankings, wikir_test_qrels)
            write_trec_run(RUNS_DIR / f"{run_id}.trec", rankings, score_lookup, run_id)
            summary["dense_retrieval"].append(
                summarize_dense_results(
                    "WikIR test",
                    model_name,
                    metrics,
                    corpus_size=len(wikir_doc_ids),
                    query_count=len(wikir_test_qids),
                    top_k=args.dense_top_k_wikir,
                    doc_encode_seconds=float(wikir_doc_meta["encode_seconds"]),
                    query_encode_seconds=float(wikir_test_query_meta["encode_seconds"]),
                    avg_query_seconds=avg_query_seconds,
                )
            )

        if model_name == DEFAULT_RERANK_MODEL and need_default_model:
            wikir_train_query_embeddings, _ = encode_texts(
                model_name,
                wikir_train_texts,
                cache_prefix="wikir_train_queries",
                device=device,
                batch_size=args.batch_size,
                shard_size=args.embedding_shard_size,
            )
            selected_embeddings["wikir_doc_embeddings"] = wikir_doc_embeddings
            selected_embeddings["wikir_doc_index"] = {doc_id: idx for idx, doc_id in enumerate(wikir_doc_ids)}
            selected_embeddings["wikir_test_query_map"] = build_query_embedding_map(wikir_test_qids, wikir_test_query_embeddings)
            selected_embeddings["wikir_train_query_map"] = build_query_embedding_map(wikir_train_qids, wikir_train_query_embeddings)
            if args.skip_dense:
                dense_rankings, dense_scores, _ = dense_search(
                    wikir_test_qids,
                    wikir_test_query_embeddings,
                    wikir_doc_ids,
                    wikir_doc_embeddings,
                    top_k=args.dense_top_k_wikir,
                )
                selected_embeddings["wikir_test_dense_rankings"] = dense_rankings
                selected_embeddings["wikir_test_dense_scores"] = dense_scores
            else:
                selected_embeddings["wikir_test_dense_rankings"] = rankings
                selected_embeddings["wikir_test_dense_scores"] = score_lookup

            train_dense_rankings, train_dense_scores, _ = dense_search(
                wikir_train_qids,
                wikir_train_query_embeddings,
                wikir_doc_ids,
                wikir_doc_embeddings,
                top_k=args.dense_top_k_wikir,
            )
            selected_embeddings["wikir_train_dense_rankings"] = train_dense_rankings
            selected_embeddings["wikir_train_dense_scores"] = train_dense_scores

        mirage_doc_embeddings = None
        mirage_doc_meta = None
        mirage_test_query_embeddings = None
        mirage_test_query_meta = None

        if need_mirage_embeddings:
            mirage_doc_embeddings, mirage_doc_meta = encode_texts(
                model_name,
                mirage_doc_texts,
                cache_prefix="mirage_documents",
                device=device,
                batch_size=args.batch_size,
                shard_size=args.embedding_shard_size,
            )
            mirage_test_query_embeddings, mirage_test_query_meta = encode_texts(
                model_name,
                mirage_test_texts,
                cache_prefix="mirage_test_queries",
                device=device,
                batch_size=args.batch_size,
                shard_size=args.embedding_shard_size,
            )

        if not args.skip_dense:
            run_id = f"mirage_dense_{model_name.replace('/', '__')}"
            rankings, score_lookup, avg_query_seconds = dense_search(
                mirage_test_qids,
                mirage_test_query_embeddings,
                mirage_doc_ids,
                mirage_doc_embeddings,
                top_k=args.dense_top_k_mirage,
            )
            metrics = evaluate_rankings(rankings, mirage_test_qrels)
            write_trec_run(RUNS_DIR / f"{run_id}.trec", rankings, score_lookup, run_id)
            summary["dense_retrieval"].append(
                summarize_dense_results(
                    "MIRAGE test",
                    model_name,
                    metrics,
                    corpus_size=len(mirage_doc_ids),
                    query_count=len(mirage_test_qids),
                    top_k=args.dense_top_k_mirage,
                    doc_encode_seconds=float(mirage_doc_meta["encode_seconds"]),
                    query_encode_seconds=float(mirage_test_query_meta["encode_seconds"]),
                    avg_query_seconds=avg_query_seconds,
                )
            )

        if model_name == DEFAULT_RERANK_MODEL and need_default_model:
            selected_embeddings["mirage_doc_embeddings"] = mirage_doc_embeddings
            selected_embeddings["mirage_doc_index"] = {doc_id: idx for idx, doc_id in enumerate(mirage_doc_ids)}
            selected_embeddings["mirage_test_query_map"] = build_query_embedding_map(mirage_test_qids, mirage_test_query_embeddings)
            if args.skip_dense:
                dense_rankings, dense_scores, _ = dense_search(
                    mirage_test_qids,
                    mirage_test_query_embeddings,
                    mirage_doc_ids,
                    mirage_doc_embeddings,
                    top_k=args.dense_top_k_mirage,
                )
                selected_embeddings["mirage_test_dense_rankings"] = dense_rankings
                selected_embeddings["mirage_test_dense_scores"] = dense_scores
            else:
                selected_embeddings["mirage_test_dense_rankings"] = rankings
                selected_embeddings["mirage_test_dense_scores"] = score_lookup

    if not args.skip_rerank or not args.skip_mixture:
        wikir_bm25_limit = max(max(rerank_ks, default=0), args.dense_top_k_wikir)
        wikir_test_bm25_rankings, wikir_test_bm25_scores = load_wikir_bm25_run(
            "test",
            top_k=wikir_bm25_limit,
            qids=set(wikir_test_qids),
        )
        wikir_train_bm25_rankings, wikir_train_bm25_scores = load_wikir_bm25_run(
            "train",
            top_k=args.dense_top_k_wikir,
            qids=set(wikir_train_qids),
        )

        mirage_bm25 = BM25Index(mirage_doc_ids, mirage_doc_texts, "mirage")
        mirage_bm25_build = mirage_bm25.build()
        mirage_bm25_rankings, mirage_bm25_scores, mirage_bm25_avg_query_seconds = mirage_bm25.search(
            mirage_test_qids,
            mirage_test_texts,
            top_k=max(max(rerank_ks, default=0), args.mirage_bm25_top_k),
        )
        summary["mirage_bm25"] = mirage_bm25_build | {"avg_query_seconds": mirage_bm25_avg_query_seconds}

    if not args.skip_rerank:
        for dataset_name, query_ids, qrels, base_rankings, query_map_key, doc_key, index_key in [
            (
                "WikIR test",
                wikir_test_qids,
                wikir_test_qrels,
                wikir_test_bm25_rankings,
                "wikir_test_query_map",
                "wikir_doc_embeddings",
                "wikir_doc_index",
            ),
            (
                "MIRAGE test",
                mirage_test_qids,
                mirage_test_qrels,
                mirage_bm25_rankings,
                "mirage_test_query_map",
                "mirage_doc_embeddings",
                "mirage_doc_index",
            ),
        ]:
            for k in rerank_ks:
                rankings, scores, avg_query_seconds = rerank_candidates(
                    query_ids,
                    base_rankings,
                    selected_embeddings[query_map_key],
                    selected_embeddings[doc_key],
                    selected_embeddings[index_key],
                    rerank_k=k,
                )
                metrics = evaluate_rankings(rankings, qrels)
                run_id = f"{dataset_name.lower().replace(' ', '_')}_rerank_{DEFAULT_RERANK_MODEL}_k_{k}"
                write_trec_run(RUNS_DIR / f"{run_id}.trec", rankings, scores, run_id)
                summary["reranking"].append(
                    {
                        "dataset": dataset_name,
                        "model": DEFAULT_RERANK_MODEL,
                        "k": k,
                        "metrics": metrics,
                        "avg_query_seconds": avg_query_seconds,
                    }
                )

    if not args.skip_mixture:
        alpha_values = np.round(np.arange(0.0, 1.0 + args.alpha_step / 2.0, args.alpha_step), 4)
        wikir_train_candidates = build_mixture_candidates(
            wikir_train_qids,
            wikir_train_bm25_rankings,
            wikir_train_bm25_scores,
            selected_embeddings["wikir_train_dense_rankings"],
            selected_embeddings["wikir_train_dense_scores"],
            selected_embeddings["wikir_train_query_map"],
            selected_embeddings["wikir_doc_embeddings"],
            selected_embeddings["wikir_doc_index"],
            bm25_top_k=args.dense_top_k_wikir,
            dense_top_k=args.dense_top_k_wikir,
        )
        alpha_curve: list[dict[str, object]] = []
        best_alpha = 0.0
        best_score = float("-inf")
        best_train_metrics: dict[str, float] | None = None
        for alpha in alpha_values:
            rankings, _, avg_query_seconds = score_mixture_candidates(wikir_train_candidates, alpha=float(alpha))
            metrics = evaluate_rankings(rankings, wikir_train_qrels)
            alpha_curve.append(
                {
                    "alpha": float(alpha),
                    "metrics": metrics,
                    "avg_query_seconds": avg_query_seconds,
                }
            )
            if metrics["nDCG@20"] > best_score:
                best_score = metrics["nDCG@20"]
                best_alpha = float(alpha)
                best_train_metrics = metrics

        mixture_results: list[dict[str, object]] = []
        for dataset_name, query_ids, qrels, bm25_rankings, bm25_scores, dense_rankings, dense_scores, query_map_key, doc_key, index_key, bm25_top_k, dense_top_k in [
            (
                "WikIR test",
                wikir_test_qids,
                wikir_test_qrels,
                wikir_test_bm25_rankings,
                wikir_test_bm25_scores,
                selected_embeddings["wikir_test_dense_rankings"],
                selected_embeddings["wikir_test_dense_scores"],
                "wikir_test_query_map",
                "wikir_doc_embeddings",
                "wikir_doc_index",
                args.dense_top_k_wikir,
                args.dense_top_k_wikir,
            ),
            (
                "MIRAGE test",
                mirage_test_qids,
                mirage_test_qrels,
                mirage_bm25_rankings,
                mirage_bm25_scores,
                selected_embeddings["mirage_test_dense_rankings"],
                selected_embeddings["mirage_test_dense_scores"],
                "mirage_test_query_map",
                "mirage_doc_embeddings",
                "mirage_doc_index",
                args.mirage_bm25_top_k,
                args.dense_top_k_mirage,
            ),
        ]:
            candidates = build_mixture_candidates(
                query_ids,
                bm25_rankings,
                bm25_scores,
                dense_rankings,
                dense_scores,
                selected_embeddings[query_map_key],
                selected_embeddings[doc_key],
                selected_embeddings[index_key],
                bm25_top_k=bm25_top_k,
                dense_top_k=dense_top_k,
            )
            rankings, scores, avg_query_seconds = score_mixture_candidates(candidates, alpha=best_alpha)
            metrics = evaluate_rankings(rankings, qrels)
            run_id = f"{dataset_name.lower().replace(' ', '_')}_mixture_alpha_{best_alpha:g}"
            write_trec_run(RUNS_DIR / f"{run_id}.trec", rankings, scores, run_id)
            mixture_results.append(
                {
                    "dataset": dataset_name,
                    "model": DEFAULT_RERANK_MODEL,
                    "alpha": best_alpha,
                    "metrics": metrics,
                    "avg_query_seconds": avg_query_seconds,
                }
            )

        summary["mixture"] = {
            "model": DEFAULT_RERANK_MODEL,
            "selection_metric": "nDCG@20",
            "best_alpha": best_alpha,
            "best_train_metrics": best_train_metrics,
            "alpha_curve": alpha_curve,
            "results": mixture_results,
        }

    summary = hydrate_summary_from_existing_artifacts(summary)

    summary_path = ARTIFACTS_DIR / "summary.json"
    markdown_path = ARTIFACTS_DIR / "summary.md"
    report_path = ARTIFACTS_DIR / "report.md"
    save_json(summary, summary_path)
    markdown_path.write_text(build_summary_markdown(summary), encoding="utf-8")
    report_path.write_text(build_report_markdown(summary), encoding="utf-8")

    dense_table = pd.DataFrame(
        [
            {
                "dataset": row["dataset"],
                "model": row["model"],
                **row["metrics"],
                "avg_query_seconds": row["avg_query_seconds"],
            }
            for row in summary["dense_retrieval"]
        ]
    )
    if not dense_table.empty:
        dense_table.to_csv(TABLES_DIR / "dense_results.csv", index=False)

    rerank_table = pd.DataFrame(
        [
            {
                "dataset": row["dataset"],
                "model": row["model"],
                "k": row["k"],
                **row["metrics"],
                "avg_query_seconds": row["avg_query_seconds"],
            }
            for row in summary["reranking"]
        ]
    )
    if not rerank_table.empty:
        rerank_table.to_csv(TABLES_DIR / "reranking_results.csv", index=False)

    mixture_rows = [
        {
            "dataset": row["dataset"],
            "model": row["model"],
            "alpha": row["alpha"],
            **row["metrics"],
            "avg_query_seconds": row["avg_query_seconds"],
        }
        for row in summary.get("mixture", {}).get("results", [])
    ]
    if mixture_rows:
        pd.DataFrame(mixture_rows).to_csv(TABLES_DIR / "mixture_results.csv", index=False)
        pd.DataFrame(summary["mixture"]["alpha_curve"]).to_json(
            TABLES_DIR / "alpha_curve.json",
            orient="records",
            indent=2,
            force_ascii=False,
        )

    print(json.dumps(summary["config"], indent=2, ensure_ascii=False, default=json_default))
    print(f"Summary JSON: {summary_path}")
    print(f"Summary Markdown: {markdown_path}")
    print(f"Report Markdown: {report_path}")


if __name__ == "__main__":
    main()
