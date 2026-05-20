from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import csv

import numpy as np


def build_group_sizes(qids: np.ndarray) -> list[int]:
    if len(qids) == 0:
        return []

    sizes: list[int] = []
    current = 1
    for prev_qid, qid in zip(qids[:-1], qids[1:]):
        if qid == prev_qid:
            current += 1
            continue
        sizes.append(current)
        current = 1
    sizes.append(current)
    return sizes


def dcg_at_k(labels: np.ndarray, scores: np.ndarray, k: int) -> float:
    if len(labels) == 0:
        return 0.0
    order = np.argsort(scores)[::-1][:k]
    ranked = labels[order]
    discounts = 1.0 / np.log2(np.arange(2, len(ranked) + 2))
    gains = np.power(2.0, ranked) - 1.0
    return float(np.sum(gains * discounts))


def ndcg_at_k(labels: np.ndarray, scores: np.ndarray, k: int) -> float:
    ideal = dcg_at_k(labels, labels, k)
    if ideal <= 0.0:
        return 0.0
    return dcg_at_k(labels, scores, k) / ideal


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    if len(labels) == 0:
        return 0.0
    binary = (labels > 0).astype(np.int32)
    positives = int(binary.sum())
    if positives == 0:
        return 0.0

    order = np.argsort(scores)[::-1]
    ranked = binary[order]
    cumulative = np.cumsum(ranked)
    precision = cumulative / np.arange(1, len(ranked) + 1)
    return float(np.sum(precision * ranked) / positives)


def grouped_metrics(
    qids: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray,
    ks: tuple[int, ...] = (10, 20),
) -> dict[str, float]:
    rows: dict[str, list[float]] = defaultdict(list)
    start = 0
    for size in build_group_sizes(qids):
        end = start + size
        group_labels = labels[start:end]
        group_scores = scores[start:end]
        rows["ap"].append(average_precision(group_labels, group_scores))
        for k in ks:
            rows[f"ndcg@{k}"].append(ndcg_at_k(group_labels, group_scores, k))
        start = end

    return {metric: float(np.mean(values)) for metric, values in rows.items()}


def read_trec_qrels(path: Path) -> dict[str, dict[str, float]]:
    qrels: dict[str, dict[str, float]] = defaultdict(dict)
    with path.open("r", encoding="utf-8", newline="") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            qid, _, doc_id, relevance = line.split()
            qrels[qid][doc_id] = float(relevance)
    return qrels


def read_queries_csv(path: Path, id_field: str, text_field: str) -> dict[str, str]:
    data: dict[str, str] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            data[str(row[id_field]).strip()] = row[text_field].strip()
    return data
