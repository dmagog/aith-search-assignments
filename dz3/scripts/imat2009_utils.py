from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from statistics import median
import json

import numpy as np


@dataclass
class ImatDataset:
    labels: np.ndarray
    qids: np.ndarray
    feature_rows: list[list[tuple[int, float]]]
    num_features: int
    source_path: Path

    def to_dense(self, dtype: np.dtype = np.float32) -> np.ndarray:
        matrix = np.zeros((len(self.labels), self.num_features), dtype=dtype)
        for row_idx, feature_row in enumerate(self.feature_rows):
            for feature_idx, value in feature_row:
                matrix[row_idx, feature_idx - 1] = value
        return matrix

    def subset(self, mask: np.ndarray) -> "ImatDataset":
        indices = np.flatnonzero(mask)
        return ImatDataset(
            labels=self.labels[indices],
            qids=self.qids[indices],
            feature_rows=[self.feature_rows[idx] for idx in indices],
            num_features=self.num_features,
            source_path=self.source_path,
        )

    def write_letor(self, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            for label, qid, feature_row in zip(self.labels, self.qids, self.feature_rows):
                features = " ".join(f"{feature_idx}:{value:.6f}" for feature_idx, value in feature_row)
                handle.write(f"{label:g} qid:{qid} {features} # {qid}\n")

    def stats(self) -> dict:
        query_sizes = compute_query_sizes(self.qids)
        non_zero_counts = [len(row) for row in self.feature_rows]
        unique_labels, label_counts = np.unique(self.labels, return_counts=True)
        density = sum(non_zero_counts) / (len(self.labels) * self.num_features)

        return {
            "source_path": str(self.source_path),
            "rows": int(len(self.labels)),
            "queries": int(len(query_sizes)),
            "num_features": int(self.num_features),
            "label_min": float(np.min(self.labels)),
            "label_max": float(np.max(self.labels)),
            "label_distribution": {
                f"{label:g}": int(count) for label, count in zip(unique_labels, label_counts)
            },
            "query_size": {
                "min": int(min(query_sizes)),
                "median": float(median(query_sizes)),
                "mean": float(sum(query_sizes) / len(query_sizes)),
                "max": int(max(query_sizes)),
            },
            "nnz_per_row": {
                "min": int(min(non_zero_counts)),
                "median": float(median(non_zero_counts)),
                "mean": float(sum(non_zero_counts) / len(non_zero_counts)),
                "max": int(max(non_zero_counts)),
            },
            "density": float(density),
        }


def parse_imat2009(path: Path) -> ImatDataset:
    labels: list[float] = []
    qids: list[int] = []
    feature_rows: list[list[tuple[int, float]]] = []
    max_feature_idx = 0

    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue

            try:
                left_part, comment = line.split("#", 1)
            except ValueError as exc:
                raise ValueError(f"Missing qid comment at {path}:{line_number}") from exc

            tokens = left_part.split()
            if not tokens:
                raise ValueError(f"Empty row at {path}:{line_number}")

            label = float(tokens[0])
            qid = int(comment.strip())

            feature_row: list[tuple[int, float]] = []
            for token in tokens[1:]:
                feature_idx_str, value_str = token.split(":", 1)
                feature_idx = int(feature_idx_str)
                value = float(value_str)
                feature_row.append((feature_idx, value))
                max_feature_idx = max(max_feature_idx, feature_idx)

            labels.append(label)
            qids.append(qid)
            feature_rows.append(feature_row)

    return ImatDataset(
        labels=np.asarray(labels, dtype=np.float32),
        qids=np.asarray(qids, dtype=np.int64),
        feature_rows=feature_rows,
        num_features=max_feature_idx,
        source_path=path,
    )


def compute_query_sizes(qids: np.ndarray) -> list[int]:
    if len(qids) == 0:
        return []

    query_sizes: list[int] = []
    current_size = 1
    for prev_qid, qid in zip(qids[:-1], qids[1:]):
        if qid == prev_qid:
            current_size += 1
            continue
        query_sizes.append(current_size)
        current_size = 1
    query_sizes.append(current_size)
    return query_sizes


def save_json(data: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=True)


def build_summary_markdown(train_stats: dict, test_stats: dict) -> str:
    return "\n".join(
        [
            "# Сводка по Internet Mathematics 2009",
            "",
            "## Обучающая выборка",
            format_stats_block(train_stats),
            "",
            "## Тестовая выборка",
            format_stats_block(test_stats),
            "",
        ]
    )


def format_stats_block(stats: dict) -> str:
    query_size = stats["query_size"]
    nnz_per_row = stats["nnz_per_row"]

    return "\n".join(
        [
            f"- число пар запрос-документ: {stats['rows']}",
            f"- число запросов: {stats['queries']}",
            f"- число признаков: {stats['num_features']}",
            f"- диапазон меток релевантности: {stats['label_min']:g} .. {stats['label_max']:g}",
            (
                f"- документов на запрос min/median/mean/max: "
                f"{query_size['min']} / {query_size['median']:.1f} / "
                f"{query_size['mean']:.2f} / {query_size['max']}"
            ),
            (
                f"- ненулевых признаков в строке min/median/mean/max: "
                f"{nnz_per_row['min']} / {nnz_per_row['median']:.1f} / "
                f"{nnz_per_row['mean']:.2f} / {nnz_per_row['max']}"
            ),
            f"- плотность матрицы признаков: {stats['density']:.6f}",
        ]
    )
