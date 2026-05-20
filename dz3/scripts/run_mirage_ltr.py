from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha1
from heapq import heappop, heappush
from pathlib import Path

from catboost import CatBoostRanker, Pool
import numpy as np
import pandas as pd

from imat2009_utils import save_json
from ltr_metrics import grouped_metrics


@dataclass
class PassageEntry:
    title: str
    body: str
    title_tokens: list[str]
    body_tokens: list[str]
    title_counts: Counter[str]
    body_counts: Counter[str]
    title_bigrams: set[tuple[str, str]]
    body_positions: dict[str, list[int]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Раздел 4: ranker для MIRAGE.")
    parser.add_argument(
        "--parquet-path",
        type=Path,
        default=Path("../dz2/data/mirage/train.parquet"),
        help="Путь к parquet-файлу MIRAGE.",
    )
    parser.add_argument(
        "--signals-path",
        type=Path,
        default=Path("artifacts/mirage_ltr/wiki_signals.csv"),
        help="CSV с внешними wiki-сигналами.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/mirage_ltr"),
        help="Каталог для результатов MIRAGE LTR.",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.2,
        help="Доля запросов под тест, стратифицированная по source.",
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.1,
        help="Доля запросов от train под валидацию, стратифицированная по source.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=300,
        help="Максимальное число итераций CatBoost.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Начальное значение генератора случайных чисел.",
    )
    return parser.parse_args()


def passage_id(title: str, body: str) -> str:
    digest = sha1(f"{title}\n{body}".encode("utf-8")).hexdigest()[:16]
    return f"mirage-{digest}"


def stratified_query_split(
    query_meta: pd.DataFrame,
    test_fraction: float,
    validation_fraction: float,
    random_seed: int,
) -> tuple[set[str], set[str], set[str]]:
    rng = np.random.default_rng(random_seed)
    train_ids: set[str] = set()
    validation_ids: set[str] = set()
    test_ids: set[str] = set()

    for source, group in query_meta.groupby("source"):
        qids = group["query_id"].to_numpy()
        shuffled = rng.permutation(qids)
        test_count = max(1, int(round(len(shuffled) * test_fraction)))
        test_part = shuffled[:test_count]
        remaining = shuffled[test_count:]
        validation_count = max(1, int(round(len(remaining) * validation_fraction)))
        validation_part = remaining[:validation_count]
        train_part = remaining[validation_count:]

        test_ids.update(map(str, test_part))
        validation_ids.update(map(str, validation_part))
        train_ids.update(map(str, train_part))

    return train_ids, validation_ids, test_ids


def build_collection_df(df: pd.DataFrame) -> tuple[dict[str, int], int]:
    counter: dict[str, int] = defaultdict(int)
    seen_passages: set[str] = set()
    for row in df.itertuples(index=False):
        candidates = list(zip(row.doc_pool["doc_name"], row.doc_pool["doc_chunk"], row.doc_pool["support"], strict=True))
        candidates.append((row.oracle["doc_name"], row.oracle["doc_chunk"], row.oracle["support"]))
        for title, body, _ in candidates:
            pid = passage_id(str(title).strip(), str(body).strip())
            if pid in seen_passages:
                continue
            seen_passages.add(pid)
            for token in set(str(body).strip().split()):
                counter[token] += 1
    return dict(counter), len(seen_passages)


def best_span_length(position_lists: list[list[int]]) -> int:
    if len(position_lists) < 2 or any(not positions for positions in position_lists):
        return 0

    heap: list[tuple[int, int, int]] = []
    current_max = -1
    for list_idx, positions in enumerate(position_lists):
        value = positions[0]
        heappush(heap, (value, list_idx, 0))
        current_max = max(current_max, value)

    best = float("inf")
    while True:
        current_min, list_idx, offset = heappop(heap)
        best = min(best, current_max - current_min + 1)
        next_offset = offset + 1
        if next_offset >= len(position_lists[list_idx]):
            break
        next_value = position_lists[list_idx][next_offset]
        heappush(heap, (next_value, list_idx, next_offset))
        current_max = max(current_max, next_value)
    return 0 if best == float("inf") else int(best)


def compute_idf(token: str, df_counter: dict[str, int], num_docs: int) -> float:
    df = df_counter.get(token, 0)
    return float(np.log((num_docs - df + 0.5) / (df + 0.5) + 1.0))


def model_name(model_key: str) -> str:
    mapping = {
        "lexical_only": "Лексическая модель ранжирования",
        "enhanced_with_wiki_signals": "Модель с признаками заголовка, текста и сигналами Wikipedia",
    }
    return mapping.get(model_key, model_key)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.parquet_path)
    query_meta = df[["query_id", "source", "query"]].copy()
    train_ids, validation_ids, test_ids = stratified_query_split(
        query_meta,
        test_fraction=args.test_fraction,
        validation_fraction=args.validation_fraction,
        random_seed=args.random_seed,
    )

    df_counter, num_docs = build_collection_df(df)
    signals = pd.read_csv(args.signals_path) if args.signals_path.exists() else pd.DataFrame(columns=["title"])
    signal_map = {
        str(row["title"]): {
            "pageviews_sum_12m": float(row.get("pageviews_sum_12m", 0.0)),
            "pageviews_mean_12m": float(row.get("pageviews_mean_12m", 0.0)),
            "incoming_links": float(row.get("incoming_links", 0.0)),
            "redirects": float(row.get("redirects", 0.0)),
        }
        for _, row in signals.iterrows()
    }

    @lru_cache(maxsize=50000)
    def passage_entry(title: str, body: str) -> PassageEntry:
        title_tokens = title.split()
        body_tokens = body.split()
        positions: dict[str, list[int]] = defaultdict(list)
        for idx, token in enumerate(body_tokens):
            positions[token].append(idx)
        return PassageEntry(
            title=title,
            body=body,
            title_tokens=title_tokens,
            body_tokens=body_tokens,
            title_counts=Counter(title_tokens),
            body_counts=Counter(body_tokens),
            title_bigrams=set(zip(title_tokens[:-1], title_tokens[1:])),
            body_positions=dict(positions),
        )

    rows: list[dict[str, object]] = []
    for row in df.itertuples(index=False):
        query_id = str(row.query_id)
        query_text = str(row.query).strip()
        source = str(row.source)
        q_tokens = query_text.split()
        q_unique = list(dict.fromkeys(q_tokens))

        candidates: dict[str, dict[str, object]] = {}
        for title, body, support in zip(
            row.doc_pool["doc_name"],
            row.doc_pool["doc_chunk"],
            row.doc_pool["support"],
            strict=True,
        ):
            title = str(title).strip()
            body = str(body).strip()
            pid = passage_id(title, body)
            candidates[pid] = {"title": title, "body": body, "label": float(support)}

        oracle_title = str(row.oracle["doc_name"]).strip()
        oracle_body = str(row.oracle["doc_chunk"]).strip()
        oracle_pid = passage_id(oracle_title, oracle_body)
        candidates.setdefault(oracle_pid, {"title": oracle_title, "body": oracle_body, "label": float(row.oracle["support"])})
        candidates[oracle_pid]["label"] = max(float(candidates[oracle_pid]["label"]), float(row.oracle["support"]))

        for pid, candidate in candidates.items():
            entry = passage_entry(candidate["title"], candidate["body"])
            title_counts = [entry.title_counts.get(token, 0) for token in q_unique]
            body_counts = [entry.body_counts.get(token, 0) for token in q_unique]
            title_matches = [token for token in q_unique if entry.title_counts.get(token, 0) > 0]
            body_matches = [token for token in q_unique if entry.body_counts.get(token, 0) > 0]
            body_idfs = [compute_idf(token, df_counter, num_docs) for token in q_unique] if q_unique else [0.0]
            body_match_positions = [entry.body_positions[token] for token in body_matches]
            best_span = best_span_length(body_match_positions) if len(body_match_positions) >= 2 else len(entry.body_tokens) + 1
            wiki = signal_map.get(candidate["title"], {})

            rows.append(
                {
                    "query_id": query_id,
                    "source": source,
                    "label": float(candidate["label"]),
                    "passage_id": pid,
                    "query_length": float(len(q_tokens)),
                    "title_length": float(len(entry.title_tokens)),
                    "body_length": float(len(entry.body_tokens)),
                    "title_matches": float(len(title_matches)),
                    "body_matches": float(len(body_matches)),
                    "title_coverage_ratio": float(len(title_matches) / max(len(q_unique), 1)),
                    "body_coverage_ratio": float(len(body_matches) / max(len(q_unique), 1)),
                    "title_sum_tf": float(sum(title_counts)),
                    "title_max_tf": float(max(title_counts) if title_counts else 0.0),
                    "body_sum_tf": float(sum(body_counts)),
                    "body_max_tf": float(max(body_counts) if body_counts else 0.0),
                    "body_mean_tf": float(np.mean(body_counts) if body_counts else 0.0),
                    "body_sum_idf": float(sum(body_idfs)),
                    "body_max_idf": float(max(body_idfs) if body_idfs else 0.0),
                    "body_best_span": float(best_span),
                    "body_span_reciprocal": float(1.0 / max(best_span, 1)),
                    "title_exact_fraction": float(sum(tf > 0 for tf in title_counts) / max(len(q_unique), 1)),
                    "pageviews_sum_12m": float(wiki.get("pageviews_sum_12m", 0.0)),
                    "pageviews_mean_12m": float(wiki.get("pageviews_mean_12m", 0.0)),
                    "incoming_links": float(wiki.get("incoming_links", 0.0)),
                    "redirects": float(wiki.get("redirects", 0.0)),
                    "log_pageviews_sum_12m": float(np.log1p(wiki.get("pageviews_sum_12m", 0.0))),
                    "log_incoming_links": float(np.log1p(wiki.get("incoming_links", 0.0))),
                }
            )

    pairs = pd.DataFrame(rows).sort_values(["query_id", "passage_id"]).reset_index(drop=True)
    train_df = pairs[pairs["query_id"].isin(train_ids)].copy()
    validation_df = pairs[pairs["query_id"].isin(validation_ids)].copy()
    test_df = pairs[pairs["query_id"].isin(test_ids)].copy()

    feature_sets = {
        "lexical_only": [
            "query_length",
            "body_length",
            "body_matches",
            "body_coverage_ratio",
            "body_sum_tf",
            "body_max_tf",
            "body_mean_tf",
            "body_sum_idf",
            "body_max_idf",
            "body_best_span",
            "body_span_reciprocal",
        ],
        "enhanced_with_wiki_signals": [
            "query_length",
            "title_length",
            "body_length",
            "title_matches",
            "body_matches",
            "title_coverage_ratio",
            "body_coverage_ratio",
            "title_sum_tf",
            "title_max_tf",
            "body_sum_tf",
            "body_max_tf",
            "body_mean_tf",
            "body_sum_idf",
            "body_max_idf",
            "body_best_span",
            "body_span_reciprocal",
            "title_exact_fraction",
            "log_pageviews_sum_12m",
            "pageviews_mean_12m",
            "log_incoming_links",
            "redirects",
        ],
    }

    results: list[dict[str, object]] = []
    for model_key, feature_columns in feature_sets.items():
        train_pool = Pool(
            train_df[feature_columns].to_numpy(dtype=np.float32),
            label=train_df["label"].to_numpy(dtype=np.float32),
            group_id=train_df["query_id"].to_numpy(),
        )
        validation_pool = Pool(
            validation_df[feature_columns].to_numpy(dtype=np.float32),
            label=validation_df["label"].to_numpy(dtype=np.float32),
            group_id=validation_df["query_id"].to_numpy(),
        )
        test_pool = Pool(
            test_df[feature_columns].to_numpy(dtype=np.float32),
            label=test_df["label"].to_numpy(dtype=np.float32),
            group_id=test_df["query_id"].to_numpy(),
        )

        model_dir = args.output_dir / model_key
        model_dir.mkdir(parents=True, exist_ok=True)
        model = CatBoostRanker(
            loss_function="PairLogit",
            eval_metric="NDCG:top=5",
            iterations=args.iterations,
            learning_rate=0.05,
            depth=6,
            random_seed=args.random_seed,
            od_type="Iter",
            od_wait=50,
            train_dir=str(model_dir),
        )
        model.fit(train_pool, eval_set=validation_pool, use_best_model=True, verbose=False)

        validation_scores = model.predict(validation_pool)
        test_scores = model.predict(test_pool)
        results.append(
            {
                "model_key": model_key,
                "model_name": model_name(model_key),
                "feature_columns": feature_columns,
                "best_iteration": int(model.get_best_iteration()),
                "validation_metrics": grouped_metrics(
                    validation_df["query_id"].to_numpy(),
                    validation_df["label"].to_numpy(dtype=np.float32),
                    validation_scores,
                    ks=(3, 5),
                ),
                "test_metrics": grouped_metrics(
                    test_df["query_id"].to_numpy(),
                    test_df["label"].to_numpy(dtype=np.float32),
                    test_scores,
                    ks=(3, 5),
                ),
            }
        )

    split_summary = (
        query_meta.assign(
            split=query_meta["query_id"].map(
                lambda qid: "train" if qid in train_ids else "validation" if qid in validation_ids else "test"
            )
        )
        .groupby(["split", "source"])
        .size()
        .reset_index(name="queries")
    )

    payload = {
        "dataset": {
            "queries_total": int(query_meta.shape[0]),
            "pairs_total": int(pairs.shape[0]),
            "unique_passages": int(pairs["passage_id"].nunique()),
        },
        "split_summary": split_summary.to_dict(orient="records"),
        "results": results,
    }
    save_json(payload, args.output_dir / "results.json")
    (args.output_dir / "results.md").write_text(
        "\n".join(
            [
                "# Раздел 4: обучение ранжированию на MIRAGE",
                "",
                "## Разбиение запросов",
                "",
                "| Подвыборка | Источник | Число запросов |",
                "| --- | --- | ---: |",
                *[
                    f"| {row['split']} | {row['source']} | {row['queries']} |"
                    for row in payload["split_summary"]
                ],
                "",
                "## Качество",
                "",
                "| Модель | Лучшая итерация | Validation AP | Validation NDCG@5 | Test AP | Test NDCG@5 |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
                *[
                    (
                        f"| {result['model_name']} | {result['best_iteration']} | "
                        f"{result['validation_metrics']['ap']:.5f} | {result['validation_metrics']['ndcg@5']:.5f} | "
                        f"{result['test_metrics']['ap']:.5f} | {result['test_metrics']['ndcg@5']:.5f} |"
                    )
                    for result in results
                ],
                "",
            ]
        ),
        encoding="utf-8",
    )

    print(payload)


if __name__ == "__main__":
    main()
