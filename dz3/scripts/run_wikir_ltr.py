from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from heapq import heappop, heappush
from pathlib import Path

from catboost import CatBoostRanker, Pool
import numpy as np
import pandas as pd

from imat2009_utils import save_json
from ltr_metrics import grouped_metrics, read_queries_csv, read_trec_qrels


WIKIR_ROOT = Path("../dz1/dz1/wikIR1k/wikIR1k")
DZ2_TABLES = Path("../dz2/artifacts/tables")


@dataclass
class DocCacheEntry:
    tokens: list[str]
    term_counts: Counter[str]
    positions: dict[str, list[int]]
    bigrams: set[tuple[str, str]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Пункты 3.1 и 3.2: LTR на WikIR.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/wikir_ltr"),
        help="Каталог для результатов WikIR LTR.",
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


def read_documents(path: Path) -> dict[str, str]:
    rows = pd.read_csv(path, dtype={"id_right": str})
    return dict(zip(rows["id_right"].astype(str), rows["text_right"].fillna("").astype(str), strict=True))


def read_bm25_run(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        sep=" ",
        header=None,
        names=["qid", "Q0", "doc_id", "rank", "bm25_score", "run_id"],
        dtype={"qid": str, "doc_id": str},
    )


def read_bm25_train_labels(path: Path) -> pd.DataFrame:
    rows = pd.read_csv(path, dtype={"id_left": str, "id_right": str})
    return rows.rename(columns={"id_left": "qid", "id_right": "doc_id", "label": "label"})[
        ["qid", "doc_id", "label"]
    ]


def qrels_to_frame(path: Path) -> pd.DataFrame:
    qrels = read_trec_qrels(path)
    rows: list[dict[str, object]] = []
    for qid, docs in qrels.items():
        for doc_id, label in docs.items():
            rows.append({"qid": qid, "doc_id": doc_id, "label": float(label)})
    return pd.DataFrame(rows)


def merge_candidates(run_df: pd.DataFrame, qrels_df: pd.DataFrame, top_k: int) -> pd.DataFrame:
    top_df = run_df[run_df["rank"] < top_k].copy()
    merged = top_df.merge(qrels_df, on=["qid", "doc_id"], how="left")
    merged["label"] = merged["label"].fillna(0.0)
    return merged.sort_values(["qid", "rank"]).reset_index(drop=True)


def sample_train_pairs(train_df: pd.DataFrame, random_seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(random_seed)
    sampled_groups: list[pd.DataFrame] = []
    for _, group in train_df.groupby("qid", sort=False):
        positives = group[group["label"] > 0]
        negatives = group[group["label"] <= 0]
        if positives.empty or negatives.empty:
            continue

        negative_count = min(len(positives), len(negatives))
        # Жесткие отрицательные примеры: выбираем из верхней части BM25-выдачи.
        hard_negatives = negatives.nsmallest(negative_count * 3, "rank")
        if len(hard_negatives) > negative_count:
            chosen_idx = rng.choice(hard_negatives.index.to_numpy(), size=negative_count, replace=False)
            chosen_negatives = hard_negatives.loc[np.sort(chosen_idx)]
        else:
            chosen_negatives = hard_negatives
        sampled_groups.append(pd.concat([positives, chosen_negatives], ignore_index=False).sort_values("rank"))

    return pd.concat(sampled_groups, ignore_index=True)


def build_collection_df(documents: dict[str, str]) -> tuple[dict[str, int], int]:
    df_counter: dict[str, int] = defaultdict(int)
    for text in documents.values():
        tokens = set(text.split())
        for token in tokens:
            df_counter[token] += 1
    return dict(df_counter), len(documents)


def default_idf(num_docs: int) -> float:
    return float(np.log((num_docs + 0.5) / 0.5))


def compute_idf(token: str, df_counter: dict[str, int], num_docs: int) -> float:
    df = df_counter.get(token, 0)
    return float(np.log((num_docs - df + 0.5) / (df + 0.5) + 1.0))


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


def build_doc_cache(tokens: list[str]) -> DocCacheEntry:
    positions: dict[str, list[int]] = defaultdict(list)
    for idx, token in enumerate(tokens):
        positions[token].append(idx)
    bigrams = set(zip(tokens[:-1], tokens[1:]))
    return DocCacheEntry(
        tokens=tokens,
        term_counts=Counter(tokens),
        positions=dict(positions),
        bigrams=bigrams,
    )


def model_display_name(model_key: str) -> str:
    mapping = {
        "feature_augmented_yetirank": "CatBoost с расширенными признаками (YetiRank)",
        "feature_augmented_pairlogit": "CatBoost с расширенными признаками (PairLogit)",
        "bm25_reconstruction": "CatBoost по компонентам BM25",
    }
    return mapping.get(model_key, model_key)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    documents = read_documents(WIKIR_ROOT / "documents.csv")
    queries_train = read_queries_csv(WIKIR_ROOT / "train" / "queries.csv", "id_left", "text_left")
    queries_validation = read_queries_csv(WIKIR_ROOT / "validation" / "queries.csv", "id_left", "text_left")
    queries_test = read_queries_csv(WIKIR_ROOT / "test" / "queries.csv", "id_left", "text_left")

    train_run = read_bm25_run(WIKIR_ROOT / "train" / "BM25.res")
    validation_run = read_bm25_run(WIKIR_ROOT / "validation" / "BM25.res")
    test_run = read_bm25_run(WIKIR_ROOT / "test" / "BM25.res")

    train_labels = read_bm25_train_labels(WIKIR_ROOT / "train" / "BM25.qrels.csv")
    validation_labels = qrels_to_frame(WIKIR_ROOT / "validation" / "qrels")
    test_labels = qrels_to_frame(WIKIR_ROOT / "test" / "qrels")

    train_candidates = train_run.merge(train_labels, on=["qid", "doc_id"], how="left")
    train_candidates["label"] = train_candidates["label"].fillna(0.0)
    train_candidates = train_candidates.sort_values(["qid", "rank"]).reset_index(drop=True)
    sampled_train = sample_train_pairs(train_candidates, random_seed=args.random_seed)

    validation_candidates = merge_candidates(validation_run, validation_labels, top_k=100)
    test_candidates = merge_candidates(test_run, test_labels, top_k=100)

    df_counter, num_docs = build_collection_df(documents)
    unknown_idf = default_idf(num_docs)

    @lru_cache(maxsize=50000)
    def doc_entry(doc_id: str) -> DocCacheEntry:
        return build_doc_cache(documents[doc_id].split())

    def make_feature_row(query_text: str, doc_id: str, bm25_score: float) -> dict[str, float]:
        q_tokens = query_text.split()
        q_unique = list(dict.fromkeys(q_tokens))
        entry = doc_entry(doc_id)
        d_tokens = entry.tokens
        d_len = len(d_tokens)

        tfs = [entry.term_counts.get(token, 0) for token in q_unique]
        idfs = [compute_idf(token, df_counter, num_docs) for token in q_unique] if q_unique else []
        matched_terms = [token for token in q_unique if entry.term_counts.get(token, 0) > 0]
        matched_positions = [entry.positions[token] for token in matched_terms]
        matched_ratio = len(matched_terms) / len(q_unique) if q_unique else 0.0
        bigram_hits = 0
        for left, right in zip(q_tokens[:-1], q_tokens[1:]):
            if (left, right) in entry.bigrams:
                bigram_hits += 1

        best_span = best_span_length(matched_positions) if len(matched_positions) >= 2 else d_len + 1
        matched_idfs = [compute_idf(token, df_counter, num_docs) for token in matched_terms] if matched_terms else [0.0]

        feature_row = {
            "query_length": float(len(q_tokens)),
            "document_length": float(d_len),
            "bm25_score": float(bm25_score),
            "bm25_rank": 0.0,
            "reciprocal_bm25_rank": 0.0,
            "bm25_per_query_term": float(bm25_score / max(len(q_tokens), 1)),
            "matched_terms": float(len(matched_terms)),
            "coverage_ratio": float(matched_ratio),
            "sum_tf": float(sum(tfs)),
            "mean_tf": float(np.mean(tfs) if tfs else 0.0),
            "max_tf": float(max(tfs) if tfs else 0.0),
            "sum_idf": float(sum(idfs)),
            "mean_idf": float(np.mean(idfs) if idfs else unknown_idf),
            "max_idf": float(max(idfs) if idfs else unknown_idf),
            "matched_idf_sum": float(sum(matched_idfs)),
            "best_span": float(best_span),
            "span_reciprocal": float(1.0 / max(best_span, 1)),
            "query_bigram_hits": float(bigram_hits),
            "exact_match_fraction": float(sum(tf > 0 for tf in tfs) / max(len(q_tokens), 1)),
        }
        return feature_row

    def build_features(df: pd.DataFrame, query_map: dict[str, str]) -> pd.DataFrame:
        rows: list[dict[str, float | str]] = []
        for row in df.itertuples(index=False):
            feature_row = make_feature_row(query_map[row.qid], row.doc_id, row.bm25_score)
            feature_row.update(
                {
                    "qid": row.qid,
                    "doc_id": row.doc_id,
                    "label": float(row.label),
                    "bm25_rank": int(row.rank),
                    "reciprocal_bm25_rank": float(1.0 / (int(row.rank) + 1)),
                }
            )
            rows.append(feature_row)
        return pd.DataFrame(rows)

    train_features = build_features(sampled_train, queries_train)
    validation_features = build_features(validation_candidates, queries_validation)
    test_features = build_features(test_candidates, queries_test)

    feature_sets = {
        "feature_augmented_yetirank": [
            "query_length",
            "document_length",
            "bm25_score",
            "bm25_rank",
            "reciprocal_bm25_rank",
            "bm25_per_query_term",
            "matched_terms",
            "coverage_ratio",
            "sum_tf",
            "mean_tf",
            "max_tf",
            "sum_idf",
            "mean_idf",
            "max_idf",
            "matched_idf_sum",
            "best_span",
            "span_reciprocal",
            "query_bigram_hits",
            "exact_match_fraction",
        ],
        "feature_augmented_pairlogit": [
            "query_length",
            "document_length",
            "bm25_score",
            "bm25_rank",
            "reciprocal_bm25_rank",
            "bm25_per_query_term",
            "matched_terms",
            "coverage_ratio",
            "sum_tf",
            "mean_tf",
            "max_tf",
            "sum_idf",
            "mean_idf",
            "max_idf",
            "matched_idf_sum",
            "best_span",
            "span_reciprocal",
            "query_bigram_hits",
            "exact_match_fraction",
        ],
        "bm25_reconstruction": [
            "query_length",
            "document_length",
            "matched_terms",
            "coverage_ratio",
            "sum_tf",
            "mean_tf",
            "max_tf",
            "sum_idf",
            "mean_idf",
            "max_idf",
            "matched_idf_sum",
        ],
    }

    baseline_validation = grouped_metrics(
        validation_features["qid"].to_numpy(),
        validation_features["label"].to_numpy(dtype=np.float32),
        validation_features["bm25_score"].to_numpy(dtype=np.float32),
    )
    baseline_test = grouped_metrics(
        test_features["qid"].to_numpy(),
        test_features["label"].to_numpy(dtype=np.float32),
        test_features["bm25_score"].to_numpy(dtype=np.float32),
    )

    results: list[dict[str, object]] = []
    for model_key, feature_columns in feature_sets.items():
        train_pool = Pool(
            train_features[feature_columns].to_numpy(dtype=np.float32),
            label=train_features["label"].to_numpy(dtype=np.float32),
            group_id=train_features["qid"].to_numpy(),
        )
        validation_pool = Pool(
            validation_features[feature_columns].to_numpy(dtype=np.float32),
            label=validation_features["label"].to_numpy(dtype=np.float32),
            group_id=validation_features["qid"].to_numpy(),
        )
        test_pool = Pool(
            test_features[feature_columns].to_numpy(dtype=np.float32),
            label=test_features["label"].to_numpy(dtype=np.float32),
            group_id=test_features["qid"].to_numpy(),
        )

        model_dir = args.output_dir / model_key
        model_dir.mkdir(parents=True, exist_ok=True)
        model = CatBoostRanker(
            loss_function="PairLogit" if model_key == "feature_augmented_pairlogit" else "YetiRank",
            eval_metric="NDCG:top=20",
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
                "model_name": model_display_name(model_key),
                "feature_columns": feature_columns,
                "best_iteration": int(model.get_best_iteration()),
                "validation_metrics": grouped_metrics(
                    validation_features["qid"].to_numpy(),
                    validation_features["label"].to_numpy(dtype=np.float32),
                    validation_scores,
                ),
                "test_metrics": grouped_metrics(
                    test_features["qid"].to_numpy(),
                    test_features["label"].to_numpy(dtype=np.float32),
                    test_scores,
                ),
            }
        )

    dz2_baseline = pd.read_csv(DZ2_TABLES / "baseline_results_summary.csv")
    dz2_best = dz2_baseline.loc[dz2_baseline["run_id"] == "bm25_stemmed_test_k1_1.5_b_0.75"].iloc[0]

    payload = {
        "dataset": {
            "name": "WikIR en1k",
            "train_sample_rows": int(len(train_features)),
            "validation_rows": int(len(validation_features)),
            "test_rows": int(len(test_features)),
            "train_queries": int(train_features["qid"].nunique()),
            "validation_queries": int(validation_features["qid"].nunique()),
            "test_queries": int(test_features["qid"].nunique()),
        },
        "baselines": {
            "bm25_top100_validation": baseline_validation,
            "bm25_top100_test": baseline_test,
            "assignment2_best_bm25": {
                "ap": float(dz2_best["AP"]),
                "ndcg@20": float(dz2_best["nDCG@20"]),
            },
        },
        "results": results,
    }

    save_json(payload, args.output_dir / "results.json")
    (args.output_dir / "results.md").write_text(
        "\n".join(
            [
                "# Пункты 3.1 и 3.2: обучение ранжированию на WikIR",
                "",
                "## Данные",
                "",
                "| Подвыборка | Строк | Запросов |",
                "| --- | ---: | ---: |",
                f"| Обучение после выборки отрицательных примеров | {payload['dataset']['train_sample_rows']} | {payload['dataset']['train_queries']} |",
                f"| Валидация top-100 BM25 | {payload['dataset']['validation_rows']} | {payload['dataset']['validation_queries']} |",
                f"| Тест top-100 BM25 | {payload['dataset']['test_rows']} | {payload['dataset']['test_queries']} |",
                "",
                "## Базовый BM25",
                "",
                "| Базовая система | AP | NDCG@10 | NDCG@20 |",
                "| --- | ---: | ---: | ---: |",
                (
                    f"| BM25 на validation top-100 | {baseline_validation['ap']:.5f} | "
                    f"{baseline_validation['ndcg@10']:.5f} | {baseline_validation['ndcg@20']:.5f} |"
                ),
                (
                    f"| BM25 на test top-100 | {baseline_test['ap']:.5f} | "
                    f"{baseline_test['ndcg@10']:.5f} | {baseline_test['ndcg@20']:.5f} |"
                ),
                (
                    f"| Лучший BM25 из ДЗ2 | {payload['baselines']['assignment2_best_bm25']['ap']:.5f} | "
                    f"n/a | {payload['baselines']['assignment2_best_bm25']['ndcg@20']:.5f} |"
                ),
                "",
                "## Результаты LTR",
                "",
                "| Модель | Лучшая итерация | Validation AP | Validation NDCG@20 | Test AP | Test NDCG@20 |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
                *[
                    (
                        f"| {result['model_name']} | {result['best_iteration']} | "
                        f"{result['validation_metrics']['ap']:.5f} | {result['validation_metrics']['ndcg@20']:.5f} | "
                        f"{result['test_metrics']['ap']:.5f} | {result['test_metrics']['ndcg@20']:.5f} |"
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
