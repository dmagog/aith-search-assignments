from __future__ import annotations

import argparse
from pathlib import Path

from catboost import CatBoostRanker, Pool
from catboost.datasets import msrank_10k
import numpy as np
import pandas as pd

from ltr_metrics import grouped_metrics
from imat2009_utils import save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Раздел 1: CatBoost на данных MSRank / LETOR.")
    parser.add_argument(
        "--train-csv",
        type=Path,
        default=None,
        help="Локальный train.csv для MSRank. Если не задан, используется catboost.datasets.msrank_10k().",
    )
    parser.add_argument(
        "--test-csv",
        type=Path,
        default=None,
        help="Локальный test.csv для MSRank. Если не задан, используется catboost.datasets.msrank_10k().",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/msrank"),
        help="Каталог для результатов раздела 1.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=400,
        help="Максимальное число итераций бустинга.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Начальное значение генератора случайных чисел.",
    )
    return parser.parse_args()


def split_dataframe(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = df.iloc[:, 0].to_numpy(dtype=np.float32)
    qids = df.iloc[:, 1].to_numpy(dtype=np.int64)
    features = df.iloc[:, 2:].to_numpy(dtype=np.float32)
    return labels, qids, features


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.train_csv is not None and args.test_csv is not None:
        train_df = pd.read_csv(args.train_csv, header=None)
        test_df = pd.read_csv(args.test_csv, header=None)
    else:
        train_df, test_df = msrank_10k()
    train_labels, train_qids, train_features = split_dataframe(train_df)
    test_labels, test_qids, test_features = split_dataframe(test_df)

    unique_train_qids = np.unique(train_qids)
    rng = np.random.default_rng(args.random_seed)
    validation_qids = set(int(qid) for qid in rng.permutation(unique_train_qids)[: max(1, len(unique_train_qids) // 10)])
    validation_mask = np.asarray([int(qid) in validation_qids for qid in train_qids], dtype=bool)
    fit_mask = ~validation_mask

    fit_pool = Pool(train_features[fit_mask], label=train_labels[fit_mask], group_id=train_qids[fit_mask])
    validation_pool = Pool(
        train_features[validation_mask],
        label=train_labels[validation_mask],
        group_id=train_qids[validation_mask],
    )
    test_pool = Pool(test_features, label=test_labels, group_id=test_qids)

    params = {
        "loss_function": "YetiRank",
        "eval_metric": "NDCG:top=10",
        "iterations": args.iterations,
        "learning_rate": 0.1,
        "depth": 6,
        "random_seed": args.random_seed,
        "od_type": "Iter",
        "od_wait": 50,
        "train_dir": str(args.output_dir / "catboost_info"),
    }
    model = CatBoostRanker(**params)
    model.fit(fit_pool, eval_set=validation_pool, use_best_model=True, verbose=False)

    validation_predictions = model.predict(validation_pool)
    test_predictions = model.predict(test_pool)

    payload = {
        "dataset": {
            "name": "msrank_10k",
            "train_rows": int(train_df.shape[0]),
            "test_rows": int(test_df.shape[0]),
            "features": int(train_df.shape[1] - 2),
            "fit_queries": int(len(np.unique(train_qids[fit_mask]))),
            "validation_queries": int(len(np.unique(train_qids[validation_mask]))),
            "test_queries": int(len(np.unique(test_qids))),
        },
        "model": {
            "loss_function": params["loss_function"],
            "eval_metric": params["eval_metric"],
            "best_iteration": int(model.get_best_iteration()),
        },
        "validation_metrics": grouped_metrics(
            train_qids[validation_mask],
            train_labels[validation_mask],
            validation_predictions,
        ),
        "test_metrics": grouped_metrics(test_qids, test_labels, test_predictions),
    }

    save_json(payload, args.output_dir / "results.json")
    (args.output_dir / "results.md").write_text(
        "\n".join(
            [
                "# Раздел 1: CatBoost на MSRank / LETOR",
                "",
                "## Набор данных",
                "",
                "| Показатель | Значение |",
                "| --- | ---: |",
                f"| Строк в обучении | {payload['dataset']['train_rows']} |",
                f"| Строк в тесте | {payload['dataset']['test_rows']} |",
                f"| Признаков | {payload['dataset']['features']} |",
                f"| Запросов в fit | {payload['dataset']['fit_queries']} |",
                f"| Запросов в validation | {payload['dataset']['validation_queries']} |",
                f"| Запросов в test | {payload['dataset']['test_queries']} |",
                "",
                "## Качество",
                "",
                "| Выборка | AP | NDCG@10 | NDCG@20 |",
                "| --- | ---: | ---: | ---: |",
                (
                    f"| Валидация | {payload['validation_metrics']['ap']:.5f} | "
                    f"{payload['validation_metrics']['ndcg@10']:.5f} | "
                    f"{payload['validation_metrics']['ndcg@20']:.5f} |"
                ),
                (
                    f"| Тест | {payload['test_metrics']['ap']:.5f} | "
                    f"{payload['test_metrics']['ndcg@10']:.5f} | "
                    f"{payload['test_metrics']['ndcg@20']:.5f} |"
                ),
                "",
            ]
        ),
        encoding="utf-8",
    )

    print(payload)


if __name__ == "__main__":
    main()
