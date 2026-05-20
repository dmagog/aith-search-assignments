from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from catboost import CatBoostRanker, Pool
from sklearn.metrics import ndcg_score

from imat2009_utils import parse_imat2009, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Запуск базовых ранжирующих моделей CatBoost на imat2009.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("imat2009_new_split"),
        help="Каталог с файлами imat2009.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/imat2009"),
        help="Каталог для метрик и итоговых артефактов.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=300,
        help="Максимальное число итераций бустинга.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Начальное значение генератора случайных чисел.",
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.1,
        help="Доля запросов из обучающей выборки, выделяемая под валидацию.",
    )
    return parser.parse_args()


def build_group_slices(qids: np.ndarray) -> list[tuple[int, int]]:
    if len(qids) == 0:
        return []

    slices: list[tuple[int, int]] = []
    start = 0
    for idx in range(1, len(qids)):
        if qids[idx] == qids[idx - 1]:
            continue
        slices.append((start, idx))
        start = idx
    slices.append((start, len(qids)))
    return slices


def mean_ndcg(y_true: np.ndarray, y_pred: np.ndarray, qids: np.ndarray, k: int) -> float:
    per_query_scores: list[float] = []
    for start, end in build_group_slices(qids):
        if end - start < 2:
            continue
        score = ndcg_score([y_true[start:end]], [y_pred[start:end]], k=k)
        per_query_scores.append(float(score))
    return float(np.mean(per_query_scores))


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, qids: np.ndarray) -> dict[str, float]:
    return {
        "ndcg@5": mean_ndcg(y_true, y_pred, qids, k=5),
        "ndcg@10": mean_ndcg(y_true, y_pred, qids, k=10),
        "ndcg@20": mean_ndcg(y_true, y_pred, qids, k=20),
    }


def make_query_level_split(
    qids: np.ndarray,
    validation_fraction: float,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    unique_qids = np.unique(qids)
    rng = np.random.default_rng(random_seed)
    shuffled_qids = rng.permutation(unique_qids)

    validation_query_count = max(1, int(round(len(unique_qids) * validation_fraction)))
    validation_qids = set(int(qid) for qid in shuffled_qids[:validation_query_count])
    validation_mask = np.asarray([int(qid) in validation_qids for qid in qids], dtype=bool)
    train_mask = ~validation_mask
    return train_mask, validation_mask


def fit_and_evaluate(
    model_name: str,
    params: dict,
    output_dir: Path,
    train_pool: Pool,
    validation_pool: Pool,
    test_pool: Pool,
    y_validation: np.ndarray,
    qids_validation: np.ndarray,
    y_test: np.ndarray,
    qids_test: np.ndarray,
) -> dict:
    model_output_dir = output_dir / model_name
    model_output_dir.mkdir(parents=True, exist_ok=True)

    model = CatBoostRanker(
        **params,
        train_dir=str(model_output_dir),
    )
    model.fit(train_pool, eval_set=validation_pool, use_best_model=True, verbose=False)
    validation_predictions = model.predict(validation_pool)
    test_predictions = model.predict(test_pool)
    validation_metrics = evaluate_predictions(y_validation, validation_predictions, qids_validation)
    test_metrics = evaluate_predictions(y_test, test_predictions, qids_test)

    result = {
        "model": model_name,
        "params": params,
        "best_iteration": int(model.get_best_iteration()),
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
    }
    return result


def display_model_name(model_name: str) -> str:
    mapping = {
        "catboost_yetirank": "CatBoost YetiRank",
        "catboost_pairlogit": "CatBoost PairLogit",
    }
    return mapping.get(model_name, model_name)


def build_markdown(results: list[dict], split_summary: dict) -> str:
    lines = [
        "# Результаты ранжирования на imat2009",
        "",
        "## Схема разбиения",
        "",
        "| Подвыборка | Число пар | Число запросов |",
        "| --- | ---: | ---: |",
        f"| Обучение | {split_summary['train_rows']} | {split_summary['train_queries']} |",
        f"| Валидация | {split_summary['validation_rows']} | {split_summary['validation_queries']} |",
        f"| Тест | {split_summary['test_rows']} | {split_summary['test_queries']} |",
        "",
        "## Качество моделей",
        "",
        "| Модель | Лучшая итерация | Валидация NDCG@5 | Валидация NDCG@10 | Валидация NDCG@20 | Тест NDCG@5 | Тест NDCG@10 | Тест NDCG@20 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for result in results:
        validation_metrics = result["validation_metrics"]
        test_metrics = result["test_metrics"]
        lines.append(
            "| "
            f"{display_model_name(result['model'])} | "
            f"{result['best_iteration']} | "
            f"{validation_metrics['ndcg@5']:.5f} | "
            f"{validation_metrics['ndcg@10']:.5f} | "
            f"{validation_metrics['ndcg@20']:.5f} | "
            f"{test_metrics['ndcg@5']:.5f} | "
            f"{test_metrics['ndcg@10']:.5f} | "
            f"{test_metrics['ndcg@20']:.5f} |"
        )

    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    full_train_dataset = parse_imat2009(args.input_dir / "imat2009_train_new.txt")
    test_dataset = parse_imat2009(args.input_dir / "imat2009_test_new.txt")

    train_mask, validation_mask = make_query_level_split(
        qids=full_train_dataset.qids,
        validation_fraction=args.validation_fraction,
        random_seed=args.random_seed,
    )

    train_dataset = full_train_dataset.subset(train_mask)
    validation_dataset = full_train_dataset.subset(validation_mask)

    x_train = train_dataset.to_dense()
    x_validation = validation_dataset.to_dense()
    x_test = test_dataset.to_dense()

    train_pool = Pool(x_train, label=train_dataset.labels, group_id=train_dataset.qids)
    validation_pool = Pool(
        x_validation,
        label=validation_dataset.labels,
        group_id=validation_dataset.qids,
    )
    test_pool = Pool(x_test, label=test_dataset.labels, group_id=test_dataset.qids)

    common_params = {
        "iterations": args.iterations,
        "depth": 6,
        "learning_rate": 0.05,
        "random_seed": args.random_seed,
        "eval_metric": "NDCG:top=10",
        "od_type": "Iter",
        "od_wait": 50,
    }

    experiments = [
        (
            "catboost_yetirank",
            {
                **common_params,
                "loss_function": "YetiRank",
            },
        ),
        (
            "catboost_pairlogit",
            {
                **common_params,
                "loss_function": "PairLogit",
            },
        ),
    ]

    results = []
    for model_name, params in experiments:
        result = fit_and_evaluate(
            model_name=model_name,
            params=params,
            output_dir=args.output_dir,
            train_pool=train_pool,
            validation_pool=validation_pool,
            test_pool=test_pool,
            y_validation=validation_dataset.labels,
            qids_validation=validation_dataset.qids,
            y_test=test_dataset.labels,
            qids_test=test_dataset.qids,
        )
        results.append(result)
        print(model_name, result["test_metrics"])

    split_summary = {
        "train_rows": int(len(train_dataset.labels)),
        "train_queries": int(len(np.unique(train_dataset.qids))),
        "validation_rows": int(len(validation_dataset.labels)),
        "validation_queries": int(len(np.unique(validation_dataset.qids))),
        "test_rows": int(len(test_dataset.labels)),
        "test_queries": int(len(np.unique(test_dataset.qids))),
    }

    save_json(
        {
            "split_summary": split_summary,
            "results": results,
        },
        args.output_dir / "results.json",
    )
    (args.output_dir / "results.md").write_text(
        build_markdown(results, split_summary),
        encoding="utf-8",
    )

    print(f"Сохранен файл {args.output_dir / 'results.json'}")
    print(f"Сохранен файл {args.output_dir / 'results.md'}")


if __name__ == "__main__":
    main()
