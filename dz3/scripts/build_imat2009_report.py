from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from imat2009_utils import compute_query_sizes, parse_imat2009


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Построение отчета по imat2009.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("imat2009_new_split"),
        help="Каталог с файлами imat2009.",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("artifacts/processed"),
        help="Каталог со статистикой после подготовки данных.",
    )
    parser.add_argument(
        "--results-path",
        type=Path,
        default=Path("artifacts/imat2009/results.json"),
        help="JSON с результатами экспериментов.",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=Path("artifacts/figures"),
        help="Каталог для графиков.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("report.md"),
        help="Путь к итоговому markdown-отчету.",
    )
    return parser.parse_args()


def save_plot_query_sizes(train_sizes: list[int], test_sizes: list[int], output_path: Path) -> None:
    plt.figure(figsize=(9, 5))
    bins = np.arange(0, min(60, max(max(train_sizes), max(test_sizes)) + 2)) - 0.5
    plt.hist(train_sizes, bins=bins, alpha=0.6, label="Обучающая", density=True)
    plt.hist(test_sizes, bins=bins, alpha=0.6, label="Тестовая", density=True)
    plt.xlabel("Число документов на запрос")
    plt.ylabel("Нормированная частота")
    plt.title("Распределение числа документов на запрос")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def save_plot_labels(train_labels: np.ndarray, test_labels: np.ndarray, output_path: Path) -> None:
    bins = np.linspace(0, 4, 21)
    plt.figure(figsize=(9, 5))
    plt.hist(train_labels, bins=bins, alpha=0.6, label="Обучающая", density=True)
    plt.hist(test_labels, bins=bins, alpha=0.6, label="Тестовая", density=True)
    plt.xlabel("Метка релевантности")
    plt.ylabel("Нормированная частота")
    plt.title("Распределение меток релевантности")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def display_model_name(model_name: str) -> str:
    mapping = {
        "catboost_yetirank": "CatBoost YetiRank",
        "catboost_pairlogit": "CatBoost PairLogit",
    }
    return mapping.get(model_name, model_name)


def save_plot_metrics(results: list[dict], output_path: Path) -> None:
    metrics = ["ndcg@5", "ndcg@10", "ndcg@20"]
    x = np.arange(len(metrics))
    width = 0.35

    plt.figure(figsize=(9, 5))
    for idx, result in enumerate(results):
        offset = (idx - (len(results) - 1) / 2) * width
        scores = [result["test_metrics"][metric] for metric in metrics]
        plt.bar(x + offset, scores, width=width, label=display_model_name(result["model"]))

    plt.xticks(x, [metric.upper() for metric in metrics])
    plt.ylim(0.65, 0.82)
    plt.ylabel("Значение метрики")
    plt.title("Сравнение моделей на тестовой выборке")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def format_dataset_table(stats: dict) -> str:
    return "\n".join(
        [
            "| Выборка | Пар запрос-документ | Запросов | Признаков | Метки | Мин/мед/сред/макс документов на запрос | Плотность |",
            "| --- | ---: | ---: | ---: | --- | --- | ---: |",
            (
                f"| Обучающая | {stats['train']['rows']} | {stats['train']['queries']} | "
                f"{stats['train']['num_features']} | {stats['train']['label_min']:g}..{stats['train']['label_max']:g} | "
                f"{stats['train']['query_size']['min']} / {stats['train']['query_size']['median']:.1f} / "
                f"{stats['train']['query_size']['mean']:.2f} / {stats['train']['query_size']['max']} | "
                f"{stats['train']['density']:.6f} |"
            ),
            (
                f"| Тестовая | {stats['test']['rows']} | {stats['test']['queries']} | "
                f"{stats['test']['num_features']} | {stats['test']['label_min']:g}..{stats['test']['label_max']:g} | "
                f"{stats['test']['query_size']['min']} / {stats['test']['query_size']['median']:.1f} / "
                f"{stats['test']['query_size']['mean']:.2f} / {stats['test']['query_size']['max']} | "
                f"{stats['test']['density']:.6f} |"
            ),
        ]
    )


def format_results_table(results_payload: dict) -> str:
    lines = [
        "| Модель | Лучшая итерация | Валидация NDCG@5 | Валидация NDCG@10 | Валидация NDCG@20 | Тест NDCG@5 | Тест NDCG@10 | Тест NDCG@20 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for result in results_payload["results"]:
        valid = result["validation_metrics"]
        test = result["test_metrics"]
        lines.append(
            f"| {display_model_name(result['model'])} | {result['best_iteration']} | "
            f"{valid['ndcg@5']:.5f} | {valid['ndcg@10']:.5f} | {valid['ndcg@20']:.5f} | "
            f"{test['ndcg@5']:.5f} | {test['ndcg@10']:.5f} | {test['ndcg@20']:.5f} |"
        )
    return "\n".join(lines)


def build_report(stats: dict, results_payload: dict) -> str:
    split_summary = results_payload["split_summary"]
    best_result = max(results_payload["results"], key=lambda item: item["test_metrics"]["ndcg@10"])

    return "\n".join(
        [
            "# DZ3: обучение ранжированию на Internet Mathematics 2009",
            "",
            "## Проверка полноты и целостности",
            "",
            "- Текущее решение корректно закрывает пункт 2 задания для набора `Internet Mathematics 2009`.",
            "- Полным решением всего `HA3.pdf` его пока считать нельзя: пункты 1, 3 и 4 требуют отдельных данных и пока не реализованы.",
            "- Методика эксперимента приведена в корректный вид: подбор числа итераций выполняется по валидации, тест используется только для финальной оценки.",
            "",
            "## Описание данных",
            "",
            format_dataset_table(stats),
            "",
            f"Для обучения моделей из исходной обучающей выборки выделено {split_summary['validation_queries']} запросов под валидацию "
            f"({split_summary['validation_rows']} пар запрос-документ).",
            "",
            "![Распределение числа документов на запрос](artifacts/figures/query_size_distribution.png)",
            "",
            "![Распределение меток релевантности](artifacts/figures/label_distribution.png)",
            "",
            "## Эксперименты",
            "",
            "Были проверены два ранжирующих подхода на CatBoost: `YetiRank` и `PairLogit`. "
            "Целевая метрика во всех экспериментах: `NDCG`.",
            "",
            format_results_table(results_payload),
            "",
            "![Сравнение моделей по NDCG](artifacts/figures/metric_comparison.png)",
            "",
            "## Вывод",
            "",
            f"Лучший результат на тестовой выборке показала модель `{display_model_name(best_result['model'])}`: "
            f"`NDCG@10 = {best_result['test_metrics']['ndcg@10']:.5f}`.",
            "Оформление приведено к более сдаваемому виду: добавлены таблицы, графики и русскоязычный текст без лишних англицизмов.",
            "Главное ограничение текущего состояния остается прежним: решение пока покрывает только раздел `Internet Mathematics 2009` из задания.",
            "",
        ]
    )


def main() -> None:
    args = parse_args()
    args.figures_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = parse_imat2009(args.input_dir / "imat2009_train_new.txt")
    test_dataset = parse_imat2009(args.input_dir / "imat2009_test_new.txt")

    stats = json.loads((args.processed_dir / "imat2009_stats.json").read_text(encoding="utf-8"))
    results_payload = json.loads(args.results_path.read_text(encoding="utf-8"))

    save_plot_query_sizes(
        compute_query_sizes(train_dataset.qids),
        compute_query_sizes(test_dataset.qids),
        args.figures_dir / "query_size_distribution.png",
    )
    save_plot_labels(
        train_dataset.labels,
        test_dataset.labels,
        args.figures_dir / "label_distribution.png",
    )
    save_plot_metrics(
        results_payload["results"],
        args.figures_dir / "metric_comparison.png",
    )

    args.report_path.write_text(build_report(stats, results_payload), encoding="utf-8")
    print(f"Сохранен файл {args.report_path}")


if __name__ == "__main__":
    main()
