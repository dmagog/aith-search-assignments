from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Построение итоговых графиков для DZ3.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/final_figures"),
        help="Каталог для итоговых графиков.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save_wikir_plot(data: dict, output_path: Path) -> None:
    labels = ["BM25 top-100", "BM25 из ДЗ2"]
    values = [
        data["baselines"]["bm25_top100_test"]["ndcg@20"],
        data["baselines"]["assignment2_best_bm25"]["ndcg@20"],
    ]
    for result in data["results"]:
        labels.append(result["model_name"])
        values.append(result["test_metrics"]["ndcg@20"])

    plt.figure(figsize=(10, 5))
    plt.bar(np.arange(len(labels)), values)
    plt.xticks(np.arange(len(labels)), labels, rotation=15, ha="right")
    plt.ylabel("NDCG@20")
    plt.title("WikIR: сравнение систем на тесте")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def save_mirage_plot(data: dict, output_path: Path) -> None:
    labels = [result["model_name"] for result in data["results"]]
    ndcg5 = [result["test_metrics"]["ndcg@5"] for result in data["results"]]
    ap = [result["test_metrics"]["ap"] for result in data["results"]]
    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(9, 5))
    plt.bar(x - width / 2, ndcg5, width=width, label="NDCG@5")
    plt.bar(x + width / 2, ap, width=width, label="AP")
    plt.xticks(x, labels, rotation=10)
    plt.ylabel("Значение метрики")
    plt.title("MIRAGE: качество моделей ранжирования на тесте")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    wikir = load_json(Path("artifacts/wikir_ltr/results.json"))
    mirage = load_json(Path("artifacts/mirage_ltr/results.json"))

    save_wikir_plot(wikir, args.output_dir / "wikir_test_comparison.png")
    save_mirage_plot(mirage, args.output_dir / "mirage_test_comparison.png")
    print(args.output_dir)


if __name__ == "__main__":
    main()
