from __future__ import annotations

import argparse
from pathlib import Path

from imat2009_utils import build_summary_markdown, parse_imat2009, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Преобразование imat2009 в LETOR-совместимый формат.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("imat2009_new_split"),
        help="Каталог с исходными файлами imat2009.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/processed"),
        help="Каталог для преобразованных файлов и статистики.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    train_path = args.input_dir / "imat2009_train_new.txt"
    test_path = args.input_dir / "imat2009_test_new.txt"

    train_dataset = parse_imat2009(train_path)
    test_dataset = parse_imat2009(test_path)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_letor_path = args.output_dir / "imat2009_train.letor.txt"
    test_letor_path = args.output_dir / "imat2009_test.letor.txt"
    train_dataset.write_letor(train_letor_path)
    test_dataset.write_letor(test_letor_path)

    train_stats = train_dataset.stats()
    test_stats = test_dataset.stats()

    summary = {
        "train": train_stats,
        "test": test_stats,
    }

    save_json(summary, args.output_dir / "imat2009_stats.json")
    (args.output_dir / "imat2009_summary.md").write_text(
        build_summary_markdown(train_stats, test_stats),
        encoding="utf-8",
    )

    print(f"Сохранен файл {train_letor_path}")
    print(f"Сохранен файл {test_letor_path}")
    print(f"Сохранен файл {args.output_dir / 'imat2009_stats.json'}")
    print(f"Сохранен файл {args.output_dir / 'imat2009_summary.md'}")


if __name__ == "__main__":
    main()
