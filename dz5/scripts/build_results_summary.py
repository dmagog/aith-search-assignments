from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[1]
METRICS_DIR = ROOT / "artifacts" / "metrics"
JUDGE_SUMMARY = ROOT / "artifacts" / "judge" / "llm_judge_qwen2_5_1_5b_v2_summary.json"
TABLES_DIR = ROOT / "artifacts" / "tables"
FIGURES_DIR = ROOT / "artifacts" / "figures"

DISPLAY = {
    "roberta_oracle_full": "RoBERTa oracle",
    "roberta_top1_mixture_full": "RoBERTa top-1 mixture",
    "roberta_top1_dense_full": "RoBERTa top-1 dense",
    "qwen2_5_1_5b_instruct_closed_book": "Qwen без контекста",
    "qwen2_5_1_5b_instruct_oracle": "Qwen oracle",
    "qwen2_5_1_5b_instruct_top1_mixture": "Qwen top-1 mixture",
    "qwen2_5_1_5b_instruct_top1_dense": "Qwen top-1 dense",
    "qwen2_5_1_5b_instruct_top5_mixture": "Qwen top-5 mixture",
    "qwen2_5_1_5b_instruct_top5_dense": "Qwen top-5 dense",
    "qwen2_5_1_5b_instruct_mirage_mixed": "Qwen MIRAGE mixed",
    "t5gemma2_270m_squad_lora_closed_book": "T5Gemma без контекста",
    "t5gemma2_270m_squad_lora_oracle": "T5Gemma oracle",
    "t5gemma2_270m_squad_lora_top1_mixture": "T5Gemma top-1 mixture",
    "t5gemma2_270m_squad_lora_top1_dense": "T5Gemma top-1 dense",
}

CONTEXT_RU = {
    "closed_book": "без контекста",
    "oracle": "oracle",
    "top1_mixture": "top-1 mixture",
    "top1_dense": "top-1 dense",
    "top5_mixture": "top-5 mixture",
    "top5_dense": "top-5 dense",
    "mirage_mixed": "MIRAGE mixed",
}


def family(experiment_id: str) -> str:
    if experiment_id.startswith("qwen2_5_1_5b_instruct"):
        return "generative_slm"
    if experiment_id.startswith("t5gemma2_270m_squad_lora"):
        return "fine_tuned_slm"
    if experiment_id.startswith("roberta_"):
        return "extractive_qa"
    return "other"


def context_label(experiment_id: str) -> str:
    for label in [
        "closed_book",
        "mirage_mixed",
        "oracle",
        "top5_dense",
        "top5_mixture",
        "top1_dense",
        "top1_mixture",
    ]:
        if experiment_id.endswith(label) or experiment_id.endswith(f"{label}_full"):
            return label
    return "unknown"


def model_label(experiment_id: str, payload: dict[str, Any]) -> str:
    prediction_file = payload.get("prediction_file", "")
    if "qwen2_5_1_5b" in experiment_id or "qwen2_5_1_5b" in prediction_file:
        return "Qwen2.5-1.5B-Instruct"
    if "t5gemma2_270m_squad_lora" in experiment_id:
        return "google/t5gemma-2-270m-270m + LoRA SQuAD"
    if experiment_id.startswith("roberta_") and experiment_id.endswith("_full"):
        return "deepset/roberta-base-squad2"
    if "distilbert" in experiment_id:
        return "distilbert-base-cased-distilled-squad"
    return "unknown"


def read_rows() -> list[dict[str, Any]]:
    judge_by_experiment = {}
    if JUDGE_SUMMARY.exists():
        judge_payload = json.loads(JUDGE_SUMMARY.read_text(encoding="utf-8"))
        judge_by_experiment = judge_payload.get("by_experiment", {})

    rows = []
    for path in sorted(METRICS_DIR.glob("*_metrics.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        exp = path.name.removesuffix("_metrics.json")
        agg = payload.get("aggregate", {})
        judge_key = exp
        if exp == "roberta_oracle_full":
            judge_key = "roberta_oracle"
        elif exp == "roberta_top1_mixture_full":
            judge_key = "roberta_top1_mixture"
        elif exp == "roberta_top1_dense_full":
            judge_key = "roberta_top1_dense"
        judge = judge_by_experiment.get(judge_key, {})
        rows.append(
            {
                "experiment_id": exp,
                "family": family(exp),
                "model": model_label(exp, payload),
                "context": context_label(exp),
                "count": payload.get("count", ""),
                "squad_em": agg.get("squad_em", ""),
                "squad_f1": agg.get("squad_f1", ""),
                "mirage_f1": agg.get("mirage_f1", ""),
                "mirage_em_strict": agg.get("mirage_em_strict", ""),
                "mirage_em_loose": agg.get("mirage_em_loose", ""),
                "bertscore_f1": agg.get("bertscore_f1", ""),
                "judge_correct_rate": judge.get("correct_rate", ""),
                "judge_correct_or_partial_rate": judge.get("correct_or_partial_rate", ""),
                "judge_correct": judge.get("correct", ""),
                "judge_partially_correct": judge.get("partially_correct", ""),
                "judge_incorrect": judge.get("incorrect", ""),
                "judge_unanswerable_or_bad_context": judge.get("unanswerable_or_bad_context", ""),
                "metrics_file": str(path.relative_to(ROOT)),
            }
        )
    return rows


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "experiment_id",
        "family",
        "model",
        "context",
        "count",
        "squad_em",
        "squad_f1",
        "mirage_f1",
        "mirage_em_strict",
        "mirage_em_loose",
        "bertscore_f1",
        "judge_correct_rate",
        "judge_correct_or_partial_rate",
        "judge_correct",
        "judge_partially_correct",
        "judge_incorrect",
        "judge_unanswerable_or_bad_context",
        "metrics_file",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def md_float(value: Any) -> str:
    if value == "":
        return ""
    return f"{float(value):.3f}"


def write_markdown(rows: list[dict[str, Any]], path: Path) -> None:
    full = full_rows(rows)

    lines = [
        "# Сводка результатов HA5",
        "",
        "Все основные эксперименты выполнены на одной фиксированной выборке MIRAGE из 1000 вопросов.",
        "",
        "| Эксперимент | Тип | Контекст | N | SQuAD EM | SQuAD F1 | MIRAGE loose | BERTScore F1 | LLM-судья: верно | LLM-судья: верно или частично верно |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in full:
        lines.append(
            "| {experiment_id} | {family} | {context} | {count} | {em} | {f1} | {loose} | {bertscore} | {j_correct} | {j_cp} |".format(
                experiment_id=DISPLAY.get(row["experiment_id"], row["experiment_id"]),
                family={
                    "generative_slm": "генеративная SLM",
                    "fine_tuned_slm": "дообученная SLM",
                    "extractive_qa": "экстрактивная QA",
                }.get(row["family"], row["family"]),
                context=CONTEXT_RU.get(row["context"], row["context"]),
                count=row["count"],
                em=md_float(row["squad_em"]),
                f1=md_float(row["squad_f1"]),
                loose=md_float(row["mirage_em_loose"]),
                bertscore=md_float(row["bertscore_f1"]),
                j_correct=md_float(row["judge_correct_rate"]),
                j_cp=md_float(row["judge_correct_or_partial_rate"]),
            )
        )

    by_id = {row["experiment_id"]: row for row in full}
    qwen_oracle = by_id.get("qwen2_5_1_5b_instruct_oracle")
    qwen_closed = by_id.get("qwen2_5_1_5b_instruct_closed_book")
    roberta_oracle = by_id.get("roberta_oracle_full")
    if qwen_oracle and qwen_closed and roberta_oracle:
        lines.extend(
            [
                "",
                "## Ключевые наблюдения",
                "",
                "- Контекст oracle резко улучшает генерацию: SQuAD F1 у Qwen растет с "
                f"{md_float(qwen_closed['squad_f1'])} до {md_float(qwen_oracle['squad_f1'])}.",
                "- Экстрактивная RoBERTa на oracle-пассажах дает лучший EM "
                f"({md_float(roberta_oracle['squad_em'])}) среди oracle-постановок.",
            ]
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def full_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    full = [row for row in rows if row["count"] == 1000 and row["experiment_id"] in DISPLAY]
    order = {
        "roberta_oracle_full": 0,
        "roberta_top1_mixture_full": 1,
        "roberta_top1_dense_full": 2,
        "qwen2_5_1_5b_instruct_closed_book": 3,
        "qwen2_5_1_5b_instruct_oracle": 4,
        "qwen2_5_1_5b_instruct_top1_mixture": 5,
        "qwen2_5_1_5b_instruct_top1_dense": 6,
        "qwen2_5_1_5b_instruct_top5_mixture": 7,
        "qwen2_5_1_5b_instruct_top5_dense": 8,
        "qwen2_5_1_5b_instruct_mirage_mixed": 9,
        "t5gemma2_270m_squad_lora_closed_book": 10,
        "t5gemma2_270m_squad_lora_oracle": 11,
        "t5gemma2_270m_squad_lora_top1_mixture": 12,
        "t5gemma2_270m_squad_lora_top1_dense": 13,
    }
    return sorted(full, key=lambda row: order[row["experiment_id"]])


def write_full_csv(rows: list[dict[str, Any]], path: Path) -> None:
    selected = full_rows(rows)
    fields = [
        "name",
        "experiment_id",
        "family",
        "context",
        "count",
        "squad_em",
        "squad_f1",
        "mirage_em_strict",
        "mirage_em_loose",
        "bertscore_f1",
        "judge_correct_rate",
        "judge_correct_or_partial_rate",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in selected:
            writer.writerow(
                {
                    "name": DISPLAY[row["experiment_id"]],
                    **{field: row.get(field, "") for field in fields if field != "name"},
                }
            )


def plot_metric(rows: list[dict[str, Any]], metric: str, title: str, filename: str) -> None:
    selected = full_rows(rows)
    labels = [DISPLAY[row["experiment_id"]] for row in selected]
    values = [float(row[metric]) if row[metric] != "" else 0.0 for row in selected]
    palette = {
        "generative_slm": "#3b82f6",
        "fine_tuned_slm": "#f59e0b",
        "extractive_qa": "#10b981",
    }
    colors = [palette[row["family"]] for row in selected]

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig_height = max(5.0, 0.42 * len(selected))
    fig, ax = plt.subplots(figsize=(9, fig_height))
    ax.barh(labels, values, color=colors)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Значение метрики")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25)
    ax.invert_yaxis()
    ax.legend(
        handles=[
            Patch(facecolor=palette["extractive_qa"], label="Экстрактивная QA"),
            Patch(facecolor=palette["generative_slm"], label="Генеративная SLM"),
            Patch(facecolor=palette["fine_tuned_slm"], label="Дообученная SLM"),
        ],
        loc="lower right",
        frameon=True,
    )
    for i, value in enumerate(values):
        ax.text(min(value + 0.015, 0.98), i, f"{value:.3f}", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / filename, dpi=180)
    plt.close(fig)


def write_figures(rows: list[dict[str, Any]]) -> None:
    plot_metric(rows, "squad_f1", "SQuAD F1 на полной выборке", "squad_f1_full.png")
    plot_metric(rows, "mirage_em_loose", "MIRAGE loose EM на полной выборке", "mirage_loose_full.png")
    plot_metric(rows, "bertscore_f1", "BERTScore F1 на полной выборке", "bertscore_f1_full.png")
    plot_metric(
        rows,
        "judge_correct_or_partial_rate",
        "LLM-судья: верно или частично верно",
        "judge_correct_or_partial_full.png",
    )
    write_t5gemma_empty_rate_figure()


def write_t5gemma_empty_rate_figure() -> None:
    experiments = [
        ("t5gemma2_270m_squad_lora_closed_book", "без контекста"),
        ("t5gemma2_270m_squad_lora_oracle", "oracle"),
        ("t5gemma2_270m_squad_lora_top1_mixture", "top-1 mixture"),
        ("t5gemma2_270m_squad_lora_top1_dense", "top-1 dense"),
    ]
    labels = []
    values = []
    for experiment_id, label in experiments:
        path = ROOT / "artifacts" / "predictions" / f"{experiment_id}.jsonl"
        if not path.exists():
            continue
        total = 0
        empty = 0
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                total += 1
                row = json.loads(line)
                if not str(row.get("prediction", "")).strip():
                    empty += 1
        if total:
            labels.append(label)
            values.append(empty / total)

    if not values:
        return

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar(labels, values, color="#f59e0b")
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Доля пустых ответов")
    ax.set_title("T5Gemma: доля пустых ответов", pad=14)
    ax.grid(axis="y", alpha=0.25)
    for i, value in enumerate(values):
        ax.text(i, min(value + 0.025, 1.055), f"{value:.1%}", ha="center", va="bottom", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "t5gemma_empty_rate.png", dpi=180)
    plt.close(fig)


def write_coverage(path: Path) -> None:
    lines = [
        "# Проверка покрытия пунктов HA5",
        "",
        "| Пункт | Статус | Что сделано |",
        "| --- | --- | --- |",
        "| 1. SLM без контекста, 0-shot | закрыто | Qwen2.5-1.5B-Instruct, 1000 вопросов |",
        "| 2. Экстрактивная QA / MRC | закрыто | deepset/roberta-base-squad2 по oracle-пассажам, 1000 вопросов |",
        "| 3. SLM с oracle-контекстом | закрыто | Qwen2.5-1.5B-Instruct с oracle-контекстом, 1000 вопросов |",
        "| 4. Top-1 контекст из ранжировщиков HA4 | закрыто | top-1 mixture/dense для RoBERTa и Qwen, 1000 вопросов |",
        "| 5. Top-5 контекст из ранжировщиков HA4 | закрыто | top-5 mixture/dense для Qwen + сравнение с MIRAGE mixed, 1000 вопросов |",
        "| 6. Дообучение T5Gemma 2 | закрыто | LoRA-дообучение на SQuAD, оценка без контекста/oracle/top-1 mixture/top-1 dense на 1000 вопросах |",
        "| 7. Оценка LLM-судьей | закрыто | Qwen-судья v2 на 14000 ответов, включая 4000 ответов T5Gemma |",
        "| Метрики EM/F1/MIRAGE | закрыто | SQuAD EM/F1, MIRAGE strict/loose для всех полных прогонов |",
        "| BERTScore | закрыто | BERTScore F1 посчитан для всех 14 полных прогонов с `bert-base-uncased` |",
        "| Таблицы | закрыто | CSV + Markdown-сводка |",
        "| Графики | закрыто | столбчатые диаграммы по SQuAD F1, MIRAGE loose, BERTScore F1, оценке LLM-судьи и доле пустых ответов T5Gemma |",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    rows = read_rows()
    write_csv(rows, TABLES_DIR / "dz5_all_metrics_summary.csv")
    write_full_csv(rows, TABLES_DIR / "dz5_full_results_for_report.csv")
    write_markdown(rows, TABLES_DIR / "dz5_results_summary.md")
    write_figures(rows)
    write_coverage(TABLES_DIR / "dz5_task_coverage.md")
    print(f"wrote {TABLES_DIR / 'dz5_all_metrics_summary.csv'}")
    print(f"wrote {TABLES_DIR / 'dz5_full_results_for_report.csv'}")
    print(f"wrote {TABLES_DIR / 'dz5_results_summary.md'}")
    print(f"wrote {TABLES_DIR / 'dz5_task_coverage.md'}")


if __name__ == "__main__":
    main()
