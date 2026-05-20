# DZ5: Question Answering

Решение домашней работы HA5 по question answering для MIRAGE и SQuAD.

## С чего читать

- [artifacts/report.md](artifacts/report.md) — итоговый текстовый отчёт.
- [artifacts/report.pdf](artifacts/report.pdf) — собранный PDF-отчёт.
- [artifacts/report_landscape.pdf](artifacts/report_landscape.pdf) — PDF-отчёт в альбомной ориентации.
- [artifacts/tables/dz5_results_summary.md](artifacts/tables/dz5_results_summary.md) — краткая таблица итоговых результатов.

## Что внутри

- [scripts/prepare_data.py](scripts/prepare_data.py) — подготовка MIRAGE-сэмпла и контекстов.
- [scripts/evaluate_predictions.py](scripts/evaluate_predictions.py) — расчёт метрик.
- [scripts/run_extractive_qa.py](scripts/run_extractive_qa.py) — extractive QA.
- [scripts/run_llm_qa.py](scripts/run_llm_qa.py) — генеративные QA-эксперименты.
- [scripts/run_llm_judge.py](scripts/run_llm_judge.py) — LLM-as-a-judge.
- [scripts/build_results_summary.py](scripts/build_results_summary.py) — сборка итоговых таблиц.
- [notebooks/ha5_colab_llm_inference.ipynb](notebooks/ha5_colab_llm_inference.ipynb) — Colab-ноутбук для LLM inference.
- [COLAB.md](COLAB.md) — инструкция для запуска Colab-части.
- [artifacts/tables](artifacts/tables/) — итоговые таблицы.
- [artifacts/figures](artifacts/figures/) — графики для отчёта.
- [artifacts/metrics](artifacts/metrics/) — сохранённые метрики экспериментов.

## Быстрый запуск

Prepare the fixed MIRAGE evaluation sample and ranker contexts:

```bash
python scripts/prepare_data.py --sample-size 1000 --seed 42
python scripts/prepare_data.py --sample-size 1000 --seed 42 --limit 50
```

Run metric self-checks:

```bash
python scripts/evaluate_predictions.py --self-test
```

Run QA inference smoke tests after model access is available:

```bash
python3 scripts/run_llm_qa.py --experiment gemma_closed_book --limit 10 --local-files-only
TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 python3.10 scripts/run_extractive_qa.py \
  --experiment roberta_oracle \
  --model models/distilbert-base-cased-distilled-squad \
  --limit 10 \
  --local-files-only
```

Evaluate a prediction JSONL:

```bash
python scripts/evaluate_predictions.py \
  --predictions artifacts/predictions/example.jsonl \
  --output artifacts/metrics/example_metrics.json
```

All full experiments should use `artifacts/data/mirage_sample_1000_qids.txt`.
