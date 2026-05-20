# DZ2

Этот каталог содержит решение `ДЗ2` по курсу IR/Search: основную задачу по `WikIR` и две дополнительные задачи.

## Основная часть

Отчет:

- [report.md](report.md)
- [report.pdf](report.pdf)

Ключевые артефакты:

- [baseline_results_summary.csv](artifacts/tables/baseline_results_summary.csv)
- [bm25_tuning_best_params_summary.csv](artifacts/tables/bm25_tuning_best_params_summary.csv)
- [bm25_default_vs_tuned_summary.csv](artifacts/tables/bm25_default_vs_tuned_summary.csv)
- [assignment_coverage_checklist.csv](artifacts/tables/assignment_coverage_checklist.csv)

Основные скрипты:

- [audit_wikir.py](scripts/audit_wikir.py)
- [run_retrieval.py](scripts/run_retrieval.py)
- [evaluate_run.py](scripts/evaluate_run.py)
- [tune_bm25.py](scripts/tune_bm25.py)

## Дополнительные задачи

### 1. MIRAGE collection

Отчет:

- [report_mirage.md](report_mirage.md)
- [report_mirage.pdf](report_mirage.pdf)

Ключевые артефакты:

- [mirage_collection_summary.csv](artifacts/mirage/tables/mirage_collection_summary.csv)
- [mirage_experiment_summary.csv](artifacts/mirage/tables/mirage_experiment_summary.csv)
- [mirage_source_effects.csv](artifacts/mirage/tables/mirage_source_effects.csv)
- [mirage_query_length_effects.csv](artifacts/mirage/tables/mirage_query_length_effects.csv)
- [mirage_error_cases_bm25_stemmed.csv](artifacts/mirage/tables/mirage_error_cases_bm25_stemmed.csv)

Основные скрипты:

- [prepare_mirage.py](scripts/prepare_mirage.py)
- [run_mirage_retrieval.py](scripts/run_mirage_retrieval.py)
- [analyze_mirage.py](scripts/analyze_mirage.py)
- [analyze_mirage_queries.py](scripts/analyze_mirage_queries.py)

### 2. Tolstoy & Dostoevsky

Отчет:

- [report_tolstoy_dostoevsky.md](report_tolstoy_dostoevsky.md)
- [report_tolstoy_dostoevsky.pdf](report_tolstoy_dostoevsky.pdf)

Ключевые артефакты:

- [paragraph_corpus_summary.csv](artifacts/authors_search/paragraph_corpus_summary.csv)
- [similarity_summary.csv](artifacts/authors_search/similarity_summary.csv)
- [idf_scheme_comparison.csv](artifacts/authors_search/idf_scheme_comparison.csv)
- [manual_cross_book_analysis.csv](artifacts/authors_search/manual_cross_book_analysis.csv)
- [manual_cross_book_category_summary.csv](artifacts/authors_search/manual_cross_book_category_summary.csv)

Основные скрипты:

- [prepare_authors_paragraphs.py](scripts/prepare_authors_paragraphs.py)
- [run_authors_similarity.py](scripts/run_authors_similarity.py)
- [analyze_authors_similarity.py](scripts/analyze_authors_similarity.py)

## Структура каталога

- [scripts](scripts/) — подготовка данных, retrieval, evaluation, analysis
- [artifacts](artifacts/) — таблицы, графики, timing, tuning
- [data](data/) — локальные данные и вспомогательные README

## С чего читать

Если нужен только финальный результат:

1. Основная часть: [report.pdf](report.pdf)
2. Дополнительная `MIRAGE`: [report_mirage.pdf](report_mirage.pdf)
3. Дополнительная `Tolstoy & Dostoevsky`: [report_tolstoy_dostoevsky.pdf](report_tolstoy_dostoevsky.pdf)

Если нужна воспроизводимость:

1. Посмотреть [requirements.txt](requirements.txt)
2. Запускать скрипты из [scripts](scripts/) с опорой на сохраненные CSV в [artifacts](artifacts/)
