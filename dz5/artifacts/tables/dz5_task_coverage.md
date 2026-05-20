# Проверка покрытия пунктов HA5

| Пункт | Статус | Что сделано |
| --- | --- | --- |
| 1. SLM без контекста, 0-shot | закрыто | Qwen2.5-1.5B-Instruct, 1000 вопросов |
| 2. Экстрактивная QA / MRC | закрыто | deepset/roberta-base-squad2 по oracle-пассажам, 1000 вопросов |
| 3. SLM с oracle-контекстом | закрыто | Qwen2.5-1.5B-Instruct с oracle-контекстом, 1000 вопросов |
| 4. Top-1 контекст из ранжировщиков HA4 | закрыто | top-1 mixture/dense для RoBERTa и Qwen, 1000 вопросов |
| 5. Top-5 контекст из ранжировщиков HA4 | закрыто | top-5 mixture/dense для Qwen + сравнение с MIRAGE mixed, 1000 вопросов |
| 6. Дообучение T5Gemma 2 | закрыто | LoRA-дообучение на SQuAD, оценка без контекста/oracle/top-1 mixture/top-1 dense на 1000 вопросах |
| 7. Оценка LLM-судьей | закрыто | Qwen-судья v2 на 14000 ответов, включая 4000 ответов T5Gemma |
| Метрики EM/F1/MIRAGE | закрыто | SQuAD EM/F1, MIRAGE strict/loose для всех полных прогонов |
| BERTScore | закрыто | BERTScore F1 посчитан для всех 14 полных прогонов с `bert-base-uncased` |
| Таблицы | закрыто | CSV + Markdown-сводка |
| Графики | закрыто | столбчатые диаграммы по SQuAD F1, MIRAGE loose, BERTScore F1, оценке LLM-судьи и доле пустых ответов T5Gemma |
