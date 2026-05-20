# Раздел 4: обучение ранжированию на MIRAGE

## Разбиение запросов

| Подвыборка | Источник | Число запросов |
| --- | --- | ---: |
| test | drop | 15 |
| test | ifqa | 50 |
| test | naturalqa | 716 |
| test | popqa | 615 |
| test | triviaqa | 117 |
| train | drop | 54 |
| train | ifqa | 178 |
| train | naturalqa | 2576 |
| train | popqa | 2214 |
| train | triviaqa | 420 |
| validation | drop | 6 |
| validation | ifqa | 20 |
| validation | naturalqa | 286 |
| validation | popqa | 246 |
| validation | triviaqa | 47 |

## Качество

| Модель | Лучшая итерация | Validation AP | Validation NDCG@5 | Test AP | Test NDCG@5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Лексическая модель ранжирования | 41 | 0.67175 | 0.75813 | 0.69796 | 0.77774 |
| Модель с признаками заголовка, текста и сигналами Wikipedia | 289 | 0.73069 | 0.80205 | 0.73564 | 0.80551 |
