# DZ4 summary

## Dense retrieval

| Dataset | Model | P@1 | P@10 | P@20 | MAP@20 | nDCG@20 | Avg query seconds |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| WikIR test | all-MiniLM-L6-v2 | 0.6500 | 0.2430 | 0.1705 | 0.1929 | 0.4705 | 0.0016 |
| WikIR test | paraphrase-multilingual-MiniLM-L12-v2 | 0.4400 | 0.1300 | 0.0900 | 0.0952 | 0.2757 | 0.0019 |
| MIRAGE test | all-MiniLM-L6-v2 | 0.5922 | 0.1129 | 0.0582 | 0.7039 | 0.7694 | 0.0005 |
| MIRAGE test | paraphrase-multilingual-MiniLM-L12-v2 | 0.5109 | 0.0927 | 0.0491 | 0.5841 | 0.6469 | 0.0005 |

## Re-ranking

| Dataset | Model | k | P@1 | P@10 | P@20 | MAP@20 | nDCG@20 | Avg query seconds |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| WikIR test | all-MiniLM-L6-v2 | 10 | 0.4800 | 0.1770 | 0.1295 | 0.1579 | 0.3413 | 0.0023 |
| WikIR test | all-MiniLM-L6-v2 | 20 | 0.5300 | 0.2050 | 0.1295 | 0.1686 | 0.3626 | 0.0013 |
| WikIR test | all-MiniLM-L6-v2 | 50 | 0.5300 | 0.2120 | 0.1365 | 0.1711 | 0.3721 | 0.0033 |
| WikIR test | all-MiniLM-L6-v2 | 100 | 0.5300 | 0.2250 | 0.1465 | 0.1807 | 0.3910 | 0.0014 |
| MIRAGE test | all-MiniLM-L6-v2 | 10 | 0.4686 | 0.0742 | 0.0391 | 0.5158 | 0.5524 | 0.0000 |
| MIRAGE test | all-MiniLM-L6-v2 | 20 | 0.4812 | 0.0781 | 0.0391 | 0.5316 | 0.5652 | 0.0000 |
| MIRAGE test | all-MiniLM-L6-v2 | 50 | 0.5030 | 0.0839 | 0.0423 | 0.5638 | 0.6011 | 0.0004 |
| MIRAGE test | all-MiniLM-L6-v2 | 100 | 0.5102 | 0.0859 | 0.0435 | 0.5755 | 0.6145 | 0.0006 |

## Mixture model

Selected alpha on WikIR train: `0.1` using `nDCG@20`.

| Dataset | Model | P@1 | P@10 | P@20 | MAP@20 | nDCG@20 | Avg query seconds |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| WikIR test | all-MiniLM-L6-v2 | 0.6800 | 0.2620 | 0.1795 | 0.2105 | 0.4900 | 0.000014 |
| MIRAGE test | all-MiniLM-L6-v2 | 0.6134 | 0.1141 | 0.0585 | 0.7194 | 0.7820 | 0.000012 |

## Previous best references

- DZ2 WikIR best BM25: `bm25_stemmed_test_k1_1.5_b_0.75`
- DZ2 MIRAGE best lexical baseline: `mirage_bm25_stemmed_k1_1.5_b_0.75`
- DZ3 WikIR best LTR: `CatBoost с расширенными признаками (PairLogit)`
- DZ3 MIRAGE best LTR: `Модель с признаками заголовка, текста и сигналами Wikipedia`

## Notes

- MIRAGE test split is reproduced via the same stratified split by `source` as in DZ3.
- The optional cross-encoder fine-tuning task is not part of the current implementation.
