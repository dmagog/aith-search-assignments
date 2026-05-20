from __future__ import annotations

import csv
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WIKIR_ROOT = PROJECT_ROOT.parent / "dz1" / "wikIR1k" / "wikIR1k"

RUN_PATH = PROJECT_ROOT / "artifacts" / "runs" / "bm25_stemmed_test_k1_1.5_b_0.75.trec"
OUTPUT_PATH = PROJECT_ROOT / "artifacts" / "tables" / "error_cases_bm25_stemmed.csv"

SELECTED = {
    "5453": "lexical false positive",
    "81634": "topic-related but not qrel-positive",
    "15686": "likely disputable qrel / semantically relevant",
}


def main() -> None:
    queries = {}
    with (WIKIR_ROOT / "test" / "queries.csv").open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            queries[row["id_left"]] = row["text_left"]

    positives: dict[str, set[str]] = {}
    with (WIKIR_ROOT / "test" / "qrels").open("r", encoding="utf-8") as f:
        for line in f:
            qid, _, docid, rel = line.split()
            if int(rel) > 0:
                positives.setdefault(qid, set()).add(docid)

    docs = {}
    with (WIKIR_ROOT / "documents.csv").open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            docs[row["id_right"]] = row["text_right"]

    first_bad: dict[str, tuple[int, str, float]] = {}
    with RUN_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            qid, _, docid, rank, score, _ = line.split()
            if qid in first_bad:
                continue
            if qid in SELECTED and docid not in positives.get(qid, set()):
                first_bad[qid] = (int(rank), docid, float(score))

    rows = []
    for qid, note in SELECTED.items():
        rank, docid, score = first_bad[qid]
        rows.append(
            {
                "query_id": qid,
                "query_text": queries[qid],
                "rank": rank,
                "doc_id": docid,
                "score": score,
                "case_type": note,
                "doc_snippet": docs[docid][:400],
            }
        )

    with OUTPUT_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "query_id",
                "query_text",
                "rank",
                "doc_id",
                "score",
                "case_type",
                "doc_snippet",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
