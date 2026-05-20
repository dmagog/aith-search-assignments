from __future__ import annotations

from pathlib import Path

import ir_measures
import matplotlib.pyplot as plt
import pandas as pd
from ir_measures import AP, P, nDCG


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MIRAGE_DIR = PROJECT_ROOT / "artifacts" / "mirage"
TABLES_DIR = MIRAGE_DIR / "tables"
FIGURES_DIR = MIRAGE_DIR / "figures"
BEST_RUN = MIRAGE_DIR / "runs" / "mirage_bm25_stemmed_k1_1.5_b_0.75.trec"
MEASURES = [P@1, P@10, P@20, AP, nDCG@20]


def load_per_query_metrics() -> pd.DataFrame:
    qrels = ir_measures.read_trec_qrels(str(MIRAGE_DIR / "qrels.trec"))
    run = ir_measures.read_trec_run(str(BEST_RUN))
    rows = [
        {"query_id": metric.query_id, "measure": str(metric.measure), "value": metric.value}
        for metric in ir_measures.iter_calc(MEASURES, qrels, run)
    ]
    df = pd.DataFrame(rows)
    pivot = df.pivot(index="query_id", columns="measure", values="value").reset_index()
    return pivot


def load_top1_docs() -> pd.DataFrame:
    columns = ["query_id", "Q0", "doc_id", "rank", "score", "run_id"]
    run_df = pd.read_csv(BEST_RUN, sep=r"\s+", names=columns)
    return run_df[run_df["rank"] == 1][["query_id", "doc_id", "score"]].rename(
        columns={"doc_id": "top1_doc_id", "score": "top1_score"}
    )


def load_relevance() -> dict[tuple[str, str], int]:
    rows = []
    with (MIRAGE_DIR / "qrels.trec").open("r", encoding="utf-8") as f:
        for line in f:
            qid, _, docid, rel = line.strip().split()
            rows.append((qid, docid, int(rel)))
    return {(qid, docid): rel for qid, docid, rel in rows}


def bucket_query_length(length: int) -> str:
    if length <= 5:
        return "<=5"
    if length <= 10:
        return "6-10"
    return "11+"


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    metrics = load_per_query_metrics()
    audit = pd.read_csv(TABLES_DIR / "mirage_query_audit.csv")
    queries = pd.read_csv(MIRAGE_DIR / "queries.csv")[["query_id", "query", "source"]]
    docs = pd.read_csv(MIRAGE_DIR / "documents.csv")[["doc_id", "doc_name", "text"]]
    top1 = load_top1_docs()
    relevance = load_relevance()

    merged = metrics.merge(audit, on="query_id", how="left").merge(queries, on=["query_id", "source"], how="left")
    merged["length_bucket"] = merged["query_length_words"].map(bucket_query_length)
    merged.to_csv(TABLES_DIR / "mirage_bm25_stemmed_per_query_metrics.csv", index=False)

    easy = merged.sort_values(["AP", "nDCG@20"], ascending=False).head(15)
    hard = merged.sort_values(["AP", "nDCG@20"], ascending=True).head(15)
    easy.to_csv(TABLES_DIR / "mirage_easy_queries_bm25_stemmed.csv", index=False)
    hard.to_csv(TABLES_DIR / "mirage_hard_queries_bm25_stemmed.csv", index=False)

    source_summary = (
        merged.groupby("source", as_index=False)
        .agg(
            queries=("query_id", "count"),
            mean_ap=("AP", "mean"),
            mean_ndcg20=("nDCG@20", "mean"),
            mean_p1=("P@1", "mean"),
            mean_query_length=("query_length_words", "mean"),
        )
        .sort_values("mean_ap", ascending=False)
    )
    source_summary.to_csv(TABLES_DIR / "mirage_source_effects.csv", index=False)

    length_summary = (
        merged.groupby("length_bucket", as_index=False)
        .agg(
            queries=("query_id", "count"),
            mean_ap=("AP", "mean"),
            mean_ndcg20=("nDCG@20", "mean"),
            mean_p1=("P@1", "mean"),
        )
        .sort_values("length_bucket")
    )
    length_summary.to_csv(TABLES_DIR / "mirage_query_length_effects.csv", index=False)

    error_cases = merged.merge(top1, on="query_id", how="left").merge(
        docs, left_on="top1_doc_id", right_on="doc_id", how="left"
    )
    error_cases["top1_relevance"] = error_cases.apply(
        lambda row: relevance.get((row["query_id"], row["top1_doc_id"]), 0), axis=1
    )
    error_cases = error_cases[error_cases["top1_relevance"] == 0].copy()
    error_cases = error_cases.sort_values(["P@1", "AP", "nDCG@20", "top1_score"], ascending=[True, True, True, False])
    error_cases["top1_text_snippet"] = error_cases["text"].fillna("").str.slice(0, 280)
    error_cases[
        [
            "query_id",
            "source",
            "query",
            "query_length_words",
            "relevant_passages",
            "AP",
            "nDCG@20",
            "top1_doc_id",
            "doc_name",
            "top1_score",
            "top1_text_snippet",
        ]
    ].head(15).to_csv(TABLES_DIR / "mirage_error_cases_bm25_stemmed.csv", index=False)

    plt.figure(figsize=(8, 4))
    plt.bar(source_summary["source"], source_summary["mean_ap"], color="#1b9e77")
    plt.title("MIRAGE BM25 + stemmed: средний AP по источникам")
    plt.ylabel("Средний AP")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "mirage_ap_by_source.png", dpi=160)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.bar(length_summary["length_bucket"], length_summary["mean_ap"], color="#7570b3")
    plt.title("MIRAGE BM25 + stemmed: средний AP по длине запроса")
    plt.ylabel("Средний AP")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "mirage_ap_by_query_length.png", dpi=160)
    plt.close()

    print(TABLES_DIR / "mirage_bm25_stemmed_per_query_metrics.csv")


if __name__ == "__main__":
    main()
