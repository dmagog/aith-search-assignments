from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLES_DIR = PROJECT_ROOT / "artifacts" / "tables"

BASE_RUNS = [
    "tfidf_original_test",
    "tfidf_stemmed_test",
    "tfidf_lemmatized_test",
    "bm25_original_test_k1_1.5_b_0.75",
    "bm25_stemmed_test_k1_1.5_b_0.75",
    "bm25_lemmatized_test_k1_1.5_b_0.75",
]


def load_base_query_frame() -> pd.DataFrame:
    queries = pd.read_csv(TABLES_DIR / "test_queries_audit.csv")
    qrels = pd.read_csv(TABLES_DIR / "test_qrels_per_query.csv")
    df = queries.merge(qrels, on="query_id", how="left")
    df["query_id"] = df["query_id"].astype(str)
    df["length_bucket"] = pd.cut(
        df["length_words"],
        bins=[0, 1, 2, 10],
        labels=["1 word", "2 words", "3+ words"],
        include_lowest=True,
    )
    df["relevant_docs_bucket"] = pd.cut(
        df["relevant_docs"],
        bins=[0, 10, 20, 1000],
        labels=["<=10", "11-20", "21+"],
        include_lowest=True,
    )
    return df


def load_per_query_metrics(run_id: str) -> pd.DataFrame:
    df = pd.read_csv(TABLES_DIR / f"{run_id}_per_query_metrics.csv")
    df["query_id"] = df["query_id"].astype(str)
    pivot = df.pivot(index="query_id", columns="measure", values="value").reset_index()
    renamed = {col: f"{run_id}__{col}" for col in pivot.columns if col != "query_id"}
    return pivot.rename(columns=renamed)


def main() -> None:
    base = load_base_query_frame()
    merged = base.copy()
    for run_id in BASE_RUNS:
        merged = merged.merge(load_per_query_metrics(run_id), on="query_id", how="left")

    merged.to_csv(TABLES_DIR / "test_query_analysis_full.csv", index=False)

    best_run = "bm25_stemmed_test_k1_1.5_b_0.75"
    score_col = f"{best_run}__nDCG@20"

    easy = (
        merged.sort_values(score_col, ascending=False)
        [["query_id", "text", "length_words", "relevant_docs", score_col]]
        .head(15)
        .rename(columns={score_col: "nDCG@20"})
    )
    hard = (
        merged.sort_values(score_col, ascending=True)
        [["query_id", "text", "length_words", "relevant_docs", score_col]]
        .head(15)
        .rename(columns={score_col: "nDCG@20"})
    )
    easy.to_csv(TABLES_DIR / "easy_queries_bm25_stemmed_ndcg20.csv", index=False)
    hard.to_csv(TABLES_DIR / "hard_queries_bm25_stemmed_ndcg20.csv", index=False)

    length_effects = []
    rel_effects = []
    for run_id in BASE_RUNS:
        ap_col = f"{run_id}__AP"
        ndcg_col = f"{run_id}__nDCG@20"

        length_group = (
            merged.groupby("length_bucket", observed=True)[[ap_col, ndcg_col]]
            .mean()
            .reset_index()
            .rename(columns={ap_col: "AP", ndcg_col: "nDCG@20"})
        )
        length_group["run_id"] = run_id
        length_effects.append(length_group)

        rel_group = (
            merged.groupby("relevant_docs_bucket", observed=True)[[ap_col, ndcg_col]]
            .mean()
            .reset_index()
            .rename(columns={ap_col: "AP", ndcg_col: "nDCG@20"})
        )
        rel_group["run_id"] = run_id
        rel_effects.append(rel_group)

    pd.concat(length_effects, ignore_index=True).to_csv(
        TABLES_DIR / "query_length_effects.csv", index=False
    )
    pd.concat(rel_effects, ignore_index=True).to_csv(
        TABLES_DIR / "relevant_docs_effects.csv", index=False
    )

    correlations = []
    for run_id in BASE_RUNS:
        ap_col = f"{run_id}__AP"
        ndcg_col = f"{run_id}__nDCG@20"
        correlations.append(
            {
                "run_id": run_id,
                "corr_length_ap": merged["length_words"].corr(merged[ap_col], method="spearman"),
                "corr_length_ndcg20": merged["length_words"].corr(merged[ndcg_col], method="spearman"),
                "corr_relevant_docs_ap": merged["relevant_docs"].corr(merged[ap_col], method="spearman"),
                "corr_relevant_docs_ndcg20": merged["relevant_docs"].corr(merged[ndcg_col], method="spearman"),
            }
        )
    pd.DataFrame(correlations).to_csv(TABLES_DIR / "query_metric_correlations.csv", index=False)

    print(TABLES_DIR)


if __name__ == "__main__":
    main()
