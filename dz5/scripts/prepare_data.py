from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DZ5_ROOT = Path(__file__).resolve().parents[1]
SEARCH_ROOT = DZ5_ROOT.parent
ARTIFACTS_DIR = DZ5_ROOT / "artifacts"
DATA_DIR = ARTIFACTS_DIR / "data"
TABLES_DIR = ARTIFACTS_DIR / "tables"

MIRAGE_PARQUET = SEARCH_ROOT / "dz2" / "data" / "mirage" / "train.parquet"
DZ4_RUNS_DIR = SEARCH_ROOT / "dz4" / "artifacts" / "runs"
HA4_RUNS = {
    "ha4_mixture": DZ4_RUNS_DIR / "mirage_test_mixture_alpha_0.1.trec",
    "ha4_dense": DZ4_RUNS_DIR / "mirage_dense_all-MiniLM-L6-v2.trec",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare fixed MIRAGE sample and HA4 contexts for HA5.")
    parser.add_argument("--sample-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, help="Write a small smoke subset after the fixed sample is selected.")
    parser.add_argument("--top-k", type=int, default=5, help="How many ranker passages to attach.")
    parser.add_argument("--output-prefix", default="mirage_eval")
    return parser.parse_args()


def ensure_dirs() -> None:
    for path in (ARTIFACTS_DIR, DATA_DIR, TABLES_DIR):
        path.mkdir(parents=True, exist_ok=True)


def passage_id(title: str, body: str) -> str:
    digest = hashlib.sha1(f"{title}\n{body}".encode("utf-8")).hexdigest()[:16]
    return f"mirage-{digest}"


def passage_text(title: str, body: str) -> str:
    title = str(title).strip()
    body = str(body).strip()
    if not title:
        return body
    if not body:
        return title
    return f"{title}. {body}"


def to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return value


def answer_aliases(value: Any) -> list[str]:
    value = to_jsonable(value)
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    text = str(value).strip()
    return [text] if text else []


def stratified_mirage_split(
    query_meta: pd.DataFrame,
    *,
    test_fraction: float = 0.2,
    validation_fraction: float = 0.1,
    random_seed: int = 42,
) -> tuple[set[str], set[str], set[str]]:
    rng = np.random.default_rng(random_seed)
    train_ids: set[str] = set()
    validation_ids: set[str] = set()
    test_ids: set[str] = set()

    for _, group in query_meta.groupby("source", sort=True):
        qids = group["query_id"].astype(str).to_numpy()
        shuffled = rng.permutation(qids)
        test_count = max(1, int(round(len(shuffled) * test_fraction)))
        test_part = shuffled[:test_count]
        remaining = shuffled[test_count:]
        validation_count = max(1, int(round(len(remaining) * validation_fraction))) if len(remaining) > 1 else 0
        validation_part = remaining[:validation_count]
        train_part = remaining[validation_count:]
        test_ids.update(map(str, test_part))
        validation_ids.update(map(str, validation_part))
        train_ids.update(map(str, train_part))

    return train_ids, validation_ids, test_ids


def read_trec_run(path: Path, qids: set[str] | None = None, top_k: int = 5) -> dict[str, list[dict[str, Any]]]:
    if not path.exists():
        raise FileNotFoundError(path)

    rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.split()
            if len(parts) < 6:
                continue
            qid, _, doc_id, rank, score, run_id = parts[:6]
            if qids is not None and qid not in qids:
                continue
            rank_i = int(rank)
            if rank_i > top_k:
                continue
            rows[qid].append(
                {
                    "doc_id": doc_id,
                    "rank": rank_i,
                    "score": float(score),
                    "run_id": run_id,
                }
            )

    for qid in rows:
        rows[qid].sort(key=lambda item: (item["rank"], -item["score"], item["doc_id"]))
    return dict(rows)


def read_trec_qids(path: Path) -> set[str]:
    qids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                qids.add(line.split()[0])
    return qids


def stratified_sample_qids(
    frame: pd.DataFrame,
    eligible_qids: set[str],
    *,
    sample_size: int,
    seed: int,
) -> list[str]:
    eligible = frame[frame["query_id"].astype(str).isin(eligible_qids)][["query_id", "source"]].copy()
    eligible["query_id"] = eligible["query_id"].astype(str)
    if sample_size >= len(eligible):
        selected = set(eligible["query_id"].tolist())
        return [str(qid) for qid in frame["query_id"].astype(str).tolist() if qid in selected]

    rng = np.random.default_rng(seed)
    group_sizes = eligible.groupby("source")["query_id"].size().sort_index()
    raw_counts = group_sizes / int(group_sizes.sum()) * sample_size
    counts = np.floor(raw_counts).astype(int)
    remainder = sample_size - int(counts.sum())
    if remainder > 0:
        fractions = (raw_counts - counts).sort_values(ascending=False)
        for source in fractions.index[:remainder]:
            counts[source] += 1

    selected_qids: set[str] = set()
    for source, count in counts.items():
        source_qids = sorted(eligible.loc[eligible["source"] == source, "query_id"].tolist())
        if count <= 0:
            continue
        chosen = rng.choice(source_qids, size=min(int(count), len(source_qids)), replace=False)
        selected_qids.update(map(str, chosen))

    return [str(qid) for qid in frame["query_id"].astype(str).tolist() if qid in selected_qids]


def build_examples(frame: pd.DataFrame) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    examples: dict[str, dict[str, Any]] = {}
    passages: dict[str, dict[str, Any]] = {}

    for row in frame.itertuples(index=False):
        qid = str(row.query_id)
        mixed_contexts: list[dict[str, Any]] = []
        doc_pool = to_jsonable(row.doc_pool)
        for title, body, support in zip(
            doc_pool["doc_name"],
            doc_pool["doc_chunk"],
            doc_pool["support"],
            strict=True,
        ):
            title_s = str(title).strip()
            body_s = str(body).strip()
            pid = passage_id(title_s, body_s)
            text = passage_text(title_s, body_s)
            support_i = int(support)
            passages.setdefault(pid, {"doc_id": pid, "title": title_s, "text": text})
            mixed_contexts.append(
                {
                    "doc_id": pid,
                    "title": title_s,
                    "text": text,
                    "support": support_i,
                }
            )

        oracle = to_jsonable(row.oracle)
        oracle_title = str(oracle["doc_name"]).strip()
        oracle_body = str(oracle["doc_chunk"]).strip()
        oracle_pid = passage_id(oracle_title, oracle_body)
        oracle_text = passage_text(oracle_title, oracle_body)
        passages.setdefault(oracle_pid, {"doc_id": oracle_pid, "title": oracle_title, "text": oracle_text})

        examples[qid] = {
            "query_id": qid,
            "source": str(row.source),
            "question": str(row.query).strip(),
            "answers": answer_aliases(row.answer),
            "oracle_doc_id": oracle_pid,
            "oracle_context": oracle_text,
            "mixed_contexts": mixed_contexts,
            "ranker_contexts": {},
        }

    return examples, passages


def attach_ranker_contexts(
    examples: dict[str, dict[str, Any]],
    passages: dict[str, dict[str, Any]],
    *,
    selected_qids: list[str],
    top_k: int,
) -> dict[str, int]:
    selected = set(selected_qids)
    missing_counts: dict[str, int] = {}
    for run_name, path in HA4_RUNS.items():
        run = read_trec_run(path, qids=selected, top_k=top_k)
        missing = 0
        for qid in selected_qids:
            docs: list[dict[str, Any]] = []
            for item in run.get(qid, []):
                passage = passages.get(item["doc_id"])
                if passage is None:
                    missing += 1
                    continue
                docs.append(
                    {
                        "doc_id": item["doc_id"],
                        "rank": item["rank"],
                        "score": item["score"],
                        "title": passage["title"],
                        "text": passage["text"],
                    }
                )
            examples[qid]["ranker_contexts"][run_name] = {
                "top1": docs[:1],
                "top5": docs[:5],
            }
        missing_counts[run_name] = missing
    return missing_counts


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_summary(
    *,
    frame: pd.DataFrame,
    selected_qids: list[str],
    eligible_qids: set[str],
    output_path: Path,
    missing_counts: dict[str, int],
    args: argparse.Namespace,
) -> None:
    selected_frame = frame[frame["query_id"].astype(str).isin(set(selected_qids))].copy()
    source_counts = selected_frame["source"].value_counts().sort_index()
    summary_rows = [
        {"metric": "mirage_rows", "value": int(len(frame))},
        {"metric": "eligible_test_qids", "value": int(len(eligible_qids))},
        {"metric": "sample_qids", "value": int(len(selected_qids))},
        {"metric": "written_examples", "value": int(args.limit or len(selected_qids))},
        {"metric": "seed", "value": int(args.seed)},
    ]
    for name, count in missing_counts.items():
        summary_rows.append({"metric": f"missing_ranker_passages_{name}", "value": int(count)})
    for source, count in source_counts.items():
        summary_rows.append({"metric": f"sample_source_{source}", "value": int(count)})
    pd.DataFrame(summary_rows).to_csv(output_path, index=False)


def main() -> None:
    args = parse_args()
    ensure_dirs()

    frame = pd.read_parquet(MIRAGE_PARQUET)
    frame["query_id"] = frame["query_id"].astype(str)
    _, _, test_ids = stratified_mirage_split(frame[["query_id", "source", "query"]], random_seed=42)

    run_qids: set[str] | None = None
    for path in HA4_RUNS.values():
        current = read_trec_qids(path)
        run_qids = current if run_qids is None else run_qids & current
    if run_qids is None:
        raise RuntimeError("No HA4 runs configured.")
    eligible_qids = test_ids & run_qids

    selected_qids = stratified_sample_qids(frame, eligible_qids, sample_size=args.sample_size, seed=args.seed)
    if len(selected_qids) != min(args.sample_size, len(eligible_qids)):
        raise RuntimeError(f"Expected {args.sample_size} selected qids, got {len(selected_qids)}.")

    qids_path = DATA_DIR / f"mirage_sample_{len(selected_qids)}_qids.txt"
    qids_path.write_text("\n".join(selected_qids) + "\n", encoding="utf-8")

    examples, passages = build_examples(frame)
    missing_counts = attach_ranker_contexts(examples, passages, selected_qids=selected_qids, top_k=args.top_k)

    output_qids = selected_qids[: args.limit] if args.limit else selected_qids
    suffix = f"sample_{len(selected_qids)}"
    if args.limit:
        suffix += f"_limit_{args.limit}"
    output_path = DATA_DIR / f"{args.output_prefix}_{suffix}.jsonl"
    output_rows = [examples[qid] for qid in output_qids]
    write_jsonl(output_path, output_rows)

    summary_path = TABLES_DIR / f"data_summary_{suffix}.csv"
    write_summary(
        frame=frame,
        selected_qids=selected_qids,
        eligible_qids=eligible_qids,
        output_path=summary_path,
        missing_counts=missing_counts,
        args=args,
    )

    print(json.dumps(
        {
            "output": str(output_path),
            "qids": str(qids_path),
            "summary": str(summary_path),
            "selected_qids": len(selected_qids),
            "written_examples": len(output_rows),
            "eligible_qids": len(eligible_qids),
            "missing_ranker_passages": missing_counts,
        },
        indent=2,
        ensure_ascii=False,
    ))


if __name__ == "__main__":
    main()
