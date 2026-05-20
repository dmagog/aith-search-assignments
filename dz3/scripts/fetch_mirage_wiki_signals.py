from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Сбор внешних wiki-сигналов для MIRAGE.")
    parser.add_argument(
        "--parquet-path",
        type=Path,
        default=Path("../dz2/data/mirage/train.parquet"),
        help="Путь к parquet-файлу MIRAGE.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("artifacts/mirage_ltr/wiki_signals.csv"),
        help="CSV-файл с сигналами по страницам.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=12,
        help="Число параллельных запросов.",
    )
    return parser.parse_args()


def month_window() -> tuple[str, str]:
    today = date.today()
    month = today.month - 1
    year = today.year
    if month == 0:
        month = 12
        year -= 1

    end_year = year
    end_month = month
    start_year = end_year
    start_month = end_month - 11
    while start_month <= 0:
        start_month += 12
        start_year -= 1

    return f"{start_year:04d}{start_month:02d}0100", f"{end_year:04d}{end_month:02d}0100"


def iter_titles(parquet_path: Path) -> list[str]:
    df = pd.read_parquet(parquet_path)
    titles: set[str] = set()
    for row in df.itertuples(index=False):
        titles.add(str(row.doc_name).strip())
        for title in row.doc_pool["doc_name"]:
            titles.add(str(title).strip())
        titles.add(str(row.oracle["doc_name"]).strip())
    return sorted(title for title in titles if title)


def fetch_one(title: str, start_ts: str, end_ts: str) -> dict[str, object]:
    article = quote(title.replace(" ", "_"), safe="")
    pageviews_url = (
        "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
        f"en.wikipedia.org/all-access/user/{article}/monthly/{start_ts}/{end_ts}"
    )
    linkcount_url = f"https://linkcount.toolforge.org/api/?page={article}&project=en.wikipedia.org"

    pageviews_sum = 0
    pageviews_mean = 0.0
    incoming_links = 0
    redirects = 0
    status = "ok"

    try:
        pageviews_response = requests.get(pageviews_url, timeout=30)
        if pageviews_response.ok:
            items = pageviews_response.json().get("items", [])
            views = [int(item.get("views", 0)) for item in items]
            pageviews_sum = int(sum(views))
            pageviews_mean = float(sum(views) / len(views)) if views else 0.0
        else:
            status = f"pageviews_http_{pageviews_response.status_code}"
    except Exception as exc:  # noqa: BLE001
        status = f"pageviews_error:{type(exc).__name__}"

    try:
        linkcount_response = requests.get(linkcount_url, timeout=30)
        if linkcount_response.ok:
            payload = linkcount_response.json()
            incoming_links = int(((payload.get("wikilinks") or {}).get("all")) or 0)
            redirects = int(payload.get("redirects") or 0)
        else:
            status = status if status != "ok" else f"linkcount_http_{linkcount_response.status_code}"
    except Exception as exc:  # noqa: BLE001
        if status == "ok":
            status = f"linkcount_error:{type(exc).__name__}"

    return {
        "title": title,
        "pageviews_sum_12m": pageviews_sum,
        "pageviews_mean_12m": pageviews_mean,
        "incoming_links": incoming_links,
        "redirects": redirects,
        "status": status,
    }


def main() -> None:
    args = parse_args()
    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    titles = iter_titles(args.parquet_path)
    existing = {}
    if args.output_path.exists():
        cached = pd.read_csv(args.output_path)
        existing = {str(row["title"]): row for _, row in cached.iterrows()}

    pending = [title for title in titles if title not in existing]
    start_ts, end_ts = month_window()

    rows = [dict(row) for row in existing.values()]
    if pending:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(fetch_one, title, start_ts, end_ts): title for title in pending}
            for future in as_completed(futures):
                rows.append(future.result())

    output_df = pd.DataFrame(rows).sort_values("title").reset_index(drop=True)
    output_df.to_csv(args.output_path, index=False)
    print(
        {
            "titles_total": len(titles),
            "cached": len(existing),
            "fetched_now": len(pending),
            "output_path": str(args.output_path),
        }
    )


if __name__ == "__main__":
    main()
