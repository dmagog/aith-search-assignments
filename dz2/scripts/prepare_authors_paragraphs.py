from __future__ import annotations

import csv
import html
import json
import re
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT.parent / "dz1" / "additional_data" / "authors"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "authors_search"

BOOK_SOURCES = {
    "tolstoy": {
        "book": "War and Peace",
        "files": [
            "tolstoy_vol1.shtml",
            "tolstoy_vol2.shtml",
            "tolstoy_vol3.shtml",
            "tolstoy_vol4.shtml",
        ],
        "start_markers": [
            "ТОМ ПЕРВЫЙ",
            "ТОМ ВТОРОЙ",
            "ТОМ ТРЕТИЙ",
            "ТОМ ЧЕТВЕРТЫЙ",
        ],
    },
    "dostoevsky": {
        "book": "The Brothers Karamazov",
        "files": [
            "dostoevsky_part1.shtml",
            "dostoevsky_part2.shtml",
            "dostoevsky_part3.shtml",
            "dostoevsky_part4.shtml",
        ],
        "start_markers": [
            "Братья Карамазовы",
            "ОТ АВТОРА.",
        ],
    },
}

TOKEN_RE = re.compile(r"[A-Za-zА-Яа-яЁё]+")
SPACE_RE = re.compile(r"\s+")
TAG_RE = re.compile(r"<[^>]+>")
SUP_RE = re.compile(r"<sup.*?>.*?</sup>", re.IGNORECASE | re.DOTALL)
BR_RE = re.compile(r"<br\s*/?>", re.IGNORECASE)
BAD_TEXT_PATTERNS = [
    "комментарии:",
    "обновлено:",
    "ваша оценка",
    "связаться с программистом сайта",
    "статистика.",
    "lib.ru",
    "readingtolstoy.ru",
    "tolstoy.ru",
    "az.lib.ru",
    "список конъектур",
    "см. т.",
    " р. в.",
    "изд.",
]
BAD_SECTION_PATTERNS = [
    "ПРЕДИСЛОВИЕ",
    "ПРИМЕЧАНИЯ",
    "КОММЕНТАРИИ",
]


@dataclass
class ParagraphRow:
    paragraph_id: str
    author: str
    book: str
    source_file: str
    section: str
    paragraph_index: int
    text: str
    token_count: int
    char_count: int


def normalize_text(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = SPACE_RE.sub(" ", text)
    return text.strip()


def token_count(text: str) -> int:
    return len(TOKEN_RE.findall(text))


def block_to_text(block: str) -> str:
    block = SUP_RE.sub(" ", block)
    block = BR_RE.sub(" ", block)
    text = TAG_RE.sub(" ", block)
    text = html.unescape(text)
    return normalize_text(text)


def iter_blocks(html: str) -> list[tuple[str, str]]:
    blocks: list[tuple[str, str]] = []
    current_tag: str | None = None
    current_lines: list[str] = []

    for line in html.splitlines():
        stripped = line.lstrip()
        next_tag: str | None = None
        if stripped.lower().startswith("<h4"):
            next_tag = "h4"
        elif stripped.lower().startswith("<dd"):
            next_tag = "dd"

        if next_tag is not None:
            if current_tag is not None and current_lines:
                blocks.append((current_tag, "\n".join(current_lines)))
            current_tag = next_tag
            current_lines = [line]
            continue

        if current_tag is not None:
            current_lines.append(line)

    if current_tag is not None and current_lines:
        blocks.append((current_tag, "\n".join(current_lines)))
    return blocks


def is_substantive_paragraph(text: str, current_section: str) -> bool:
    if len(text) < 80:
        return False
    if token_count(text) < 12:
        return False
    lowered = text.lower()
    if any(pattern in lowered for pattern in BAD_TEXT_PATTERNS):
        return False
    if text.startswith(("Вместо", "После слов", "В дополнение", "Главы , соответствующей", "Глава ")):
        return False
    if current_section and any(pattern in current_section.upper() for pattern in BAD_SECTION_PATTERNS):
        return False
    digit_ratio = sum(char.isdigit() for char in text) / max(len(text), 1)
    if digit_ratio > 0.08:
        return False
    return True


def extract_rows_for_author(author: str, config: dict[str, object]) -> list[ParagraphRow]:
    rows: list[ParagraphRow] = []
    current_section = ""
    paragraph_idx = 0
    started = False
    markers = [marker.lower() for marker in config["start_markers"]]
    book = str(config["book"])

    for file_name in config["files"]:
        path = SOURCE_ROOT / file_name
        html = path.read_bytes().decode("cp1251", errors="ignore")
        start_offsets = [html.find(marker) for marker in config["start_markers"] if html.find(marker) >= 0]
        if start_offsets:
            start_pos = min(start_offsets)
            tag_start = max(html.rfind("<h4", 0, start_pos), html.rfind("<dd", 0, start_pos))
            html = html[(tag_start if tag_start >= 0 else start_pos) :]

        for tag_name, block in iter_blocks(html):
            text = block_to_text(block)
            if not text:
                continue

            if not started and any(marker in text.lower() for marker in markers):
                started = True
                if tag_name == "h4":
                    current_section = text
                continue

            if not started:
                continue

            if tag_name == "h4":
                current_section = text
                continue

            if not is_substantive_paragraph(text, current_section):
                continue

            paragraph_idx += 1
            rows.append(
                ParagraphRow(
                    paragraph_id=f"{author}:{Path(file_name).stem}:{paragraph_idx:05d}",
                    author=author,
                    book=book,
                    source_file=file_name,
                    section=current_section,
                    paragraph_index=paragraph_idx,
                    text=text,
                    token_count=token_count(text),
                    char_count=len(text),
                )
            )

    return rows


def write_rows(rows: list[ParagraphRow]) -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    paragraphs_csv = ARTIFACTS_DIR / "paragraphs.csv"
    with paragraphs_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "paragraph_id",
                "author",
                "book",
                "source_file",
                "section",
                "paragraph_index",
                "text",
                "token_count",
                "char_count",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)

    summary: list[dict[str, object]] = []
    for author in sorted({row.author for row in rows}):
        author_rows = [row for row in rows if row.author == author]
        summary.append(
            {
                "author": author,
                "book": author_rows[0].book,
                "paragraphs": len(author_rows),
                "mean_tokens": round(sum(row.token_count for row in author_rows) / len(author_rows), 2),
                "median_tokens_approx": sorted(row.token_count for row in author_rows)[len(author_rows) // 2],
                "min_tokens": min(row.token_count for row in author_rows),
                "max_tokens": max(row.token_count for row in author_rows),
            }
        )

    summary_csv = ARTIFACTS_DIR / "paragraph_corpus_summary.csv"
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "author",
                "book",
                "paragraphs",
                "mean_tokens",
                "median_tokens_approx",
                "min_tokens",
                "max_tokens",
            ],
        )
        writer.writeheader()
        writer.writerows(summary)

    metadata = {
        "paragraphs_csv": str(paragraphs_csv),
        "summary_csv": str(summary_csv),
        "authors": summary,
    }
    with (ARTIFACTS_DIR / "paragraph_corpus_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def main() -> None:
    rows: list[ParagraphRow] = []
    for author, config in BOOK_SOURCES.items():
        rows.extend(extract_rows_for_author(author, config))
    write_rows(rows)
    print(ARTIFACTS_DIR / "paragraphs.csv")


if __name__ == "__main__":
    main()
