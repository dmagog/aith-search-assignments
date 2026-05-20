from __future__ import annotations

import random
import re
from collections import Counter
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from pymorphy3 import MorphAnalyzer

matplotlib.use("Agg")
plt.style.use("ggplot")
pd.set_option("display.max_colwidth", 200)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "additional_data" / "authors"
OUT_DIR = ROOT / "artifacts" / "additional_authors"
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

TOLSTOY_URLS = {
    "vol1": "http://az.lib.ru/t/tolstoj_lew_nikolaewich/text_0040.shtml",
    "vol2": "http://az.lib.ru/t/tolstoj_lew_nikolaewich/text_0050.shtml",
    "vol3": "http://az.lib.ru/t/tolstoj_lew_nikolaewich/text_0060.shtml",
    "vol4": "http://az.lib.ru/t/tolstoj_lew_nikolaewich/text_0070.shtml",
}

DOSTOEVSKY_URLS = {
    "part1": "http://az.lib.ru/d/dostoewskij_f_m/text_0100.shtml",
    "part2": "http://az.lib.ru/d/dostoewskij_f_m/text_0110.shtml",
    "part3": "http://az.lib.ru/d/dostoewskij_f_m/text_0120.shtml",
    "part4": "http://az.lib.ru/d/dostoewskij_f_m/text_0130.shtml",
}

TOKEN_PATTERN = re.compile(r"[а-яё]+", flags=re.IGNORECASE)
NOISE_TOKENS = {"стр", "строка", "изд", "чт", "сноске", "гл", "ч"}
MORPH = MorphAnalyzer()


def download_file(url: str, target_path: Path) -> Path:
    if target_path.exists():
        return target_path
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    target_path.write_bytes(response.content)
    return target_path


def extract_book_text(html_path: Path, start_pattern: str, min_start_index: int = 40) -> str:
    html = html_path.read_bytes().decode("cp1251", errors="ignore")
    text = BeautifulSoup(html, "html.parser").get_text("\n")
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    start_idx = next(
        i for i, line in enumerate(lines) if i >= min_start_index and re.search(start_pattern, line)
    )
    end_candidates = [i for i, line in enumerate(lines) if line.startswith("Комментарии:")]
    end_idx = end_candidates[-1]

    return "\n".join(lines[start_idx:end_idx])


def tokenize_russian(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_PATTERN.findall(text)]


def lemmatize_tokens(tokens: list[str]) -> tuple[list[str], dict[str, str]]:
    lemma_map = {token: MORPH.parse(token)[0].normal_form for token in set(tokens)}
    return [lemma_map[token] for token in tokens], lemma_map


def compute_vocab_growth(tokens: list[str], step: int = 5_000) -> pd.DataFrame:
    seen = set()
    xs: list[int] = []
    ys: list[int] = []

    for i, token in enumerate(tokens, start=1):
        seen.add(token)
        if i % step == 0:
            xs.append(i)
            ys.append(len(seen))

    if not xs or xs[-1] != len(tokens):
        xs.append(len(tokens))
        ys.append(len(seen))

    return pd.DataFrame({"tokens_seen": xs, "vocab_size": ys})


def characteristic_words(
    tokens_a: list[str], tokens_b: list[str], top_n: int = 20, min_count: int = 10
) -> tuple[pd.DataFrame, pd.DataFrame]:
    counts_a = Counter(tokens_a)
    counts_b = Counter(tokens_b)
    vocab = set(counts_a) | set(counts_b)
    total_a = sum(counts_a.values())
    total_b = sum(counts_b.values())
    alpha = 1

    rows = []
    for token in vocab:
        if token in NOISE_TOKENS:
            continue
        count_a = counts_a[token]
        count_b = counts_b[token]
        if count_a + count_b < min_count:
            continue
        score = np.log((count_a + alpha) / (total_a + alpha * len(vocab))) - np.log(
            (count_b + alpha) / (total_b + alpha * len(vocab))
        )
        rows.append({"token": token, "score": score, "count_a": count_a, "count_b": count_b})

    df = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    return df.head(top_n), df.tail(top_n).iloc[::-1].reset_index(drop=True)


def basic_stats(tokens: list[str], author: str) -> dict[str, float | int | str]:
    unique_tokens = len(set(tokens))
    return {
        "author": author,
        "tokens": len(tokens),
        "unique_tokens": unique_tokens,
        "type_token_ratio": unique_tokens / len(tokens),
        "avg_token_length": float(np.mean([len(token) for token in tokens])) if tokens else 0.0,
    }


def save_growth_plot(
    tolstoy_df: pd.DataFrame,
    dostoevsky_df: pd.DataFrame,
    output_path: Path,
    title: str,
    tolstoy_label: str,
    dostoevsky_label: str,
) -> None:
    plt.figure(figsize=(10, 6))
    plt.plot(tolstoy_df["tokens_seen"], tolstoy_df["vocab_size"], label=tolstoy_label, linewidth=2.2)
    plt.plot(dostoevsky_df["tokens_seen"], dostoevsky_df["vocab_size"], label=dostoevsky_label, linewidth=2.2)
    plt.xlabel("Число просмотренных токенов")
    plt.ylabel("Размер словаря")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def save_characteristic_plot(df: pd.DataFrame, title: str, output_path: Path) -> None:
    plot_df = df.copy().sort_values("score")
    plt.figure(figsize=(10, 6))
    plt.barh(plot_df["token"], plot_df["score"], color="#4C78A8")
    plt.xlabel("Скор")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def build_summary(
    stats_df: pd.DataFrame,
    lemma_stats_df: pd.DataFrame,
    tolstoy_growth_df: pd.DataFrame,
    dostoevsky_growth_df: pd.DataFrame,
    tolstoy_growth_shuffled_df: pd.DataFrame,
    dostoevsky_growth_shuffled_df: pd.DataFrame,
    tolstoy_top_df: pd.DataFrame,
    dostoevsky_top_df: pd.DataFrame,
    tolstoy_lemma_top_df: pd.DataFrame,
    dostoevsky_lemma_top_df: pd.DataFrame,
) -> str:
    tolstoy_vocab = int(stats_df.loc[stats_df["author"] == "Толстой", "unique_tokens"].iloc[0])
    dostoevsky_vocab = int(stats_df.loc[stats_df["author"] == "Достоевский", "unique_tokens"].iloc[0])
    tolstoy_tokens = int(stats_df.loc[stats_df["author"] == "Толстой", "tokens"].iloc[0])
    dostoevsky_tokens = int(stats_df.loc[stats_df["author"] == "Достоевский", "tokens"].iloc[0])
    tolstoy_unique_lemmas = int(lemma_stats_df.loc[lemma_stats_df["author"] == "Толстой", "unique_lemmas"].iloc[0])
    dostoevsky_unique_lemmas = int(lemma_stats_df.loc[lemma_stats_df["author"] == "Достоевский", "unique_lemmas"].iloc[0])
    tolstoy_reduction_pct = float(lemma_stats_df.loc[lemma_stats_df["author"] == "Толстой", "reduction_pct"].iloc[0])
    dostoevsky_reduction_pct = float(lemma_stats_df.loc[lemma_stats_df["author"] == "Достоевский", "reduction_pct"].iloc[0])
    fmt = lambda n: f"{n:,}".replace(",", " ")

    tolstoy_merged = tolstoy_growth_df.merge(
        tolstoy_growth_shuffled_df, on="tokens_seen", suffixes=("_orig", "_shuf")
    )
    dostoevsky_merged = dostoevsky_growth_df.merge(
        dostoevsky_growth_shuffled_df, on="tokens_seen", suffixes=("_orig", "_shuf")
    )
    tolstoy_merged["gap"] = tolstoy_merged["vocab_size_shuf"] - tolstoy_merged["vocab_size_orig"]
    dostoevsky_merged["gap"] = dostoevsky_merged["vocab_size_shuf"] - dostoevsky_merged["vocab_size_orig"]

    tolstoy_max_gap = int(tolstoy_merged["gap"].max())
    dostoevsky_max_gap = int(dostoevsky_merged["gap"].max())
    tolstoy_gap_at_100k = int(
        tolstoy_merged.loc[tolstoy_merged["tokens_seen"].sub(100_000).abs().idxmin(), "gap"]
    )
    dostoevsky_gap_at_100k = int(
        dostoevsky_merged.loc[dostoevsky_merged["tokens_seen"].sub(100_000).abs().idxmin(), "gap"]
    )

    return "\n".join(
        [
            "# Дополнительная задача: Толстой и Достоевский",
            "",
            "## Ключевые результаты",
            "",
            f"- Толстой: {fmt(tolstoy_tokens)} токенов, {fmt(tolstoy_vocab)} уникальных слов.",
            f"- Достоевский: {fmt(dostoevsky_tokens)} токенов, {fmt(dostoevsky_vocab)} уникальных слов.",
            f"- В выбранных текстах словарь Толстого больше на {fmt(tolstoy_vocab - dostoevsky_vocab)} слов.",
            f"- После лемматизации словарь Толстого сокращается до {fmt(tolstoy_unique_lemmas)} лемм ({tolstoy_reduction_pct:.2f}% к словоформам), а словарь Достоевского — до {fmt(dostoevsky_unique_lemmas)} лемм ({dostoevsky_reduction_pct:.2f}%).",
            f"- На отметке около 100 000 токенов shuffled-версия дает словарь больше на {tolstoy_gap_at_100k} слов у Толстого и на {dostoevsky_gap_at_100k} слов у Достоевского.",
            f"- Максимальный разрыв между shuffled и исходной кривой достигает {tolstoy_max_gap} слов у Толстого и {dostoevsky_max_gap} слов у Достоевского.",
            "",
            "## Интерпретация",
            "",
            "- Исходные кривые роста показывают, что новые слова появляются неравномерно: на это влияет структура романа, смена сцен, персонажей и тематических блоков.",
            "- После перемешивания исчезает локальная тематическая концентрация слов, поэтому рост словаря становится ближе к усредненному случаю.",
            "- Список характерных слов хорошо отделяет лексику авторов: у Толстого чаще проявляются слова, связанные с военной и бытовой сценой, у Достоевского сильнее заметна философская и психологическая лексика.",
            "",
            "## Примеры характерных слов",
            "",
            f"- Толстой: {', '.join(tolstoy_top_df['token'].head(10))}.",
            f"- Достоевский: {', '.join(dostoevsky_top_df['token'].head(10))}.",
            "",
            "## Примеры характерных лемм",
            "",
            f"- Толстой: {', '.join(tolstoy_lemma_top_df['token'].head(10))}.",
            f"- Достоевский: {', '.join(dostoevsky_lemma_top_df['token'].head(10))}.",
        ]
    )


def main() -> None:
    tolstoy_paths = {
        name: download_file(url, DATA_DIR / f"tolstoy_{name}.shtml") for name, url in TOLSTOY_URLS.items()
    }
    dostoevsky_paths = {
        name: download_file(url, DATA_DIR / f"dostoevsky_{name}.shtml") for name, url in DOSTOEVSKY_URLS.items()
    }

    tolstoy_text = "\n".join(
        extract_book_text(path, r"^ТОМ [А-ЯЁ]+$|^ЧАСТЬ [А-ЯЁ]+\.$") for _, path in tolstoy_paths.items()
    )
    dostoevsky_text = "\n".join(
        extract_book_text(path, r"^ОТ АВТОРА\.$|^ЧАСТЬ [А-ЯЁ]+\.$|^КНИГА [А-ЯЁ]+\.$")
        for _, path in dostoevsky_paths.items()
    )

    tolstoy_tokens = tokenize_russian(tolstoy_text)
    dostoevsky_tokens = tokenize_russian(dostoevsky_text)
    tolstoy_lemma_tokens, tolstoy_lemma_map = lemmatize_tokens(tolstoy_tokens)
    dostoevsky_lemma_tokens, dostoevsky_lemma_map = lemmatize_tokens(dostoevsky_tokens)

    stats_df = pd.DataFrame(
        [basic_stats(tolstoy_tokens, "Толстой"), basic_stats(dostoevsky_tokens, "Достоевский")]
    )
    stats_df.to_csv(OUT_DIR / "authors_basic_stats.csv", index=False)

    lemma_stats_df = pd.DataFrame(
        [
            {
                "author": "Толстой",
                "tokens": len(tolstoy_tokens),
                "unique_wordforms": len(set(tolstoy_tokens)),
                "unique_lemmas": len(set(tolstoy_lemma_tokens)),
                "reduction_abs": len(set(tolstoy_tokens)) - len(set(tolstoy_lemma_tokens)),
                "reduction_pct": (len(set(tolstoy_tokens)) - len(set(tolstoy_lemma_tokens))) / len(set(tolstoy_tokens)) * 100,
            },
            {
                "author": "Достоевский",
                "tokens": len(dostoevsky_tokens),
                "unique_wordforms": len(set(dostoevsky_tokens)),
                "unique_lemmas": len(set(dostoevsky_lemma_tokens)),
                "reduction_abs": len(set(dostoevsky_tokens)) - len(set(dostoevsky_lemma_tokens)),
                "reduction_pct": (len(set(dostoevsky_tokens)) - len(set(dostoevsky_lemma_tokens))) / len(set(dostoevsky_tokens)) * 100,
            },
        ]
    )
    lemma_stats_df.to_csv(OUT_DIR / "authors_lemmatized_stats.csv", index=False)

    lemma_preview_df = pd.DataFrame(
        [
            {"wordform": token, "lemma": tolstoy_lemma_map[token], "author": "Толстой"}
            for token in sorted(list(set(tolstoy_tokens)))[:25]
        ]
        + [
            {"wordform": token, "lemma": dostoevsky_lemma_map[token], "author": "Достоевский"}
            for token in sorted(list(set(dostoevsky_tokens)))[:25]
        ]
    )
    lemma_preview_df.to_csv(OUT_DIR / "authors_lemma_preview.csv", index=False)

    tolstoy_growth_df = compute_vocab_growth(tolstoy_tokens)
    dostoevsky_growth_df = compute_vocab_growth(dostoevsky_tokens)
    tolstoy_growth_df.to_csv(OUT_DIR / "tolstoy_vocab_growth.csv", index=False)
    dostoevsky_growth_df.to_csv(OUT_DIR / "dostoevsky_vocab_growth.csv", index=False)

    rng = random.Random(42)
    tolstoy_tokens_shuffled = tolstoy_tokens.copy()
    dostoevsky_tokens_shuffled = dostoevsky_tokens.copy()
    rng.shuffle(tolstoy_tokens_shuffled)
    rng.shuffle(dostoevsky_tokens_shuffled)

    tolstoy_growth_shuffled_df = compute_vocab_growth(tolstoy_tokens_shuffled)
    dostoevsky_growth_shuffled_df = compute_vocab_growth(dostoevsky_tokens_shuffled)
    tolstoy_growth_shuffled_df.to_csv(OUT_DIR / "tolstoy_vocab_growth_shuffled.csv", index=False)
    dostoevsky_growth_shuffled_df.to_csv(OUT_DIR / "dostoevsky_vocab_growth_shuffled.csv", index=False)

    tolstoy_top_df, dostoevsky_top_df = characteristic_words(tolstoy_tokens, dostoevsky_tokens)
    tolstoy_top_df.to_csv(OUT_DIR / "tolstoy_characteristic_words.csv", index=False)
    dostoevsky_top_df.to_csv(OUT_DIR / "dostoevsky_characteristic_words.csv", index=False)
    tolstoy_lemma_top_df, dostoevsky_lemma_top_df = characteristic_words(tolstoy_lemma_tokens, dostoevsky_lemma_tokens)
    tolstoy_lemma_top_df.to_csv(OUT_DIR / "tolstoy_characteristic_lemmas.csv", index=False)
    dostoevsky_lemma_top_df.to_csv(OUT_DIR / "dostoevsky_characteristic_lemmas.csv", index=False)

    save_growth_plot(
        tolstoy_growth_df,
        dostoevsky_growth_df,
        OUT_DIR / "authors_growth_original.png",
        "Рост словаря в исходных текстах",
        "Толстой",
        "Достоевский",
    )
    save_growth_plot(
        tolstoy_growth_shuffled_df,
        dostoevsky_growth_shuffled_df,
        OUT_DIR / "authors_growth_shuffled.png",
        "Рост словаря после перемешивания",
        "Толстой shuffled",
        "Достоевский shuffled",
    )
    save_characteristic_plot(
        tolstoy_top_df,
        "Наиболее характерные слова Толстого",
        OUT_DIR / "tolstoy_characteristic_words.png",
    )
    save_characteristic_plot(
        dostoevsky_top_df,
        "Наиболее характерные слова Достоевского",
        OUT_DIR / "dostoevsky_characteristic_words.png",
    )

    summary = build_summary(
        stats_df,
        lemma_stats_df,
        tolstoy_growth_df,
        dostoevsky_growth_df,
        tolstoy_growth_shuffled_df,
        dostoevsky_growth_shuffled_df,
        tolstoy_top_df,
        dostoevsky_top_df,
        tolstoy_lemma_top_df,
        dostoevsky_lemma_top_df,
    )
    (OUT_DIR / "authors_summary.md").write_text(summary, encoding="utf-8")

    print(stats_df.to_string(index=False))
    print()
    print("Толстой, характерные слова:", ", ".join(tolstoy_top_df["token"].head(10)))
    print("Достоевский, характерные слова:", ", ".join(dostoevsky_top_df["token"].head(10)))
    print()
    print(f"Артефакты сохранены в {OUT_DIR}")


if __name__ == "__main__":
    main()
