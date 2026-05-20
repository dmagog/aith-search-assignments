from __future__ import annotations

from collections import Counter
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")
plt.style.use("ggplot")
pd.set_option("display.max_colwidth", 200)

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MAIN_FIGURES_DIR = FIGURES_DIR / "main"
AUTHORS_FIGURES_DIR = FIGURES_DIR / "authors"
COAT_FIGURES_DIR = FIGURES_DIR / "coat"

for path in [REPORTS_DIR, FIGURES_DIR, MAIN_FIGURES_DIR, AUTHORS_FIGURES_DIR, COAT_FIGURES_DIR]:
    path.mkdir(parents=True, exist_ok=True)


def fig_rel(path: Path) -> str:
    return path.relative_to(REPORTS_DIR).as_posix()


def format_int(value: int) -> str:
    return f"{int(value):,}".replace(",", " ")


def format_float(value: float, digits: int = 4) -> str:
    return f"{float(value):.{digits}f}"


def dataframe_to_markdown(df: pd.DataFrame, round_map: dict[str, int] | None = None) -> str:
    out = df.copy()
    round_map = round_map or {}
    for col, digits in round_map.items():
        if col in out.columns:
            out[col] = out[col].map(lambda x: round(float(x), digits) if pd.notna(x) else x)
    return out.to_markdown(index=False)


def load_main_documents() -> list[list[str]]:
    docs_path = ROOT / "wikIR1k" / "wikIR1k" / "documents.csv"
    df = pd.read_csv(docs_path)
    text_col = "text" if "text" in df.columns else df.columns[-1]
    return [str(text).split() for text in df[text_col].fillna("")]


def build_main_figures() -> dict[str, Path]:
    tokenized_docs = load_main_documents()
    all_tokens = [token for doc in tokenized_docs for token in doc]
    word_counts = Counter(all_tokens)

    top20_words = pd.read_csv(ROOT / "top30_words.csv").head(20)
    stopword_summary = pd.read_csv(ROOT / "stopword_summary.csv")
    top20_bigrams = pd.read_csv(ROOT / "top30_bigrams.csv").head(20)
    heaps_df = pd.read_csv(ROOT / "heaps_points.csv")
    morph_df = pd.read_csv(ROOT / "morph_comparison_stats.csv")

    stopword_occurrences = int(stopword_summary.loc[stopword_summary["metric"] == "stopword_occurrences", "value"].iloc[0])
    total_tokens = int(morph_df.loc[morph_df["Версия"] == "Исходная коллекция", "Размер коллекции в токенах"].iloc[0])

    paths: dict[str, Path] = {}

    path = MAIN_FIGURES_DIR / "top20_words.png"
    plt.figure(figsize=(11, 6))
    plt.bar(top20_words["token"], top20_words["frequency"], color="#4C78A8")
    plt.title("Топ-20 самых частотных слов")
    plt.xlabel("Слово")
    plt.ylabel("Частота")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    paths["top20_words"] = path

    path = MAIN_FIGURES_DIR / "stopword_share.png"
    plt.figure(figsize=(7, 5))
    plt.bar(["Stopwords", "Остальные токены"], [stopword_occurrences, total_tokens - stopword_occurrences], color=["#E45756", "#4C78A8"])
    plt.title("Доля stopwords в коллекции")
    plt.ylabel("Число токенов")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    paths["stopword_share"] = path

    path = MAIN_FIGURES_DIR / "zipf_plot.png"
    frequencies = np.array(sorted(word_counts.values(), reverse=True), dtype=np.int64)
    ranks = np.arange(1, len(frequencies) + 1)
    plt.figure(figsize=(8, 6))
    plt.loglog(ranks, frequencies, linewidth=1.6, color="#4C78A8")
    plt.title("Проверка закона Ципфа")
    plt.xlabel("Ранг слова")
    plt.ylabel("Частота")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    paths["zipf"] = path

    path = MAIN_FIGURES_DIR / "heaps_plot.png"
    plt.figure(figsize=(8, 6))
    plt.loglog(heaps_df["tokens_seen"], heaps_df["vocab_size"], linewidth=2, color="#F58518")
    plt.title("Проверка закона Хипса")
    plt.xlabel("Число просмотренных токенов")
    plt.ylabel("Размер словаря")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    paths["heaps"] = path

    path = MAIN_FIGURES_DIR / "top20_bigrams.png"
    labels = (top20_bigrams["token_1"] + " " + top20_bigrams["token_2"]).tolist()
    plt.figure(figsize=(12, 6))
    plt.bar(labels, top20_bigrams["frequency"], color="#72B7B2")
    plt.title("Топ-20 биграмм по частоте")
    plt.xlabel("Биграмма")
    plt.ylabel("Частота")
    plt.xticks(rotation=55, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    paths["top20_bigrams"] = path

    path = MAIN_FIGURES_DIR / "morph_vocab.png"
    plt.figure(figsize=(8, 5))
    plt.bar(morph_df["Версия"], morph_df["Число уникальных токенов"], color="#54A24B")
    plt.title("Размер словаря после разных способов нормализации")
    plt.ylabel("Число уникальных токенов")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    paths["morph_vocab"] = path

    path = MAIN_FIGURES_DIR / "morph_doc_len.png"
    plt.figure(figsize=(8, 5))
    plt.bar(morph_df["Версия"], morph_df["Средняя длина документа"], color="#B279A2")
    plt.title("Средняя длина документа после нормализации")
    plt.ylabel("Средняя длина документа")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    paths["morph_doc_len"] = path

    return paths


def copy_additional_figures() -> dict[str, dict[str, Path]]:
    mapping = {
        "authors": {
            "growth_original": ROOT / "artifacts" / "additional_authors" / "authors_growth_original.png",
            "growth_shuffled": ROOT / "artifacts" / "additional_authors" / "authors_growth_shuffled.png",
            "tolstoy_words": ROOT / "artifacts" / "additional_authors" / "tolstoy_characteristic_words.png",
            "dostoevsky_words": ROOT / "artifacts" / "additional_authors" / "dostoevsky_characteristic_words.png",
        },
        "coat": {
            "class_distribution": ROOT / "artifacts" / "additional_generated_texts" / "coat_train_class_distribution.png",
            "word_len_hist": ROOT / "artifacts" / "additional_generated_texts" / "coat_train_word_len_hist.png",
            "results_comparison": ROOT / "artifacts" / "additional_generated_texts" / "coat_results_comparison.png",
            "confusion_lexical": ROOT / "artifacts" / "additional_generated_texts" / "coat_confusion_lexical.png",
            "confusion_tfidf": ROOT / "artifacts" / "additional_generated_texts" / "coat_confusion_tfidf.png",
        },
    }
    out: dict[str, dict[str, Path]] = {"authors": {}, "coat": {}}
    for section, items in mapping.items():
        target_dir = AUTHORS_FIGURES_DIR if section == "authors" else COAT_FIGURES_DIR
        for key, src in items.items():
            dst = target_dir / src.name
            dst.write_bytes(src.read_bytes())
            out[section][key] = dst
    return out


def build_main_report(main_figs: dict[str, Path]) -> None:
    morph_df = pd.read_csv(ROOT / "morph_comparison_stats.csv")
    top_words_df = pd.read_csv(ROOT / "top30_words.csv").head(10)
    top_bigrams_df = pd.read_csv(ROOT / "top30_bigrams.csv").head(10)
    selected_bigrams_source = pd.read_csv(ROOT / "selected_bigrams_top50.csv").copy()
    stopword_summary = pd.read_csv(ROOT / "stopword_summary.csv")

    orig = morph_df.loc[morph_df["Версия"] == "Исходная коллекция"].iloc[0]
    porter = morph_df.loc[morph_df["Версия"] == "Стемминг (Porter)"].iloc[0]
    spacy = morph_df.loc[morph_df["Версия"] == "Лемматизация (spaCy)"].iloc[0]
    bert = morph_df.loc[morph_df["Версия"] == "BERT-токенизация"].iloc[0]

    stopword_occurrences = int(stopword_summary.loc[stopword_summary["metric"] == "stopword_occurrences", "value"].iloc[0])
    stopword_share = float(stopword_summary.loc[stopword_summary["metric"] == "stopword_share", "value"].iloc[0])

    basic_stats_df = pd.DataFrame(
        [
            {
                "Показатель": "Число документов",
                "Значение": format_int(orig["Число документов"]),
            },
            {
                "Показатель": "Размер коллекции в токенах",
                "Значение": format_int(orig["Размер коллекции в токенах"]),
            },
            {
                "Показатель": "Средняя длина документа",
                "Значение": format_float(orig["Средняя длина документа"], 4),
            },
            {
                "Показатель": "Число уникальных слов",
                "Значение": format_int(orig["Число уникальных токенов"]),
            },
            {
                "Показатель": "Средняя длина слова",
                "Значение": format_float(orig["Средняя длина токена"], 4),
            },
            {
                "Показатель": "Средняя длина уникального слова",
                "Значение": format_float(orig["Средняя длина уникального токена"], 4),
            },
        ]
    )

    top_words_df = top_words_df.rename(columns={"token": "Слово", "frequency": "Частота", "rank": "Ранг", "is_stopword": "Stopword"})
    top_words_df["Stopword"] = top_words_df["Stopword"].map(lambda x: "Да" if bool(x) else "Нет")
    top_bigrams_df["Биграмма"] = top_bigrams_df["token_1"] + " " + top_bigrams_df["token_2"]
    top_bigrams_df = top_bigrams_df[["Биграмма", "frequency"]].rename(columns={"frequency": "Частота"})
    selected_examples = [
        "united states",
        "according to",
        "2010 census",
        "census bureau",
        "square mile",
        "population density",
        "housing units",
        "racial makeup",
        "per square",
        "total area",
    ]
    selected_bigrams_df = selected_bigrams_source.copy()
    selected_bigrams_df["Биграмма"] = selected_bigrams_df["token_1"] + " " + selected_bigrams_df["token_2"]
    selected_bigrams_df = (
        selected_bigrams_df[selected_bigrams_df["Биграмма"].isin(selected_examples)]
        .set_index("Биграмма")
        .loc[selected_examples]
        .reset_index()
    )
    selected_bigrams_df = selected_bigrams_df[["Биграмма", "frequency", "pmi", "both_stopwords"]].rename(
        columns={"frequency": "Частота", "pmi": "PMI", "both_stopwords": "Оба stopword"}
    )
    selected_bigrams_df["Оба stopword"] = selected_bigrams_df["Оба stopword"].map(lambda x: "Да" if bool(x) else "Нет")
    morph_view = morph_df.copy()
    for col in ["Число документов", "Размер коллекции в токенах", "Число уникальных токенов"]:
        morph_view[col] = morph_view[col].map(format_int)
    for col in ["Средняя длина документа", "Средняя длина токена", "Средняя длина уникального токена"]:
        morph_view[col] = morph_view[col].round(4)

    report = "\n".join(
        [
            "# Отчет по основной части ДЗ1",
            "",
            "## 1. Постановка задачи",
            "",
            "В основной части работы нужно проанализировать коллекцию `WikIR en1k`: посчитать базовые статистики, исследовать частотное распределение слов, проверить законы Ципфа и Хипса, проанализировать биграммы и сравнить несколько способов нормализации текста.",
            "",
            "Для анализа использовался файл `documents.csv` из коллекции `wikIR1k`. В условии указано, что документы уже очищены, приведены к нижнему регистру и токенизированы, поэтому базовая единица анализа в отчете — готовый токен.",
            "",
            "## 2. Базовые статистики коллекции",
            "",
            dataframe_to_markdown(basic_stats_df),
            "",
            "Эти числа показывают, что коллекция достаточно крупная для корпусного анализа: почти сто тысяч документов и около двадцати миллионов токенов. При этом словарь корпуса очень большой, что естественно для энциклопедического домена.",
            "",
            "## 3. Частотный словарь слов и stopwords",
            "",
            "### Топ-10 самых частотных слов",
            "",
            dataframe_to_markdown(top_words_df),
            "",
            f"Stopwords дают {format_int(stopword_occurrences)} вхождений, то есть {stopword_share:.2%} всех токенов коллекции. Не все слова из top-30 являются стоп-словами: среди частотных содержательных токенов встречаются `census`, `population`, а также числовые токены `0` и `1`.",
            "",
            f"![Топ-20 слов]({fig_rel(main_figs['top20_words'])})",
            "",
            f"![Доля stopwords]({fig_rel(main_figs['stopword_share'])})",
            "",
            "По этим результатам видно, что классический stopword list полезен, но не должен расширяться автоматически по сырым частотам корпуса: частотное слово может быть доменно значимым, а не служебным.",
            "",
            "## 4. Законы Ципфа и Хипса",
            "",
            f"![Закон Ципфа]({fig_rel(main_figs['zipf'])})",
            "",
            "На log-log графике зависимости `rank-frequency` наблюдается типичное почти линейное поведение: небольшое число слов имеет очень высокую частоту, а основная масса слов редка. Это соответствует закону Ципфа.",
            "",
            f"![Закон Хипса]({fig_rel(main_figs['heaps'])})",
            "",
            "График роста словаря показывает сублинейный рост: по мере чтения новых токенов словарь продолжает расширяться, но скорость появления новых слов постепенно падает. Это соответствует закону Хипса.",
            "",
            "## 5. Частотный словарь биграмм",
            "",
            "В коллекции получилось `4 083 973` уникальных биграммы. Верхушка списка по сырой частоте сильно засорена служебными сочетаниями, поэтому для отбора полезных биграмм нужен отдельный критерий.",
            "",
            "### Топ-10 биграмм по частоте",
            "",
            dataframe_to_markdown(top_bigrams_df),
            "",
            f"![Топ-20 биграмм]({fig_rel(main_figs['top20_bigrams'])})",
            "",
            "### Биграммы, прошедшие формальный критерий отбора",
            "",
            "В качестве рабочего критерия использовалась комбинация условий: `frequency >= 20`, биграмма не должна состоять из двух stopwords одновременно, а мера ассоциации `PMI` должна быть не меньше `3`.",
            "",
            dataframe_to_markdown(selected_bigrams_df, round_map={"PMI": 4}),
            "",
            "Такой критерий лучше отделяет реально содержательные словосочетания от служебных пар. Среди отобранных биграмм особенно полезными для индекса выглядят `united states`, `according to`, `2010 census`, `population was` и другие устойчивые сочетания с высокой ассоциацией.",
            "",
            "## 6. Сравнение способов нормализации текста",
            "",
            dataframe_to_markdown(morph_view),
            "",
            f"![Размер словаря после нормализации]({fig_rel(main_figs['morph_vocab'])})",
            "",
            f"![Средняя длина документа после нормализации]({fig_rel(main_figs['morph_doc_len'])})",
            "",
            f"Стемминг через Porter сокращает словарь сильнее всего: с {format_int(orig['Число уникальных токенов'])} до {format_int(porter['Число уникальных токенов'])}, то есть примерно на 17.31%. Лемматизация через spaCy уменьшает словарь мягче, до {format_int(spacy['Число уникальных токенов'])}, то есть примерно на 7.33%. BERT-токенизация дает принципиально другой эффект: словарь резко сжимается до {format_int(bert['Число уникальных токенов'])}, но средняя длина документа возрастает до {format_float(bert['Средняя длина документа'], 4)}, потому что слова разбиваются на подслова.",
            "",
            "## 7. Итоговые выводы",
            "",
            "- Коллекция `WikIR en1k` демонстрирует типичную для IR-корпусов структуру: тяжелый хвост распределения слов, большое число редких токенов и высокую долю stopwords.",
            "- Законы Ципфа и Хипса на данных подтверждаются визуально: это согласуется с ожидаемым поведением естественного текста в больших коллекциях.",
            "- Для биграмм одной частоты недостаточно: нужны формальные критерии на основе частоты, stopwords и ассоциативной меры вроде PMI.",
            "- Стемминг сильнее всего сжимает словарь, лемматизация делает это аккуратнее, а BERT-токенизация не является прямой заменой словарной нормализации, потому что переходит к subword-представлению.",
            "",
        ]
    )

    (REPORTS_DIR / "dz1_main_report.md").write_text(report, encoding="utf-8")


def build_authors_report(figs: dict[str, Path]) -> None:
    stats_df = pd.read_csv(ROOT / "artifacts" / "additional_authors" / "authors_basic_stats.csv")
    lemma_stats_df = pd.read_csv(ROOT / "artifacts" / "additional_authors" / "authors_lemmatized_stats.csv")
    tolstoy_df = pd.read_csv(ROOT / "artifacts" / "additional_authors" / "tolstoy_characteristic_words.csv").head(10)
    dost_df = pd.read_csv(ROOT / "artifacts" / "additional_authors" / "dostoevsky_characteristic_words.csv").head(10)
    tolstoy_lemma_df = pd.read_csv(ROOT / "artifacts" / "additional_authors" / "tolstoy_characteristic_lemmas.csv").head(10)
    dost_lemma_df = pd.read_csv(ROOT / "artifacts" / "additional_authors" / "dostoevsky_characteristic_lemmas.csv").head(10)
    tolstoy_df = tolstoy_df.rename(columns={"token": "Слово", "score": "Скор", "count_a": "Частота у Толстого", "count_b": "Частота у Достоевского"})
    dost_df = dost_df.rename(columns={"token": "Слово", "score": "Скор", "count_a": "Частота у Толстого", "count_b": "Частота у Достоевского"})
    tolstoy_lemma_df = tolstoy_lemma_df.rename(columns={"token": "Лемма", "score": "Скор", "count_a": "Частота у Толстого", "count_b": "Частота у Достоевского"})
    dost_lemma_df = dost_lemma_df.rename(columns={"token": "Лемма", "score": "Скор", "count_a": "Частота у Толстого", "count_b": "Частота у Достоевского"})
    stats_view = stats_df.copy()
    stats_view["tokens"] = stats_view["tokens"].map(format_int)
    stats_view["unique_tokens"] = stats_view["unique_tokens"].map(format_int)
    stats_view["type_token_ratio"] = stats_view["type_token_ratio"].round(4)
    stats_view["avg_token_length"] = stats_view["avg_token_length"].round(4)
    stats_view = stats_view.rename(columns={"author": "Автор", "tokens": "Токены", "unique_tokens": "Уникальные слова", "type_token_ratio": "Type-token ratio", "avg_token_length": "Средняя длина токена"})
    lemma_view = lemma_stats_df.copy()
    for col in ["tokens", "unique_wordforms", "unique_lemmas", "reduction_abs"]:
        lemma_view[col] = lemma_view[col].map(format_int)
    lemma_view["reduction_pct"] = lemma_view["reduction_pct"].map(lambda x: round(float(x), 2))
    lemma_view = lemma_view.rename(
        columns={
            "author": "Автор",
            "tokens": "Токены",
            "unique_wordforms": "Уникальные словоформы",
            "unique_lemmas": "Уникальные леммы",
            "reduction_abs": "Сокращение словаря",
            "reduction_pct": "Сокращение, %",
        }
    )

    report = "\n".join(
        [
            "# Отчет по дополнительной задаче: Толстой и Достоевский",
            "",
            "## 1. Постановка задачи",
            "",
            "В этой задаче нужно сравнить словарное поведение двух авторов на материале `Войны и мира` Льва Толстого и `Братьев Карамазовых` Федора Достоевского. Нас интересуют три вещи: рост словаря в исходных текстах, влияние случайного перемешивания токенов и наиболее характерные слова каждого автора.",
            "",
            "Важно, что анализ проводится не по всему творчеству авторов, а по выбранным романам. Поэтому результаты ниже отражают одновременно и авторскую манеру, и лексику конкретных произведений.",
            "",
            "## 2. Базовые статистики",
            "",
            dataframe_to_markdown(stats_view),
            "",
            "В абсолютном размере словарь Толстого больше, но относительная лексическая насыщенность по `type-token ratio` выше у Достоевского.",
            "",
            "## 3. Рост словаря в исходных текстах",
            "",
            f"![Рост словаря в исходных текстах]({fig_rel(figs['growth_original'])})",
            "",
            "На исходных текстах кривые растут неравномерно: это означает, что новые слова приходят не случайным равномерным потоком, а блоками, связанными со сценами, персонажами и тематическими переходами.",
            "",
            "## 4. Рост словаря после перемешивания",
            "",
            f"![Рост словаря после перемешивания]({fig_rel(figs['growth_shuffled'])})",
            "",
            "После случайного перемешивания токенов рост словаря становится более гладким. На отметке около `100 000` токенов shuffled-версия дает словарь больше на `1663` слова у Толстого и на `547` слов у Достоевского. Максимальный разрыв между shuffled и исходной кривой достигает `1819` и `937` слов соответственно.",
            "",
            "## 5. Наиболее характерные слова авторов",
            "",
            f"![Характерные слова Толстого]({fig_rel(figs['tolstoy_words'])})",
            "",
            dataframe_to_markdown(tolstoy_df, round_map={"Скор": 4}),
            "",
            f"![Характерные слова Достоевского]({fig_rel(figs['dostoevsky_words'])})",
            "",
            dataframe_to_markdown(dost_df, round_map={"Скор": 4}),
            "",
            "После очистки технического шума у Толстого особенно выделяются слова `пьер`, `пьера`, `ростов`, `княжна`, `князь`, `граф`, `кутузов`, а у Достоевского — `алеша`, `митя`, `смердяков`, `грушенька`, `старец`, `карамазов`, `ракитин`.",
            "",
            "## 6. Сравнение словарей после лемматизации",
            "",
            "Дополнительно был проведен анализ на уровне лемм. Это позволяет убрать различия между словоформами и посмотреть, насколько сильно морфология раздувает исходный словарь.",
            "",
            dataframe_to_markdown(lemma_view),
            "",
            "После лемматизации словарь Толстого сокращается сильнее в абсолютном выражении, потому что у него изначально больше словоформ. При этом у обоих авторов заметная часть различий между словоформами схлопывается в общие леммы.",
            "",
            "### Наиболее характерные леммы Толстого",
            "",
            dataframe_to_markdown(tolstoy_lemma_df, round_map={"Скор": 4}),
            "",
            "### Наиболее характерные леммы Достоевского",
            "",
            dataframe_to_markdown(dost_lemma_df, round_map={"Скор": 4}),
            "",
            "После лемматизации видно, что формы типа `пьер` и `пьера` или `алеша` и `алеши` схлопываются, поэтому сравнение становится немного ближе к тематическому и понятийному уровню, а не к чистой словоформенной поверхности текста.",
            "",
            "При этом автоматическая лемматизация художественного текста и имен собственных неидеальна: для некоторых форм появляются артефакты вроде `алеш` или `грушенек`. Поэтому лемматизированный анализ здесь полезен именно как дополнительное сравнение словаря, а не как абсолютно точное представление авторской лексики.",
            "",
            "## 7. Итоговые выводы",
            "",
            "- Словарь Толстого в выбранных текстах больше в абсолютном размере, но у Достоевского выше относительная лексическая насыщенность.",
            "- Перемешивание токенов меняет форму кривой роста словаря: это подтверждает, что распределение слов по тексту неслучайно.",
            "- Лемматизация заметно сокращает словарь у обоих авторов и показывает, какая часть различий объясняется морфологическими вариантами слов.",
            "- Списки характерных слов и лемм хорошо отделяют лексические миры романов, но отражают не только авторский стиль, а еще и содержание конкретных произведений.",
            "",
        ]
    )
    (REPORTS_DIR / "dz1_additional_authors_report.md").write_text(report, encoding="utf-8")


def build_coat_report(figs: dict[str, Path]) -> None:
    splits_df = pd.read_csv(ROOT / "artifacts" / "additional_generated_texts" / "coat_splits_stats.csv")
    class_share_df = pd.read_csv(ROOT / "artifacts" / "additional_generated_texts" / "coat_train_class_share.csv")
    length_df = pd.read_csv(ROOT / "artifacts" / "additional_generated_texts" / "coat_text_length_stats.csv")
    lexical_df = pd.read_csv(ROOT / "artifacts" / "additional_generated_texts" / "coat_lexical_feature_means.csv")
    results_df = pd.read_csv(ROOT / "artifacts" / "additional_generated_texts" / "coat_baseline_results.csv")
    pred_share_df = pd.read_csv(ROOT / "artifacts" / "additional_generated_texts" / "coat_test_prediction_share.csv")

    splits_view = splits_df.copy()
    splits_view["rows"] = splits_view["rows"].map(format_int)
    splits_view = splits_view.rename(columns={"split": "Сплит", "rows": "Число текстов"})
    class_view = class_share_df.copy()
    class_view["share"] = class_view["share"].round(3)
    class_view = class_view.rename(columns={"label": "Класс", "share": "Доля"})
    length_view = length_df.copy().round(3)
    length_view = length_view.rename(columns={"split": "Сплит", "avg_char_len": "Средняя длина в символах", "avg_word_len": "Средняя длина в словах", "median_word_len": "Медианная длина в словах"})
    lexical_view = lexical_df.copy().round(4)
    lexical_view = lexical_view.rename(columns={"label": "Класс", "token_count": "token_count", "unique_count": "unique_count", "avg_token_length": "avg_token_length", "type_token_ratio": "type_token_ratio", "hapax_ratio": "hapax_ratio", "digit_share": "digit_share"})
    results_view = results_df.copy()
    for col in results_view.columns:
        if col != "Модель":
            results_view[col] = results_view[col].map(lambda x: "—" if pd.isna(x) else round(float(x), 4))
    pred_share_view = pred_share_df.copy().round(4)
    pred_share_view = pred_share_view.rename(columns={"model": "Модель", "predicted_label_0_share": "Доля предсказаний класса 0", "predicted_label_1_share": "Доля предсказаний класса 1"})

    report = "\n".join(
        [
            "# Отчет по дополнительной задаче: детекция искусственно сгенерированных текстов",
            "",
            "## 1. Постановка задачи",
            "",
            "В этой задаче требуется различать тексты, написанные человеком, и машинно-сгенерированные тексты на материале корпуса `CoAT` в конфигурации `binary`.",
            "",
            "Работа построена как baseline-анализ: сначала рассматриваются простые свойства датасета и текста, затем обучаются две модели — на лексических агрегатах и на `TF-IDF` признаках.",
            "",
            "## 2. Структура датасета",
            "",
            dataframe_to_markdown(splits_view),
            "",
            dataframe_to_markdown(class_view),
            "",
            f"![Распределение классов]({fig_rel(figs['class_distribution'])})",
            "",
            "Обучающая выборка полностью сбалансирована: в `train` одинаковое число текстов обоих классов.",
            "",
            "## 3. Длины текстов и простые признаки",
            "",
            dataframe_to_markdown(length_view),
            "",
            f"![Распределение длины текстов]({fig_rel(figs['word_len_hist'])})",
            "",
            "Средние лексические признаки по `train`:",
            "",
            dataframe_to_markdown(lexical_view),
            "",
            "У класса `1` тексты в среднем длиннее и показывают более высокие `unique_count`, `type_token_ratio` и `hapax_ratio`. Значит, даже простые агрегированные характеристики уже содержат полезный сигнал для классификации.",
            "",
            "## 4. Baseline-модели",
            "",
            dataframe_to_markdown(results_view),
            "",
            f"![Сравнение baseline-моделей]({fig_rel(figs['results_comparison'])})",
            "",
            f"![Confusion matrix: лексические признаки]({fig_rel(figs['confusion_lexical'])})",
            "",
            f"![Confusion matrix: TF-IDF]({fig_rel(figs['confusion_tfidf'])})",
            "",
            "Модель на простых лексических признаках дает `Val macro F1 = 0.6203`, а модель `TF-IDF + LogisticRegression` заметно сильнее и достигает `0.7157`.",
            "",
            "## 5. Особенность test-сплита",
            "",
            "У опубликованного `test`-сплита в конфигурации `binary` нет истинных меток: там во всех строках `label = -1`. Поэтому корректно считать тестовые `accuracy` и `F1` здесь нельзя; вместо этого можно анализировать только предсказания моделей на test.",
            "",
            dataframe_to_markdown(pred_share_view),
            "",
            "На неразмеченном `test` модель на лексических признаках относит примерно `44.26%` текстов к классу `1`, а `TF-IDF`-модель — примерно `47.16%`.",
            "",
            "## 6. Итоговые выводы",
            "",
            "- Простые статистические признаки текста дают рабочий baseline, но заметно уступают частотной модели на словах и биграммах.",
            "- Для задачи детекции важны не только длина и разнообразие словаря, но и конкретное содержимое текста.",
            "- Даже линейная модель на `TF-IDF` показывает, что у human и generated текстов есть устойчивые различия, заметные уже без сложных нейросетевых моделей.",
            "- При интерпретации результатов важно учитывать, что test-сплит в текущей конфигурации не размечен.",
            "",
        ]
    )
    (REPORTS_DIR / "dz1_additional_generated_texts_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    main_figs = build_main_figures()
    additional_figs = copy_additional_figures()
    build_main_report(main_figs)
    build_authors_report(additional_figs["authors"])
    build_coat_report(additional_figs["coat"])
    print(f"Markdown-отчеты сохранены в {REPORTS_DIR}")


if __name__ == "__main__":
    main()
