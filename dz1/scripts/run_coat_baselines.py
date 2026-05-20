from __future__ import annotations

import os
import re
from collections import Counter
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

matplotlib.use("Agg")
plt.style.use("ggplot")
pd.set_option("display.max_colwidth", 200)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "artifacts" / "additional_generated_texts"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TOKEN_PATTERN = re.compile(r"[а-яёa-z]+", flags=re.IGNORECASE)


def tokenize_text(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_PATTERN.findall(text)]


def extract_lexical_features(text: str) -> dict[str, float]:
    tokens = tokenize_text(text)
    token_count = len(tokens)
    unique_count = len(set(tokens))
    avg_token_length = float(np.mean([len(token) for token in tokens])) if tokens else 0.0
    type_token_ratio = unique_count / token_count if token_count else 0.0
    hapax_ratio = sum(1 for _, count in Counter(tokens).items() if count == 1) / unique_count if unique_count else 0.0
    digit_share = sum(char.isdigit() for char in text) / len(text) if text else 0.0
    return {
        "token_count": token_count,
        "unique_count": unique_count,
        "avg_token_length": avg_token_length,
        "type_token_ratio": type_token_ratio,
        "hapax_ratio": hapax_ratio,
        "digit_share": digit_share,
    }


def metrics_row(name: str, y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float | str]:
    return {
        "Модель": name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Macro F1": f1_score(y_true, y_pred, average="macro"),
    }


def plot_class_distribution(train_df: pd.DataFrame, output_path: Path) -> None:
    value_counts = train_df["label"].value_counts().sort_index()
    plt.figure(figsize=(7, 5))
    plt.bar(value_counts.index.astype(str), value_counts.values, color=["#4C78A8", "#E45756"])
    plt.title("Распределение классов в train")
    plt.xlabel("label")
    plt.ylabel("Число текстов")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_word_length_distribution(train_df: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(10, 5))
    plt.hist(train_df["word_len"], bins=50, color="#72B7B2")
    plt.title("Распределение длины текстов в словах")
    plt.xlabel("Число слов")
    plt.ylabel("Частота")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_results(results_df: pd.DataFrame, output_path: Path) -> None:
    plot_df = results_df.set_index("Модель")[["Val macro F1", "Test macro F1"]]
    ax = plot_df.plot(kind="bar", figsize=(9, 5), color=["#4C78A8", "#F58518"])
    ax.set_title("Сравнение baseline-моделей")
    ax.set_ylabel("Macro F1")
    ax.set_xlabel("")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_confusion_matrix(cm: np.ndarray, title: str, output_path: Path) -> None:
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, cmap="Blues")
    plt.title(title)
    plt.xlabel("Предсказание")
    plt.ylabel("Истинный класс")
    plt.xticks([0, 1], ["0", "1"])
    plt.yticks([0, 1], ["0", "1"])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, int(cm[i, j]), ha="center", va="center", color="black")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def build_summary(
    splits_stats_df: pd.DataFrame,
    class_share_df: pd.DataFrame,
    text_length_stats_df: pd.DataFrame,
    lexical_feature_means_df: pd.DataFrame,
    results_df: pd.DataFrame,
    test_prediction_share_df: pd.DataFrame,
    has_test_labels: bool,
) -> str:
    best_row = results_df.sort_values("Val macro F1", ascending=False).iloc[0]
    return "\n".join(
        [
            "# Дополнительная задача: детекция искусственно сгенерированных текстов",
            "",
            "## Ключевые результаты",
            "",
            f"- Размеры сплитов: train={int(splits_stats_df.loc[splits_stats_df['split'] == 'train', 'rows'].iloc[0])}, validation={int(splits_stats_df.loc[splits_stats_df['split'] == 'validation', 'rows'].iloc[0])}, test={int(splits_stats_df.loc[splits_stats_df['split'] == 'test', 'rows'].iloc[0])}.",
            f"- На train доля класса 0 равна {class_share_df.loc[class_share_df['label'] == 0, 'share'].iloc[0]:.3f}, доля класса 1 равна {class_share_df.loc[class_share_df['label'] == 1, 'share'].iloc[0]:.3f}.",
            f"- Лучшая baseline-модель на validation: {best_row['Модель']} с Val macro F1 = {best_row['Val macro F1']:.4f}.",
            "- У сплита test в опубликованном датасете нет истинных меток: там label = -1, поэтому полноценные test-метрики считать нельзя." if not has_test_labels else "- Для test доступны истинные метки, поэтому ниже можно интерпретировать и test-метрики.",
            "",
            "## Интерпретация",
            "",
            "- Простые лексические признаки дают рабочий ориентир, но заметно уступают модели на TF-IDF признаках.",
            "- Это означает, что для различения human и generated текстов важны не только агрегированные свойства длины и разнообразия слов, но и сами слова и их сочетания.",
            "- Даже базовая линейная модель на мешке слов уже подхватывает устойчивые шаблоны генерации.",
            "",
            "## Распределение предсказаний на unlabeled test",
            "",
            test_prediction_share_df.to_markdown(index=False),
            "",
            "## Средние лексические признаки по train",
            "",
            lexical_feature_means_df.to_markdown(index=False),
            "",
            "## Сводка по длинам текстов",
            "",
            text_length_stats_df.to_markdown(index=False),
        ]
    )


def main() -> None:
    dataset = load_dataset("RussianNLP/coat", "binary")

    train_df = dataset["train"].to_pandas()
    val_df = dataset["validation"].to_pandas() if "validation" in dataset else dataset["val"].to_pandas()
    test_df = dataset["test"].to_pandas()
    has_test_labels = sorted(test_df["label"].dropna().unique().tolist()) != [-1]

    splits_stats_df = pd.DataFrame(
        [
            {"split": "train", "rows": len(train_df)},
            {"split": "validation", "rows": len(val_df)},
            {"split": "test", "rows": len(test_df)},
        ]
    )
    splits_stats_df.to_csv(OUT_DIR / "coat_splits_stats.csv", index=False)

    class_share_df = (
        train_df["label"].value_counts(normalize=True).sort_index().rename_axis("label").reset_index(name="share")
    )
    class_share_df.to_csv(OUT_DIR / "coat_train_class_share.csv", index=False)

    for df in [train_df, val_df, test_df]:
        df["char_len"] = df["text"].str.len()
        df["word_len"] = df["text"].str.split().str.len()

    text_length_stats_df = pd.DataFrame(
        [
            {
                "split": name,
                "avg_char_len": df["char_len"].mean(),
                "avg_word_len": df["word_len"].mean(),
                "median_word_len": df["word_len"].median(),
            }
            for name, df in [("train", train_df), ("validation", val_df), ("test", test_df)]
        ]
    )
    text_length_stats_df.to_csv(OUT_DIR / "coat_text_length_stats.csv", index=False)

    plot_class_distribution(train_df, OUT_DIR / "coat_train_class_distribution.png")
    plot_word_length_distribution(train_df, OUT_DIR / "coat_train_word_len_hist.png")

    train_features_df = pd.DataFrame([extract_lexical_features(text) for text in train_df["text"]])
    val_features_df = pd.DataFrame([extract_lexical_features(text) for text in val_df["text"]])
    test_features_df = pd.DataFrame([extract_lexical_features(text) for text in test_df["text"]])

    lexical_feature_means_df = (
        pd.concat([train_df[["label"]], train_features_df], axis=1).groupby("label", as_index=False).mean()
    )
    lexical_feature_means_df.to_csv(OUT_DIR / "coat_lexical_feature_means.csv", index=False)

    lex_clf = LogisticRegression(max_iter=1000, random_state=42)
    lex_clf.fit(train_features_df, train_df["label"])
    val_pred_lex = lex_clf.predict(val_features_df)
    test_pred_lex = lex_clf.predict(test_features_df)

    tfidf_vectorizer = TfidfVectorizer(
        lowercase=True,
        analyzer="word",
        ngram_range=(1, 2),
        min_df=3,
        max_features=50_000,
    )
    X_train_tfidf = tfidf_vectorizer.fit_transform(train_df["text"])
    X_val_tfidf = tfidf_vectorizer.transform(val_df["text"])
    X_test_tfidf = tfidf_vectorizer.transform(test_df["text"])

    tfidf_clf = LogisticRegression(max_iter=1000, random_state=42)
    tfidf_clf.fit(X_train_tfidf, train_df["label"])
    val_pred_tfidf = tfidf_clf.predict(X_val_tfidf)
    test_pred_tfidf = tfidf_clf.predict(X_test_tfidf)

    results_rows = [
        {
            "Модель": "Лексические признаки",
            "Val accuracy": accuracy_score(val_df["label"], val_pred_lex),
            "Val macro F1": f1_score(val_df["label"], val_pred_lex, average="macro"),
        },
        {
            "Модель": "TF-IDF + LogisticRegression",
            "Val accuracy": accuracy_score(val_df["label"], val_pred_tfidf),
            "Val macro F1": f1_score(val_df["label"], val_pred_tfidf, average="macro"),
        },
    ]
    if has_test_labels:
        results_rows[0]["Test accuracy"] = accuracy_score(test_df["label"], test_pred_lex)
        results_rows[0]["Test macro F1"] = f1_score(test_df["label"], test_pred_lex, average="macro")
        results_rows[1]["Test accuracy"] = accuracy_score(test_df["label"], test_pred_tfidf)
        results_rows[1]["Test macro F1"] = f1_score(test_df["label"], test_pred_tfidf, average="macro")
    else:
        results_rows[0]["Test accuracy"] = np.nan
        results_rows[0]["Test macro F1"] = np.nan
        results_rows[1]["Test accuracy"] = np.nan
        results_rows[1]["Test macro F1"] = np.nan

    results_df = pd.DataFrame(results_rows)
    results_df.to_csv(OUT_DIR / "coat_baseline_results.csv", index=False)

    test_predictions_df = pd.DataFrame(
        {
            "id": test_df["id"],
            "pred_lexical": test_pred_lex,
            "pred_tfidf": test_pred_tfidf,
        }
    )
    test_predictions_df.to_csv(OUT_DIR / "coat_test_predictions.csv", index=False)
    test_prediction_share_df = pd.DataFrame(
        [
            {
                "model": "Лексические признаки",
                "predicted_label_0_share": float((test_pred_lex == 0).mean()),
                "predicted_label_1_share": float((test_pred_lex == 1).mean()),
            },
            {
                "model": "TF-IDF + LogisticRegression",
                "predicted_label_0_share": float((test_pred_tfidf == 0).mean()),
                "predicted_label_1_share": float((test_pred_tfidf == 1).mean()),
            },
        ]
    )
    test_prediction_share_df.to_csv(OUT_DIR / "coat_test_prediction_share.csv", index=False)

    plot_results(results_df, OUT_DIR / "coat_results_comparison.png")
    plot_confusion_matrix(
        confusion_matrix(val_df["label"], val_pred_lex),
        "Confusion matrix: validation, лексические признаки",
        OUT_DIR / "coat_confusion_lexical.png",
    )
    plot_confusion_matrix(
        confusion_matrix(val_df["label"], val_pred_tfidf),
        "Confusion matrix: validation, TF-IDF",
        OUT_DIR / "coat_confusion_tfidf.png",
    )

    summary = build_summary(
        splits_stats_df,
        class_share_df,
        text_length_stats_df,
        lexical_feature_means_df,
        results_df,
        test_prediction_share_df,
        has_test_labels,
    )
    (OUT_DIR / "coat_summary.md").write_text(summary, encoding="utf-8")

    print(splits_stats_df.to_string(index=False))
    print()
    print(results_df.to_string(index=False))
    print()
    print(f"Артефакты сохранены в {OUT_DIR}")


if __name__ == "__main__":
    main()
