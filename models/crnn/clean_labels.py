"""Cleaning utilities for OCR label text and vocabulary extraction."""

import re

import pandas as pd

LABEL_PATTERN = re.compile(r"^[A-Za-z0-9\s€$.,\-/:%#()&'*ÁÉÍÓÚáéíóúÑñ]{1,35}$")


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Remove invalid rows and normalize label text.

    Drops rows with a missing label or image path, discards labels that
    don't match `LABEL_PATTERN` (disallowed characters or too long), and
    collapses repeated whitespace in the remaining labels.

    Args:
        df: DataFrame with at least "label" and "image_path" columns.

    Returns:
        A new, cleaned DataFrame.
    """
    df = df.dropna(subset=["label", "image_path"]).copy()
    df["label"] = df["label"].astype(str)

    df = df[df["label"].str.match(LABEL_PATTERN)].copy()
    df["label"] = df["label"].str.replace(r"\s+", " ", regex=True)

    return df


def extract_vocab(*dataframes: pd.DataFrame) -> str:
    """Return the sorted set of unique characters found across label columns.

    Args:
        *dataframes: One or more DataFrames with a "label" column.

    Returns:
        A string containing every distinct character, sorted.
    """
    chars = set()
    for df in dataframes:
        chars.update("".join(df["label"]))
    return "".join(sorted(chars))


if __name__ == "__main__":
    train_df = clean_dataframe(pd.read_csv("train.csv"))
    test_df = clean_dataframe(pd.read_csv("test.csv"))

    train_df.to_csv("train.csv", index=False)
    test_df.to_csv("test.csv", index=False)

    vocab = extract_vocab(train_df, test_df)
    with open("vocab.txt", "w", encoding="utf-8") as f:
        f.write(vocab)

    print(f"Train: {len(train_df)} muestras | Test: {len(test_df)} muestras")
    print(f"Vocabulario ({len(vocab)} caracteres): {vocab!r}")
