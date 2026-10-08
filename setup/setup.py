"""Builds the OCR training data: downloads the SROIE dataset, crops every
labeled word box into its own image, cleans the label text, and writes the
resulting train/test CSVs and character vocabulary.
"""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

from clean_labels import clean_dataframe, extract_vocab

DATASET_NAME = "jsdnrs/ICDAR2019-SROIE"
CROPS_ROOT = Path("./dataset_crops")
MIN_CROP_SIZE = 5


def crop_split(split_name: str, split_data) -> pd.DataFrame:
    """Crop every labeled word box in a dataset split and save it to disk.

    Args:
        split_name: Name of the split ("train" or "test"); also used as
            the output subdirectory under `dataset_crops/`.
        split_data: A HuggingFace dataset split with "image", "words" and
            "bboxes" fields.

    Returns:
        A DataFrame with one row per cropped word: "image_path", "label".
    """
    output_dir = CROPS_ROOT / split_name
    output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    crop_count = 0

    for item in tqdm(split_data, desc=f"Procesando {split_name}"):
        image = cv2.cvtColor(np.array(item["image"]), cv2.COLOR_RGB2BGR)

        for box, text in zip(item.get("bboxes", []), item.get("words", [])):
            label = str(text).strip()
            if not label:
                continue

            try:
                x_min, y_min, x_max, y_max = map(int, box[:4])
                crop = image[y_min:y_max, x_min:x_max]
            except (TypeError, ValueError):
                continue

            if crop.shape[0] < MIN_CROP_SIZE or crop.shape[1] < MIN_CROP_SIZE:
                continue

            crop_path = output_dir / f"crop_{crop_count:06d}.png"
            cv2.imwrite(str(crop_path), crop)

            records.append({"image_path": str(crop_path), "label": label})
            crop_count += 1

    return pd.DataFrame(records)


def main():
    """Run the full pipeline and write train.csv, test.csv and vocab.txt."""
    dataset = load_dataset(DATASET_NAME)

    train_df = clean_dataframe(crop_split("train", dataset["train"]))
    test_df = clean_dataframe(crop_split("test", dataset["test"]))

    train_df.to_csv("train.csv", index=False)
    test_df.to_csv("test.csv", index=False)

    vocab = extract_vocab(train_df, test_df)
    with open("vocab.txt", "w", encoding="utf-8") as f:
        f.write(vocab)

    print(f"Train: {len(train_df)} muestras | Test: {len(test_df)} muestras")
    print(f"Vocabulario ({len(vocab)} caracteres): {vocab!r}")


if __name__ == "__main__":
    main()
