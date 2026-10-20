"""Dataset preparation for the U-Net text segmentation model.

Downloads the ICDAR2019-SROIE receipt dataset from the Hugging Face Hub and
exports its images to disk. For the training split, it also generates
segmentation masks from the text bounding boxes, both as class-index masks
for training and as high-contrast masks for visual inspection.
"""

from pathlib import Path

import cv2
import os
import random
import numpy as np

from datasets import load_dataset

DATASET_NAME = "jsdnrs/ICDAR2019-SROIE"
IMAGES_PATH = Path("./datasets/u-net")

def paint_bbox(img, bboxes, fill = 1, border = 2):
    """Paint bounding boxes onto a segmentation mask.

    All boxes are first drawn as filled rectangles and then outlined, so
    borders always take precedence over fills where boxes overlap.

    The resulting mask uses the following classes:
        - 0: background
        - 1: fill (box interior)
        - 2: border (box outline)

    Args:
        img (np.ndarray): Single-channel mask to draw on, typically
            initialised to zeros (background).
        bboxes (Iterable[Sequence[int]]): Bounding boxes in
            ``(x_min, y_min, x_max, y_max)`` pixel coordinates.
        fill (int, optional): Value used for the box interiors. Defaults to 1.
        border (int, optional): Value used for the box outlines, drawn with a
            thickness of 2 pixels. Defaults to 2.

    Returns:
        np.ndarray: The mask with all bounding boxes painted.
    """
    for bbox in bboxes:
        img = cv2.rectangle(img=img,pt1=(bbox[0], bbox[1]), pt2=(bbox[2], bbox[3]), color=fill, thickness=cv2.FILLED)

    for bbox in bboxes:
        img = cv2.rectangle(img=img,pt1=(bbox[0], bbox[1]), pt2=(bbox[2], bbox[3]), color=border, thickness=2)

    return img


def save_images(dataset, mode="train"):
    """Export a dataset split to disk as PNG files.

    Images are written to ``IMAGES_PATH / mode`` and named sequentially as
    ``img_XXXXXX.png``. When ``mode`` is ``"train"``, two masks are also
    generated for every image under the same file name:

        - ``train_masks``: class-index masks (0, 1, 2) used as training
          targets.
        - ``train_masks_visible``: the same masks rescaled to 0, 128 and 255
          for visual inspection.

    Output directories are created if they do not already exist.

    Args:
        dataset (datasets.Dataset): Dataset split whose items provide the
            ``image``, ``image_size`` and ``bboxes`` fields.
        mode (str, optional): Name of the split, used as the output
            subdirectory. Masks are only generated for ``"train"``.
            Defaults to ``"train"``.
    """
    path = IMAGES_PATH / mode
    path.mkdir(parents=True, exist_ok=True)

    count = 0
    for item in dataset:
        img = item["image"]

        if mode == "train":
            m_path = IMAGES_PATH / "train_masks"
            m_path.mkdir(parents=True, exist_ok=True)

            mask = np.zeros((int(item["image_size"]["height"]), int(item["image_size"]["width"])), dtype=np.uint8)
            mask = paint_bbox(mask, item["bboxes"])

            cv2.imwrite(m_path / f"img_{count:06d}.png", mask)

            m_path_visible = IMAGES_PATH / "train_masks_visible"

            m_path_visible.mkdir(parents=True, exist_ok=True)

            mask = np.zeros((int(item["image_size"]["height"]), int(item["image_size"]["width"])), dtype=np.uint8)
            mask = paint_bbox(mask, item["bboxes"], fill=128, border=255)

            cv2.imwrite(m_path_visible / f"img_{count:06d}.png", mask)


        cv2.imwrite(path / f"img_{count:06d}.png", np.array(img))

        count += 1

def main():
    """Download the dataset and export the U-Net training data.

    Loads the ICDAR2019-SROIE dataset and saves the train split (images and
    masks) and the test split (images only) under ``IMAGES_PATH``.
    """
    dataset = load_dataset(DATASET_NAME)

    train = dataset["train"]
    test = dataset["test"]

    save_images(train, "train")
    save_images(test, "test")


if __name__ == "__main__":
    main()


