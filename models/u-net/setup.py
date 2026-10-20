"""Dataset preparation for the U-Net text segmentation model.

Downloads the ICDAR2019-SROIE receipt dataset from the Hugging Face Hub and
exports its images to disk. Every image is normalised to a fixed width so
that text has a similar scale across the dataset: full-page A4 scans are
first cropped to the receipt area, and then all images are resized to
``TARGET_WIDTH`` keeping their aspect ratio. For the training split, it also
generates segmentation masks from the text bounding boxes at the final
resolution, both as class-index masks for training and as high-contrast
masks for visual inspection.
"""

from pathlib import Path

import cv2
import numpy as np

from datasets import load_dataset

DATASET_NAME = "jsdnrs/ICDAR2019-SROIE"
IMAGES_PATH = Path("./datasets/u-net")

TARGET_WIDTH = 640
A4_MIN_WIDTH = 4000
CROP_MARGIN = 50

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


def crop_to_bboxes(img, bboxes, margin=CROP_MARGIN):
    """Crop an image to the area covered by its bounding boxes.

    Used for full-page A4 scans, where the receipt only covers part of the
    page. The crop is the union of all boxes plus a margin, clipped to the
    image limits, and the boxes are shifted to the new origin.

    Args:
        img (np.ndarray): Image to crop.
        bboxes (np.ndarray): ``(N, 4)`` array of boxes in
            ``(x_min, y_min, x_max, y_max)`` pixel coordinates.
        margin (int, optional): Extra pixels kept around the boxes.
            Defaults to ``CROP_MARGIN``.

    Returns:
        tuple[np.ndarray, np.ndarray]: The cropped image and the shifted boxes.
    """
    h, w = img.shape[:2]
    x0 = int(max(bboxes[:, 0].min() - margin, 0))
    y0 = int(max(bboxes[:, 1].min() - margin, 0))
    x1 = int(min(bboxes[:, 2].max() + margin, w))
    y1 = int(min(bboxes[:, 3].max() + margin, h))

    return img[y0:y1, x0:x1], bboxes - np.array([x0, y0, x0, y0])


def resize_to_width(img, bboxes, width=TARGET_WIDTH):
    """Resize an image to a fixed width keeping its aspect ratio.

    The boxes are scaled by the same factor and rounded to integer pixels.

    Args:
        img (np.ndarray): Image to resize.
        bboxes (np.ndarray): ``(N, 4)`` array of boxes in
            ``(x_min, y_min, x_max, y_max)`` pixel coordinates.
        width (int, optional): Target width in pixels. Defaults to
            ``TARGET_WIDTH``.

    Returns:
        tuple[np.ndarray, np.ndarray]: The resized image and the scaled boxes
        as integers.
    """
    h, w = img.shape[:2]
    scale = width / w
    height = round(h * scale)

    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    img = cv2.resize(img, (width, height), interpolation=interpolation)

    return img, np.round(bboxes * scale).astype(np.int32)


def prepare_item(item):
    """Normalise a dataset item to the U-Net input scale.

    A4 scans (wider than ``A4_MIN_WIDTH``) are cropped to the receipt area
    first, and then every image is resized to ``TARGET_WIDTH``.

    Args:
        item (dict): Dataset item with ``image`` and ``bboxes`` fields.

    Returns:
        tuple[np.ndarray, np.ndarray]: The image in BGR order, ready for
        ``cv2.imwrite``, and its boxes in the same coordinates.
    """
    img = np.array(item["image"])
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    bboxes = np.array(item["bboxes"], dtype=np.float64).reshape(-1, 4)

    if img.shape[1] > A4_MIN_WIDTH:
        img, bboxes = crop_to_bboxes(img, bboxes)

    return resize_to_width(img, bboxes)


def save_images(dataset, mode="train"):
    """Export a dataset split to disk as PNG files.

    Images are normalised with ``prepare_item`` and written to
    ``IMAGES_PATH / mode``, named sequentially as ``img_XXXXXX.png``. When
    ``mode`` is ``"train"``, two masks are also generated for every image
    under the same file name and at the same resolution:

        - ``train_masks``: class-index masks (0, 1, 2) used as training
          targets.
        - ``train_masks_visible``: the same masks rescaled to 0, 128 and 255
          for visual inspection.

    Output directories are created if they do not already exist.

    Args:
        dataset (datasets.Dataset): Dataset split whose items provide the
            ``image`` and ``bboxes`` fields.
        mode (str, optional): Name of the split, used as the output
            subdirectory. Masks are only generated for ``"train"``.
            Defaults to ``"train"``.
    """
    path = IMAGES_PATH / mode
    path.mkdir(parents=True, exist_ok=True)

    count = 0
    for item in dataset:
        img, bboxes = prepare_item(item)

        m_path = IMAGES_PATH / f"{mode}_masks"
        m_path.mkdir(parents=True, exist_ok=True)

        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        mask = paint_bbox(mask, bboxes)

        cv2.imwrite(m_path / f"img_{count:06d}.png", mask)

        m_path_visible = IMAGES_PATH / f"{mode}_masks_visible"

        m_path_visible.mkdir(parents=True, exist_ok=True)

        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        mask = paint_bbox(mask, bboxes, fill=128, border=255)

        cv2.imwrite(m_path_visible / f"img_{count:06d}.png", mask)


        cv2.imwrite(path / f"img_{count:06d}.png", img)

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


